import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.modules import *  # 기존 DoubleConv, SEBlock 등

# CBAM: Channel + Spatial Attention [web:30][web:33]
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=3):
        super().__init__()
        self.ca = ChannelAttention(channels, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class SuperEnhancedAFMSUNet2D(nn.Module):
    def __init__(self, out_channels, context=0, encoder_block=SEBlock):
        super().__init__()
        self.context = context
        self.depth = 2 * context + 1
        in_channels = self.depth
        
        # Encoder (SEBlock 강화)
        self.enc1 = DoubleConv(in_channels, 64, block=encoder_block)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128, block=encoder_block)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256, block=encoder_block)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512, block=encoder_block)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck + CBAM + coarse seg
        self.bottleneck = DoubleConv(512, 1024, block=encoder_block)
        self.cbam_bottle = CBAM(1024)
        self.coarse_seg = nn.Conv2d(1024, 1, 1)
        self.seg4_coarse = nn.Conv2d(512, 1, 1)  # multi-level aux
        
        # Decoder (F.interpolate + conv 업샘플링)
        self.upconv4 = nn.Conv2d(1024, 512, 1)
        self.dec4 = DoubleConv(1024, 512)  # 512(up)+512(skip)
        self.upconv3 = nn.Conv2d(512, 256, 1)
        self.dec3 = DoubleConv(512, 256)   # 256+256
        self.upconv2 = nn.Conv2d(256, 128, 1)
        self.dec2 = DoubleConv(256, 128)   # 128+128
        self.upconv1 = nn.Conv2d(128, 64, 1)
        self.dec1 = DoubleConv(128, 64)    # 64+64
        self.conv_last = nn.Conv2d(64, out_channels, 1)

    def forward(self, X, pe=None):
        # Encoder
        enc1 = self.enc1(X)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        bottleneck = self.cbam_bottle(bottleneck)  # enhanced bottleneck
        
        # Multi-level coarse masks
        coarse_mask = torch.sigmoid(self.coarse_seg(bottleneck))
        seg4_mask = torch.sigmoid(self.seg4_coarse(enc4))
        
        # Decoder (F.interpolate 업샘플링 + plain skip 연결)
        dec4 = self.upconv4(F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False))
        dec4 = self.dec4(torch.cat([dec4, enc4], dim=1))

        dec3 = self.upconv3(F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=False))
        dec3 = self.dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.upconv2(F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=False))
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.upconv1(F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=False))
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        
        seg_out = self.conv_last(dec1)
        return seg_out, coarse_mask, seg4_mask  # main + two aux [web:4][web:20]
