import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.eps = alpha, beta, eps

    def forward(self, logits, target):
        # (B,1,H,W) 또는 (B,H,W)
        p = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        if p.dim() == 3:
            p, target = p.unsqueeze(1), target.unsqueeze(1)

        # per-image reduction (FN에 민감)
        dims = tuple(range(2, p.dim()))
        tp = (p * target).sum(dim=dims)
        fp = (p * (1 - target)).sum(dim=dims)
        fn = ((1 - p) * target).sum(dim=dims)

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return (1 - tversky).mean()

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, eps=1e-6):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, eps)
        self.gamma = gamma

    def forward(self, logits, target):
        tl = self.tversky(logits, target)  # scalar
        return tl ** self.gamma

class BCEWithPosWeight(nn.Module):
    def __init__(self, max_pw=20.0):
        super().__init__()
        self.max_pw = max_pw

    def forward(self, logits, target):
        if logits.dim() == 3:
            logits, target = logits.unsqueeze(1), target.unsqueeze(1)

        # per-sample pos_weight
        with torch.no_grad():
            pos = target.flatten(1).sum(1) + 1e-6
            tot = torch.tensor(target[0].numel(), device=target.device, dtype=pos.dtype)
            neg = tot - pos + 1e-6
            pw  = torch.clamp(neg / pos, max=self.max_pw)

        loss_list = []
        for i in range(logits.size(0)):
            loss_list.append(F.binary_cross_entropy_with_logits(
                logits[i], target[i], pos_weight=pw[i]
            ))
        return torch.stack(loss_list).mean()

class BCE_TopK(nn.Module):
    def __init__(self, k=0.3):
        super().__init__()
        self.k = k  # 0<k<=1, 상위 k 비율만 사용

    def forward(self, logits, target):
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        # per-sample top-k
        B = loss.size(0)
        loss = loss.view(B, -1)
        k = (loss.size(1) * self.k)
        k = max(1, int(k))
        topk_vals, _ = torch.topk(loss, k, dim=1, largest=True, sorted=False)
        return topk_vals.mean()

class BCE(nn.Module):
    def __init__(self, alpha=1, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        
    def forward(self, p, y, weight=None):
        p = torch.sigmoid(p)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)  # NaN 방지        
        p = p.reshape(p.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        
        bce = -(y*torch.log(p+self.smooth) + (1-y)*torch.log(1-p+self.smooth))
        if weight!=None: bce = bce*weight.reshape(p.shape[0], -1)
        
        bce = torch.mean(bce)
        
        return bce*self.alpha

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, p, y, w=None):
        p = torch.sigmoid(p)
        p = torch.clamp(p, 1e-6, 1 - 1e-6)  # NaN 방지        
        p = p.reshape(p.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        if w is not None:
            w = w.reshape(w.shape[0], -1)
        else:
            w = 1
        
        hat = torch.sum(w*(p*y*2))
        cup = torch.sum(w*(p+y)) + self.smooth
        
        res = 1 - hat/cup
        
        return res
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.95, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target, w=None):

        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce

        return focal.mean()
    
class GeneralUnionLoss(nn.Module):
    def __init__(self, alpha=0.15, r=0.85, epsilon=1e-7):
        super(GeneralUnionLoss, self).__init__()
        self.alpha = alpha
        self.beta = 1 - alpha
        self.r = r
        self.epsilon = epsilon

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, self.epsilon, 1 - self.epsilon)
        assert pred.shape == target.shape
        pr = pred ** self.r
        num = torch.sum(pr * target)
        denom = torch.sum(self.alpha * pred + self.beta * target)
        loss = 1 - num / (denom + self.epsilon)
        return loss

class BoundaryLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        boundary_pred = pred - F.avg_pool2d(pred, 3, stride=1, padding=1)  # max_pool → avg_pool!
        boundary_gt = target.float() - F.avg_pool2d(target.float(), 3, stride=1, padding=1)
        return F.binary_cross_entropy(boundary_pred, boundary_gt)