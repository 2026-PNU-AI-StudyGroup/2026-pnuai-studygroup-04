"""
CBAS(Segmentation) + YOLOv11x(Detection) 결합 추론.

두 모델은 별도로 학습된 독립적인 가중치이며, "합쳐진 하나의 가중치"는 없다.
같은 슬라이스를 두 모델에 각각 넣고, CBAS가 예측한 이진 마스크 중
YOLO가 검출한 bounding box 밖에 있는 픽셀을 제거(spatial gating)해서
최종 병변 마스크를 만든다.

필요한 가중치 (repo에는 포함되어 있지 않음 - 별도로 전달받아야 함):
  weights/cbas_best.pt   (CBAS-UNet2D, fold0 best.pt)
  weights/yolo_best.pt   (YOLOv11x, ultralytics 형식 best.pt)

사용 예:
  python predict_fusion.py --dicom_dir /path/to/patient/post --out_dir ./output
"""
import os
import sys
import argparse
import numpy as np
import pydicom
import torch
import cv2
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from networks.CBAS import SuperEnhancedAFMSUNet2D  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit('ultralytics가 설치되어 있지 않습니다: pip install ultralytics')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CBAS_THRESHOLD = 0.5
YOLO_CONF = 0.25
TARGET_SIZE = (192, 192)

TF = transforms.Compose([
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
])


def get_bvalue(ds):
    """DWI b-value 추출 (Siemens private tag 우선, 없으면 표준 태그)."""
    tag = ds.get((0x0019, 0x100c), None)
    if tag is not None:
        try:
            return int(str(tag.value).split('\\')[0])
        except (ValueError, TypeError):
            pass
    bval = getattr(ds, 'DiffusionBValue', None)
    if bval is not None:
        try:
            return int(bval)
        except (ValueError, TypeError):
            pass
    return -1


def resize_2d(arr, size=TARGET_SIZE):
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return x.squeeze().numpy()


def load_cbas(weights_path):
    model = SuperEnhancedAFMSUNet2D(out_channels=1, context=0)
    sd = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(sd)
    model.to(DEVICE).eval()
    return model


def gate_mask(binary_mask, boxes_xyxy):
    """YOLO box 밖의 마스크 픽셀을 0으로 지운다."""
    if len(boxes_xyxy) == 0:
        return np.zeros_like(binary_mask)
    gate = np.zeros_like(binary_mask)
    h, w = binary_mask.shape
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        gate[y1:y2, x1:x2] = 1
    return binary_mask * gate


def predict_slice(dcm_path, cbas_model, yolo_model):
    """DICOM 한 장에 대해 CBAS 마스크, YOLO 박스, 게이팅된 최종 마스크를 반환."""
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    resized = resize_2d(arr)

    disp = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    img_u8 = (disp * 255).astype(np.uint8)
    img_rgb = np.stack([img_u8] * 3, axis=-1)

    x = TF(torch.from_numpy(resized.astype(np.float32)))
    x = x.unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p_main, _, _ = cbas_model(x)
        prob = torch.sigmoid(p_main)[0, 0].cpu().numpy()
    cbas_binary = (prob > CBAS_THRESHOLD).astype(np.uint8)

    results = yolo_model.predict(img_rgb, verbose=False, conf=YOLO_CONF)
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results[0].boxes) else np.zeros((0, 4))

    gated = gate_mask(cbas_binary, boxes)
    return {
        'prob_map': prob,
        'cbas_mask': cbas_binary,
        'yolo_boxes': boxes,
        'final_mask': gated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dicom_dir', required=True, help='b=1000 DWI DICOM이 들어있는 폴더')
    parser.add_argument('--out_dir', required=True, help='최종 마스크(PNG) 저장 폴더')
    parser.add_argument('--cbas_weights', default=os.path.join(os.path.dirname(__file__), '..', 'weights', 'cbas_best.pt'))
    parser.add_argument('--yolo_weights', default=os.path.join(os.path.dirname(__file__), '..', 'weights', 'yolo_best.pt'))
    parser.add_argument('--only_b1000', action='store_true', default=True,
                         help='DICOM 헤더 b-value로 b=1000 슬라이스만 선별 (기본값 True)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f'모델 로드 중... (device={DEVICE})')
    cbas_model = load_cbas(args.cbas_weights)
    yolo_model = YOLO(args.yolo_weights)

    files = sorted(f for f in os.listdir(args.dicom_dir) if f.endswith('.dcm'))
    n_saved = 0
    for f in files:
        dcm_path = os.path.join(args.dicom_dir, f)
        if args.only_b1000:
            ds_head = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            if get_bvalue(ds_head) != 1000:
                continue

        result = predict_slice(dcm_path, cbas_model, yolo_model)
        stem = os.path.splitext(f)[0]
        out_path = os.path.join(args.out_dir, f'{stem}.png')
        Image.fromarray((result['final_mask'] * 255).astype(np.uint8)).save(out_path)
        n_saved += 1
        n_lesion_px = int(result['final_mask'].sum())
        print(f'  {f}: YOLO box {len(result["yolo_boxes"])}개, 최종 병변 픽셀 {n_lesion_px}개 -> {out_path}')

    print(f'\n완료: {n_saved}장 저장 -> {args.out_dir}')


if __name__ == '__main__':
    main()
