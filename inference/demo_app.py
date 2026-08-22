"""시연 영상 녹화용 로컬 데모 웹페이지 (Gradio).

이전 버전은 원내 환자 데이터 폴더를 드롭다운으로 나열해서 보여줬는데, 화면 녹화(→ 유튜브 등으로
제출)에 실제 환자 MRI 영상이 찍히는 것 자체가 바람직하지 않다는 판단으로 방식을 바꿨다. 이 버전은
환자 데이터를 전혀 스캔하지 않고, **사용자가 직접 업로드/드래그한 이미지 한 장**만 추론한다 —
즉 화면에 뭐가 나올지는 100% 업로드하는 사람이 고른 이미지에 달려있다.

시연용 이미지 추천: 원내 데이터 대신 공개 데이터셋(ISLES 2022, README의 "데이터" 섹션 참고)에서
DWI 슬라이스를 하나씩 받아 병변 있는 것/없는 것 각 1장을 준비해서 사용하는 걸 권장한다. 정 원내
이미지를 쓰고 싶다면 그건 팀의 판단이지만, 화면 녹화물이 외부로 공개될 수 있다는 점을 고려해서
신중히 고르는 걸 권장한다.

실행:
  pip install gradio
  python demo_app.py --gpu 0
  (콘솔에 뜨는 http://127.0.0.1:7860 접속)
"""
import os
import sys
import argparse

import cv2
cv2.setNumThreads(2)

import numpy as np
import torch
torch.set_num_threads(2)
import pydicom
import gradio as gr

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, '..', 'inference'))
sys.path.insert(0, os.path.join(_here, '..', 'docs', 'repo_bundle', 'inference'))

from predict_fusion import load_cbas, resize_2d, gate_mask, TF, CBAS_THRESHOLD, YOLO_CONF

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit('ultralytics가 설치되어 있지 않습니다: pip install ultralytics')

DICOM_EXTS = ('.dcm', '.dicom', '.ima')


def load_as_grayscale_array(file_path):
    """업로드된 파일을 (H,W) float32 grayscale numpy 배열로 변환. DICOM과 일반 이미지(png/jpg 등) 둘 다 지원."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in DICOM_EXTS:
        ds = pydicom.dcmread(file_path)
        return ds.pixel_array.astype(np.float32)

    try:
        ds = pydicom.dcmread(file_path, force=True)
        if hasattr(ds, 'pixel_array'):
            return ds.pixel_array.astype(np.float32)
    except Exception:
        pass

    data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('이미지를 읽을 수 없습니다 (DICOM 또는 PNG/JPG 등 일반 이미지 파일이어야 합니다).')
    return img.astype(np.float32)


def overlay_mask(gray_u8, mask_binary, color_bgr, alpha=0.45):
    rgb = np.stack([gray_u8] * 3, axis=-1).astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32)
    m = (mask_binary > 0)
    rgb[m] = rgb[m] * (1 - alpha) + color * alpha
    return rgb.astype(np.uint8)


class Demo:
    def __init__(self, cbas_weights, yolo_weights, device):
        print(f'CBAS 가중치 로드: {cbas_weights}')
        self.cbas_model = load_cbas(cbas_weights)
        print(f'YOLO 가중치 로드: {yolo_weights}')
        self.yolo_model = YOLO(yolo_weights)
        self.device = device

    def run(self, file):
        if file is None:
            return None, None, None, '이미지를 업로드하세요.'

        file_path = file if isinstance(file, str) else file.name
        arr = load_as_grayscale_array(file_path)
        resized = resize_2d(arr)

        disp = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        img_u8 = (disp * 255).astype(np.uint8)
        img_rgb = np.stack([img_u8] * 3, axis=-1)

        x = TF(torch.from_numpy(resized.astype(np.float32)))
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p_main, _, _ = self.cbas_model(x)
            prob = torch.sigmoid(p_main)[0, 0].cpu().numpy()
        cbas_binary = (prob > CBAS_THRESHOLD).astype(np.uint8)

        yolo_res = self.yolo_model.predict(img_rgb, verbose=False, conf=YOLO_CONF)
        boxes = yolo_res[0].boxes.xyxy.cpu().numpy() if len(yolo_res[0].boxes) else np.zeros((0, 4))
        gated = gate_mask(cbas_binary, boxes)

        original = np.stack([img_u8] * 3, axis=-1)
        cbas_view = overlay_mask(img_u8, cbas_binary, (0, 0, 255))     # 빨강: CBAS 단독
        gated_view = overlay_mask(img_u8, gated, (0, 200, 0))          # 초록: 게이팅 후

        cbas_px = int(cbas_binary.sum())
        gated_px = int(gated.sum())
        if gated_px > 0:
            verdict = f'🔴 **병변 의심 영역 검출됨** (게이팅 후 {gated_px}px, YOLO 박스 {len(boxes)}개)'
        elif cbas_px > 0:
            verdict = (f'🟢 **병변 미검출로 판단** (CBAS는 {cbas_px}px를 의심했지만 YOLO가 뒷받침하지 '
                       f'않아 게이팅으로 제거됨)')
        else:
            verdict = '🟢 **병변 미검출**'

        return original, cbas_view, gated_view, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cbas_weights', default=os.path.join(_here, '..', 'weights', 'cbas_best.pt'))
    parser.add_argument('--yolo_weights', default=os.path.join(_here, '..', 'weights', 'yolo_best.pt'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--share', action='store_true',
                         help='gradio 공개 링크 생성 - 업로드하는 이미지가 외부 서버를 거치니 신중히 사용')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    demo_runner = Demo(args.cbas_weights, args.yolo_weights, device)

    with gr.Blocks(title='CBAS + YOLOv11x 병변 탐지 데모') as app:
        gr.Markdown(
            '# 뇌 MRI 병변 Detection-Segmentation Fusion 데모\n'
            'DWI MRI 슬라이스 이미지(DICOM 또는 PNG/JPG)를 업로드하면, CBAS-UNet2D 단독 분할(빨강)과 '
            'YOLOv11x 게이팅 적용 후(초록) 결과를 비교해서 보여줍니다.\n\n'
            '**이 페이지는 업로드한 이미지 외 어떤 환자 데이터도 불러오지 않습니다** — 시연 영상에는 '
            '직접 선택한 이미지만 나옵니다. 원내 데이터 대신 공개 데이터셋(ISLES 2022)의 예시 슬라이스 '
            '사용을 권장합니다.'
        )
        with gr.Row():
            file_input = gr.File(label='DICOM 또는 이미지 업로드', file_count='single')
            run_btn = gr.Button('추론 실행', variant='primary')
        verdict_text = gr.Markdown()
        with gr.Row():
            img_original = gr.Image(label='원본', type='numpy')
            img_cbas = gr.Image(label='CBAS 단독 (빨강)', type='numpy')
            img_gated = gr.Image(label='CBAS + YOLO 게이팅 (초록)', type='numpy')

        run_btn.click(fn=demo_runner.run, inputs=[file_input], outputs=[img_original, img_cbas, img_gated, verdict_text])
        file_input.change(fn=demo_runner.run, inputs=[file_input], outputs=[img_original, img_cbas, img_gated, verdict_text])

    app.launch(share=args.share)


if __name__ == '__main__':
    main()
