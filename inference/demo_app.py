"""시연 영상 녹화용 로컬 데모 웹페이지 (Gradio).

CBAS(Segmentation) 단독 vs CBAS+YOLOv11x(게이팅) 결과를 슬라이스 단위로 나란히 보여준다 —
이 프로젝트의 핵심 결과(게이팅이 위양성을 90% 가까이 줄인다)를 시각적으로 바로 보여주기 위한
용도. 환자 선택 드롭다운은 실명이 아닌 익명 Case ID만 노출한다(화면 녹화 시 개인정보가 찍히지
않도록 하기 위함 — 절대로 실제 폴더명/환자명을 UI에 표시하지 않는다).

실행:
  pip install gradio
  python demo_app.py --gpu 0
  (브라우저가 자동으로 열리지 않으면 콘솔에 출력된 http://127.0.0.1:7860 접속)

가중치(cbas_best.pt, yolo_best.pt)와 원내 데이터는 로컬에만 있고 저장소에는 포함되지 않는다.
이 앱은 순수 로컬 실행용이며, 어디에도 배포/공개하지 않는다는 전제로 만들어졌다.
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

from predict_fusion import load_cbas, get_bvalue, resize_2d, gate_mask, TF, CBAS_THRESHOLD, YOLO_CONF

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit('ultralytics가 설치되어 있지 않습니다: pip install ultralytics')


def resolve_case_dir(patient_dir, case):
    for name in (case, case + 'op'):
        p = os.path.join(patient_dir, name)
        if os.path.isdir(p):
            return p, name
    return None, None


def get_b1000_files(phase_dir):
    files = sorted(f for f in os.listdir(phase_dir) if f.endswith('.dcm'))
    keep = []
    for f in files:
        ds = pydicom.dcmread(os.path.join(phase_dir, f), stop_before_pixels=True)
        if get_bvalue(ds) == 1000:
            keep.append(f)
    return keep


def build_case_index(data_root):
    """환자 폴더를 스캔해 익명 Case ID -> (patient_dir, phase_dir, phase_name) 매핑을 만든다.
    UI에는 실제 환자 폴더명이 절대 노출되지 않고 case_001 같은 익명 ID만 보인다."""
    index = {}
    labels = []
    counter = 0
    if not os.path.isdir(data_root):
        return index, labels
    for group in sorted(os.listdir(data_root)):
        group_path = os.path.join(data_root, group)
        if not os.path.isdir(group_path):
            continue
        for patient in sorted(os.listdir(group_path)):
            patient_dir = os.path.join(group_path, patient)
            if not os.path.isdir(patient_dir):
                continue
            for case in ('post', 'pre'):
                case_dir, phase_name = resolve_case_dir(patient_dir, case)
                if case_dir is None or not any(f.endswith('.dcm') for f in os.listdir(case_dir)):
                    continue
                b1000 = get_b1000_files(case_dir)
                if not b1000:
                    continue
                counter += 1
                case_id = f'case_{counter:03d} ({phase_name}, {len(b1000)}장)'
                gt_dir = os.path.join(patient_dir, 'labelmask', phase_name.replace('op', ''))
                index[case_id] = {
                    'phase_dir': case_dir, 'gt_dir': gt_dir, 'files': b1000,
                }
                labels.append(case_id)
    return index, labels


def overlay_mask(gray_u8, mask_binary, color_bgr, alpha=0.45):
    """흑백 슬라이스 위에 이진 마스크를 반투명 색으로 얹는다."""
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
        self._cache = {}  # case_id -> list of per-slice result dicts

    def _run_case(self, case_id, case_info):
        results = []
        for f in case_info['files']:
            dcm_path = os.path.join(case_info['phase_dir'], f)
            ds = pydicom.dcmread(dcm_path)
            arr = ds.pixel_array.astype(np.float32)
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

            gt_path = os.path.join(case_info['gt_dir'], os.path.splitext(f)[0] + '.png')
            gt_mask = None
            if os.path.exists(gt_path):
                raw = cv2.imdecode(np.fromfile(gt_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if raw is not None:
                    gt_mask = cv2.resize((raw > 0).astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST)

            results.append({
                'img_u8': img_u8, 'cbas_mask': cbas_binary, 'gated_mask': gated,
                'gt_mask': gt_mask, 'n_boxes': len(boxes),
            })
        return results

    def load_case(self, case_id, case_index):
        if case_id not in self._cache:
            self._cache[case_id] = self._run_case(case_id, case_index[case_id])
        n = len(self._cache[case_id])
        slider_update = gr.update(minimum=0, maximum=max(1, n - 1), value=0, step=1)
        original, cbas_view, gated_view, gt_view, summary = self.render_slice(case_id, 0)
        return slider_update, original, cbas_view, gated_view, gt_view, summary

    def render_slice(self, case_id, slice_idx):
        if case_id not in self._cache:
            return None, None, None, None, ''
        slice_idx = int(slice_idx)
        r = self._cache[case_id][slice_idx]
        original = np.stack([r['img_u8']] * 3, axis=-1)
        cbas_view = overlay_mask(r['img_u8'], r['cbas_mask'], (0, 0, 255))       # 빨강: CBAS 단독
        gated_view = overlay_mask(r['img_u8'], r['gated_mask'], (0, 200, 0))     # 초록: 게이팅 후
        gt_view = (overlay_mask(r['img_u8'], r['gt_mask'], (255, 200, 0))
                   if r['gt_mask'] is not None else original)

        cbas_px = int(r['cbas_mask'].sum())
        gated_px = int(r['gated_mask'].sum())
        removed = cbas_px - gated_px
        pct = (removed / cbas_px * 100) if cbas_px > 0 else 0.0
        summary = (
            f'슬라이스 {slice_idx + 1}/{len(self._cache[case_id])}  |  '
            f'YOLO 검출 박스 {r["n_boxes"]}개  |  '
            f'CBAS 단독 병변 픽셀 {cbas_px}  →  게이팅 후 {gated_px}  '
            f'(제거 {removed}px, {pct:.1f}%)'
        )
        return original, cbas_view, gated_view, gt_view, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=os.path.join(_here, '..', 'data', 'results_train_cleaned'))
    parser.add_argument('--cbas_weights', default=os.path.join(_here, '..', 'weights', 'cbas_best.pt'))
    parser.add_argument('--yolo_weights', default=os.path.join(_here, '..', 'weights', 'yolo_best.pt'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--share', action='store_true', help='gradio 공개 링크 생성(팀 외부 공유 시에만 사용)')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    case_index, case_labels = build_case_index(args.data_root)
    if not case_labels:
        raise SystemExit(f'{args.data_root} 아래에서 사용 가능한 케이스를 찾지 못했습니다.')

    demo_runner = Demo(args.cbas_weights, args.yolo_weights, device)

    with gr.Blocks(title='CBAS + YOLOv11x 병변 탐지 데모') as app:
        gr.Markdown(
            '# 뇌 MRI 병변 Detection-Segmentation Fusion 데모\n'
            'CBAS-UNet2D 단독 분할(빨강) vs YOLOv11x 게이팅 적용 후(초록)을 슬라이스 단위로 비교합니다. '
            '환자 식별 정보는 표시하지 않고 익명 Case ID만 사용합니다.'
        )
        with gr.Row():
            case_dropdown = gr.Dropdown(choices=case_labels, label='Case 선택 (익명)', value=case_labels[0])
            load_btn = gr.Button('케이스 로드 & 추론 실행', variant='primary')
        slice_slider = gr.Slider(minimum=0, maximum=1, step=1, value=0, label='슬라이스')
        summary_text = gr.Markdown()
        with gr.Row():
            img_original = gr.Image(label='원본', type='numpy')
            img_cbas = gr.Image(label='CBAS 단독 (빨강)', type='numpy')
            img_gated = gr.Image(label='CBAS + YOLO 게이팅 (초록)', type='numpy')
            img_gt = gr.Image(label='GT 라벨 (있는 경우, 주황)', type='numpy')

        case_index_state = gr.State(case_index)

        load_btn.click(
            fn=demo_runner.load_case,
            inputs=[case_dropdown, case_index_state],
            outputs=[slice_slider, img_original, img_cbas, img_gated, img_gt, summary_text],
        )
        slice_slider.change(
            fn=demo_runner.render_slice,
            inputs=[case_dropdown, slice_slider],
            outputs=[img_original, img_cbas, img_gated, img_gt, summary_text],
        )

    app.launch(share=args.share)


if __name__ == '__main__':
    main()
