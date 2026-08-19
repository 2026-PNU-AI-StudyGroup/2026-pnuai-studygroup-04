"""
GT(사람 라벨) vs CBAS-only vs CBAS+YOLO(게이팅) 병변 개수/부피/위치 비교.

data_root 아래 모든 환자·phase를 스캔해서:
  1) GT labelmask 폴더로 부피 계산
  2) CBAS만으로 예측한 마스크로 부피 계산
  3) CBAS+YOLO 게이팅한 마스크로 부피 계산
  4) (CBASOnly, Gated) 각각을 GT와 3D 병변 단위 IoU로 매칭해 TP/FP/FN/Precision/Recall/F1 계산
     — 병변 "개수"나 "총 부피"가 GT와 비슷해도 실제로는 다른 위치의 병변일 수 있어(자문 검토가
     지적한 낙관적 편향), lesion_matching.py로 위치까지 맞는지 확인한다.
세 결과를 표로 만들어 CSV/Excel로 저장한다 (환자 실명/등록번호 등 개인정보는 제외, 익명 케이스 ID만 사용).

사용 예:
  python measure_and_compare.py \
    --data_root ../data/results_train_cleaned \
    --cbas_weights ../weights/cbas_best.pt \
    --yolo_weights ../weights/yolo_best.pt \
    --out_prefix result
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import pydicom
import torch
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'inference'))
sys.path.insert(0, os.path.dirname(__file__))

from predict_fusion import load_cbas, get_bvalue, resize_2d, gate_mask, TF, DEVICE  # noqa: E402
from functions import imwrite_unicode, load_and_normalize, summarize_df  # noqa: E402
from volume_tracking import process_mask_dir  # noqa: E402
from lesion_matching import match_lesions_from_dirs  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit('ultralytics가 설치되어 있지 않습니다: pip install ultralytics')

CBAS_THRESHOLD = 0.5
YOLO_CONF = 0.25
CBAS_ONLY_SUBDIR = '_pred_cbas_only'
GATED_SUBDIR = '_pred_cbas_yolo_gated'
LESION_IOU_THRESHOLD = 0.1  # 병변이 작고 불규칙해 0.5 같은 높은 IoU 기준은 비현실적이라 낮게 설정


def resolve_case_dir(patient_dir, case):
    for name in (case, case + 'op'):
        p = os.path.join(patient_dir, name)
        if os.path.isdir(p):
            return p, name
    return None, None


def scan_pairs(data_root):
    """data_root/<그룹>/<환자>/{pre|post|preop|postop} 전부 스캔."""
    pairs = []
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
                if case_dir and any(f.endswith('.dcm') for f in os.listdir(case_dir)):
                    pairs.append((patient_dir, phase_name))
    return pairs


def get_b1000_files(phase_dir):
    files = sorted(f for f in os.listdir(phase_dir) if f.endswith('.dcm'))
    keep = []
    for f in files:
        ds = pydicom.dcmread(os.path.join(phase_dir, f), stop_before_pixels=True)
        if get_bvalue(ds) == 1000:
            keep.append(f)
    return keep


def predict_and_save(cbas_model, yolo_model, patient_dir, phase):
    phase_dir = os.path.join(patient_dir, phase)
    cbas_dir = os.path.join(patient_dir, CBAS_ONLY_SUBDIR, phase)
    gated_dir = os.path.join(patient_dir, GATED_SUBDIR, phase)
    os.makedirs(cbas_dir, exist_ok=True)
    os.makedirs(gated_dir, exist_ok=True)

    for f in get_b1000_files(phase_dir):
        dcm_path = os.path.join(phase_dir, f)
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

        stem = os.path.splitext(f)[0]
        imwrite_unicode(os.path.join(cbas_dir, f'{stem}.png'), (cbas_binary * 255).astype(np.uint8))
        imwrite_unicode(os.path.join(gated_dir, f'{stem}.png'), (gated * 255).astype(np.uint8))

    return cbas_dir, gated_dir


def volume_summary(mask_dir):
    if not os.path.isdir(mask_dir) or not any(f.endswith('.png') for f in os.listdir(mask_dir)):
        return summarize_df(pd.DataFrame())
    process_mask_dir(mask_dir)
    vol_txt = os.path.join(mask_dir, 'volume.txt')
    if os.path.exists(vol_txt):
        return summarize_df(load_and_normalize(vol_txt))
    return summarize_df(pd.DataFrame())


def lesion_matching_summary(gt_dir, pred_dir, iou_threshold=LESION_IOU_THRESHOLD):
    """개수/부피 총량 비교(volume_summary)와 달리, GT·예측 병변을 3D IoU로 1:1 매칭해서
    '위치까지 맞는' 진짜 TP/FP/FN을 센다. (자문 검토가 지적한, 개수만 맞으면 일치로 보이는
    낙관적 편향을 보정)"""
    result = match_lesions_from_dirs(gt_dir, pred_dir, iou_threshold=iou_threshold)
    return {
        'TP': result['tp'], 'FP': result['fp'], 'FN': result['fn'],
        'Precision': round(result['precision'], 4),
        'Recall': round(result['recall'], 4),
        'F1': round(result['f1'], 4),
        'MeanIoU_TP': round(result['mean_iou_tp'], 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--cbas_weights', default='../weights/cbas_best.pt')
    parser.add_argument('--yolo_weights', default='../weights/yolo_best.pt')
    parser.add_argument('--out_prefix', default='result')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--lesion_iou_threshold', type=float, default=LESION_IOU_THRESHOLD)
    args = parser.parse_args()

    pairs = scan_pairs(args.data_root)
    if args.limit:
        pairs = pairs[:args.limit]

    print(f'모델 로드 중... (device={DEVICE})')
    cbas_model = load_cbas(args.cbas_weights)
    yolo_model = YOLO(args.yolo_weights)
    print(f'대상 (환자,phase) 수: {len(pairs)}')

    rows = []
    for i, (patient_dir, phase) in enumerate(pairs):
        case_id = f'case_{i+1:03d}'  # 익명 케이스 ID (실명/등록번호 미포함)
        print(f'[{case_id}] {phase}')

        cbas_dir, gated_dir = predict_and_save(cbas_model, yolo_model, patient_dir, phase)

        gt_phase = phase.replace('op', '')
        gt_dir = os.path.join(patient_dir, 'labelmask', gt_phase)

        row = {'CaseID': case_id, 'Phase': phase}
        for prefix, mdir in [('GT', gt_dir), ('CBASOnly', cbas_dir), ('Gated', gated_dir)]:
            summary = volume_summary(mdir)
            for k, v in summary.items():
                row[f'{prefix}_{k}'] = v

        # 병변 단위 IoU 매칭 (개수/부피 총량 비교와 별개로, 위치까지 맞는지 확인)
        for prefix, mdir in [('CBASOnly', cbas_dir), ('Gated', gated_dir)]:
            lesion_result = lesion_matching_summary(gt_dir, mdir, iou_threshold=args.lesion_iou_threshold)
            for k, v in lesion_result.items():
                row[f'{prefix}_Lesion_{k}'] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = f'{args.out_prefix}_volume_comparison_anonymized.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'저장 완료: {csv_path}')

    # 병변 단위 micro-averaged precision/recall/F1 (케이스별 TP/FP/FN을 합산 후 계산)
    for prefix in ('CBASOnly', 'Gated'):
        tp_sum = df[f'{prefix}_Lesion_TP'].sum()
        fp_sum = df[f'{prefix}_Lesion_FP'].sum()
        fn_sum = df[f'{prefix}_Lesion_FN'].sum()
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 1.0
        recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f'[{prefix}] 병변 단위(IoU>={args.lesion_iou_threshold}) 전체 TP={tp_sum} FP={fp_sum} FN={fn_sum} '
              f'Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}')

    xlsx_path = f'{args.out_prefix}_volume_comparison_anonymized.xlsx'
    df.to_excel(xlsx_path, index=False)
    print(f'저장 완료: {xlsx_path}')

    return df


if __name__ == '__main__':
    main()
