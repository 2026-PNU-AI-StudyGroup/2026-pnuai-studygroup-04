"""fold별 held-out test 환자에 대해, 그 fold의 실제 best.pt(+공용 YOLO)로 CBAS-only /
CBAS+YOLO-gated 예측을 만들고, GT와 3D 병변 단위 IoU 매칭(TP/FP/FN)을 수행한다.

evaluate_test.py가 슬라이스 단위 Dice/IoU를 재는 반면, 이 스크립트는
volume_measurement/lesion_matching.py를 실제 held-out 환자에 적용해서 "위치까지 맞는"
병변 단위 Precision/Recall/F1을 낸다. 각 fold는 자기 test 환자만, 그 fold의 학습에
한 번도 쓰이지 않은 가중치로 평가하므로 전체 99명 전원이 정확히 한 번씩 진짜
held-out으로 평가된다(5-fold cross-validation 방식의 pooled 평가).

사용 예:
  python evaluate_lesion_matching.py --name CBAGS_grouped --gpu 6
"""
import os
import sys
import json
import argparse

import cv2
cv2.setNumThreads(2)

import numpy as np
import torch
torch.set_num_threads(2)
import pydicom
from torchvision import transforms

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
# 저장소 배치(src/ + inference/ + volume_measurement/가 형제 디렉토리)와
# 이 프로젝트의 원 작업 디렉토리 배치(functions.py 등이 상위 루트에 바로 있음)를 모두 지원
sys.path.insert(0, os.path.join(_here, '..', 'inference'))
sys.path.insert(0, os.path.join(_here, '..', 'volume_measurement'))
sys.path.insert(0, os.path.join(_here, '..', 'docs', 'repo_bundle', 'inference'))
sys.path.insert(0, os.path.join(_here, '..'))

from utils.dataset import Dicom2dDataset
from options.hyper_parameters import HP
from train import get_patient_groups, grouped_kfold_patients
from predict_fusion import load_cbas, get_bvalue, resize_2d, gate_mask, TF, CBAS_THRESHOLD, YOLO_CONF
from functions import imwrite_unicode
from lesion_matching import match_lesions_from_dirs

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit('ultralytics가 설치되어 있지 않습니다: pip install ultralytics')

CBAS_ONLY_SUBDIR = '_pred_cbas_only'
GATED_SUBDIR = '_pred_cbas_yolo_gated'
LESION_IOU_THRESHOLD = 0.1


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


def predict_and_save(cbas_model, yolo_model, phase_dir, cbas_dir, gated_dir, device):
    os.makedirs(cbas_dir, exist_ok=True)
    os.makedirs(gated_dir, exist_ok=True)
    saved = []
    for f in get_b1000_files(phase_dir):
        dcm_path = os.path.join(phase_dir, f)
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(np.float32)
        resized = resize_2d(arr)

        disp = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        img_u8 = (disp * 255).astype(np.uint8)
        img_rgb = np.stack([img_u8] * 3, axis=-1)

        x = TF(torch.from_numpy(resized.astype(np.float32)))
        x = x.unsqueeze(0).unsqueeze(0).to(device)
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
        saved.append(f'{stem}.png')
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='CBAGS_grouped')
    parser.add_argument('--root_dir', default='../data/h5', help='h5 데이터셋 경로 (환자 목록 확보용)')
    parser.add_argument('--data_root', default='../data/results_train_cleaned', help='원본 DICOM/labelmask 경로')
    parser.add_argument('--out_root', default='../res/eval_lesion_matching', help='예측 마스크 저장 경로')
    parser.add_argument('--context', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--yolo_weights', default='../weights/yolo_best.pt')
    parser.add_argument('--lesion_iou_threshold', type=float, default=LESION_IOU_THRESHOLD)
    parser.add_argument('--out_csv', default='lesion_matching_results.csv')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hp = HP(model=None, name=args.name)
    tf = transforms.Compose([
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    ])
    target_dir = ['5mm 미만', '5mm 이상', 'normal', '추가 병변']

    print('환자 목록 확보용 데이터셋 로드 중...')
    full_dataset = Dicom2dDataset(root_dir=args.root_dir, target_dir=target_dir, transform=tf,
                                   context=args.context, only_lesion=False)
    full_groups = get_patient_groups(full_dataset, args.context)
    unique_patients = set(full_groups.tolist())

    print(f'YOLO 로드 중: {args.yolo_weights}')
    yolo_model = YOLO(args.yolo_weights)

    rows = []
    case_counter = 0
    for fold, train_patients, val_patients, test_patients in grouped_kfold_patients(unique_patients, args.k, hp.seed):
        ckpt_path = f'{hp.path_model}/fold_{fold}/best.pt'
        cbas_model = load_cbas(ckpt_path)
        print(f'[Fold {fold}] test 환자 {len(test_patients)}명, 가중치={ckpt_path}')

        for patient in sorted(test_patients):
            lesion_group, patient_folder = patient.split('/', 1)
            patient_dir = os.path.join(args.data_root, lesion_group, patient_folder)
            if not os.path.isdir(patient_dir):
                print(f'  [skip] raw DICOM 폴더 없음: {patient_dir}')
                continue

            for case in ('post', 'pre'):
                case_dir, phase_name = resolve_case_dir(patient_dir, case)
                if case_dir is None or not any(f.endswith('.dcm') for f in os.listdir(case_dir)):
                    continue

                case_counter += 1
                case_id = f'case_{case_counter:03d}'

                out_dir = os.path.join(args.out_root, f'fold{fold}', case_id)
                cbas_dir = os.path.join(out_dir, CBAS_ONLY_SUBDIR)
                gated_dir = os.path.join(out_dir, GATED_SUBDIR)
                saved = predict_and_save(cbas_model, yolo_model, case_dir, cbas_dir, gated_dir, device)
                if not saved:
                    continue

                gt_phase = phase_name.replace('op', '')
                gt_dir = os.path.join(patient_dir, 'labelmask', gt_phase)

                row = {'Fold': fold, 'CaseID': case_id, 'Phase': phase_name}
                for prefix, pred_dir in [('CBASOnly', cbas_dir), ('Gated', gated_dir)]:
                    result = match_lesions_from_dirs(gt_dir, pred_dir, iou_threshold=args.lesion_iou_threshold)
                    row[f'{prefix}_n_gt'] = result['n_gt']
                    row[f'{prefix}_n_pred'] = result['n_pred']
                    row[f'{prefix}_TP'] = result['tp']
                    row[f'{prefix}_FP'] = result['fp']
                    row[f'{prefix}_FN'] = result['fn']
                    row[f'{prefix}_Precision'] = round(result['precision'], 4)
                    row[f'{prefix}_Recall'] = round(result['recall'], 4)
                rows.append(row)
                print(f'  [{case_id}] {phase_name}: CBASOnly TP={row["CBASOnly_TP"]} FP={row["CBASOnly_FP"]} FN={row["CBASOnly_FN"]}  |  '
                      f'Gated TP={row["Gated_TP"]} FP={row["Gated_FP"]} FN={row["Gated_FN"]}')

        del cbas_model
        torch.cuda.empty_cache()

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False, encoding='utf-8-sig')
    print(f'\n저장 완료: {args.out_csv} ({len(df)}건, 환자 실명/등록번호 미포함, 익명 CaseID만 사용)')

    summary = {}
    for prefix in ('CBASOnly', 'Gated'):
        tp, fp, fn = df[f'{prefix}_TP'].sum(), df[f'{prefix}_FP'].sum(), df[f'{prefix}_FN'].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        summary[prefix] = {'TP': int(tp), 'FP': int(fp), 'FN': int(fn),
                            'Precision': round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}
        print(f'[{prefix}] 전체(pooled, 5-fold held-out 전원) TP={tp} FP={fp} FN={fn} '
              f'Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}')

    with open(args.out_csv.replace('.csv', '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
