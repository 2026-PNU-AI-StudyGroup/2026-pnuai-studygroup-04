"""fold별 held-out test set에 대한 실제 Dice/IoU/Precision/Recall 평가.

train.py는 학습 중 valid-set 기준으로 best checkpoint를 골랐을 뿐, test_subset(각 fold의
held-out 환자)에 대해서는 한 번도 평가를 돌리지 않았다(만들기만 하고 안 씀). 자문 검토에서
명시적으로 요청한 "fold별 Dice/IoU ± SD"를 내려면 별도로 이 스크립트가 필요하다.

각 fold의 test 환자 집합은 train.py의 grouped_kfold_patients(seed 고정)로 재현하므로,
실제 학습 때 held-out으로 뺐던 환자와 정확히 동일한 환자에 대해서만 평가한다(train과 겹치지
않음을 다시 assert로 확인).

사용 예:
  python evaluate_test.py --name CBAGS_grouped --gpu 6
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.dataset import Dicom2dDataset
from utils.utils import custom_collate_fn
from networks.CBAS import SuperEnhancedAFMSUNet2D
from options.hyper_parameters import HP
from train import get_patient_groups, grouped_kfold_patients


@torch.no_grad()
def evaluate_fold(model, loader, device, alpha=0.5, eps=1e-8):
    """배치가 아니라 슬라이스(샘플) 단위로 Dice/IoU/Precision/Recall을 계산.

    자연 비율 test set은 93%가 정상(병변 없는) 슬라이스라, 전체 슬라이스 평균만 보면
    "병변 없다고 잘 맞춘" 쉬운 케이스가 점수를 끌어올려 실제 병변 분할 성능을 과대평가하게
    된다(자문 검토가 지적한 낙관적 평가와 같은 함정). 그래서 (1) 전체 슬라이스 기준과
    (2) GT에 실제 병변이 있는 슬라이스만 기준, 두 가지를 모두 계산해서 반환한다.
    """
    dices, ious, precisions, recalls, has_lesion = [], [], [], [], []
    for data in loader:
        X, y = data[0].to(device), data[1].to(device)
        p_main, _, _ = model(X)
        p_bin = (torch.sigmoid(p_main) > alpha).float()
        y_bin = (y > 0.5).float()

        dims = tuple(range(1, p_bin.dim()))
        inter = (p_bin * y_bin).sum(dim=dims)
        p_sum = p_bin.sum(dim=dims)
        y_sum = y_bin.sum(dim=dims)
        union = p_sum + y_sum - inter

        dice = (2 * inter + eps) / (p_sum + y_sum + eps)
        iou = (inter + eps) / (union + eps)
        precision = torch.where(p_sum > 0, inter / p_sum.clamp(min=eps), torch.zeros_like(inter))
        recall = torch.where(y_sum > 0, inter / y_sum.clamp(min=eps), torch.zeros_like(inter))
        # GT/예측 둘 다 병변이 없는 완전-음성 슬라이스는 "잘 맞춘 것"으로 Dice/IoU=1 처리
        both_empty = (p_sum == 0) & (y_sum == 0)
        dice = torch.where(both_empty, torch.ones_like(dice), dice)
        iou = torch.where(both_empty, torch.ones_like(iou), iou)
        precision = torch.where(both_empty, torch.ones_like(precision), precision)
        recall = torch.where(both_empty, torch.ones_like(recall), recall)

        dices.append(dice.cpu())
        ious.append(iou.cpu())
        precisions.append(precision.cpu())
        recalls.append(recall.cpu())
        has_lesion.append((y_sum > 0).cpu())

    dices = torch.cat(dices)
    ious = torch.cat(ious)
    precisions = torch.cat(precisions)
    recalls = torch.cat(recalls)
    has_lesion = torch.cat(has_lesion)

    def _stats(mask):
        if mask.sum() == 0:
            return {'n_slices': 0, 'Dice_mean': None, 'Dice_std': None,
                    'IoU_mean': None, 'IoU_std': None,
                    'Precision_mean': None, 'Recall_mean': None}
        return {
            'n_slices': int(mask.sum()),
            'Dice_mean': float(dices[mask].mean()), 'Dice_std': float(dices[mask].std()),
            'IoU_mean': float(ious[mask].mean()), 'IoU_std': float(ious[mask].std()),
            'Precision_mean': float(precisions[mask].mean()),
            'Recall_mean': float(recalls[mask].mean()),
        }

    all_mask = torch.ones_like(has_lesion, dtype=torch.bool)
    return {'all_slices': _stats(all_mask), 'lesion_slices_only': _stats(has_lesion)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='CBAGS_grouped')
    parser.add_argument('--root_dir', default='../data/h5')
    parser.add_argument('--context', type=int, default=0)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--out_json', default='test_eval_results.json')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hp = HP(model=None, name=args.name)  # seed/batch_size 등 학습 때와 동일한 기본값 재사용

    tf = transforms.Compose([
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    ])
    target_dir = ['5mm 미만', '5mm 이상', 'normal', '추가 병변']

    print('전체 데이터셋(자연 비율) 로드 중...')
    full_dataset = Dicom2dDataset(root_dir=args.root_dir, target_dir=target_dir, transform=tf,
                                   context=args.context, only_lesion=False)
    full_groups = get_patient_groups(full_dataset, args.context)
    unique_patients = set(full_groups.tolist())

    results = {}
    for fold, train_patients, val_patients, test_patients in grouped_kfold_patients(unique_patients, args.k, hp.seed):
        assert not (train_patients & test_patients), f'fold{fold}: train/test 환자 겹침'
        assert not (val_patients & test_patients), f'fold{fold}: val/test 환자 겹침'

        test_idx = [i for i, g in enumerate(full_groups) if g in test_patients]
        test_subset = Subset(full_dataset, test_idx)
        test_loader = DataLoader(test_subset, batch_size=hp.batch_size, shuffle=False,
                                  collate_fn=custom_collate_fn)

        ckpt_path = f'{hp.path_model}/fold_{fold}/best.pt'
        model = SuperEnhancedAFMSUNet2D(out_channels=1, context=args.context)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device).eval()

        print(f'[Fold {fold}] test 환자 {len(test_patients)}명, {len(test_subset)}슬라이스 평가 중...')
        metrics = evaluate_fold(model, test_loader, device)
        metrics['n_test_patients'] = len(test_patients)
        results[f'fold_{fold}'] = metrics
        a, l = metrics['all_slices'], metrics['lesion_slices_only']
        print(f'[Fold {fold}] 전체슬라이스(n={a["n_slices"]}) Dice={a["Dice_mean"]:.4f} IoU={a["IoU_mean"]:.4f}  |  '
              f'병변슬라이스만(n={l["n_slices"]}) Dice={l["Dice_mean"]:.4f} IoU={l["IoU_mean"]:.4f} '
              f'Precision={l["Precision_mean"]:.4f} Recall={l["Recall_mean"]:.4f}')

        del model
        torch.cuda.empty_cache()

    def _summary(key):
        dice = np.array([results[f'fold_{f}'][key]['Dice_mean'] for f in range(args.k)])
        iou = np.array([results[f'fold_{f}'][key]['IoU_mean'] for f in range(args.k)])
        return {
            'Dice_mean_across_folds': float(dice.mean()), 'Dice_std_across_folds': float(dice.std()),
            'IoU_mean_across_folds': float(iou.mean()), 'IoU_std_across_folds': float(iou.std()),
        }, dice, iou

    results['summary'] = {'all_slices': _summary('all_slices')[0], 'lesion_slices_only': _summary('lesion_slices_only')[0]}
    _, dice_a, iou_a = _summary('all_slices')
    _, dice_l, iou_l = _summary('lesion_slices_only')
    print(f'\n[전체 슬라이스 기준] Dice = {dice_a.mean():.4f} ± {dice_a.std():.4f}  IoU = {iou_a.mean():.4f} ± {iou_a.std():.4f}  (5-fold)')
    print(f'[병변 슬라이스만 기준] Dice = {dice_l.mean():.4f} ± {dice_l.std():.4f}  IoU = {iou_l.mean():.4f} ± {iou_l.std():.4f}  (5-fold)')

    with open(args.out_json, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'저장 완료: {args.out_json}')


if __name__ == '__main__':
    main()
