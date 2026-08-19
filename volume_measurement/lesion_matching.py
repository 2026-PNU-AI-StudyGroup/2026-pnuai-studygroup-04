"""GT vs 예측 마스크의 3D 병변 단위 IoU 매칭 (TP/FP/FN).

기존 measure_and_compare.py의 비교는 케이스별 "병변 개수/총 부피"만 GT와 예측 사이에서
집계해서 비교했다. 이 경우 예측이 GT와 개수·부피가 비슷해도 실제로는 다른 위치의 병변을
찾은 것일 수 있어(엉뚱한 곳 병변 3개 예측 + 실제 병변 3개 놓침 == "3개 vs 3개, 완벽히 일치"로
보임) 성능을 낙관적으로 보이게 만드는 문제가 있다(자문 검토에서 지적된 세 번째 편향 요소).

이 모듈은 GT 마스크와 예측 마스크를 각각 3D 볼륨으로 쌓아 3D connected-component로 개별
병변 인스턴스를 뽑고, 병변 쌍의 3D voxel IoU로 비용행렬을 만들어 헝가리안 알고리즘으로
매칭한다. IoU가 임계값 이상인 매칭만 TP로 인정하고, 매칭되지 않은 GT는 FN, 매칭되지 않은
예측은 FP로 센다.
"""
import os
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from functions import imread_unicode

# 3D 26-연결(모든 면/모서리/꼭짓점 인접) 구조 요소
_STRUCT_26 = np.ones((3, 3, 3), dtype=np.uint8)


def load_mask_volume_sorted(mask_dir, filenames=None):
    """mask_dir의 png들을 (전달받았으면 그 순서로, 아니면 파일명 정렬 순서로) 쌓아
    (Z, H, W) 이진(0/1) 볼륨으로 반환. GT/예측 두 볼륨을 같은 z 순서로 쌓기 위해
    반드시 같은 filenames 리스트를 넘겨서 호출해야 한다."""
    if filenames is None:
        filenames = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith('.png'))

    slices = []
    for fname in filenames:
        path = os.path.join(mask_dir, fname)
        if os.path.exists(path):
            mask = imread_unicode(path)
            slices.append((mask > 0).astype(np.uint8) if mask is not None else None)
        else:
            slices.append(None)

    shape = next((s.shape for s in slices if s is not None), None)
    if shape is None:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    slices = [s if s is not None else np.zeros(shape, dtype=np.uint8) for s in slices]
    return np.stack(slices, axis=0)


def label_3d_lesions(volume, min_voxels=1):
    """(Z,H,W) 이진 볼륨 -> 3D connected-component 라벨링. 너무 작은(노이즈성) 인스턴스는 제거."""
    labeled, n = ndimage.label(volume, structure=_STRUCT_26)
    if n == 0:
        return labeled, 0

    sizes = ndimage.sum(volume, labeled, index=np.arange(1, n + 1))
    keep_ids = [i + 1 for i, sz in enumerate(sizes) if sz >= min_voxels]
    if len(keep_ids) == n:
        return labeled, n

    remap = np.zeros(n + 1, dtype=np.int64)
    for new_id, old_id in enumerate(keep_ids, start=1):
        remap[old_id] = new_id
    return remap[labeled], len(keep_ids)


def _pairwise_iou(gt_labeled, n_gt, pred_labeled, n_pred):
    """모든 (gt_lesion, pred_lesion) 쌍의 voxel IoU를 bincount로 한 번에 계산."""
    iou = np.zeros((n_gt, n_pred), dtype=np.float64)
    if n_gt == 0 or n_pred == 0:
        return iou

    gt_sizes = ndimage.sum(np.ones_like(gt_labeled), gt_labeled, index=np.arange(1, n_gt + 1))
    pred_sizes = ndimage.sum(np.ones_like(pred_labeled), pred_labeled, index=np.arange(1, n_pred + 1))

    combined = gt_labeled.astype(np.int64) * (n_pred + 1) + pred_labeled.astype(np.int64)
    mask = (gt_labeled > 0) & (pred_labeled > 0)
    counts = np.bincount(combined[mask], minlength=(n_gt + 1) * (n_pred + 1))

    for gi in range(1, n_gt + 1):
        for pj in range(1, n_pred + 1):
            inter = counts[gi * (n_pred + 1) + pj]
            if inter == 0:
                continue
            union = gt_sizes[gi - 1] + pred_sizes[pj - 1] - inter
            iou[gi - 1, pj - 1] = inter / union if union > 0 else 0.0

    return iou


def match_lesions_iou(gt_volume, pred_volume, iou_threshold=0.1, min_voxels=1):
    """GT/예측 3D 볼륨을 병변 단위로 매칭.

    Returns: dict(
        n_gt, n_pred, tp, fp, fn,
        precision, recall, f1,
        mean_iou_tp: 매칭된 쌍(TP)의 평균 IoU,
        matches: [(gt_id, pred_id, iou), ...]  (TP만)
    )
    """
    gt_labeled, n_gt = label_3d_lesions(gt_volume, min_voxels=min_voxels)
    pred_labeled, n_pred = label_3d_lesions(pred_volume, min_voxels=min_voxels)

    iou_matrix = _pairwise_iou(gt_labeled, n_gt, pred_labeled, n_pred)

    matches = []
    matched_gt, matched_pred = set(), set()
    if n_gt > 0 and n_pred > 0:
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r + 1, c + 1, float(iou_matrix[r, c])))
                matched_gt.add(r + 1)
                matched_pred.add(c + 1)

    tp = len(matches)
    fn = n_gt - tp
    fp = n_pred - tp

    # tp+fp == n_pred, tp+fn == n_gt: 예측/GT가 아예 없으면 관례적으로 1.0(vacuously true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou_tp = float(np.mean([m[2] for m in matches])) if matches else 0.0

    return {
        'n_gt': n_gt, 'n_pred': n_pred,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'mean_iou_tp': mean_iou_tp,
        'matches': matches,
    }


def match_lesions_from_dirs(gt_dir, pred_dir, iou_threshold=0.1, min_voxels=1, filenames=None):
    """GT/예측 마스크 PNG 디렉토리 두 개를 받아 바로 매칭 결과를 반환하는 편의 함수.
    filenames를 넘기지 않으면 두 디렉토리 각각의 파일명 정렬 순서를 사용하므로,
    두 폴더의 파일 구성이 동일(같은 슬라이스 stem 집합)해야 z 순서가 맞는다."""
    if filenames is None:
        gt_files = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith('.png')) if os.path.isdir(gt_dir) else []
        pred_files = sorted(f for f in os.listdir(pred_dir) if f.lower().endswith('.png')) if os.path.isdir(pred_dir) else []
        filenames = sorted(set(gt_files) | set(pred_files))

    gt_volume = load_mask_volume_sorted(gt_dir, filenames)
    pred_volume = load_mask_volume_sorted(pred_dir, filenames)

    if gt_volume.size == 0 and pred_volume.size == 0:
        return {'n_gt': 0, 'n_pred': 0, 'tp': 0, 'fp': 0, 'fn': 0,
                'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'mean_iou_tp': 0.0, 'matches': []}
    if gt_volume.size == 0:
        gt_volume = np.zeros_like(pred_volume)
    if pred_volume.size == 0:
        pred_volume = np.zeros_like(gt_volume)

    return match_lesions_iou(gt_volume, pred_volume, iou_threshold=iou_threshold, min_voxels=min_voxels)
