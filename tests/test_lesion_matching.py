"""3D 병변 단위 IoU 매칭(volume_measurement/lesion_matching.py)의 정확성 테스트.
합성 numpy 볼륨만으로 동작하며 실제 데이터/모델이 필요 없다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'volume_measurement'))

import numpy as np

from lesion_matching import match_lesions_iou, label_3d_lesions


def _empty_volume(shape=(4, 32, 32)):
    return np.zeros(shape, dtype=np.uint8)


def _cube(volume, z0, z1, y0, y1, x0, x1):
    v = volume.copy()
    v[z0:z1, y0:y1, x0:x1] = 1
    return v


def test_perfect_match_is_single_tp():
    gt = _cube(_empty_volume(), 1, 3, 5, 10, 5, 10)
    pred = gt.copy()

    result = match_lesions_iou(gt, pred, iou_threshold=0.1)

    assert result['n_gt'] == 1 and result['n_pred'] == 1
    assert result['tp'] == 1 and result['fp'] == 0 and result['fn'] == 0
    assert result['mean_iou_tp'] == 1.0
    assert result['precision'] == 1.0 and result['recall'] == 1.0 and result['f1'] == 1.0


def test_disjoint_lesions_are_fp_and_fn_not_a_match():
    gt = _cube(_empty_volume(), 1, 3, 2, 6, 2, 6)
    pred = _cube(_empty_volume(), 1, 3, 20, 24, 20, 24)  # 완전히 다른 위치

    result = match_lesions_iou(gt, pred, iou_threshold=0.1)

    assert result['tp'] == 0
    assert result['fn'] == 1  # GT 병변을 못 찾음
    assert result['fp'] == 1  # 예측 병변은 실제 GT와 무관한 위치


def test_aggregate_count_can_hide_location_mismatch():
    """자문 검토가 지적한 핵심 문제 재현: GT 2개, 예측 2개로 '개수는 일치'하지만
    실제 위치가 전부 어긋나면 병변 단위로는 전부 오탐/누락이어야 한다."""
    gt = _empty_volume()
    gt = _cube(gt, 0, 2, 2, 6, 2, 6)
    gt = _cube(gt, 0, 2, 20, 24, 2, 6)

    pred = _empty_volume()
    pred = _cube(pred, 0, 2, 2, 6, 20, 24)
    pred = _cube(pred, 0, 2, 20, 24, 20, 24)

    result = match_lesions_iou(gt, pred, iou_threshold=0.1)

    assert result['n_gt'] == 2 and result['n_pred'] == 2  # 개수 집계로는 "일치"
    assert result['tp'] == 0  # 그러나 병변 단위 IoU 매칭으로는 전부 불일치
    assert result['fn'] == 2 and result['fp'] == 2


def test_low_overlap_below_threshold_does_not_count_as_tp():
    gt = _cube(_empty_volume(), 1, 3, 0, 10, 0, 10)      # 10x10x2 = 200 voxel
    pred = _cube(_empty_volume(), 1, 3, 9, 11, 9, 11)    # GT와 1 voxel(z=1,2 x y=9,x=9)만 겹침

    result = match_lesions_iou(gt, pred, iou_threshold=0.5)

    assert result['tp'] == 0
    assert result['fn'] == 1 and result['fp'] == 1


def test_hungarian_avoids_double_assignment_with_multiple_lesions():
    gt = _empty_volume()
    gt = _cube(gt, 0, 2, 2, 6, 2, 6)
    gt = _cube(gt, 0, 2, 20, 24, 20, 24)

    pred = _empty_volume()
    pred = _cube(pred, 0, 2, 2, 6, 2, 6)      # GT 병변 1과 완전 일치
    pred = _cube(pred, 0, 2, 20, 24, 20, 24)  # GT 병변 2와 완전 일치

    result = match_lesions_iou(gt, pred, iou_threshold=0.1)

    assert result['tp'] == 2 and result['fp'] == 0 and result['fn'] == 0
    matched_gt_ids = {m[0] for m in result['matches']}
    matched_pred_ids = {m[1] for m in result['matches']}
    assert len(matched_gt_ids) == 2 and len(matched_pred_ids) == 2  # 1:1 매칭, 중복 없음


def test_empty_gt_and_empty_pred_is_trivially_perfect():
    gt = _empty_volume()
    pred = _empty_volume()

    result = match_lesions_iou(gt, pred, iou_threshold=0.1)

    assert result['n_gt'] == 0 and result['n_pred'] == 0
    assert result['tp'] == 0 and result['fp'] == 0 and result['fn'] == 0
    assert result['precision'] == 1.0 and result['recall'] == 1.0


def test_empty_gt_with_prediction_is_pure_false_positive():
    gt = _empty_volume()
    pred = _cube(_empty_volume(), 1, 3, 5, 10, 5, 10)

    result = match_lesions_iou(gt, pred, iou_threshold=0.1)

    assert result['fp'] == 1 and result['tp'] == 0 and result['fn'] == 0
    assert result['precision'] == 0.0
    assert result['recall'] == 1.0  # GT 병변이 없으므로 recall은 정의상 1.0


def test_min_voxels_filters_tiny_noise_components():
    volume = _empty_volume()
    volume[0, 0, 0] = 1  # 1voxel짜리 노이즈

    labeled, n = label_3d_lesions(volume, min_voxels=1)
    assert n == 1

    labeled, n = label_3d_lesions(volume, min_voxels=5)
    assert n == 0
