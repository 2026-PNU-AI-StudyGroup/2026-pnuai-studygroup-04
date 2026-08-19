"""
main.py의 병변 추적(process_patient_dir) + functions.py의 volume_cal만 떼어낸 모듈.
main.py 원본은 상단에서 utils.py/hyper_parameters.py/model/cbas.py(우리 src/ 구조와 다른,
이 컨테이너엔 없는 파일들)를 import해서 그대로는 로드가 안 되기 때문에,
그 인퍼런스 편의 함수들은 빼고 라벨 추적 + 볼륨 계산 로직만 그대로 옮겼다.
(로직 자체는 원본 main.py의 process_patient_dir / functions.py의 volume_cal과 100% 동일)
"""
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

from functions import (
    imread_unicode, extract_boxes_from_mask, save_yolo_boxes,
    load_yolo_boxes, count_mask_pixels_yolo_box, volume_cal,
    postprocess_binary_mask,
)


def calculate_iou(box1, box2):
    box1_x1, box1_y1 = box1['x'] - box1['w'] / 2, box1['y'] - box1['h'] / 2
    box1_x2, box1_y2 = box1['x'] + box1['w'] / 2, box1['y'] + box1['h'] / 2
    box2_x1, box2_y1 = box2['x'] - box2['w'] / 2, box2['y'] - box2['h'] / 2
    box2_x2, box2_y2 = box2['x'] + box2['w'] / 2, box2['y'] + box2['h'] / 2

    inter_x1 = max(box1_x1, box2_x1)
    inter_y1 = max(box1_y1, box2_y1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = box1['w'] * box1['h']
    box2_area = box2['w'] * box2['h']
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def process_patient_dir(mask_dir, save_txt='lesions.txt', w_iou=0.5, w_dist=20.0, w_size=100.0,
                         max_cost_threshold=1.8, split_iou_min=0.05, split_dist_max=0.08,
                         split_child_ratio=0.85, min_lesion_total_px=0, min_lesion_slices=1):
    filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
    all_boxes = {}
    lesion_id_counter = 0

    for i in range(len(filenames)):
        img_name = filenames[i]
        z = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(mask_dir, img_name)
        txt_path = os.path.join(mask_dir, f'{z}.txt')

        mask = imread_unicode(img_path)
        if mask is None:
            continue

        current_boxes = load_yolo_boxes(txt_path, mask.shape, z)
        for b in current_boxes:
            if 'matched' not in b: b['matched'] = False
            if 'id' not in b:      b['id'] = None

        all_boxes[z] = current_boxes

        if i == 0:
            for box in current_boxes:
                box['id'] = lesion_id_counter
                lesion_id_counter += 1
            continue

        prev_z = int(os.path.splitext(filenames[i - 1])[0])
        prev_boxes = all_boxes.get(prev_z, [])
        for b in prev_boxes:    b['matched'] = False
        for b in current_boxes: b['matched'] = False

        if not prev_boxes or not current_boxes:
            for box in current_boxes:
                if box['id'] is None:
                    box['id'] = lesion_id_counter
                    lesion_id_counter += 1
            continue

        num_prev, num_curr = len(prev_boxes), len(current_boxes)
        cost_matrix = np.full((num_prev, num_curr), 1e8)
        iou_matrix = np.zeros((num_prev, num_curr))
        for r in range(num_prev):
            for c in range(num_curr):
                pbox, cbox = prev_boxes[r], current_boxes[c]
                iou = calculate_iou(pbox, cbox)
                iou_matrix[r, c] = iou
                if iou < 0.1:
                    continue
                iou_cost = 1 - iou
                size_cost = abs(pbox['w'] * pbox['h'] - cbox['w'] * cbox['h'])
                dist_cost = np.sqrt((pbox['x'] - cbox['x']) ** 2 + (pbox['y'] - cbox['y']) ** 2)
                total_cost = w_iou * iou_cost + w_size * size_cost + w_dist * dist_cost
                cost_matrix[r, c] = total_cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost < max_cost_threshold:
                prev_box, current_box = prev_boxes[r], current_boxes[c]
                current_box['id'] = prev_box['id']
                current_box['matched'] = True
                prev_box['matched'] = True

        leftovers = [idx for idx, b in enumerate(current_boxes) if not b['matched']]
        if leftovers:
            for c in leftovers:
                best_r = None
                best_score = -1.0
                for r in range(num_prev):
                    pbox, cbox = prev_boxes[r], current_boxes[c]
                    iou = iou_matrix[r, c]
                    dist = np.hypot(pbox['x'] - cbox['x'], pbox['y'] - cbox['y'])
                    area_p = pbox['w'] * pbox['h']
                    area_c = cbox['w'] * cbox['h']
                    if (iou >= split_iou_min or dist <= split_dist_max) and (area_c <= area_p * split_child_ratio):
                        if iou > best_score:
                            best_score = iou
                            best_r = r
                if best_r is not None:
                    current_boxes[c]['id'] = prev_boxes[best_r]['id']
                    current_boxes[c]['matched'] = True

        for box in current_boxes:
            if not box['matched']:
                box['id'] = lesion_id_counter
                lesion_id_counter += 1

    result = []
    lesion_total_px = {}
    lesion_slice_set = {}
    for z in sorted(all_boxes.keys()):
        mask_path = os.path.join(mask_dir, f"{z}.png")
        mask = imread_unicode(mask_path)
        if mask is None:
            continue
        for b in all_boxes[z]:
            if b['id'] is not None:
                lesion_size = count_mask_pixels_yolo_box(mask, b, mask.shape)
                lesion_id = int(b['id'])
                lesion_total_px[lesion_id] = lesion_total_px.get(lesion_id, 0) + lesion_size
                lesion_slice_set.setdefault(lesion_id, set()).add(int(b['z']))
                result.append((lesion_id, f"{b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f} {b['z']} {b['id']} {lesion_size}"))

    if min_lesion_total_px > 0 or min_lesion_slices > 1:
        keep_ids = {
            lesion_id for lesion_id, total_px in lesion_total_px.items()
            if total_px >= min_lesion_total_px and len(lesion_slice_set.get(lesion_id, set())) >= min_lesion_slices
        }
        result = [(lesion_id, line) for lesion_id, line in result if lesion_id in keep_ids]

    save_path = os.path.join(mask_dir, save_txt)
    with open(save_path, 'w') as f:
        f.write('\n'.join(line for _, line in result))

    return save_path


def process_mask_dir(mask_dir, w_iou=0.5, w_dist=20.0, w_size=100.0,
                      max_cost_threshold=1.8, split_iou_min=0.05,
                      split_dist_max=0.08, split_child_ratio=0.85,
                      area_threshold_px=0, postprocess_kwargs=None,
                      min_lesion_total_px=0, min_lesion_slices=1):
    """PNG(마스크) -> YOLO boxes -> 병변 추적 -> volume.txt 저장. (main.py._process_mask_dir와 동일 로직)"""
    for png in [f for f in os.listdir(mask_dir) if f.lower().endswith('.png')]:
        mask_path = os.path.join(mask_dir, png)
        txt_path = os.path.splitext(mask_path)[0] + '.txt'
        if postprocess_kwargs:
            mask = postprocess_binary_mask(mask_path, **postprocess_kwargs)
            if mask is None:
                continue
            boxes = extract_boxes_from_mask(mask, area_threshold_px=area_threshold_px)
        else:
            boxes = extract_boxes_from_mask(mask_path, area_threshold_px=area_threshold_px)
        save_yolo_boxes(boxes, txt_path)

    process_patient_dir(
        mask_dir,
        w_iou=w_iou, w_dist=w_dist, w_size=w_size,
        max_cost_threshold=max_cost_threshold,
        split_iou_min=split_iou_min, split_dist_max=split_dist_max,
        split_child_ratio=split_child_ratio,
        min_lesion_total_px=min_lesion_total_px,
        min_lesion_slices=min_lesion_slices,
    )
    return volume_cal(mask_dir)
