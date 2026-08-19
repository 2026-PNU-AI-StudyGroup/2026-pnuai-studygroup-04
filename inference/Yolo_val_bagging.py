import os
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

DATA_YAML = r"/ds/want2704/mri/mydata/data.yaml"

RUNS_BASE = r"runs/detect/runs_bagging_map50_951.0"
RUN_NAME_FMT = "y12n_bag_{:02d}"
WEIGHTS_REL = r"weights/best.pt"
BAG_RANGE = range(0, 5)

IMG_SIZE = 640
DEVICE = 2

PRED_CONF = 0.50
PRED_IOU = 0.80

WBF_IOU = 0.95
WBF_SKIP_BOX_THR = 0.005
WBF_CONF_TYPE = "avg"
MODEL_WEIGHTS = None

MERGE_IOU = 0.95
MERGE_CONTAIN_THR = 0.95
SCORE_THR = 0.10
SCORE_TYPE = "prob"

SAVE_MERGED_IMAGES = True
SAVE_SUBDIR = "val_images_strict_alpha50_raw_green_merged_red"
SAVE_EXT = ".jpg"

DRAW_THICKNESS_RAW = 1
DRAW_THICKNESS_MERGED = 2
BOX_ALPHA_RAW = 0.35
BOX_ALPHA_MERGED = 0.65

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_data_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    nc = data.get("nc")

    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    if nc is None:
        nc = len(names)

    return data, names, nc


def resolve_split_images(data, yaml_path, split="val"):
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))
    dataset_root = data.get("path", yaml_dir)

    if dataset_root is None:
        dataset_root = yaml_dir

    if not os.path.isabs(dataset_root):
        dataset_root = os.path.normpath(os.path.join(yaml_dir, dataset_root))

    split_path = data[split]

    if isinstance(split_path, list):
        imgs = []
        for p in split_path:
            p_abs = p if os.path.isabs(p) else os.path.normpath(os.path.join(dataset_root, p))
            imgs.extend(collect_images_from_path(p_abs))
        return sorted(imgs)

    p_abs = split_path if os.path.isabs(split_path) else os.path.normpath(os.path.join(dataset_root, split_path))
    return collect_images_from_path(p_abs)


def collect_images_from_path(p_abs):
    if os.path.isdir(p_abs):
        imgs = []
        for root, _, files in os.walk(p_abs):
            for fn in files:
                if fn.lower().endswith(IMG_EXTS):
                    imgs.append(os.path.join(root, fn))
        return sorted(imgs)

    if os.path.isfile(p_abs) and p_abs.lower().endswith(".txt"):
        txt_base = os.path.dirname(p_abs)
        imgs = []
        with open(p_abs, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue

                imgp = ln if os.path.isabs(ln) else os.path.normpath(os.path.join(txt_base, ln))

                if os.path.isfile(imgp):
                    imgs.append(imgp)

        return sorted(imgs)

    raise FileNotFoundError(f"Image path not found: {p_abs}")


def image_to_label_path(img_path):
    p = img_path.replace("\\", "/")

    if "/images/" in p:
        p = p.replace("/images/", "/labels/")
    else:
        p = p.replace("/valid/", "/valid/labels/")

    p = os.path.splitext(p)[0] + ".txt"
    return p.replace("/", os.sep)


def load_yolo_labels(label_path, w, h):
    if not os.path.isfile(label_path):
        return np.zeros((0, 5), dtype=np.float32)

    out = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])

            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            out.append([cls, x1, y1, x2, y2])

    if len(out) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    return np.array(out, dtype=np.float32)


def box_iou(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a[:, None, :]
    b = b[None, :, :]

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])

    return inter / (area_a + area_b - inter + 1e-9)


def box_intersection_over_min(a, b):
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    a = a[:, None, :]
    b = b[None, :, :]

    ix1 = np.maximum(a[..., 0], b[..., 0])
    iy1 = np.maximum(a[..., 1], b[..., 1])
    ix2 = np.minimum(a[..., 2], b[..., 2])
    iy2 = np.minimum(a[..., 3], b[..., 3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = np.maximum(0.0, a[..., 2] - a[..., 0]) * np.maximum(0.0, a[..., 3] - a[..., 1])
    area_b = np.maximum(0.0, b[..., 2] - b[..., 0]) * np.maximum(0.0, b[..., 3] - b[..., 1])
    area_min = np.minimum(area_a, area_b)

    return inter / (area_min + 1e-9)


def process_batch(det, lab, iouv):
    correct = np.zeros((det.shape[0], iouv.shape[0]), dtype=bool)

    if det.shape[0] == 0 or lab.shape[0] == 0:
        return correct

    det_cls = det[:, 5].astype(int)
    lab_cls = lab[:, 0].astype(int)
    iou = box_iou(det[:, :4], lab[:, 1:5])

    for cls in np.unique(lab_cls):
        ti = np.where(lab_cls == cls)[0]
        pi = np.where(det_cls == cls)[0]

        if pi.size == 0:
            continue

        iou_sub = iou[np.ix_(pi, ti)]
        best_t = iou_sub.argmax(axis=1)
        best_iou = iou_sub[np.arange(iou_sub.shape[0]), best_t]
        order = np.argsort(-best_iou)

        used_t = set()

        for j in order:
            t = ti[best_t[j]]

            if t in used_t:
                continue

            if best_iou[j] <= 0:
                continue

            used_t.add(t)
            correct[pi[j]] = best_iou[j] >= iouv

    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    x = np.linspace(0, 1, 101)
    return np.trapz(np.interp(x, mrec, mpre), x)


def ap_per_class(correct, conf, pred_cls, target_cls, nc, iouv):
    ap = np.zeros((nc, iouv.size), dtype=np.float32)
    p = np.zeros((nc,), dtype=np.float32)
    r = np.zeros((nc,), dtype=np.float32)

    for c in range(nc):
        i = pred_cls == c
        n_p = int(i.sum())
        n_t = int((target_cls == c).sum())

        if n_t == 0 or n_p == 0:
            continue

        idx = np.argsort(-conf[i])
        correct_c = correct[i][idx].astype(np.float32)

        tp = correct_c
        fp = 1.0 - tp

        tp_cum = np.cumsum(tp, axis=0)
        fp_cum = np.cumsum(fp, axis=0)

        for k in range(iouv.size):
            rc = tp_cum[:, k] / (n_t + 1e-9)
            pr = tp_cum[:, k] / (tp_cum[:, k] + fp_cum[:, k] + 1e-9)
            ap[c, k] = compute_ap(rc, pr)

        rc0 = tp_cum[:, 0] / (n_t + 1e-9)
        pr0 = tp_cum[:, 0] / (tp_cum[:, 0] + fp_cum[:, 0] + 1e-9)
        f1 = 2 * pr0 * rc0 / (pr0 + rc0 + 1e-9)

        if f1.size > 0:
            j = int(f1.argmax())
            p[c] = float(pr0[j])
            r[c] = float(rc0[j])

    return p, r, ap


def flatten_raw_preds(all_preds):
    raw = []

    for model_idx, (boxes_xyxy, scores, clss) in enumerate(all_preds):
        if boxes_xyxy.shape[0] == 0:
            continue

        model_col = np.full((boxes_xyxy.shape[0], 1), model_idx, dtype=np.float32)

        raw.append(
            np.concatenate(
                [
                    boxes_xyxy.astype(np.float32),
                    scores.astype(np.float32)[:, None],
                    clss.astype(np.float32)[:, None],
                    model_col,
                ],
                axis=1,
            )
        )

    if len(raw) == 0:
        return np.zeros((0, 7), dtype=np.float32)

    return np.concatenate(raw, axis=0)


def wbf_fuse(all_preds, w, h):
    boxes_list = []
    scores_list = []
    labels_list = []

    for boxes_xyxy, scores, clss in all_preds:
        if boxes_xyxy.shape[0] == 0:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])
            continue

        b = boxes_xyxy.copy()
        b[:, [0, 2]] /= w
        b[:, [1, 3]] /= h

        boxes_list.append(np.clip(b, 0, 1).tolist())
        scores_list.append(scores.tolist())
        labels_list.append(clss.astype(int).tolist())

    weights = [1.0] * len(all_preds) if MODEL_WEIGHTS is None else MODEL_WEIGHTS

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=WBF_IOU,
        skip_box_thr=WBF_SKIP_BOX_THR,
        conf_type=WBF_CONF_TYPE
    )

    if len(boxes) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h

    return np.concatenate(
        [
            boxes,
            np.array(scores, dtype=np.float32)[:, None],
            np.array(labels, dtype=np.float32)[:, None],
        ],
        axis=1,
    )


def merge_big_union(det, iou_thr=0.40, contain_thr=0.60, score_thr=0.20, score_type="prob"):
    if det is None or det.shape[0] == 0:
        return np.zeros((0, 6), dtype=np.float32)

    out = []

    for cls in np.unique(det[:, 5].astype(int)):
        d = det[det[:, 5].astype(int) == cls].copy()
        d = d[np.argsort(-d[:, 4])]

        used = np.zeros((d.shape[0],), dtype=bool)

        for start in range(d.shape[0]):
            if used[start]:
                continue

            group_idx = {start}
            changed = True

            while changed:
                changed = False
                current = d[list(group_idx), :4]
                rest_idx = np.where(~used)[0]

                for ri in rest_idx:
                    if ri in group_idx:
                        continue

                    candidate = d[ri:ri + 1, :4]
                    iou = box_iou(candidate, current).max()
                    iom = box_intersection_over_min(candidate, current).max()

                    if iou >= iou_thr or iom >= contain_thr:
                        group_idx.add(int(ri))
                        changed = True

            group_idx = sorted(list(group_idx))
            used[group_idx] = True
            grp = d[group_idx]

            x1 = float(grp[:, 0].min())
            y1 = float(grp[:, 1].min())
            x2 = float(grp[:, 2].max())
            y2 = float(grp[:, 3].max())

            confs = grp[:, 4].astype(np.float32)

            if score_type == "avg":
                score = float(confs.mean())
            elif score_type == "max":
                score = float(confs.max())
            elif score_type == "sum":
                score = float(confs.sum())
            elif score_type == "avg_x_k":
                score = float(confs.mean() * grp.shape[0])
            elif score_type == "prob":
                score = float(1.0 - np.prod(1.0 - np.clip(confs, 0.0, 0.999999)))
            else:
                raise ValueError(f"Unknown score_type: {score_type}")

            if score >= score_thr:
                out.append([x1, y1, x2, y2, score, float(cls)])

    if len(out) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    return np.array(out, dtype=np.float32)


def build_weight_paths():
    return [
        os.path.join(RUNS_BASE, RUN_NAME_FMT.format(i), WEIGHTS_REL)
        for i in BAG_RANGE
    ]


def build_save_dir():
    return os.path.join(RUNS_BASE, SAVE_SUBDIR)


def draw_raw_and_merged_boxes(im, raw_det, merged_det):
    out = im.copy()
    h, w = out.shape[:2]

    raw_overlay = out.copy()
    merged_overlay = out.copy()

    # raw bagmodel boxes: green
    for row in raw_det:
        x1, y1, x2, y2, conf, cls, model_idx = row

        x1 = int(max(0, min(w - 1, round(float(x1)))))
        y1 = int(max(0, min(h - 1, round(float(y1)))))
        x2 = int(max(0, min(w - 1, round(float(x2)))))
        y2 = int(max(0, min(h - 1, round(float(y2)))))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            raw_overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            DRAW_THICKNESS_RAW,
        )

    out = cv2.addWeighted(raw_overlay, BOX_ALPHA_RAW, out, 1.0 - BOX_ALPHA_RAW, 0)

    # final merged boxes: red
    merged_overlay = out.copy()

    for row in merged_det:
        x1, y1, x2, y2, conf, cls = row

        x1 = int(max(0, min(w - 1, round(float(x1)))))
        y1 = int(max(0, min(h - 1, round(float(y1)))))
        x2 = int(max(0, min(w - 1, round(float(x2)))))
        y2 = int(max(0, min(h - 1, round(float(y2)))))

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            merged_overlay,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            DRAW_THICKNESS_MERGED,
        )

    out = cv2.addWeighted(merged_overlay, BOX_ALPHA_MERGED, out, 1.0 - BOX_ALPHA_MERGED, 0)

    cv2.putText(
        out,
        "Green: raw bag boxes | Red: merged boxes",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "Green: raw bag boxes | Red: merged boxes",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return out


def split_gt_by_size(labels):
    if labels.shape[0] == 0:
        return {
            "small": np.zeros((0,), dtype=bool),
            "large": np.zeros((0,), dtype=bool),
        }

    xyxy = labels[:, 1:5]
    bbox_w_px = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    bbox_h_px = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
    bbox_area_px = bbox_w_px * bbox_h_px

    small_mask = bbox_area_px <= 3.1 * 3.1
    large_mask = bbox_area_px > 3.1 * 3.1

    return {
        "small": small_mask,
        "large": large_mask,
    }


def assign_pred_to_best_gt(det, labels):
    if det.shape[0] == 0 or labels.shape[0] == 0:
        return (
            np.full((det.shape[0],), -1, dtype=np.int32),
            np.zeros((det.shape[0],), dtype=np.float32),
        )

    iou = box_iou(det[:, :4], labels[:, 1:5])
    best_gt_idx = iou.argmax(axis=1).astype(np.int32)
    best_iou = iou[np.arange(iou.shape[0]), best_gt_idx].astype(np.float32)

    return best_gt_idx, best_iou


def filter_predictions_for_size_bin(det, labels, gt_bin_mask):
    if det.shape[0] == 0:
        return det

    if labels.shape[0] == 0:
        return det

    best_gt_idx, best_iou = assign_pred_to_best_gt(det, labels)

    keep = np.zeros((det.shape[0],), dtype=bool)

    for i in range(det.shape[0]):
        if best_iou[i] <= 0:
            keep[i] = True
        else:
            keep[i] = bool(gt_bin_mask[best_gt_idx[i]])

    return det[keep]


def init_metric_bucket():
    return {
        "correct_all": [],
        "conf_all": [],
        "pred_cls_all": [],
        "target_cls_all": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "gt_count": 0,
        "pred_count": 0,
        "img_tp": 0,
        "img_fp": 0,
        "img_fn": 0,
        "img_tn": 0,
        "img_pos": 0,
        "img_neg": 0,
    }


def update_metric_bucket(bucket, det_use, labels_use, iouv):
    has_pred = det_use.shape[0] > 0
    has_gt = labels_use.shape[0] > 0

    correct = process_batch(det_use, labels_use, iouv)

    bucket["correct_all"].append(correct)
    bucket["conf_all"].append(det_use[:, 4] if has_pred else np.zeros((0,), dtype=np.float32))
    bucket["pred_cls_all"].append(det_use[:, 5] if has_pred else np.zeros((0,), dtype=np.float32))
    bucket["target_cls_all"].append(labels_use[:, 0] if has_gt else np.zeros((0,), dtype=np.float32))

    bucket["gt_count"] += int(labels_use.shape[0])
    bucket["pred_count"] += int(det_use.shape[0])

    if det_use.shape[0] == 0 and labels_use.shape[0] == 0:
        bucket["img_neg"] += 1
        bucket["img_tn"] += 1
        return

    if det_use.shape[0] == 0:
        bucket["fn"] += int(labels_use.shape[0])
        bucket["img_pos"] += 1
        bucket["img_fn"] += 1
        return

    if labels_use.shape[0] == 0:
        bucket["fp"] += int(det_use.shape[0])
        bucket["img_neg"] += 1
        bucket["img_fp"] += 1
        return

    correct50 = correct[:, 0]
    tp = int(correct50.sum())
    fp = int((~correct50).sum())

    iou = box_iou(det_use[:, :4], labels_use[:, 1:5])
    matched_gt = set()

    det_cls = det_use[:, 5].astype(int)
    gt_cls = labels_use[:, 0].astype(int)

    for pi in range(det_use.shape[0]):
        if not correct50[pi]:
            continue

        gi = int(iou[pi].argmax())

        if det_cls[pi] == gt_cls[gi] and iou[pi, gi] >= iouv[0]:
            matched_gt.add(gi)

    fn = int(labels_use.shape[0] - len(matched_gt))

    bucket["tp"] += tp
    bucket["fp"] += fp
    bucket["fn"] += fn

    bucket["img_pos"] += 1

    if len(matched_gt) > 0:
        bucket["img_tp"] += 1
    else:
        bucket["img_fn"] += 1


def finalize_bucket(bucket, nc, iouv):
    if len(bucket["correct_all"]) == 0:
        correct_all = np.zeros((0, iouv.size), dtype=bool)
        conf_all = np.zeros((0,), dtype=np.float32)
        pred_cls_all = np.zeros((0,), dtype=np.float32)
        target_cls_all = np.zeros((0,), dtype=np.float32)
    else:
        correct_all = np.concatenate(bucket["correct_all"], axis=0)
        conf_all = np.concatenate(bucket["conf_all"], axis=0)
        pred_cls_all = np.concatenate(bucket["pred_cls_all"], axis=0)
        target_cls_all = np.concatenate(bucket["target_cls_all"], axis=0)

    _, _, ap = ap_per_class(correct_all, conf_all, pred_cls_all, target_cls_all, nc, iouv)

    precision = bucket["tp"] / (bucket["tp"] + bucket["fp"] + 1e-9)
    recall = bucket["tp"] / (bucket["tp"] + bucket["fn"] + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    specificity = bucket["img_tn"] / (bucket["img_tn"] + bucket["img_fp"] + 1e-9)

    map50 = float(np.nanmean(ap[:, 0])) if ap.size else float("nan")
    map5095 = float(np.nanmean(ap.mean(axis=1))) if ap.size else float("nan")

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "map50": map50,
        "map5095": map5095,
        "tp": int(bucket["tp"]),
        "fp": int(bucket["fp"]),
        "fn": int(bucket["fn"]),
        "gt_count": int(bucket["gt_count"]),
        "pred_count": int(bucket["pred_count"]),
        "img_tp": int(bucket["img_tp"]),
        "img_fp": int(bucket["img_fp"]),
        "img_fn": int(bucket["img_fn"]),
        "img_tn": int(bucket["img_tn"]),
        "img_pos": int(bucket["img_pos"]),
        "img_neg": int(bucket["img_neg"]),
    }


def print_metric_block(title, result):
    print(f"\n========== {title} ==========")
    print(f"Precision               : {result['precision']:.4f}")
    print(f"Recall                  : {result['recall']:.4f}")
    print(f"F1-score                : {result['f1']:.4f}")
    print(f"Specificity(image-level): {result['specificity']:.4f}")
    print(f"mAP@0.50                : {result['map50']:.4f}")
    print(f"mAP@0.50:0.95           : {result['map5095']:.4f}")
    print(f"[LESION-LEVEL] TP={result['tp']} FP={result['fp']} FN={result['fn']}")
    print(f"[IMAGE-LEVEL ] TP={result['img_tp']} FP={result['img_fp']} FN={result['img_fn']} TN={result['img_tn']}")
    print(f"[GT-COUNT    ] {result['gt_count']}")
    print(f"[PRED-COUNT  ] {result['pred_count']}")
    print(f"[IMG POS/NEG ] POS={result['img_pos']} NEG={result['img_neg']}")
    print("======================================================")


def main():
    data, names, nc = load_data_yaml(DATA_YAML)
    val_images = resolve_split_images(data, DATA_YAML, "val")

    print(f"[INFO] Found val images: {len(val_images)}")
    if len(val_images) == 0:
        raise RuntimeError("No validation images found.")

    weight_paths = build_weight_paths()
    models = [YOLO(p) for p in weight_paths]

    iouv = np.linspace(0.5, 0.95, 10)

    save_dir = build_save_dir()
    save_dir_abs = os.path.abspath(save_dir)

    if SAVE_MERGED_IMAGES:
        os.makedirs(save_dir_abs, exist_ok=True)

    overall_bucket = init_metric_bucket()
    small_bucket = init_metric_bucket()
    large_bucket = init_metric_bucket()

    saved_count = 0
    total_gt_small = 0
    total_gt_large = 0

    for img_path in tqdm(val_images):
        im = cv2.imread(img_path)

        if im is None:
            continue

        h, w = im.shape[:2]
        labels = load_yolo_labels(image_to_label_path(img_path), w, h)

        all_preds = []

        for m in models:
            r = m.predict(
                img_path,
                conf=PRED_CONF,
                iou=PRED_IOU,
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False,
            )[0]

            if r.boxes is None or len(r.boxes) == 0:
                all_preds.append(
                    (
                        np.zeros((0, 4), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32),
                        np.zeros((0,), dtype=np.float32),
                    )
                )
                continue

            all_preds.append(
                (
                    r.boxes.xyxy.cpu().numpy().astype(np.float32),
                    r.boxes.conf.cpu().numpy().astype(np.float32),
                    r.boxes.cls.cpu().numpy().astype(np.float32),
                )
            )

        raw_det = flatten_raw_preds(all_preds)

        det_wbf = wbf_fuse(all_preds, w, h)

        det = merge_big_union(
            det_wbf,
            iou_thr=MERGE_IOU,
            contain_thr=MERGE_CONTAIN_THR,
            score_thr=SCORE_THR,
            score_type=SCORE_TYPE,
        )

        update_metric_bucket(overall_bucket, det, labels, iouv)

        size_info = split_gt_by_size(labels)

        labels_small = labels[size_info["small"]]
        labels_large = labels[size_info["large"]]

        det_small = filter_predictions_for_size_bin(det, labels, size_info["small"])
        det_large = filter_predictions_for_size_bin(det, labels, size_info["large"])

        update_metric_bucket(small_bucket, det_small, labels_small, iouv)
        update_metric_bucket(large_bucket, det_large, labels_large, iouv)

        total_gt_small += int(size_info["small"].sum())
        total_gt_large += int(size_info["large"].sum())

        if SAVE_MERGED_IMAGES:
            vis = draw_raw_and_merged_boxes(im, raw_det, det)
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(save_dir_abs, base + SAVE_EXT)

            ok = cv2.imwrite(out_path, vis)

            if ok:
                saved_count += 1
            else:
                print("[imwrite FAIL]", out_path)

    overall_result = finalize_bucket(overall_bucket, nc, iouv)
    small_result = finalize_bucket(small_bucket, nc, iouv)
    large_result = finalize_bucket(large_bucket, nc, iouv)

    print("\n[SETTING]")
    print(f"PRED_CONF={PRED_CONF}")
    print(f"PRED_IOU={PRED_IOU}")
    print(f"WBF_IOU={WBF_IOU}")
    print(f"WBF_SKIP_BOX_THR={WBF_SKIP_BOX_THR}")
    print(f"MERGE_IOU={MERGE_IOU}")
    print(f"MERGE_CONTAIN_THR={MERGE_CONTAIN_THR}")
    print(f"SCORE_THR={SCORE_THR}")
    print(f"DRAW_THICKNESS_RAW={DRAW_THICKNESS_RAW}")
    print(f"DRAW_THICKNESS_MERGED={DRAW_THICKNESS_MERGED}")
    print(f"BOX_ALPHA_RAW={BOX_ALPHA_RAW}")
    print(f"BOX_ALPHA_MERGED={BOX_ALPHA_MERGED}")
    print("- strict pseudo-labeling")
    print("- raw bagmodel boxes: green")
    print("- final merged boxes: red")
    print("- label text removed except legend")

    print("\n[SIZE RULE]")
    print("- small lesion: bbox_area <= 3.1 x 3.1 px")
    print("- large lesion: bbox_area > 3.1 x 3.1 px")
    print(f"- total small GT count: {total_gt_small}")
    print(f"- total large GT count: {total_gt_large}")

    print_metric_block("OVERALL METRICS", overall_result)
    print_metric_block("LESION SIZE <= 3x3 px", small_result)
    print_metric_block("LESION SIZE > 3x3 px", large_result)

    if SAVE_MERGED_IMAGES:
        print(f"[SAVED] {saved_count} images -> {save_dir_abs}")


if __name__ == "__main__":
    main()