import os
import cv2
import numpy as np
import pydicom
import math
from collections import defaultdict
import pandas as pd
from pathlib import Path
import glob
from pydicom.valuerep import PersonName
from pydicom.tag import Tag

def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        print(f"[ERROR] 이미지 인코딩 실패: {path}: {e}")
        return None


def imwrite_unicode(path, image):
    try:
        ext = os.path.splitext(path)[1]
        result, encoded_img = cv2.imencode(ext, image)
        if result:
            with open(path, mode='wb') as f:
                encoded_img.tofile(f)
                return True
        else:
            print(f"[ERROR] 이미지 인코딩 실패: {path}")
            return False
    except Exception as e:
        print(f"[ERROR] 이미지 저장 실패: {path}: {e}")
        return False


def postprocess_binary_mask(mask, min_area_px=0, close_kernel=0, open_kernel=0, fill_holes=False):
    """Clean a binary mask before component extraction.

    This is mainly useful for model outputs: small isolated components inflate
    lesion counts, while a light closing step can reconnect fragmented lesions.
    """
    if isinstance(mask, (str, bytes, bytearray)):
        mask = imread_unicode(mask)
        if mask is None:
            return None
    if mask.ndim == 3:
        mask = mask[..., 0]

    binm = (mask > 0).astype(np.uint8) * 255

    if close_kernel and close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, kernel)

    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, kernel)

    if fill_holes:
        flood = binm.copy()
        h, w = flood.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        binm = cv2.bitwise_or(binm, holes)

    if min_area_px and min_area_px > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binm > 0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(binm)
        for lbl in range(1, num_labels):
            area_px = stats[lbl, cv2.CC_STAT_AREA]
            if area_px >= min_area_px:
                cleaned[labels == lbl] = 255
        binm = cleaned

    return binm


def extract_boxes_from_mask(mask, area_threshold_px=0):
    if isinstance(mask, (str, bytes, bytearray)):
        mask = imread_unicode(mask)
        if mask is None:
            print("[ERROR] 마스크 로딩 실패")
            return []
    if mask.ndim == 3:
        mask = mask[..., 0]

    H, W = mask.shape[:2]
    binm = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binm, connectivity=8)

    boxes = []
    for lbl in range(1, num_labels):
        x, y, w, h, area_px = stats[lbl, 0], stats[lbl, 1], stats[lbl, 2], stats[lbl, 3], stats[lbl, 4]

        if area_px < area_threshold_px:   # h*x -> area_px로 수정
            continue

        x_c_norm = (x + w / 2) / W
        y_c_norm = (y + h / 2) / H
        w_norm = w / W
        h_norm = h / H

        # clip
        x_c_norm = min(max(x_c_norm, 0.0), 1.0)
        y_c_norm = min(max(y_c_norm, 0.0), 1.0)
        w_norm = min(w_norm, 1.0)
        h_norm = min(h_norm, 1.0)

        lesion_size = int(area_px)
        if lesion_size <= 0:
            lesion_size = 1

        boxes.append((x_c_norm, y_c_norm, w_norm, h_norm, lesion_size))

    return boxes

def save_yolo_boxes(boxes, save_path):
    with open(save_path, 'w') as f:
        for box in boxes:
            f.write(' '.join(f'{v:.6f}' for v in box) + '\n')

def load_yolo_boxes(txt_path, img_shape, z):
    h, w = img_shape
    boxes = []
    if not os.path.exists(txt_path):
        return []
    with open(txt_path, 'r') as f:
        for line in f:
            x, y, bw, bh, size = map(float, line.strip().split())
            boxes.append({
                'x': x, 'y': y, 'w': bw, 'h': bh,
                'z': z,
                'matched': False,
                'size': size,
                'id': None
            })
    return boxes

def count_mask_pixels_yolo_box(mask, box, img_shape):
    h, w = img_shape
    x1 = int((box['x'] - box['w'] / 2) * w)
    y1 = int((box['y'] - box['h'] / 2) * h)
    x2 = int((box['x'] + box['w'] / 2) * w)
    y2 = int((box['y'] + box['h'] / 2) * h)

    if x1 < x2:
        x1 = max(0, x1 - 1)
        x2 = min(w, x2 + 1)
    elif x1 > x2:
        x1 = min(w, x1 + 1)
        x2 = max(0, x2 - 1)

    if y1 < y2:
        y1 = max(0, y1 - 1)
        y2 = min(h, y2 + 1)
    elif y1 > y2:
        y1 = min(h, y1 + 1)
        y2 = max(0, y2 - 1)

    cropped = int(np.count_nonzero(mask[y1:y2, x1:x2]))

    return cropped


def draw_yolo_boxes_on_image(image_path, txt_path, box_color=(255, 0, 0), thickness=1):
    image = imread_unicode(image_path)
    if image is None:
        print(f"[ERROR] 이미지 로딩 실패: {image_path}")
        return

    h, w = image.shape[:2]

    if not os.path.exists(txt_path):
        print(f"[WARNING] .txt 파일 없음: {txt_path}")
        return

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        x_c, y_c, box_w, box_h, _ = map(float, parts)
        x_c *= w
        y_c *= h
        box_w *= w
        box_h *= h

        x1 = int(x_c - box_w / 2)
        y1 = int(y_c - box_h / 2)
        x2 = int(x_c + box_w / 2)
        y2 = int(y_c + box_h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

    cv2.imshow('YOLO Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def volume_cal(path):
    parent_dir = os.path.dirname(os.path.dirname(path))
    txt_path = os.path.join(path, 'lesions.txt')
    output_txt_path = os.path.join(path, 'volume.txt')
    phase = os.path.basename(path).lower()
    if 'pre' in phase:
        dicom_candidates = ['preop', 'pre', 'postop', 'post']
    else:
        dicom_candidates = ['postop', 'post', 'preop', 'pre']
    for post in dicom_candidates:
        dicom_path = os.path.join(parent_dir, post, '10001.dcm')
        try:
            # ===================================================================
            # 1. voxel 관련 기본 정보 계산
            # ===================================================================
            dcm = pydicom.dcmread(dicom_path)

            image_height = int(dcm.get("Rows", 0))
            image_width = int(dcm.get("Columns", 0))
            if image_height <= 0 or image_width <= 0:
                raise ValueError("DICOM Rows/Columns가 올바르지 않습니다.")

            mask_height = 192
            mask_width = 192
            magnification_height = image_height / mask_height
            magnification_width = image_width / mask_width

            pixel_spacing = dcm.get("PixelSpacing", [None, None])
            row_spacing_mm = float(pixel_spacing[0]) * magnification_height
            col_spacing_mm = float(pixel_spacing[1]) * magnification_width

            spacing_between_slices = dcm.get("SpacingBetweenSlices", None)
            if spacing_between_slices is None:
                spacing_between_slices = dcm.get("SliceThickness", None)
            spacing_z = float(spacing_between_slices) if spacing_between_slices is not None else None

            if spacing_z is None:
                raise ValueError("DICOM에 SpacingBetweenSlices/SliceThickness가 없습니다.")

            voxel_volume_mm3 = row_spacing_mm * col_spacing_mm * spacing_z
            # ===================================================================
            # 2. lesions.txt 읽기 및 통계 집계
            # ===================================================================
            lesion_voxel_count = defaultdict(int)
            lesion_slices = defaultdict(set)
            lesion_max_width = defaultdict(float)
            lesion_max_height = defaultdict(float)

            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 7:
                        continue

                    _, _, width, height, slice_id, lesion_id, pixel_count = parts
                    lesion_id = int(lesion_id)
                    slice_id = int(slice_id)
                    pixel_count = int(pixel_count)

                    width_mm = float(width) * mask_width * col_spacing_mm
                    height_mm = float(height) * mask_height * row_spacing_mm

                    lesion_max_width[lesion_id] = max(lesion_max_width[lesion_id], width_mm)
                    lesion_max_height[lesion_id] = max(lesion_max_height[lesion_id], height_mm)
                    lesion_voxel_count[lesion_id] += pixel_count
                    lesion_slices[lesion_id].add(slice_id)

            # ===================================================================
            # 3. 부피 계산
            # ===================================================================
            with open(output_txt_path, 'w') as f:
                f.write("LesionID Volume(mm3) HeightZ(mm) MaxWidth(mm) MaxHeight(mm) #in sphere-diameter\n")

                for lesion_id in sorted(lesion_voxel_count.keys()):
                    voxel_count = lesion_voxel_count[lesion_id]
                    nz_slices = lesion_slices[lesion_id]
                    n_slices = len(nz_slices)

                    max_width_mm = lesion_max_width[lesion_id]
                    max_height_mm = lesion_max_height[lesion_id]
                    volume_mm3 = voxel_count * voxel_volume_mm3
                    if n_slices == 1 and volume_mm3 <= 3:
                        diameter_mm = max(max_width_mm, max_height_mm)
                        radius_mm = diameter_mm / 2.0
                        volume_mm3 = (4.0 / 3.0) * math.pi * (radius_mm ** 3)
                        height_z_mm = spacing_z
                        f.write(f"{lesion_id} {volume_mm3:.2f} {diameter_mm:.2f} {diameter_mm:.2f} {diameter_mm:.2f}\n")
                    else:
                        height_z_mm = n_slices * spacing_z
                        f.write(
                            f"{lesion_id} {volume_mm3:.2f} {height_z_mm:.2f} {max_width_mm:.2f} {max_height_mm:.2f}\n")

            print(f"부피 결과 저장 완료: {output_txt_path}")
            print("-" * 30)
            return output_txt_path
        except Exception as e:
            print(f"[WARN] volume_cal 실패: {dicom_path} ({e})")
            continue



def format_patient_name(pn_value) -> str:
    if pn_value in (None, "", "N/A"):
        return "N/A"
    first_rep = str(pn_value).split('=')[0]  # 여러 표현 중 첫 번째만
    try:
        pn = PersonName(first_rep)
        parts = [pn.family_name, pn.given_name, pn.middle_name]
        parts = [p for p in parts if p]
        return " ".join(parts) if parts else first_rep.replace('^', ' ')
    except Exception:
        return first_rep.replace('^', ' ')


# ---------------------------
#
# ---------------------------
def get_by_str_tags(ds: pydicom.Dataset, keys: list[tuple[str, str]], default="N/A"):
    for g, e in keys:
        try:
            tag = Tag(int(g, 16), int(e, 16))
            if tag in ds:
                return ds[tag].value
        except Exception:
            pass
    return default


# ---------------------------
# volume.txt 로딩 + 컬럼 표준화
# ---------------------------
def load_and_normalize(txt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(txt_path, sep=r"\s+", engine="python")

    rename_map = {}
    for c in df.columns:
        c_norm = c.strip().lower()
        if "lesionid" in c_norm:   rename_map[c] = "LesionID"
        elif "volume" in c_norm:   rename_map[c] = "Volume_mm3"
        elif "heightz" in c_norm:  rename_map[c] = "HeightZ_mm"
        elif "maxwidth" in c_norm: rename_map[c] = "MaxWidth_mm"
        elif "maxheight" in c_norm:rename_map[c] = "MaxHeight_mm"
    df = df.rename(columns=rename_map)

    need = ["LesionID","Volume_mm3","HeightZ_mm","MaxWidth_mm","MaxHeight_mm"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"[{txt_path}] 필수 컬럼 없음: {missing}")

    return df


# ---------------------------
# 한 환자의 volume.txt 요약 통계
# ---------------------------
def summarize_df(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {
            "NumLesions": 0,
            "LargestLesionMaxLength_mm": 0.0,
            "LargestLesionVolume_mm3": 0.0,
            "TotalVolume_mm3": 0.0,
            "GlobalMaxLength_mm": 0.0,
            "GlobalMaxHeightZ_mm": 0.0,
            "GlobalMaxWidth_mm": 0.0,
            "GlobalMaxHeight_mm": 0.0,
        }

    num_lesions = int(len(df))
    idx_max_vol = df["Volume_mm3"].idxmax()
    row = df.loc[idx_max_vol]

    largest_lesion_max_len_mm = float(
        np.max([row["HeightZ_mm"], row["MaxWidth_mm"], row["MaxHeight_mm"]])
    )

    return {
        "NumLesions": num_lesions,
        "LargestLesionMaxLength_mm": float(round(largest_lesion_max_len_mm, 3)),
        "LargestLesionVolume_mm3": float(round(row["Volume_mm3"], 3)),
        "TotalVolume_mm3": float(round(df["Volume_mm3"].sum(), 3)),
        "GlobalMaxLength_mm": float(round(
            df[["HeightZ_mm","MaxWidth_mm","MaxHeight_mm"]].to_numpy().max(), 3)),
        "GlobalMaxHeightZ_mm": float(round(df["HeightZ_mm"].max(), 3)),
        "GlobalMaxWidth_mm": float(round(df["MaxWidth_mm"].max(), 3)),
        "GlobalMaxHeight_mm": float(round(df["MaxHeight_mm"].max(), 3)),
    }

def read_patient_meta(patient_root: Path) -> tuple[str, str]:
    for phase in ['post', 'postop', 'pre', 'preop']:
        dcm_path = patient_root / phase / "10001.dcm"
        if dcm_path.exists():
            try:
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
                pname_raw = get_by_str_tags(ds, [("0010", "0010")], default="N/A")
                pid_raw   = get_by_str_tags(ds, [("0010", "0020")], default="N/A")
                pname = format_patient_name(pname_raw)
                pid   = str(pid_raw) if pid_raw != "N/A" else "N/A"
                return pid, pname
            except Exception as e:
                print(f"[WARN] DICOM 읽기 실패: {dcm_path} ({e})")
    print(f"[WARN] DICOM 없음: {patient_root}")
    return "N/A", "N/A"


# ---------------------------
# volume.txt → 요약 dict (최종 CSV: PatientID/Name + 요약)
# ---------------------------
def summarize_volume_txt(txt_path: Path) -> dict:
    df = load_and_normalize(txt_path)
    summary = summarize_df(df)
    patient_root = Path(*txt_path.parts[:-3])

    patient_id, patient_name = read_patient_meta(patient_root)

    row = {
        "PatientID": patient_id,
        "PatientName": patient_name,
    }
    row.update(summary)
    out_csv = txt_path.with_name("volume.summary.csv")
    pd.DataFrame([row]).to_csv(out_csv, index=False, encoding="utf-8-sig")

    return row


# ---------------------------
# 전체 집계 CSV 생성
# ---------------------------
_NORMALIZE_PHASE = {"pre": "pre", "preop": "pre", "post": "post", "postop": "post"}

def process_all_volumes(root_dir: str, mask_subdir: str = 'labelmask') -> pd.DataFrame | None:
    """
    root_dir 아래 모든 {mask_subdir}/{phase}/volume.txt 를 재귀 탐색하여 집계 CSV를 생성.
    # MRI (3단계), 4_lesion+ / 추가 병변 (2단계) 구조를 모두 지원.
    """
    pattern = str(Path(root_dir) / "**" / mask_subdir / "**" / "volume.txt")
    txt_files = glob.glob(pattern, recursive=True)

    if not txt_files:
        print(f"[WARN] volume.txt 없음: {root_dir}")
        return None

    all_rows = []
    for f in sorted(txt_files):
        txt_path = Path(f)
        try:
            raw_phase = txt_path.parent.name.lower()
            phase = _NORMALIZE_PHASE.get(raw_phase, raw_phase)

            # Group: root_dir 바로 아래 폴더명 (4_lesion+, # MRI, 추가 병변 등)
            try:
                rel = txt_path.relative_to(root_dir)
                group = rel.parts[0]
            except ValueError:
                group = "unknown"

            row = summarize_volume_txt(txt_path)
            if not isinstance(row, dict):
                row = {"_value": row}
            row["Group"] = group
            row["Phase"] = phase
            all_rows.append(row)
            print(f"[OK] {txt_path}  group={group}  phase={phase}")
        except Exception as e:
            print(f"[ERROR] {txt_path}: {e}")

    if not all_rows:
        print("유효한 volume 요약이 없습니다.")
        return None

    df_all = pd.DataFrame(all_rows)
    front = ["PatientID", "PatientName", "Group", "Phase"]
    others = [c for c in df_all.columns if c not in front]
    df_all = df_all[[c for c in front if c in df_all.columns] + others]

    base = Path(root_dir) / "volume_summary_all"
    out_all = base.with_suffix(".csv")
    counter = 1
    while out_all.exists():
        out_all = base.parent / f"{base.name}_{counter}.csv"
        counter += 1

    df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"\n전체 집계 저장: {out_all}  ({len(df_all)}행)")
    return df_all
