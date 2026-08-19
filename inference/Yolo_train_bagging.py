import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import yaml
from ultralytics import YOLO

DATA_YAML = r"/ds/want2704/mri/mydata/data.yaml"

BAGS = 8
SEED = 42
TRAIN_RATIO = 1.0
MIN_OOB = 50
VAL_RATIO_FALLBACK = 0.2

LINK_MODE = "copy"  # "copy" | "hardlink" | "symlink"

MODEL_WEIGHTS = "yolo12n.pt"
MODEL_YAML_FALLBACK = "yolo12n.yaml"

EPOCHS = 1000
IMGSZ = 512
BATCH = 160
DEVICE = 0
PATIENCE = 100

SCALE = 0.5
MOSAIC = 1.0
MIXUP = 0.0
COPY_PASTE = 0.1
AMP = True
PLOTS = False
WORKERS = 0


BOX = 10.0
DFL = 2.5

BAGS_ROOT = "bagged_datasets_1"
PROJECT = "runs_bagging_map50_951.0"
NAME_PREFIX = "y12n_bag"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def list_images(img_dir: Path) -> List[Path]:
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def images_to_labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" not in parts:
        raise ValueError(f"'images' not found in path: {images_dir}")
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts)


def corresponding_label(img_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = img_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


def try_link_or_copy(src: Path, dst: Path):
    safe_mkdir(dst.parent)
    if dst.exists():
        return
    if LINK_MODE == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    if LINK_MODE == "symlink":
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def resolve_trainval_images_dir(base_yaml: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p.resolve()
    bases = [base_yaml.parent, base_yaml.parent.parent, base_yaml.parent.parent.parent, Path.cwd()]
    tried = []
    for b in bases:
        cand = (b / p).resolve()
        tried.append(str(cand))
        if cand.exists():
            return cand
    msg = (
        f"Could not resolve path '{rel_or_abs}' from data.yaml.\n"
        f"data.yaml: {base_yaml}\n"
        f"Tried:\n  - " + "\n  - ".join(tried)
    )
    raise FileNotFoundError(msg)


def make_bag_split(pool: List[Path], train_size: int, seed: int) -> Tuple[List[Path], List[Path], bool]:
    rnd = random.Random(seed)
    train_sel = [rnd.choice(pool) for _ in range(train_size)]
    train_set = set(train_sel)
    oob = [p for p in pool if p not in train_set]
    if len(oob) < MIN_OOB:
        pool_shuf = pool[:]
        rnd.shuffle(pool_shuf)
        n_val = max(1, int(len(pool_shuf) * VAL_RATIO_FALLBACK))
        val_sel = pool_shuf[:n_val]
        train_sel = pool_shuf[n_val:]
        return train_sel, val_sel, True
    return train_sel, oob, False


def materialize_subset(
    img_list: List[Path],
    out_images_root: Path,
    out_labels_root: Path,
    train_images_root: Path,
    train_labels_root: Path,
    val_images_root: Path,
    val_labels_root: Path,
):
    for img in img_list:
        if str(img).startswith(str(train_images_root)):
            images_root = train_images_root
            labels_root = train_labels_root
        elif str(img).startswith(str(val_images_root)):
            images_root = val_images_root
            labels_root = val_labels_root
        else:
            raise RuntimeError(f"Image not under train/val images dirs: {img}")

        rel = img.relative_to(images_root)
        dst_img = out_images_root / rel
        try_link_or_copy(img, dst_img)

        src_lbl = corresponding_label(img, images_root, labels_root)
        dst_lbl = (out_labels_root / rel).with_suffix(".txt")
        if src_lbl.exists():
            try_link_or_copy(src_lbl, dst_lbl)


def write_bag_yaml(base_yaml_path: Path, out_yaml_path: Path):
    cfg = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8"))
    cfg["train"] = "train/images"
    cfg["val"] = "valid/images"
    safe_mkdir(out_yaml_path.parent)
    out_yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def get_yolo_model_source() -> str:
    w = Path(MODEL_WEIGHTS)
    if w.exists():
        return str(w.resolve())
    if Path(MODEL_WEIGHTS).is_absolute() and Path(MODEL_WEIGHTS).exists():
        return str(Path(MODEL_WEIGHTS).resolve())
    print(f"[WARN] weights not found: {MODEL_WEIGHTS}")
    print(f"[WARN] fallback to model yaml (random init): {MODEL_YAML_FALLBACK}")
    return MODEL_YAML_FALLBACK


def _parse_w_from_source(src: str) -> Optional[List[float]]:
    import re
    m = re.search(r"w\s*=\s*\[([^\]]+)\]", src)
    if not m:
        m = re.search(r"w\s*=\s*\(([^)]+)\)", src)
    if not m:
        return None
    raw = [x.strip() for x in m.group(1).split(",") if x.strip()]
    try:
        vals = [float(x) for x in raw]
    except Exception:
        return None
    return vals if len(vals) == 4 else None


def print_metrics_class_fitness_weights():
    import inspect
    import ultralytics.utils.metrics as um

    candidates = []
    for name, obj in um.__dict__.items():
        if isinstance(obj, type) and hasattr(obj, "fitness"):
            lname = name.lower()
            if lname == "metrics" or lname.endswith("metrics") or "metrics" in lname:
                candidates.append((name, obj))

    candidates.sort(key=lambda x: (0 if x[0].lower() == "metrics" else 1, len(x[0])))

    for name, cls in candidates:
        try:
            src = inspect.getsource(getattr(cls, "fitness"))
            w = _parse_w_from_source(src)
            if w is None:
                continue
            names = ["Precision(P)", "Recall(R)", "mAP50", "mAP50-95"]
            s = sum(w) if sum(w) != 0 else 1.0
            wn = [x / s for x in w]
            print(f"\n[FITNESS] source=ultralytics.utils.metrics.{name}.fitness")
            print("[FITNESS] fitness = Σ (weight_i * metric_i)  (val 기준)")
            for n, wi, wni in zip(names, w, wn):
                print(f"  - {n:13s}: weight={wi:.4f} (normalized={wni:.3f})")
            print("  -> best.pt 선택/early stopping(patience)은 이 fitness 최대화를 기준으로 동작합니다.\n")
            return
        except Exception:
            continue

    print("\n[FITNESS] Metrics class fitness weights를 찾지 못했습니다. (출력 생략)\n")


def train_with_param_fallback(model: YOLO, kwargs: Dict[str, Any]):
    """
    Ultralytics 버전별로 train()이 받는 인자 이름이 달라서,
    TypeError(unknown arg)가 나면 해당 키를 제거하고 다시 시도하는 안전 호출.
    """
    k = dict(kwargs)
    while True:
        try:
            return model.train(**k)
        except TypeError as e:
            msg = str(e)
            bad_key = None

            # 흔한 패턴: "train() got an unexpected keyword argument 'xxx'"
            import re
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m:
                bad_key = m.group(1)

            if not bad_key:
                # 어떤 키가 문제인지 못 찾으면 그대로 에러 올림
                raise

            if bad_key in k:
                print(f"[WARN] train() does not accept '{bad_key}' in this Ultralytics version -> dropping it.")
                k.pop(bad_key, None)
                continue
            raise


def main():
    base_yaml = Path(DATA_YAML)
    if not base_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {base_yaml}")

    cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    if "train" not in cfg or "val" not in cfg:
        raise ValueError("data.yaml must contain 'train' and 'val' keys")

    train_images = resolve_trainval_images_dir(base_yaml, cfg["train"])
    val_images = resolve_trainval_images_dir(base_yaml, cfg["val"])
    train_labels = images_to_labels_dir(train_images)
    val_labels = images_to_labels_dir(val_images)

    pool = list_images(train_images) + list_images(val_images)
    if len(pool) == 0:
        raise RuntimeError("No images found in train+val")

    train_size = max(1, int(len(pool) * TRAIN_RATIO))
    model_source = get_yolo_model_source()

    print(f"[INFO] base data.yaml : {base_yaml}")
    print(f"[INFO] resolved train : {train_images}")
    print(f"[INFO] resolved val   : {val_images}")
    print(f"[INFO] pool size      : {len(pool)}")
    print(f"[INFO] train_size/bag : {train_size} (TRAIN_RATIO={TRAIN_RATIO})")
    print(f"[INFO] bags           : {BAGS}")
    print(f"[INFO] link_mode      : {LINK_MODE}")
    print(f"[INFO] model source   : {model_source}")

    # ✅ 적용 파라미터 출력
    print("\n[HYP] Applying training params:")
    print(f"  - imgsz      : {IMGSZ}")
    print(f"  - box        : {BOX}")
    print(f"  - dfl        : {DFL}\n")

    bags_root = Path(BAGS_ROOT)
    safe_mkdir(bags_root)

    for b in range(BAGS):
        bag_seed = SEED + b * 1000
        train_sel, val_sel, fallback = make_bag_split(pool, train_size, bag_seed)

        bag_dir = bags_root / f"{NAME_PREFIX}_{b:02d}"
        if bag_dir.exists():
            shutil.rmtree(bag_dir)

        out_train_img = bag_dir / "train" / "images"
        out_train_lbl = bag_dir / "train" / "labels"
        out_val_img = bag_dir / "valid" / "images"
        out_val_lbl = bag_dir / "valid" / "labels"

        safe_mkdir(out_train_img)
        safe_mkdir(out_train_lbl)
        safe_mkdir(out_val_img)
        safe_mkdir(out_val_lbl)

        materialize_subset(
            train_sel, out_train_img, out_train_lbl,
            train_images, train_labels,
            val_images, val_labels,
        )
        materialize_subset(
            val_sel, out_val_img, out_val_lbl,
            train_images, train_labels,
            val_images, val_labels,
        )

        bag_yaml = bag_dir / "data.yaml"
        write_bag_yaml(base_yaml, bag_yaml)

        run_name = f"{NAME_PREFIX}_{b:02d}"
        print(f"\n[TRAIN] bag {b+1}/{BAGS} -> {run_name}")
        print(f"  train imgs: {len(train_sel)} | val imgs: {len(val_sel)} | val={'fallback' if fallback else 'OOB'}")
        print(f"  data yaml : {bag_yaml}")

        print_metrics_class_fitness_weights()

        model = YOLO(model_source)

        # ✅ 버전별 키 차이를 고려해, auto_anchor 관련은 여러 키로 “시도” 가능하게 넣고 fallback 처리
        train_kwargs = dict(
            data=str(bag_yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            patience=PATIENCE,
            scale=SCALE,
            mosaic=MOSAIC,
            mixup=MIXUP,
            copy_paste=COPY_PASTE,
            amp=AMP,
            plots=PLOTS,
            workers=WORKERS,
            project=PROJECT,
            name=run_name,
            exist_ok=True,
            box=BOX,
            dfl=DFL,
            )

        train_with_param_fallback(model, train_kwargs)

    print("\n[DONE] All bagged trainings finished.")


if __name__ == "__main__":
    main()
