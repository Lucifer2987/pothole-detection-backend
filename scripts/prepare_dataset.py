"""
scripts/prepare_dataset.py
--------------------------
Converts, normalises, and merges all raw datasets into data/processed/
with an 80/10/10 train/val/test split, then writes data/dataset.yaml.

Sources handled:
  - data/raw/roboflow/  (YOLO .txt, 3 classes → remap to 2)
  - data/raw/kaggle/    (Pascal VOC XML → convert to YOLO .txt)
  - data/raw/rdd2022/   (Pascal VOC XML → convert to YOLO .txt, if present)

Target class map:
  0 = pothole  (Roboflow: pothole; Kaggle: pothole; RDD2022: D40, D44)
  1 = crack    (Roboflow: crocodile crack, longitudinal crack; RDD2022: D00,D10,D20)

Run from project root:
    python scripts/prepare_dataset.py [--dry-run]

    --dry-run   Print what would happen without writing any files.
"""

import argparse
import hashlib
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_OUT   = ROOT / "data" / "processed"
YAML_OUT   = ROOT / "data" / "dataset.yaml"

RF_DIR     = DATA_RAW / "roboflow"
KG_DIR     = DATA_RAW / "kaggle"
RDD_DIR    = DATA_RAW / "rdd2022"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Target class definitions
# ---------------------------------------------------------------------------

CLASS_NAMES = ["pothole", "crack"]   # index 0 = pothole, 1 = crack

# Roboflow dataset.yaml class order:  0=crocodile crack, 1=longitudinal crack, 2=pothole
# Map old_index → new_index  (None means discard)
ROBOFLOW_REMAP = {
    0: 1,    # crocodile crack  → crack
    1: 1,    # longitudinal crack → crack
    2: 0,    # pothole          → pothole
}

# Kaggle XML class names → our class index
KAGGLE_NAME_MAP = {
    "pothole": 0,
    "crack":   1,
    "alligator crack": 1,
    "transverse crack": 1,
    "longitudinal crack": 1,
}

# RDD2022 XML class names → our class index
RDD_NAME_MAP = {
    "D00": 1,   # longitudinal crack
    "D10": 1,   # transverse crack
    "D20": 1,   # alligator crack
    "D40": 0,   # pothole
    "D44": 0,   # pothole variant
    "D43": 0,   # pothole variant
    "D50": None, # manhole (ignore)
}

# Split ratios
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}

SEED = 42


# ===========================================================================
# Utilities
# ===========================================================================

def file_md5(path: Path) -> str:
    """Return MD5 hex digest of a file (used for near-duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def voc_to_yolo(xml_path: Path, name_map: dict) -> list[str]:
    """
    Parse a Pascal VOC XML annotation file and return YOLO format lines.

    Returns a list of strings like:
        "<class_idx> <cx> <cy> <w> <h>"
    where all values are normalised to [0, 1].

    Objects whose class name is not in name_map, or maps to None, are skipped.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    root_el = tree.getroot()
    size_el  = root_el.find("size")
    if size_el is None:
        return []

    W = float(size_el.findtext("width",  default="0"))
    H = float(size_el.findtext("height", default="0"))
    if W <= 0 or H <= 0:
        return []

    lines = []
    for obj in root_el.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        # Try exact lower-case match first, then original case
        cls_idx = name_map.get(name)
        if cls_idx is None:
            cls_idx = name_map.get(obj.findtext("name", "").strip())
        if cls_idx is None:
            continue

        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))

        cx = ((xmin + xmax) / 2.0) / W
        cy = ((ymin + ymax) / 2.0) / H
        bw = (xmax - xmin) / W
        bh = (ymax - ymin) / H

        # Clamp to [0, 1] in case of annotation errors
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        if bw <= 0 or bh <= 0:
            continue

        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def remap_yolo_label(txt_path: Path, remap: dict) -> list[str]:
    """
    Read an existing YOLO .txt label file and remap class indices.

    Lines whose original class index is not in remap, or maps to None, are dropped.
    """
    lines_out = []
    try:
        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return []

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            old_idx = int(parts[0])
        except ValueError:
            continue
        new_idx = remap.get(old_idx)
        if new_idx is None:
            continue
        lines_out.append(f"{new_idx} {' '.join(parts[1:])}")
    return lines_out


# ===========================================================================
# Collection step — build a flat list of (image_path, label_lines) tuples
# ===========================================================================

def collect_roboflow() -> list[tuple[Path, list[str]]]:
    """Collect Roboflow YOLO images, remapping from 3 classes → 2."""
    records = []
    if not RF_DIR.exists():
        print("[Roboflow] Directory not found, skipping.")
        return records

    for split in ["train", "valid", "test"]:
        img_dir   = RF_DIR / split / "images"
        label_dir = RF_DIR / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = label_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                label_lines = remap_yolo_label(lbl_path, ROBOFLOW_REMAP)
            else:
                label_lines = []   # image without annotation → keep (background)
            records.append((img_path, label_lines))

    print(f"[Roboflow] Collected {len(records)} images (3->2 class remap applied)")
    return records


def collect_kaggle() -> list[tuple[Path, list[str]]]:
    """Collect Kaggle Pascal VOC XML, converting to YOLO format."""
    records = []
    img_dir  = KG_DIR / "images"
    ann_dir  = KG_DIR / "annotations"
    if not img_dir.exists():
        print("[Kaggle] images/ directory not found, skipping.")
        return records

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        xml_path = ann_dir / (img_path.stem + ".xml")
        if xml_path.exists():
            label_lines = voc_to_yolo(xml_path, {
                k: v for k, v in KAGGLE_NAME_MAP.items()
            })
        else:
            label_lines = []
        records.append((img_path, label_lines))

    print(f"[Kaggle] Collected {len(records)} images (VOC->YOLO conversion applied)")
    return records


def collect_rdd2022() -> list[tuple[Path, list[str]]]:
    """Collect RDD2022 Pascal VOC XML (if present)."""
    records = []
    if not RDD_DIR.exists():
        print("[RDD2022] Directory not found, skipping.")
        return records

    # Build a case-insensitive name map for XML lookup
    name_map = {k.lower(): v for k, v in RDD_NAME_MAP.items() if v is not None}
    name_map.update({k: v for k, v in RDD_NAME_MAP.items() if v is not None})

    country_dirs = [d for d in RDD_DIR.iterdir() if d.is_dir()]
    for country_dir in sorted(country_dirs):
        for split in ["train", "test"]:
            img_dir = country_dir / split / "images"
            ann_dir = country_dir / split / "annotations" / "xmls"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                xml_path = ann_dir / (img_path.stem + ".xml")
                if xml_path.exists():
                    label_lines = voc_to_yolo(xml_path, RDD_NAME_MAP)
                else:
                    label_lines = []
                records.append((img_path, label_lines))

    print(f"[RDD2022] Collected {len(records)} images (VOC->YOLO conversion applied)")
    return records


# ===========================================================================
# Deduplication
# ===========================================================================

def deduplicate(records: list[tuple[Path, list[str]]]) -> list[tuple[Path, list[str]]]:
    """
    Remove near-duplicate images using MD5 hash.
    First occurrence of each hash is kept; subsequent ones are dropped.
    """
    seen: set[str] = set()
    unique: list[tuple[Path, list[str]]] = []
    dupes = 0
    for img_path, label_lines in records:
        try:
            digest = file_md5(img_path)
        except Exception:
            unique.append((img_path, label_lines))
            continue
        if digest in seen:
            dupes += 1
        else:
            seen.add(digest)
            unique.append((img_path, label_lines))
    if dupes:
        print(f"[Dedup] Removed {dupes} duplicate images ({len(unique)} remain)")
    else:
        print(f"[Dedup] No duplicates found ({len(unique)} images)")
    return unique


# ===========================================================================
# Split
# ===========================================================================

def split_records(
    records: list[tuple[Path, list[str]]],
    ratios: dict = SPLIT_RATIOS,
    seed: int = SEED,
) -> dict[str, list[tuple[Path, list[str]]]]:
    """Shuffle and split records into train/val/test."""
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n     = len(shuffled)
    n_tr  = int(n * ratios["train"])
    n_val = int(n * ratios["val"])

    return {
        "train": shuffled[:n_tr],
        "val":   shuffled[n_tr : n_tr + n_val],
        "test":  shuffled[n_tr + n_val :],
    }


# ===========================================================================
# Write output
# ===========================================================================

def write_split(
    split_name: str,
    records: list[tuple[Path, list[str]]],
    out_root: Path,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Copy images and write YOLO label files to out_root/{split}/images/ and /labels/.

    Returns (image_count, class0_boxes, class1_boxes).
    """
    img_dir = out_root / split_name / "images"
    lbl_dir = out_root / split_name / "labels"

    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    c0 = c1 = 0
    for img_path, label_lines in records:
        # Unique filename: prefix with source folder name to avoid collisions
        prefix   = img_path.parent.parent.parent.name  # e.g. 'roboflow', 'kaggle'
        new_stem = f"{prefix}_{img_path.stem}"
        dst_img  = img_dir / (new_stem + img_path.suffix.lower())
        dst_lbl  = lbl_dir / (new_stem + ".txt")

        if not dry_run:
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text("\n".join(label_lines), encoding="utf-8")

        for line in label_lines:
            idx = int(line.split()[0])
            if idx == 0:
                c0 += 1
            elif idx == 1:
                c1 += 1

    return len(records), c0, c1


def write_dataset_yaml(out_root: Path, dry_run: bool) -> None:
    """Write data/dataset.yaml pointing at data/processed/ splits."""
    abs_root = out_root.resolve()
    content = (
        f"# Pothole Detection — merged dataset\n"
        f"# Auto-generated by scripts/prepare_dataset.py\n"
        f"\n"
        f"path: {abs_root.as_posix()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    if not dry_run:
        YAML_OUT.write_text(content, encoding="utf-8")
        print(f"\n[YAML] Written to {YAML_OUT}")
    else:
        print(f"\n[YAML] (dry-run) Would write to {YAML_OUT}:")
        print(content)


# ===========================================================================
# Entry-point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare merged pothole dataset.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without writing files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dry_run = args.dry_run

    print("=" * 65)
    print("  Pothole Detection - Dataset Preparation")
    if dry_run:
        print("  [DRY RUN — no files will be written]")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Collect from all sources
    # ------------------------------------------------------------------
    print("\n[Step 1] Collecting images from all sources...")
    all_records: list[tuple[Path, list[str]]] = []
    all_records.extend(collect_roboflow())
    all_records.extend(collect_kaggle())
    all_records.extend(collect_rdd2022())
    print(f"\n  Total collected : {len(all_records)} images")

    if not all_records:
        print("\nNo images found. Make sure you have run download_datasets.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Deduplicate
    # ------------------------------------------------------------------
    print("\n[Step 2] Deduplicating by MD5 hash...")
    all_records = deduplicate(all_records)

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    print("\n[Step 3] Splitting 80/10/10 ...")
    splits = split_records(all_records)
    for split_name, recs in splits.items():
        print(f"  {split_name:<6}: {len(recs):>5} images")

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    print(f"\n[Step 4] Writing to {DATA_OUT} ...")
    if not dry_run:
        DATA_OUT.mkdir(parents=True, exist_ok=True)

    summary: dict[str, tuple[int,int,int]] = {}
    for split_name, recs in splits.items():
        n, c0, c1 = write_split(split_name, recs, DATA_OUT, dry_run)
        summary[split_name] = (n, c0, c1)
        action = "Would write" if dry_run else "Wrote"
        print(f"  {action} {split_name:<6}: {n} images  "
              f"(pothole boxes={c0}, crack boxes={c1})")

    # ------------------------------------------------------------------
    # 5. Write dataset.yaml
    # ------------------------------------------------------------------
    write_dataset_yaml(DATA_OUT, dry_run)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"  {'SPLIT':<8} {'IMAGES':>7}   {'POTHOLE boxes':>14}   {'CRACK boxes':>11}")
    print("=" * 65)
    total_imgs = total_c0 = total_c1 = 0
    for split_name, (n, c0, c1) in summary.items():
        print(f"  {split_name:<8} {n:>7}   {c0:>14}   {c1:>11}")
        total_imgs += n; total_c0 += c0; total_c1 += c1
    print("-" * 65)
    print(f"  {'TOTAL':<8} {total_imgs:>7}   {total_c0:>14}   {total_c1:>11}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
