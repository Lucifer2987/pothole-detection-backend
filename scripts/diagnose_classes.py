"""
Task A — Crack class diagnostic script.

1. Counts class distribution across train/val/test splits.
2. Spots-checks crack-annotated images visually (saves to data/diagnostic/).
3. Checks dataset.yaml class mapping.
4. Detects class index swaps across source datasets.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import yaml

ROOT = Path(__file__).parent.parent
DATA_PROC = ROOT / "data" / "processed"
DIAG_DIR = ROOT / "data" / "diagnostic"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────────────────────
# 1. Class distribution across splits
# ───────────────────────────────────────────────
print("=" * 60)
print("  TASK A — Crack Class Diagnostic")
print("=" * 60)

splits = ["train", "val", "test"]
grand_total = defaultdict(int)
file_totals = defaultdict(int)

for split in splits:
    ldir = DATA_PROC / split / "labels"
    if not ldir.exists():
        print(f"\n[{split}] ⚠  labels dir missing: {ldir}")
        continue
    label_files = sorted(ldir.glob("*.txt"))
    cls_count = defaultdict(int)
    files_with_cls = defaultdict(int)
    for lf in label_files:
        seen = set()
        for line in lf.read_text().strip().splitlines():
            if line.strip():
                cid = int(line.split()[0])
                cls_count[cid] += 1
                grand_total[cid] += 1
                seen.add(cid)
        for c in seen:
            files_with_cls[c] += 1
            file_totals[c] += 1
    print(f"\n[{split}]  {len(label_files)} label files")
    print(f"   class 0 (pothole) : {cls_count[0]:>6} instances  in {files_with_cls[0]} files")
    print(f"   class 1 (crack)   : {cls_count[1]:>6} instances  in {files_with_cls[1]} files")
    other = {k:v for k,v in cls_count.items() if k not in (0,1)}
    if other:
        print(f"   OTHER classes     : {dict(other)}")

print("\n" + "-" * 60)
print("GRAND TOTAL across all splits:")
print(f"   class 0 (pothole) : {grand_total[0]:>7} instances")
print(f"   class 1 (crack)   : {grand_total[1]:>7} instances")
total_all = grand_total[0] + grand_total[1]
if total_all > 0:
    crack_pct = grand_total[1] / total_all * 100
    flag = "⚠  SEVERE IMBALANCE" if crack_pct < 15 else ("⚠  MILD IMBALANCE" if crack_pct < 30 else "OK")
    print(f"   Crack share       : {crack_pct:.1f}%  [{flag}]")

# ───────────────────────────────────────────────
# 2. Check dataset YAML for class mapping
# ───────────────────────────────────────────────
print("\n" + "-" * 60)
print("Dataset YAML class mappings:")

for yaml_path in sorted((ROOT / "data").glob("**/*.yaml")):
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        if "names" in cfg:
            print(f"   {yaml_path.name}: {cfg['names']}")
    except Exception as e:
        print(f"   {yaml_path.name}: ERROR reading - {e}")

# ───────────────────────────────────────────────
# 3. Spot-check: find crack-containing label files and visualize
# ───────────────────────────────────────────────
print("\n" + "-" * 60)
print("Spot-check: drawing boxes on crack-annotated images...")

img_dir = DATA_PROC / "train" / "images"
lbl_dir = DATA_PROC / "train" / "labels"

CLASS_COLORS = {
    0: (0, 255, 0),    # pothole = green
    1: (0, 0, 255),    # crack = red
}
CLASS_NAMES = {0: "pothole", 1: "crack"}

crack_files = []
if lbl_dir.exists():
    for lf in sorted(lbl_dir.glob("*.txt")):
        anns = lf.read_text().strip().splitlines()
        if any(line.split()[0] == "1" for line in anns if line.strip()):
            crack_files.append(lf)
        if len(crack_files) >= 10:
            break

print(f"   Found {len(crack_files)} crack-containing label files (showing first 10)")

saved = []
for lf in crack_files[:10]:
    # Find matching image
    stem = lf.stem
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
        candidate = img_dir / (stem + ext)
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        print(f"   ⚠  No image found for label: {lf.name}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"   ⚠  Could not read image: {img_path}")
        continue

    h, w = img.shape[:2]
    anns = lf.read_text().strip().splitlines()
    for line in anns:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = CLASS_COLORS.get(cid, (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES.get(cid, f"cls{cid}")
        cv2.putText(img, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = DIAG_DIR / f"crack_check_{stem}.jpg"
    cv2.imwrite(str(out_path), img)
    saved.append(str(out_path))

print(f"\n   Saved {len(saved)} annotated images to: {DIAG_DIR}")
for p in saved:
    print(f"     {p}")

# ───────────────────────────────────────────────
# 4. Check source dataset label dirs for class index consistency
# ───────────────────────────────────────────────
print("\n" + "-" * 60)
print("Checking raw source datasets for class index consistency:")

raw_dirs = sorted((ROOT / "data" / "raw").glob("**"))
checked = 0
for d in raw_dirs:
    ldir = d / "labels" if (d / "labels").exists() else d
    yamls = list(d.glob("data.yaml")) + list(d.glob("*.yaml"))
    for yp in yamls:
        try:
            cfg = yaml.safe_load(open(yp))
            if "names" in cfg:
                print(f"   [{d.name}] {yp.name} -> names={cfg['names']}")
                checked += 1
        except Exception:
            pass

if checked == 0:
    print("   No source YAML files found in data/raw/ — checking was skipped.")

print("\n" + "=" * 60)
print("  Diagnostic complete. Review output above.")
print("=" * 60)
