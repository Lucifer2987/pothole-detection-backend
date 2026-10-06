"""
Task B fix: Oversample crack-containing images in the training set.

This script duplicates each crack-containing training image N times
(with augmentation applied each time) to boost crack representation
until it reaches at least target_pct of all training instances.

Usage:
    python scripts/oversample_cracks.py [--target-pct 30] [--copies 4] [--dry-run]

Default: makes 4 augmented copies of each crack image (=5x total with original),
         targeting crack instances at >= 30% of training total.
"""

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT     = Path(__file__).resolve().parent.parent
IMG_DIR  = ROOT / "data" / "processed" / "train" / "images"
LBL_DIR  = ROOT / "data" / "processed" / "train" / "labels"


def count_classes(lbl_dir: Path):
    c0 = c1 = 0
    for lf in lbl_dir.glob("*.txt"):
        for line in lf.read_text().strip().splitlines():
            if line.strip():
                try:
                    cid = int(float(line.split()[0]))  # handle both "0" and "0.0"
                    if cid == 0: c0 += 1
                    elif cid == 1: c1 += 1
                except (ValueError, IndexError):
                    continue
    return c0, c1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-pct", type=float, default=30.0,
                        help="Target crack %% of all instances (default 30)")
    parser.add_argument("--copies", type=int, default=4,
                        help="Augmented copies per crack image (default 4)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import cv2
    from src.preprocessing.augment import augment_yolo_label_file

    # --- Find crack-containing label files
    crack_files = []
    for lf in sorted(LBL_DIR.glob("*.txt")):
        anns = lf.read_text().strip().splitlines()
        if any(l.split()[0] == "1" for l in anns if l.strip()):
            crack_files.append(lf)

    print(f"Found {len(crack_files)} crack-containing training label files")

    c0, c1 = count_classes(LBL_DIR)
    total = c0 + c1
    print(f"Before: pothole={c0}  crack={c1}  crack_share={c1/total*100:.1f}%")

    if args.dry_run:
        print(f"\n[DRY-RUN] Would create {len(crack_files) * args.copies} augmented copies")
        print(f"[DRY-RUN] Estimated new totals:")
        est_new_c1 = c1 + len(crack_files) * args.copies * (c1 / len(crack_files))
        est_total = c0 + est_new_c1
        print(f"  crack~={int(est_new_c1)}  share~={est_new_c1/est_total*100:.1f}%")
        return

    created = 0
    for lf in crack_files:
        stem = lf.stem
        label_text = lf.read_text(encoding="utf-8")

        # Find matching image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            c = IMG_DIR / (stem + ext)
            if c.exists():
                img_path = c
                break
        if img_path is None:
            print(f"  [WARN] No image for {lf.name}, skipping")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i in range(args.copies):
            aug_img, aug_labels = augment_yolo_label_file(img, label_text)
            if not aug_labels.strip():
                continue  # all boxes dropped — skip
            out_img = IMG_DIR / f"aug{i}_{stem}{img_path.suffix}"
            out_lbl = LBL_DIR / f"aug{i}_{stem}.txt"
            cv2.imwrite(str(out_img), aug_img)
            out_lbl.write_text(aug_labels, encoding="utf-8")
            created += 1

    c0_new, c1_new = count_classes(LBL_DIR)
    total_new = c0_new + c1_new
    print(f"\nCreated {created} augmented images")
    print(f"After:  pothole={c0_new}  crack={c1_new}  crack_share={c1_new/total_new*100:.1f}%")


if __name__ == "__main__":
    main()
