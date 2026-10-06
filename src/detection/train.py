"""
src/detection/train.py
----------------------
YOLOv8 training script for the Pothole Detection system.

Usage
-----
Full training run (from project root):
    python src/detection/train.py

Sanity-check run (5 epochs, ~500-image subset):
    python src/detection/train.py --sanity-check

Arguments
---------
  --data PATH        Path to dataset YAML  (default: data/dataset.yaml)
  --model NAME       YOLOv8 model variant  (default: yolov8s.pt)
  --epochs N         Number of epochs      (default: 100)
  --imgsz N          Image size            (default: 960)
  --batch N          Batch size (-1 = auto)(default: -1)
  --device STR       Device: 0, cpu, etc.  (default: auto)
  --workers N        DataLoader workers    (default: 4)
  --sanity-check     Run 5-epoch test on a 500-image subset
  --project PATH     Output directory      (default: models/runs)
  --name STR         Run name              (default: pothole_yolov8s)
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Project root on PYTHONPATH
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DATA    = ROOT / "data" / "dataset.yaml"
DEFAULT_PROJECT = ROOT / "models" / "runs"
SUBSET_YAML     = ROOT / "data" / "dataset_sanity.yaml"
PRETRAINED_DIR  = ROOT / "models" / "pretrained"


# ===========================================================================
# Helpers
# ===========================================================================

def build_sanity_yaml(
    full_yaml: Path,
    n_images: int = 500,
    seed: int = 42,
) -> Path:
    """
    Create a small subset dataset YAML for the sanity-check run.

    Randomly samples *n_images* from the full training split, writes
    the images into  data/processed/sanity/images/  (symlinks avoided for
    Windows compatibility — copies instead), and writes a fresh YAML.

    Returns the path to the subset YAML.
    """
    with open(full_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    processed_root = Path(cfg["path"])
    train_img_dir  = processed_root / "train" / "images"
    train_lbl_dir  = processed_root / "train" / "labels"

    # Sample
    all_imgs = sorted(train_img_dir.iterdir())
    rng = random.Random(seed)
    sample = rng.sample(all_imgs, min(n_images, len(all_imgs)))

    # Destination directories
    sanity_root = processed_root / "sanity"
    s_img = sanity_root / "images"
    s_lbl = sanity_root / "labels"
    s_img.mkdir(parents=True, exist_ok=True)
    s_lbl.mkdir(parents=True, exist_ok=True)

    # Copy sampled images + their labels
    for img_path in sample:
        shutil.copy2(img_path, s_img / img_path.name)
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, s_lbl / img_path.name.replace(img_path.suffix, ".txt"))

    # Also reuse val/test from the full dataset (they're small already)
    subset_cfg = {
        "path":  str(sanity_root.as_posix()),
        "train": "images",
        "val":   str((processed_root / "val" / "images").as_posix()),
        "test":  str((processed_root / "test" / "images").as_posix()),
        "nc":    cfg["nc"],
        "names": cfg["names"],
    }

    SUBSET_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(SUBSET_YAML, "w") as f:
        yaml.dump(subset_cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"[Sanity] Subset YAML written: {SUBSET_YAML}")
    print(f"[Sanity] {len(sample)} training images sampled from {len(all_imgs)}")
    return SUBSET_YAML


# ===========================================================================
# Main training function
# ===========================================================================

def train(args: argparse.Namespace) -> None:
    """Run YOLOv8 training."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: python -m pip install 'ultralytics>=8.2'")
        sys.exit(1)

    data_yaml  = Path(args.data)
    run_name   = args.name
    project    = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    if args.sanity_check:
        print("\n" + "=" * 65)
        print("  SANITY-CHECK MODE: 5 epochs, ~500 image subset")
        print("=" * 65 + "\n")
        data_yaml = build_sanity_yaml(Path(args.data), n_images=500)
        args.epochs  = 5
        run_name     = run_name + "_sanity"
        args.imgsz   = 640    # smaller imgsz is fine for a sanity check
        args.batch   = 8      # conservative batch for any GPU/CPU

    if not data_yaml.exists():
        print(f"[ERROR] Dataset YAML not found: {data_yaml}")
        print("        Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    print(f"[Train] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[Train] Dataset      : {data_yaml}")
    print(f"[Train] Epochs       : {args.epochs}")
    print(f"[Train] Image size   : {args.imgsz}")
    print(f"[Train] Batch        : {args.batch}")
    print(f"[Train] Output dir   : {project / run_name}\n")

    results = model.train(
        data      = str(data_yaml),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        workers   = args.workers,
        project   = str(project),
        name      = run_name,
        exist_ok  = True,           # allow re-running without deleting old run
        pretrained= True,
        verbose   = True,
        # Augmentation tweaks suited for road-damage detection
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        flipud    = 0.0,            # road images are always upright
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.0,
    )

    # ------------------------------------------------------------------
    # Post-training summary
    # ------------------------------------------------------------------
    run_dir    = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    print("\n" + "=" * 65)
    print("  Training complete!")
    print("=" * 65)
    print(f"  Run directory   : {run_dir}")
    print(f"  best.pt exists  : {best_weights.exists()}")
    print(f"  last.pt exists  : {last_weights.exists()}")

    if hasattr(results, "results_dict"):
        d = results.results_dict
        for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                  "metrics/precision(B)", "metrics/recall(B)"]:
            if k in d:
                print(f"  {k:<30} {d[k]:.4f}")

    print(f"\n  Weights saved at: {run_dir / 'weights'}\n")

    # Copy best weights to models/finetuned/ for easy access
    finetuned_dir = ROOT / "models" / "finetuned"
    finetuned_dir.mkdir(parents=True, exist_ok=True)
    if best_weights.exists():
        dst = finetuned_dir / f"{run_name}_best.pt"
        shutil.copy2(best_weights, dst)
        print(f"  Copied best.pt -> {dst}\n")


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLOv8 on the merged pothole/crack dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data",         default=str(DEFAULT_DATA),       help="Dataset YAML path")
    p.add_argument("--model",        default="yolov8s.pt",            help="YOLOv8 checkpoint")
    p.add_argument("--epochs",       type=int,   default=100,         help="Training epochs")
    p.add_argument("--imgsz",        type=int,   default=960,         help="Image size")
    p.add_argument("--batch",        type=int,   default=-1,          help="Batch size (-1=auto)")
    p.add_argument("--device",       default="",                      help="Device (blank=auto)")
    p.add_argument("--workers",      type=int,   default=4,           help="DataLoader workers")
    p.add_argument("--project",      default=str(DEFAULT_PROJECT),    help="Output project dir")
    p.add_argument("--name",         default="pothole_yolov8s",       help="Run name")
    p.add_argument("--sanity-check", action="store_true",
                   help="Run 5-epoch test on ~500 image subset")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
