"""
scripts/download_datasets.py
----------------------------
Downloads pothole/road-damage datasets from three sources:

  1. Roboflow Universe  →  data/raw/roboflow/
  2. Kaggle             →  data/raw/kaggle/     (andrewmvd/pothole-detection)
  3. RDD2022            →  data/raw/rdd2022/    (manual — see instructions below)

After all downloads (or scans) complete the script prints a summary table:
  source | image count | annotation format

Usage
-----
  python scripts/download_datasets.py [--roboflow] [--kaggle] [--rdd-scan] [--all]

  --roboflow    Download from Roboflow
  --kaggle      Download from Kaggle
  --rdd-scan    Scan whatever RDD2022 files you have manually placed
  --all         Run all three steps (default if no flag given)

Fill in ROBOFLOW_* variables before running.
"""

# ============================================================
# USER-CONFIGURABLE VARIABLES  ← fill these in before running
# ============================================================

ROBOFLOW_API_KEY  = "uR7DmVVKqUr8en3oeQOK"   # your Roboflow API key
# IIT Madras public pothole dataset — 1906 train / 542 val / 274 test images
# Universe URL: https://universe.roboflow.com/indian-institute-of-technology-madras-xamot/pothole-detection-huf2x
ROBOFLOW_WORKSPACE = "indian-institute-of-technology-madras-xamot"
ROBOFLOW_PROJECT   = "pothole-detection-huf2x"
ROBOFLOW_VERSION   = 2                           # dataset version number to download

# Kaggle dataset identifier  (do not change unless you want a different dataset)
KAGGLE_DATASET     = "andrewmvd/pothole-detection"

# ============================================================
# RDD2022 MANUAL DOWNLOAD INSTRUCTIONS
# ============================================================
#
# RDD2022 cannot be downloaded automatically — it requires an IEEE DataPort
# account (free) or access to the GitHub-linked Google Drive folder.
#
# Steps:
#   1. Visit https://github.com/sekilab/RoadDamageDetector
#      and follow the "Dataset" link to the IEEE DataPort / Google Drive.
#   2. Download the country zip archives you need, e.g.:
#        Japan.zip, India.zip, United_States.zip, Czech.zip, Norway.zip, China.zip
#   3. Extract each archive and place the contents under:
#        data/raw/rdd2022/<CountryName>/
#      The expected structure inside each country folder is:
#        data/raw/rdd2022/Japan/
#          train/
#            images/   ← JPEG road images
#            annotations/xmls/  ← Pascal VOC XML annotation files
#   4. Re-run this script with --rdd-scan to verify the counts.
#
# Class mapping used in scripts/prepare_dataset.py:
#   D00 (longitudinal crack)  →  class 1 = crack
#   D10 (transverse crack)    →  class 1 = crack
#   D20 (alligator crack)     →  class 1 = crack
#   D40 (pothole)             →  class 0 = pothole
#   D44 (pothole, less common variant) →  class 0 = pothole
# ============================================================

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT      = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw"

RF_DIR    = DATA_RAW / "roboflow"
KG_DIR    = DATA_RAW / "kaggle"
RDD_DIR   = DATA_RAW / "rdd2022"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_images(folder: Path) -> int:
    """Recursively count image files under *folder*."""
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def detect_annotation_format(folder: Path) -> str:
    """
    Heuristically detect the annotation format present under *folder*.
    Returns one of: 'YOLO .txt', 'Pascal VOC XML', 'COCO JSON', 'unknown', 'none'.
    """
    if not folder.exists():
        return "none"
    files = list(folder.rglob("*"))
    exts  = {p.suffix.lower() for p in files}
    if ".xml" in exts:
        return "Pascal VOC XML"
    if ".json" in exts:
        # A single annotations.json is COCO; many per-image .json files are unusual
        json_files = [p for p in files if p.suffix.lower() == ".json"]
        if any("annotations" in p.name.lower() for p in json_files):
            return "COCO JSON"
        return "COCO JSON (likely)"
    if ".txt" in exts:
        # Make sure it's not just README.txt etc.
        txt_files = [p for p in files if p.suffix.lower() == ".txt"
                     and p.name.lower() not in {"readme.txt", "notes.txt"}]
        if txt_files:
            return "YOLO .txt"
    return "unknown"


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 65)
    print(f"  {'SOURCE':<15} {'IMAGES':>8}   {'ANNOTATION FORMAT'}")
    print("=" * 65)
    for r in results:
        status = str(r.get("images", "N/A"))
        fmt    = r.get("format", "unknown")
        note   = f"  [{r['note']}]" if r.get("note") else ""
        print(f"  {r['source']:<15} {status:>8}   {fmt}{note}")
    print("=" * 65)
    total = sum(r.get("images", 0) for r in results if isinstance(r.get("images"), int))
    print(f"  {'TOTAL':<15} {total:>8}")
    print("=" * 65 + "\n")


# ===========================================================================
# SOURCE 1 — Roboflow
# ===========================================================================

def download_roboflow() -> dict:
    """
    Download dataset from Roboflow Universe in YOLOv8 format.

    The dataset is saved to data/raw/roboflow/.
    Returns a summary dict for the final table.
    """
    print("\n[Roboflow] Downloading dataset ...")
    print(f"           Workspace : {ROBOFLOW_WORKSPACE}")
    print(f"           Project   : {ROBOFLOW_PROJECT}")
    print(f"           Version   : {ROBOFLOW_VERSION}")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("  [ERROR] roboflow package not installed. Run: python -m pip install roboflow")
        return {"source": "roboflow", "images": 0, "format": "N/A", "note": "not installed"}

    RF_DIR.mkdir(parents=True, exist_ok=True)

    try:
        rf      = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        version = project.version(ROBOFLOW_VERSION)

        # Download in YOLOv8 format into data/raw/roboflow/
        version.download("yolov8", location=str(RF_DIR), overwrite=True)

        imgs = count_images(RF_DIR)
        fmt  = detect_annotation_format(RF_DIR)
        print(f"[Roboflow] Done — {imgs} images found, format: {fmt}")
        return {"source": "roboflow", "images": imgs, "format": fmt}

    except Exception as exc:
        print(f"[Roboflow] ERROR: {exc}")
        # Try to report what's on disk even if download failed partway
        imgs = count_images(RF_DIR)
        return {"source": "roboflow", "images": imgs, "format": detect_annotation_format(RF_DIR),
                "note": f"error: {exc}"}


# ===========================================================================
# SOURCE 2 — Kaggle  (andrewmvd/pothole-detection)
# ===========================================================================

def download_kaggle() -> dict:
    """
    Download the Kaggle pothole detection dataset using the Kaggle API.

    Requires ~/.kaggle/kaggle.json to exist with your API credentials.
    Dataset page: https://www.kaggle.com/datasets/andrewmvd/pothole-detection
    """
    print("\n[Kaggle] Downloading dataset ...")
    print(f"         Dataset : {KAGGLE_DATASET}")

    KG_DIR.mkdir(parents=True, exist_ok=True)

    # Set Kaggle API token via env var if not already configured
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print(f"[Kaggle] WARNING: {kaggle_json} not found.")
        print("         Place your kaggle.json at ~/.kaggle/kaggle.json and retry.")
        print("         Format: {\"username\": \"YOUR_USERNAME\", \"key\": \"YOUR_API_KEY\"}")

    try:
        from kaggle import KaggleApi

        api = KaggleApi()
        api.authenticate()

        print(f"[Kaggle] Authenticated. Downloading to {KG_DIR} ...")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(KG_DIR),
            unzip=True,
            quiet=False,
        )

        imgs = count_images(KG_DIR)
        fmt  = detect_annotation_format(KG_DIR)
        print(f"[Kaggle] Done — {imgs} images found, format: {fmt}")
        return {"source": "kaggle", "images": imgs, "format": fmt}

    except Exception as exc:
        print(f"[Kaggle] ERROR: {exc}")
        imgs = count_images(KG_DIR)
        return {"source": "kaggle", "images": imgs, "format": detect_annotation_format(KG_DIR),
                "note": f"error: {exc}"}


# ===========================================================================
# SOURCE 3 — RDD2022  (manual placement, then scan)
# ===========================================================================

# Countries typically distributed in RDD2022
RDD2022_COUNTRIES = ["Japan", "India", "United_States", "Czech", "Norway", "China"]


def scan_rdd2022() -> dict:
    """
    Scan whatever RDD2022 data has been manually placed under data/raw/rdd2022/.
    Prints per-country counts and returns an aggregate summary dict.
    """
    print("\n[RDD2022] Scanning manually-placed files ...")
    print(f"          Expected root: {RDD_DIR}")
    print()

    if not RDD_DIR.exists():
        print("[RDD2022] Directory does not exist yet.")
        print("          Please follow the manual download instructions at the top of this script.")
        return {"source": "rdd2022", "images": 0, "format": "Pascal VOC XML", "note": "not downloaded"}

    total_imgs = 0
    found_any  = False

    for country in RDD2022_COUNTRIES:
        country_dir = RDD_DIR / country
        if not country_dir.exists():
            print(f"  {country:<20} NOT FOUND  (expected: {country_dir})")
            continue

        # Images are under train/images/ (and test/images/ for some countries)
        img_count  = count_images(country_dir)
        # Annotations
        xml_count  = sum(1 for p in country_dir.rglob("*.xml"))
        total_imgs += img_count
        found_any   = True

        print(f"  {country:<20} {img_count:>5} images   {xml_count:>5} XML annotations")

    # Also report any unrecognised subfolders
    if RDD_DIR.exists():
        known = {c.lower() for c in RDD2022_COUNTRIES}
        for sub in sorted(RDD_DIR.iterdir()):
            if sub.is_dir() and sub.name.lower() not in known:
                n = count_images(sub)
                print(f"  {sub.name:<20} {n:>5} images   (unrecognised country folder)")
                total_imgs += n
                found_any   = True

    if not found_any:
        print("  No country folders found yet.  See manual instructions above.")

    fmt = detect_annotation_format(RDD_DIR) if RDD_DIR.exists() else "Pascal VOC XML"
    note = None if found_any else "not downloaded"
    print(f"\n[RDD2022] Total: {total_imgs} images, format: {fmt}")
    return {"source": "rdd2022", "images": total_imgs, "format": fmt, "note": note}


# ===========================================================================
# Entry-point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pothole/road-damage datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--roboflow", action="store_true", help="Download Roboflow dataset")
    parser.add_argument("--kaggle",   action="store_true", help="Download Kaggle dataset")
    parser.add_argument("--rdd-scan", action="store_true", help="Scan manually-placed RDD2022 files")
    parser.add_argument("--all",      action="store_true", help="Run all three steps (default)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default: run everything if no flag is given
    run_all      = args.all or not (args.roboflow or args.kaggle or args.rdd_scan)
    do_roboflow  = run_all or args.roboflow
    do_kaggle    = run_all or args.kaggle
    do_rdd       = run_all or args.rdd_scan

    print("=" * 65)
    print("  Pothole Detection - Dataset Downloader")
    print("=" * 65)

    results = []

    if do_roboflow:
        results.append(download_roboflow())

    if do_kaggle:
        results.append(download_kaggle())

    if do_rdd:
        results.append(scan_rdd2022())

    print_summary(results)


if __name__ == "__main__":
    main()
