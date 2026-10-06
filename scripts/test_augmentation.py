"""
Task C — Augmentation test.

Runs a single crack-containing image through augment_image() 3 times
and saves annotated outputs to data/diagnostic/aug_test/ so you can
visually confirm boxes still align after augmentation.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import cv2

ROOT     = Path(__file__).resolve().parent.parent
LBL_DIR  = ROOT / "data" / "processed" / "train" / "labels"
IMG_DIR  = ROOT / "data" / "processed" / "train" / "images"
OUT_DIR  = ROOT / "data" / "diagnostic" / "aug_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from src.preprocessing.augment import augment_image

CLASS_COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}
CLASS_NAMES  = {0: "pothole", 1: "crack"}

def draw_boxes(image, bboxes, class_labels):
    h, w = image.shape[:2]
    img = image.copy()
    for box, cid in zip(bboxes, class_labels):
        cx, cy, bw, bh = box
        x1 = int((cx - bw/2)*w); y1 = int((cy - bh/2)*h)
        x2 = int((cx + bw/2)*w); y2 = int((cy + bh/2)*h)
        color = CLASS_COLORS.get(cid, (128,128,128))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, CLASS_NAMES.get(cid, str(cid)), (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

# Find a crack image with multiple annotations
test_lf = None
for lf in sorted(LBL_DIR.glob("*.txt")):
    anns = [l for l in lf.read_text().strip().splitlines() if l.strip()]
    has_crack = any(l.split()[0] == "1" for l in anns)
    if has_crack and len(anns) >= 2:
        test_lf = lf
        break

if test_lf is None:
    print("No suitable label file found")
    sys.exit(1)

stem = test_lf.stem
img_path = None
for ext in [".jpg", ".jpeg", ".png"]:
    c = IMG_DIR / (stem + ext)
    if c.exists(): img_path = c; break

if img_path is None:
    print(f"No image for {test_lf.name}")
    sys.exit(1)

img = cv2.imread(str(img_path))
label_text = test_lf.read_text()

bboxes = []
class_labels = []
for line in label_text.strip().splitlines():
    parts = line.strip().split()
    if len(parts) < 5: continue
    class_labels.append(int(parts[0]))
    bboxes.append([float(x) for x in parts[1:5]])

print(f"Input: {img_path.name}  |  boxes={len(bboxes)}  |  classes={class_labels}")

# Save original with boxes
orig_annotated = draw_boxes(img, bboxes, class_labels)
cv2.imwrite(str(OUT_DIR / "aug_test_0_original.jpg"), orig_annotated)
print(f"Saved: aug_test_0_original.jpg")

# Run 3 augmentation passes
for i in range(1, 4):
    aug_img, aug_boxes, aug_cls = augment_image(img, bboxes, class_labels)
    annotated = draw_boxes(aug_img, aug_boxes, aug_cls)
    out_name = f"aug_test_{i}_augmented.jpg"
    cv2.imwrite(str(OUT_DIR / out_name), annotated)
    print(f"Saved: {out_name}  |  boxes remaining={len(aug_boxes)}")

print(f"\nAll outputs saved to {OUT_DIR}")
print("Check data/diagnostic/aug_test/ to confirm boxes are correctly aligned.")
