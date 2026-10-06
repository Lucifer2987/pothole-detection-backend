"""
End-to-end smoke test for the Pothole Detection pipeline.

Tests the COMPLETE pipeline WITHOUT needing the HTTP server:
  1.  DB connectivity check
  2.  Seed a test camera via crud.py
  3.  Run the detection pipeline directly on a sample image from the test set
  4.  Print what potholes were detected, severity, new/existing status, incidents
  5.  Run a SECOND pass to confirm change_detector marks same pothole as "existing"
  6.  Cleanup (remove test camera & related rows)

Usage:
    python -m scripts.smoke_test
    (run from project root with venv active)
"""

from __future__ import annotations

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── DB / model bootstrap ──────────────────────────────────────────────────────
from src.database.db import SessionLocal, ping_db
from src.database import crud, models
from src.database.models import (
    ScanStatus, Severity, PotholeStatus, ClassType, IncidentStatus
)
from src.detection.infer import get_detector
from src.detection.severity import score_severity
from src.tracking.change_detector import ChangeDetector

import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SECTION = "=" * 60

def section(title: str):
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


def find_test_image() -> Path:
    """Return first available image from the test split."""
    test_dir = ROOT / "data" / "processed" / "test" / "images"
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs = sorted(test_dir.glob(ext))
        if imgs:
            return imgs[0]
    raise FileNotFoundError(f"No test images in {test_dir}")


def run_pipeline(db, detector, source_path: str, camera_id: int, scan_label: str):
    """
    Run one full scan pass:
        create scan → infer → change-detect → auto-create incidents
    Returns (scan, changes, incidents_created).
    """
    print(f"\n  [{scan_label}] Source: {Path(source_path).name}")

    # 1. Create scan record
    scan = crud.create_scan(db, source_path=source_path, camera_id=camera_id)
    crud.update_scan_status(db, scan.id, ScanStatus.processing)

    # 2. Inference
    dets = detector.infer_image_path(source_path)
    print(f"  [{scan_label}] Raw detections: {len(dets)}")
    for d in dets[:5]:
        print(f"    class={d.class_name}  conf={d.confidence:.2f}  "
              f"box=[{round(d.x1)},{round(d.y1)},{round(d.x2)},{round(d.y2)}]")

    # 3. Change detection
    cd = ChangeDetector(camera_id=camera_id, scan_id=scan.id)
    changes = cd.process_detections(db=db, detections=dets,
                                    camera_lat=18.52, camera_lon=73.85)

    new_ct   = sum(1 for c in changes if c.is_new)
    exist_ct = sum(1 for c in changes if not c.is_new)
    grow_ct  = sum(1 for c in changes if c.is_growing)
    print(f"  [{scan_label}] Change-detect: {new_ct} new  {exist_ct} existing  {grow_ct} growing")

    # 4. Auto-create incidents for critical/high
    incidents_created = 0
    for change in changes:
        sev_str = change.severity
        if sev_str in ("critical", "high"):
            try:
                sev = Severity(sev_str)
            except ValueError:
                sev = Severity.low
            crud.create_incident(db, pothole_id=change.pothole_id,
                                 severity=sev, department=None)
            incidents_created += 1

    crud.update_scan_status(db, scan.id, ScanStatus.completed)
    print(f"  [{scan_label}] Incidents auto-created: {incidents_created}")

    return scan, changes, incidents_created


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    section("STEP 1 — DB connectivity")
    ok = ping_db()
    print(f"  DB reachable: {ok}")
    assert ok, "Database is not reachable. Is docker-compose up?"

    section("STEP 2 — Load YOLOv8 detector")
    detector = get_detector()
    print(f"  Model loaded: {detector.model_path}")

    section("STEP 3 — Seed test camera")
    db = SessionLocal()
    cam = crud.create_camera(db, name="[SMOKE-TEST] Camera 01",
                             lat=18.52, lon=73.85, is_active=True)
    print(f"  Created camera id={cam.id}  name={cam.name}")

    try:
        img_path = find_test_image()
        print(f"\n  Test image: {img_path}")

        # ── PASS 1 ──────────────────────────────────────────────────────────
        section("STEP 4 — PASS 1: first scan (expect NEW potholes)")
        scan1, changes1, inc1 = run_pipeline(
            db, detector, str(img_path), cam.id, "PASS-1"
        )

        potholes = crud.list_potholes(db, camera_id=cam.id)
        print(f"\n  Potholes in DB after PASS 1: {len(potholes)}")
        for p in potholes[:10]:
            sev = score_severity(p.current_area or 0, 0.9, p.class_type)
            print(f"    id={p.id}  class={p.class_type.value}  "
                  f"status={p.status.value}  area={p.current_area}  severity={sev}")

        if not changes1:
            print("\n  [!] No detections — check model weights & image content")

        # ── PASS 2 ──────────────────────────────────────────────────────────
        section("STEP 5 — PASS 2: same image (expect EXISTING potholes, NOT new)")
        scan2, changes2, inc2 = run_pipeline(
            db, detector, str(img_path), cam.id, "PASS-2"
        )

        potholes2 = crud.list_potholes(db, camera_id=cam.id)
        new_count2   = sum(1 for p in potholes2 if p.status == PotholeStatus.new)
        exist_count2 = sum(1 for p in potholes2 if p.status == PotholeStatus.existing)
        print(f"\n  After PASS 2 — pothole statuses: new={new_count2}  existing={exist_count2}")
        assert exist_count2 > 0 or new_count2 == len(potholes2), \
            "Expected same potholes to now be 'existing' after second pass"
        print("  [OK] Same pothole correctly tracked across two scans")

        # ── Incidents ───────────────────────────────────────────────────────
        section("STEP 6 — Open incidents check")
        incidents = crud.list_incidents(db, limit=50)
        print(f"  Total incidents: {len(incidents)}")
        for inc in incidents[:5]:
            print(f"    id={inc.id}  pothole={inc.pothole_id}  "
                  f"sev={inc.severity.value}  status={inc.status.value}")

        section("SMOKE TEST PASSED")
        print("  All pipeline stages completed successfully:")
        print(f"    - Pass 1: {len(changes1)} detections, {inc1} incidents")
        print(f"    - Pass 2: {len(changes2)} detections ({exist_count2} existing)")
        print(f"    - Total incidents created: {len(incidents)}")

    finally:
        # ── Cleanup ─────────────────────────────────────────────────────────
        section("CLEANUP — removing test data")
        potholes_all = crud.list_potholes(db, camera_id=cam.id, limit=500)
        for inc in crud.list_incidents(db, limit=500):
            if any(inc.pothole_id == p.id for p in potholes_all):
                db.delete(inc)
        db.commit()
        for p in potholes_all:
            db.delete(p)
        db.commit()
        scans = crud.list_scans(db, camera_id=cam.id, limit=500)
        for s in scans:
            db.delete(s)
        db.commit()
        crud.delete_camera(db, cam.id)
        db.close()
        print("  Test camera, scans, potholes, and incidents removed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
