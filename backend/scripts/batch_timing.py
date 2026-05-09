#!/usr/bin/env python3
"""
batch_timing.py — run line_detection_inpaint on multiple test images and collect stats.
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from garnet.line_detection_inpaint import (
    run_line_detection_inpaint,
)

TEST_DIR = Path(__file__).resolve().parent.parent / "test"

# Representative sample across different image sets
TEST_IMAGES = [
    # PPCL test sheets (cleaner, simpler)
    "ppcl/Test-00001.jpg",
    "ppcl/Test-00004.jpg",
    "ppcl/Test-00007.jpg",
    "ppcl/Test-00009.jpg",
    # PTTEP real P&IDs (more complex)
    "pttep/images/PLCPP2/001.png",
    "pttep/images/PLCPP2/010.png",
    "pttep/images/PLCPP2/149.png",
    # Original test
    "!test01.png",
]


def profile_image(rel: str) -> dict | None:
    in_path = TEST_DIR / rel
    if not in_path.exists():
        print(f"  SKIP: {rel} (not found)")
        return None

    image_bgr = cv2.imread(str(in_path))
    if image_bgr is None:
        print(f"  SKIP: {rel} (cannot read)")
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    px = f"{w}x{h}"

    t0 = time.time()
    result = run_line_detection_inpaint(
        stage1_gray=gray,
        text_regions=[],
        object_regions=[],
        image_id=rel,
    )
    elapsed = time.time() - t0

    summary = result["summary"]
    row = {
        "image": rel,
        "size_px": px,
        "total_s": round(elapsed, 3),
        "raw_segments": summary["raw_segments"],
        "after_collinear": summary["after_collinear_merge"],
        "final_segments": summary["final_segments"],
        "horizontal": summary["horizontal_count"],
        "vertical": summary["vertical_count"],
        "corners": summary["corner_points_detected"],
    }
    print(f"  {rel:45s}  {px:10s}  {elapsed:6.3f}s  "
          f"raw={summary['raw_segments']:5d}  "
          f"final={summary['final_segments']:5d}  "
          f"H={summary['horizontal_count']:4d} V={summary['vertical_count']:4d}")
    return row


def main():
    print(f"{'Image':<45s}  {'Size':<10s}  {'Time':<7s}  {'Raw':<7s}  {'Final':<7s}  H/V")
    print("-" * 110)

    results = []
    for rel in TEST_IMAGES:
        row = profile_image(rel)
        if row:
            results.append(row)

    # Summary stats
    if results:
        times = [r["total_s"] for r in results]
        finals = [r["final_segments"] for r in results]
        print(f"\nSummary ({len(results)} images):")
        print(f"  Time  min={min(times):.3f}s  max={max(times):.3f}s  avg={sum(times)/len(times):.3f}s")
        print(f"  Final segments  min={min(finals)}  max={max(finals)}  avg={sum(finals)/len(finals):.0f}")

        # Write JSON report
        report_path = BACKEND_ROOT / "docs" / "batch_timing_results.json"
        report_path.write_text(json.dumps(results, indent=2))
        print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
