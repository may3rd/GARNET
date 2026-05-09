#!/usr/bin/env python3
"""
demo_line_inpaint.py

Standalone demo of the geometric line-extraction (Option A) pipeline.

Usage:
    cd /Volumes/Ginnungagap/maetee/Code/GARNET/backend
    python scripts/demo_line_inpaint.py \
        --input test/ppcl/Test-00001.jpg \
        --output output/demo_inpaint

Arguments:
    --input     Path to P&ID image (jpg/png)
    --output    Directory for debug artifacts (created if needed)

Artifacts written:
    adapt_01_source.jpg              — input grayscale
    adapt_02_corner_overlay.jpg      — Shi-Tomasi corners (green)
    adapt_03_inpaint_mask.jpg        — mask used for Telea
    adapt_04_cleaned_gray.jpg        — after inpainting
    adapt_05_cleaned_binary.jpg      — thresholded cleaned image
    adapt_06_segments_overlay.jpg    — H (yellow) / V (magenta) segments
    adapt_07_summary.json            — stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure backend/ is on path so `garnet` imports work
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from garnet.line_detection_inpaint import run_line_detection_inpaint, render_line_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo geometric line detection with inpainting")
    parser.add_argument("--input", required=True, help="Path to P&ID image")
    parser.add_argument("--output", default="output/demo_inpaint", help="Output directory")
    parser.add_argument("--min-seg-len", type=float, default=12.0, help="Prune segments shorter than this (px)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        return 1

    # ── Load image ──────────────────────────────────────────
    image_bgr = cv2.imread(str(in_path))
    if image_bgr is None:
        print(f"ERROR: could not read image: {in_path}")
        return 1

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # ── Run pipeline ────────────────────────────────────────
    # Pass empty text/object regions for a pure-geometry first look.
    # In real pipeline integration these come from Stage 2 (OCR) and Stage 4 (detection).
    result = run_line_detection_inpaint(
        stage1_gray=gray,
        text_regions=[],
        object_regions=[],
        image_id=str(in_path.stem),
    )

    # Optional: re-run with stricter length filter if user asked
    if args.min_seg_len != 12.0:
        from garnet.line_detection_inpaint import _prune_orphan_segments as _prune
        segs = _prune(result["segments"], min_length=args.min_seg_len)
        result["segments"] = segs
        result["summary"]["final_segments"] = len(segs)

    # ── Write artifacts ─────────────────────────────────────
    stem = "adapt"

    cv2.imwrite(str(out_dir / f"{stem}_01_source.jpg"), gray)

    # Corner overlay
    corner_overlay = image_bgr.copy()
    corners = result.get("corner_points", np.empty((0, 2)))
    for (x, y) in corners:
        cv2.circle(corner_overlay, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.imwrite(str(out_dir / f"{stem}_02_corner_overlay.jpg"), corner_overlay)

    cv2.imwrite(str(out_dir / f"{stem}_03_inpaint_mask.jpg"), result["inpaint_mask"])
    cv2.imwrite(str(out_dir / f"{stem}_04_cleaned_gray.jpg"), result["cleaned_gray"])
    cv2.imwrite(str(out_dir / f"{stem}_05_cleaned_binary.jpg"), result["cleaned_binary"])

    # Segment overlay
    seg_overlay = render_line_overlay(image_bgr, result["segments"])
    cv2.imwrite(str(out_dir / f"{stem}_06_segments_overlay.jpg"), seg_overlay)

    # Summary JSON
    summary = result["summary"]
    summary["params"] = {"min_segment_length_px": args.min_seg_len}
    with open(out_dir / f"{stem}_07_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    print("=" * 50)
    print(f"Demo complete: {in_path.name}")
    print(f"  Artifacts: {out_dir}")
    print(f"  Raw segments        : {summary['raw_segments']}")
    print(f"  After collinear merge: {summary['after_collinear_merge']}")
    print(f"  Final segments      : {summary['final_segments']} (H={summary['horizontal_count']} V={summary['vertical_count']})")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
