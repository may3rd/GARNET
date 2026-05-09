#!/usr/bin/env python3
"""
debug_timing.py — profile each phase of line_detection_inpaint on a test image.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from garnet.line_detection_inpaint import (
    _adaptive_threshold_mask,
    _detect_corner_points,
    _points_to_bboxes_fast,
    _assemble_inpaint_mask,
    _inpaint_masked_region,
    _cleaned_to_binary,
    _extract_contour_segments,
    _merge_collinear_segments,
    _split_horizontal_vertical,
    _merge_nearby_endpoints,
    _prune_orphan_segments,
)


def profile(name: str, fn, *args, **kwargs):
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"  {name}: {time.time() - t0:.3f}s")
    return result


def main() -> int:
    in_path = Path(__file__).resolve().parent.parent / "test" / "!test01.png"
    if not in_path.exists():
        print(f"Image not found: {in_path}")
        return 1

    print(f"Loading: {in_path}")
    image_bgr = cv2.imread(str(in_path))
    if image_bgr is None:
        print("Could not read image")
        return 1
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print(f"  Shape: {gray.shape}")

    shape = gray.shape[:2]
    text_regions = []
    object_regions = []

    print("\nPhase timings:")
    thresh = profile("A. adaptive_threshold_mask", _adaptive_threshold_mask, gray)

    corner_points = profile("B. detect_corner_points", _detect_corner_points, thresh)
    print(f"    -> {len(corner_points)} corners found")

    corner_boxes = profile("C1. points_to_bboxes_fast", _points_to_bboxes_fast, corner_points, 40)
    print(f"    -> {len(corner_boxes)} boxes")
    inpaint_mask = profile("C2. assemble_inpaint_mask", _assemble_inpaint_mask, shape, corner_points, [], [])
    print(f"    -> mask nonzero={np.count_nonzero(inpaint_mask)}")

    cleaned_gray = profile("D. inpaint_masked_region (Telea)", _inpaint_masked_region, gray, inpaint_mask)

    cleaned_binary = profile("E1. cleaned_to_binary", _cleaned_to_binary, cleaned_gray)
    print(f"    -> binary nonzero={np.count_nonzero(cleaned_binary)}")

    raw_segments = profile("E2. extract_contour_segments", _extract_contour_segments, cleaned_binary)
    print(f"    -> {len(raw_segments)} raw segments")

    merged = profile("F1. merge_collinear_segments", _merge_collinear_segments, raw_segments)
    horiz, vert = profile("F2. split_horizontal_vertical", _split_horizontal_vertical, merged)
    print(f"    -> H={len(horiz)} V={len(vert)}")

    all_end_merged = profile("G1. merge_nearby_endpoints", _merge_nearby_endpoints, horiz + vert)
    filtered = profile("G2. prune_orphan_segments", _prune_orphan_segments, all_end_merged)
    print(f"    -> final={len(filtered)} segments")

    final_h, final_v = _split_horizontal_vertical(filtered)
    print(f"\nFinal: {len(filtered)} segs (H={len(final_h)}, V={len(final_v)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())