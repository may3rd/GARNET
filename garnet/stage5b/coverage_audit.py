#!/usr/bin/env python
"""Pipe-coverage audit for a Stage 5b run.

Reports how much pipe mask ink NO walk covers, split into what actually matters:

  * **pipe-like uncovered** — elongated components, >= 60px long, excluding the
    sheet frame/border. These are candidate MISSING LINES.
  * **frame/other** — the border, hatching, and small blobs. Not pipe.

Why this exists: a walk count and a terminal-type histogram can both be identical
while whole pipe branches go unvisited. On sheets 0003 and 0007 two PSV legs were
100% uncovered with no change in any count. The naive headline ("85% of mask ink
uncovered") is also misleading — one component spanning the sheet is the drawing
frame. This tool separates the two so the number means something.

Usage:
    python -m garnet.stage5b.coverage_audit --out-dir output/stage5b_14780-8120-25-25-0003_ocr
    python -m garnet.stage5b.coverage_audit --out-dir <dir> --top 20 --json report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MIN_AREA = 200
DEFAULT_MIN_LENGTH = 60
DEFAULT_MIN_ELONGATION = 3.0
FRAME_SPAN_FRACTION = 0.8


def load_walks(out_dir: Path) -> dict[str, dict]:
    """Every walk from both artifacts, keyed by id."""
    walks: dict[str, dict] = {}
    t = out_dir / "stage5b_trace_results.json"
    if t.is_file():
        walks.update(json.loads(t.read_text()))
    b = out_dir / "stage5b_branch_trace_results.json"
    if b.is_file():
        walks.update(json.loads(b.read_text()).get("branches") or {})
    return walks


def audit(
    out_dir: Path,
    min_area: int = DEFAULT_MIN_AREA,
    min_length: int = DEFAULT_MIN_LENGTH,
    min_elongation: float = DEFAULT_MIN_ELONGATION,
    halo_px: int = 7,
) -> dict:
    """Measure uncovered mask ink and characterise its shape.

    `halo_px` is the half-width painted for each walk segment. A walk follows the
    ~2px centreline, so a stroke without a halo would leave the drawn line's own
    edges "uncovered" and overstate the gap. 7 matches the renderers.
    """
    mask_path = out_dir / "stage5_pipe_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read {mask_path}")
    mask = mask > 0
    h, w = mask.shape

    cov = np.zeros_like(mask, dtype=np.uint8)
    walks = load_walks(out_dir)
    n_segments = 0
    for walk in walks.values():
        for seg in walk.get("segments") or []:
            cv2.line(cov, (int(seg["x1"]), int(seg["y1"])),
                     (int(seg["x2"]), int(seg["y2"])), 1, halo_px * 2 + 1)
            n_segments += 1
    covered = cov.astype(bool)

    uncovered = np.logical_and(mask, np.logical_not(covered)).astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered, connectivity=8)

    pipe_like: list[dict] = []
    frame_px = 0
    small_px = 0
    for i in range(1, n_lab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw > FRAME_SPAN_FRACTION * w or bh > FRAME_SPAN_FRACTION * h:
            frame_px += area
            continue
        elong = max(bw, bh) / max(1, min(bw, bh))
        if area >= min_area and max(bw, bh) >= min_length and elong >= min_elongation:
            pipe_like.append({"area": area, "bbox": [x, y, x + bw, y + bh],
                              "size": [bw, bh], "elongation": round(elong, 1)})
        else:
            small_px += area

    pipe_like.sort(key=lambda p: -p["area"])
    mask_px = int(mask.sum())
    total_unc = int(uncovered.sum())
    return {
        "out_dir": str(out_dir),
        "image_size": [w, h],
        "walks": len(walks),
        "segments": n_segments,
        "mask_px": mask_px,
        "uncovered_px": total_unc,
        "uncovered_pct": round(100 * total_unc / max(mask_px, 1), 2),
        "pipe_like_uncovered_px": sum(p["area"] for p in pipe_like),
        "pipe_like_pct_of_mask": round(
            100 * sum(p["area"] for p in pipe_like) / max(mask_px, 1), 2),
        "frame_uncovered_px": frame_px,
        "small_uncovered_px": small_px,
        "pipe_like_components": len(pipe_like),
        "candidates": pipe_like,
        "thresholds": {"min_area": min_area, "min_length": min_length,
                       "min_elongation": min_elongation, "halo_px": halo_px},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit uncovered pipe in a Stage 5b run.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--json", default=None, help="Write the full report here.")
    args = p.parse_args(argv)

    r = audit(Path(args.out_dir))
    print(f"{r['out_dir']}")
    print(f"  walks {r['walks']}  segments {r['segments']}  image {r['image_size'][0]}x{r['image_size'][1]}")
    print(f"  mask ink            {r['mask_px']:>11,} px")
    print(f"  uncovered (any)     {r['uncovered_px']:>11,} px  ({r['uncovered_pct']}%)")
    print(f"    of which frame    {r['frame_uncovered_px']:>11,} px  <- border/hatching, not pipe")
    print(f"    of which small    {r['small_uncovered_px']:>11,} px  <- specks below threshold")
    print(f"    PIPE-LIKE         {r['pipe_like_uncovered_px']:>11,} px  "
          f"({r['pipe_like_pct_of_mask']}% of mask) in {r['pipe_like_components']} components")
    if r["candidates"]:
        print(f"\n  largest {min(args.top, len(r['candidates']))} candidate MISSING LINES:")
        for c in r["candidates"][:args.top]:
            b = c["bbox"]
            print(f"    area={c['area']:6d}  bbox=({b[0]},{b[1]})-({b[2]},{b[3]})  "
                  f"{c['size'][0]}x{c['size'][1]}  elong={c['elongation']}")
    if args.json:
        Path(args.json).write_text(json.dumps(r, indent=2))
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
