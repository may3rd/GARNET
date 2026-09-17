#!/usr/bin/env python
"""Draw box-only overlays for a fixture set: objects, equipment, line numbers.

Reads the three fixture JSONs for one sheet and writes one PNG per type with nothing but
thin rectangles — no labels, no banners, so the drawing underneath stays readable.

Usage:
    python garnet/draw_boxes.py --stem Test-00001
    python garnet/draw_boxes.py --stem Test-00001 --indir garnet/tests/input --outdir output
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

# BGR. One colour per type, so a type is identifiable without any text.
COLORS = {
    "objects": (255, 0, 0),        # blue
    "equipment": (40, 180, 40),    # green
    "line_numbers": (0, 0, 255),   # red
}
THICKNESS = 2


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (p if p.exists() else REPO_ROOT / p)


def _xyxy_from_objects(entry: dict) -> tuple[int, int, int, int] | None:
    b = entry.get("bbox")
    if isinstance(b, dict) and {"x_min", "y_min", "x_max", "y_max"}.issubset(b):
        return int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])
    return None


def _xyxy_from_equipment(entry: dict) -> tuple[int, int, int, int] | None:
    b = entry.get("Bounding_box_px")
    if isinstance(b, dict) and {"x_min", "y_min", "x_max", "y_max"}.issubset(b):
        return int(b["x_min"]), int(b["y_min"]), int(b["x_max"]), int(b["y_max"])
    return None


def _xyxy_from_line_number(entry: dict) -> tuple[int, int, int, int] | None:
    # Contract C is pixel xywh; some labels are rotated so w/h can be tall and thin.
    try:
        left, top = int(entry["Left"]), int(entry["Top"])
        w, h = int(entry["Width"]), int(entry["Height"])
    except (KeyError, TypeError, ValueError):
        return None
    return left, top, left + w, top + h


EXTRACTORS = {
    "objects": _xyxy_from_objects,
    "equipment": _xyxy_from_equipment,
    "line_numbers": _xyxy_from_line_number,
}

# Fixture filename suffixes on disk (they are not uniform: objects.json, equipment_bboxes.json,
# line_number_boxes.json).
FILENAMES = {
    "objects": ["{stem}_objects.json"],
    "equipment": ["{stem}_equipment_bboxes.json", "{stem}_equipment.json"],
    "line_numbers": ["{stem}_line_number_boxes.json", "{stem}_line_numbers.json"],
}


def _find_fixture(indir: Path, stem: str, kind: str) -> Path | None:
    for pattern in FILENAMES[kind]:
        candidate = indir / pattern.format(stem=stem)
        if candidate.is_file():
            return candidate
    return None


def _load_boxes(path: Path, kind: str) -> list[tuple[int, int, int, int]]:
    payload = json.loads(path.read_text())
    entries = payload.get("objects")
    if not isinstance(entries, list):
        raise ValueError(f"{path.name}: no 'objects' list")
    extract = EXTRACTORS[kind]
    boxes = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        box = extract(entry)
        if box is not None:
            boxes.append(box)
    return boxes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Box-only overlays for objects / equipment / line numbers.")
    p.add_argument("--stem", required=True, help="Fixture stem, e.g. Test-00001")
    p.add_argument("--indir", default="garnet/tests/input", help="Fixture dir. Default: garnet/tests/input")
    p.add_argument("--outdir", default="output", help="Output dir. Default: output")
    p.add_argument("--thickness", type=int, default=THICKNESS)
    args = p.parse_args(argv)

    indir = _resolve(args.indir)
    outdir = _resolve(args.outdir)
    image_path = indir / f"{args.stem}.jpg"
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: cannot read image: {image_path}", file=sys.stderr)
        return 1
    height, width = image.shape[:2]
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{args.stem}: {width}x{height}")
    for kind, color in COLORS.items():
        fixture = _find_fixture(indir, args.stem, kind)
        if fixture is None:
            print(f"  {kind:<13} SKIP (no fixture for {args.stem})")
            continue

        boxes = _load_boxes(fixture, kind)
        overlay = image.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, args.thickness)

        out_path = outdir / f"{args.stem}_{kind}_boxes.png"
        if not cv2.imwrite(str(out_path), overlay):
            print(f"error: failed to write {out_path}", file=sys.stderr)
            return 1
        print(f"  {kind:<13} {len(boxes):>4} boxes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
