#!/usr/bin/env python
"""Build a Contract-B-style equipment seed for `run_stage5b.py --equipment` from a GT delivery export.

The GT export (`~/Downloads/0002A/equipment_list.csv`) carries equipment tags, types and
`bbox_px` in 600 dpi page pixels; the traced raster is half that, so boxes are scaled by
--scale (default 0.5). Types are mapped onto the project's own EQUIPMENT_LABELS vocabulary.

    python garnet/gt_equipment_seed.py --gt ~/Downloads/0002A --out /tmp/gt_equip_0002A.json
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from garnet.detect_equipment_bboxes import AI_CLASS_MAP, EQUIPMENT_LABELS  # noqa: E402


# Prose GT types ("horizontal drum (settler)") are not class keys; the AI_CLASS_MAP keys
# are the vocabulary the detector already understands, so match those as substrings too.
# A motor driver is deliberately not a seed: it has no process nozzles of its own.
NO_SEED_HINTS = ("motor",)


def to_label(raw: str) -> str:
    t = raw.strip().lower()
    if any(h in t for h in NO_SEED_HINTS):
        return ""
    if t in EQUIPMENT_LABELS:
        return t
    if t in AI_CLASS_MAP:
        return AI_CLASS_MAP[t]
    for key in sorted(AI_CLASS_MAP, key=len, reverse=True):
        if key in t:
            return AI_CLASS_MAP[key]
    for label in sorted(EQUIPMENT_LABELS, key=len, reverse=True):   # substring, longest first
        if label in t:
            return label
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="delivery dir with equipment_list.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=0.5, help="GT px -> traced raster px")
    ap.add_argument("--sheet", default="", help="only equipment of this sheet (multi-sheet GT)")
    a = ap.parse_args()

    src = Path(a.gt) / "equipment_list.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8-sig")))
    if a.sheet:
        rows = [r for r in rows if str(r.get("sheet") or "") == a.sheet]
    objects, dropped = [], []
    for r in rows:
        label = to_label(r.get("type") or "")
        raw = (r.get("bbox_px") or "").strip().strip("[]")
        try:
            x0, y0, x1, y1 = (float(v) for v in raw.split(","))
        except ValueError:
            dropped.append((r.get("id"), "no bbox"))
            continue
        if not label:
            dropped.append((r.get("id"), f"type '{r.get('type')}' outside EQUIPMENT_LABELS"))
            continue
        objects.append({
            "Equipment_type": label,
            "Tag": (r.get("tag") or "").strip(),
            "Bounding_box_px": {"x_min": round(x0 * a.scale), "y_min": round(y0 * a.scale),
                                "x_max": round(x1 * a.scale), "y_max": round(y1 * a.scale)},
        })
    Path(a.out).write_text(json.dumps({"objects": objects}, indent=1))
    print(f"{len(objects)} equipment seeds -> {a.out}  (scale {a.scale})")
    for o in objects:
        b = o["Bounding_box_px"]
        print(f"  {o['Equipment_type']:10s} {o['Tag']:10s} {b['x_min']:5d},{b['y_min']:5d} - {b['x_max']:5d},{b['y_max']:5d}")
    if dropped:
        print("dropped:", dropped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
