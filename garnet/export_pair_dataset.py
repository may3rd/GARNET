"""Export labelled item pairs from a delivery export — the training set a learned connectivity
model needs, and the ceiling check on whether one is even worth training.

A learned model (PIDCon, Relationformer/PID2Graph) does not replace tracing; it answers a
different question: given detected items, which pairs are connected. To train or evaluate that
you need pairs labelled yes/no against your own drawings, tied to *your detector's* boxes (not to
the export's), because that is the noise a model would see in production.

This script produces that, per sheet:
  positives  - every `pipe` edge in the export whose two endpoints both matched a detected object
  negatives  - detected-item pairs on the same sheet that the export does not connect
  report     - how many export items actually matched a detection (the ceiling), and the
               positive:negative counts

    python garnet/export_pair_dataset.py --gt ~/Downloads/0002_0003 --run-dir output/trace_suite/0002 \
        --sheet 0002 --out ~/Downloads/garnet-pairs.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def load_items(gt_dir: Path, sheet: str) -> dict[str, dict]:
    """Every item the export places on this sheet, with its bbox in 600 dpi page pixels."""
    items: dict[str, dict] = {}
    for name in ("equipment_list.csv", "valve_list.csv", "instrument_list.csv",
                 "special_items_list.csv"):
        path = gt_dir / name
        if not path.exists():
            continue
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if sheet and str(row.get("sheet") or "") != sheet:
                continue
            raw = (row.get("bbox_px") or "").strip().strip("[]")
            try:
                x0, y0, x1, y1 = (float(v) for v in raw.split(","))
            except ValueError:
                continue
            items[row["id"]] = {"id": row["id"], "tag": row.get("tag", ""),
                                "type": row.get("type", ""),
                                "bbox": [x0, y0, x1, y1]}
    return items


def load_edges(gt_dir: Path, sheet: str) -> list[dict]:
    raw = json.loads((gt_dir / "connectivity_model.json").read_text())
    model = raw.get("model", raw)
    edges = []
    for e in model.get("edges", []):
        if sheet and str(e.get("sheet") or "") != sheet:
            continue
        # The two exports label the same relationship differently: 0002_0003 says "pipe",
        # 0002A says "process". Both mean "a line joins these two items".
        if str(e.get("kind") or "").lower() not in ("pipe", "process"):
            continue
        edges.append({"source": e.get("source"), "target": e.get("target"),
                      "line_id": e.get("line_id"), "line_no": e.get("line_no")})
    return edges


def load_detections(run_dir: Path, scale: float = 0.5) -> list[dict]:
    objects = json.loads((run_dir / "stage4_objects.json").read_text()).get("objects", [])
    out = []
    for o in objects:
        b = o.get("bbox")
        if not b:
            continue
        out.append({"id": o["id"], "class_name": o.get("class_name"),
                    "bbox": [b["x_min"] / scale, b["y_min"] / scale,
                             b["x_max"] / scale, b["y_max"] / scale]})
    return out


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def centre(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def match(gt_items: dict[str, dict], detections: list[dict], max_px: float = 40.0) -> dict[str, str]:
    """GT item id -> detected object id, by best bbox overlap then centre distance."""
    pairs: dict[str, str] = {}
    used: set[str] = set()
    candidates = []
    for gid, item in gt_items.items():
        for det in detections:
            overlap = iou(item["bbox"], det["bbox"])
            cx1, cy1 = centre(item["bbox"])
            cx2, cy2 = centre(det["bbox"])
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            if overlap > 0.05 or dist <= max_px:
                candidates.append((overlap, -dist, gid, det["id"]))
    candidates.sort(reverse=True)
    for overlap, neg_dist, gid, did in candidates:
        if gid in pairs or did in used:
            continue
        pairs[gid] = did
        used.add(did)
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--sheet", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--negatives-per-positive", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gt_dir, run_dir = Path(args.gt).expanduser(), Path(args.run_dir).expanduser()
    gt_items = load_items(gt_dir, args.sheet)
    edges = load_edges(gt_dir, args.sheet)
    detections = load_detections(run_dir)
    pairs = match(gt_items, detections)

    positives = []
    for e in edges:
        a, b = pairs.get(str(e["source"])), pairs.get(str(e["target"]))
        if a and b and a != b:
            positives.append({"sheet": args.sheet, "a": a, "b": b, "label": 1,
                              "line_id": e.get("line_id"), "line_no": e.get("line_no")})
    connected = {frozenset((p["a"], p["b"])) for p in positives}

    rng = random.Random(args.seed)
    det_ids = [d["id"] for d in detections]
    target = int(len(positives) * args.negatives_per_positive)
    negatives = []
    guard = 0
    while len(negatives) < target and det_ids and guard < target * 200:
        guard += 1
        a, b = rng.sample(det_ids, 2)
        if frozenset((a, b)) in connected:
            continue
        negatives.append({"sheet": args.sheet, "a": a, "b": b, "label": 0,
                          "line_id": None, "line_no": None})

    out = Path(args.out).expanduser()
    with out.open("w") as fh:
        for row in positives + negatives:
            fh.write(json.dumps(row) + "\n")

    report = {
        "sheet": args.sheet,
        "gt_items": len(gt_items),
        "gt_items_matched_to_a_detection": len(pairs),
        "match_rate": round(len(pairs) / max(len(gt_items), 1), 3),
        "gt_pipe_edges": len(edges),
        "positives": len(positives),
        "positive_rate_of_edges": round(len(positives) / max(len(edges), 1), 3),
        "negatives": len(negatives),
        "detections": len(detections),
        "out": str(out),
    }
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
