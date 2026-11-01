"""Review crops for the traces whose flow direction is still unknown (input for the flow vision agent).

    python3 -m garnet.graph.flow_candidates --stem <S> --dir <out dir> --indir <sheet dir>

Reads <dir>/graph.json, writes <dir>/flow_candidates.json and <dir>/flow_review/<trace id>.png + contact sheets.
One item per Stage 6 trace whose pipe edges are all unknown: A = where the trace starts, B = where it ends.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CONTEXT, MIN_W, MIN_H, MAX_W, MAX_H = 120, 500, 400, 1400, 1000
PER_SHEET, TILE_W, TILE_H = 6, 640, 460
TRACE_BGR, KNOWN_BGR = (0, 140, 255), (200, 80, 0)  # orange trace, blue known-flow context


def _arrow(img, poly, forward, scale, off, color):
    """Small arrowhead at the middle of a polyline pointing in its known flow direction."""
    pts = poly if forward else poly[::-1]
    half, acc, tot = 0.0, 0.0, sum(math.dist(a, b) for a, b in zip(poly, poly[1:]))
    for a, b in zip(pts, pts[1:]):
        d = math.dist(a, b)
        if acc + d >= tot / 2 and d > 0:
            f = (tot / 2 - acc) / d
            m = (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            u = ((b[0] - a[0]) / d, (b[1] - a[1]) / d)
            p0 = ((m[0] - u[0] * 8 - off[0]) * scale, (m[1] - u[1] * 8 - off[1]) * scale)
            p1 = ((m[0] + u[0] * 8 - off[0]) * scale, (m[1] + u[1] * 8 - off[1]) * scale)
            cv2.arrowedLine(img, tuple(int(v) for v in p0), tuple(int(v) for v in p1), color, 2, tipLength=0.9)
            return
        acc += d


def _poly(img, poly, scale, off, color, thick):
    pts = np.array([[(x - off[0]) * scale, (y - off[1]) * scale] for x, y in poly], np.int32)
    cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)


def render_item(raster, trace_edges, neighbours, tid, line_number, a_xy, b_xy):
    H, W = raster.shape[:2]
    xs = [p[0] for e in trace_edges for p in e["polyline"]]
    ys = [p[1] for e in trace_edges for p in e["polyline"]]
    x0, x1, y0, y1 = min(xs) - CONTEXT, max(xs) + CONTEXT, min(ys) - CONTEXT, max(ys) + CONTEXT
    w, h = max(x1 - x0, MIN_W), max(y1 - y0, MIN_H)   # grow to the minimum around the centre
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(max(0, min(cx - w / 2, W - w))), int(max(0, min(cy - h / 2, H - h)))
    x1, y1 = int(min(W, x0 + w)), int(min(H, y0 + h))
    crop = raster[y0:y1, x0:x1]
    scale = min(1.0, MAX_W / crop.shape[1], MAX_H / crop.shape[0])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else crop.copy()
    crop = cv2.addWeighted(crop, 0.75, np.full_like(crop, 255), 0.25, 0)  # lightened
    off = (x0, y0)
    layer = crop.copy()
    for e in trace_edges:
        _poly(layer, e["polyline"], scale, off, TRACE_BGR, 5)
    crop = cv2.addWeighted(layer, 0.45, crop, 0.55, 0)  # semi-transparent: symbols underneath stay visible
    for e in neighbours:
        _poly(crop, e["polyline"], scale, off, KNOWN_BGR, 1)
        _arrow(crop, e["polyline"], e["flow"] == "forward", scale, off, KNOWN_BGR)
    fs = 1.3  # screen-constant letter size, however much the crop was downscaled
    for label, xy in (("A", a_xy), ("B", b_xy)):
        p = (int((xy[0] - x0) * scale), int((xy[1] - y0) * scale))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
        org = (min(max(p[0] + 8 if label == "A" else p[0] - tw - 8, 2), crop.shape[1] - tw - 2), min(max(p[1] - 8, th + 2), crop.shape[0] - 2))
        cv2.putText(crop, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 7, cv2.LINE_AA)
        cv2.putText(crop, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 200), 3, cv2.LINE_AA)
        cv2.circle(crop, p, 6, (0, 0, 200), -1)
    bar = np.full((34, crop.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, f"{tid}   {line_number or '(no line number)'}",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, crop])


def _tile(img):
    s = min(TILE_W / img.shape[1], TILE_H / img.shape[0])
    t = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    out = np.full((TILE_H, TILE_W, 3), 255, np.uint8)
    out[:t.shape[0], :t.shape[1]] = t
    cv2.rectangle(out, (0, 0), (TILE_W - 1, TILE_H - 1), (160, 160, 160), 1)
    return out


def build(stem, out_dir, indir):
    out_dir, indir = Path(out_dir), Path(indir)
    g = json.loads((out_dir / "graph.json").read_text())
    raster = cv2.imread(str(indir / f"{stem}.jpg"))
    if raster is None:
        raise FileNotFoundError(f"cannot read {indir / f'{stem}.jpg'}")
    pipes = [e for e in g["edges"] if e["kind"] == "pipe"]
    by_trace = defaultdict(list)
    for e in pipes:  # list order = creation order = order along the trace
        by_trace[e["trace_ids"][0]].append(e)
    rdir = out_dir / "flow_review"
    rdir.mkdir(exist_ok=True)
    for old in rdir.glob("*.png"):
        old.unlink()
    items, crops = [], []
    for tid, es in by_trace.items():
        if any(e.get("flow") in ("forward", "reverse") or e.get("flow_source") == "conflict" for e in es):
            continue
        nodes = {n for e in es for n in (e["u"], e["v"])}
        nb = [e for e in pipes if e["trace_ids"][0] != tid and e.get("flow") in ("forward", "reverse")
              and (e["u"] in nodes or e["v"] in nodes)]
        a_xy, b_xy = es[0]["polyline"][0], es[-1]["polyline"][-1]
        lns = [e["line_number"] for e in es if e.get("line_number")]
        ln = max(set(lns), key=lns.count) if lns else ""
        img = render_item(raster, es, nb, tid, ln, a_xy, b_xy)
        path = f"{out_dir}/flow_review/{tid}.png"
        cv2.imwrite(path, img)
        crops.append(img)
        items.append({"id": tid, "edges": [e["id"] for e in es], "a_xy": a_xy, "b_xy": b_xy,
                      "line_number": ln, "crop": path})
    sheets = []
    for i in range(0, len(crops), PER_SHEET):
        tiles = [_tile(c) for c in crops[i:i + PER_SHEET]]
        tiles += [np.full((TILE_H, TILE_W, 3), 255, np.uint8)] * (PER_SHEET - len(tiles))
        sheet = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        sp = f"{out_dir}/flow_review/contact_{i // PER_SHEET + 1:02d}.png"
        cv2.imwrite(sp, sheet)
        sheets.append(sp)
    res = {"sheet": stem, "items": items, "contact_sheets": sheets}
    (out_dir / "flow_candidates.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--indir", required=True)
    a = ap.parse_args(argv)
    try:
        r = build(a.stem, a.dir, a.indir)
    except FileNotFoundError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 1
    print(f"{a.stem}: {len(r['items'])} traces need a flow pick, {len(r['contact_sheets'])} contact sheet(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
