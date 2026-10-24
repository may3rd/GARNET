"""Render two review images per golden connection: pair_NN.png (end points) and path_NN.png (drawn path).
python3 -m garnet.eval.render_golden --stem S [--golden F] [--graph G] [--image J] [--out DIR]"""
import argparse, html, json, os

import cv2
import numpy as np

from garnet.eval.score import _key
from garnet.graph.contracts import short_sheet

PAD, MIN_W, MIN_H, CAP = 150, 700, 500, 1800


def resolve(graph, short, gid):
    """-> list of (node_id, (x, y)) matching golden id the way score.py does (key == _key(node id))."""
    k = f"{short}/{gid}"
    return [(n["id"], tuple(n["xy_px"])) for n in graph["nodes"] if n.get("xy_px") and _key(n["id"]) == k]


def pick(cands, ref):
    if ref is None or len(cands) < 2:
        return cands
    return [min(cands, key=lambda c: (c[1][0] - ref[0]) ** 2 + (c[1][1] - ref[1]) ** 2)]


def crop(img, pts):
    """Lightened crop around pts (pad, min size, cap) -> (tile, to_tile fn)."""
    H, W = img.shape[:2]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs) - PAD, max(xs) + PAD, min(ys) - PAD, max(ys) + PAD
    if x1 - x0 < MIN_W: x0, x1 = (x0 + x1) / 2 - MIN_W / 2, (x0 + x1) / 2 + MIN_W / 2
    if y1 - y0 < MIN_H: y0, y1 = (y0 + y1) / 2 - MIN_H / 2, (y0 + y1) / 2 + MIN_H / 2
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    tile = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        tile[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]  # out-of-sheet area stays white
    tile = cv2.addWeighted(tile, 0.6, np.full_like(tile, 255), 0.4, 0)
    s = min(1.0, CAP / max(tile.shape[:2]))
    if s < 1:
        tile = cv2.resize(tile, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return tile, (lambda p: (int((p[0] - x0) * s), int((p[1] - y0) * s)))


def ring(tile, xy, label, color):
    r = 36
    cv2.circle(tile, xy, r, color, 4)
    cv2.circle(tile, xy, 4, color, -1)
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    tx = min(max(xy[0] + r + 6, 4), max(tile.shape[1] - tw - 4, 4))
    ty = min(max(xy[1] - r - 6, 24), tile.shape[0] - 6)
    cv2.putText(tile, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5)
    cv2.putText(tile, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def dashed(tile, p, q, color, dash=14):
    d = max(np.hypot(q[0] - p[0], q[1] - p[1]), 1)
    for t in np.arange(0, d, dash * 2):
        a, b = min(t + dash, d), t
        cv2.line(tile, (int(p[0] + (q[0] - p[0]) * b / d), int(p[1] + (q[1] - p[1]) * b / d)),
                 (int(p[0] + (q[0] - p[0]) * a / d), int(p[1] + (q[1] - p[1]) * a / d)), color, 2)


def with_caption(tile, text, note=""):
    w = max(tile.shape[1], 700)
    strip = np.full((64 if note else 40, w, 3), 245, np.uint8)
    cv2.putText(strip, text[:int(w / 11)], (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    if note:
        cv2.putText(strip, note[:int(w / 9)], (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
    body = np.full((tile.shape[0], w, 3), 255, np.uint8)
    body[:, :tile.shape[1]] = tile
    return np.vstack([strip, body])


RED, BLUE, GREEN = (0, 0, 255), (255, 80, 0), (0, 160, 0)


def render(golden, graph, image, out):
    os.makedirs(out, exist_ok=True)
    short = short_sheet(golden["sheet"])
    rows = []
    for n, c in enumerate(golden.get("connections", []), 1):
        path = c.get("path_px") or []
        ends, notes = [], []
        for gid, ref in ((c["a"], path[0] if path else None), (c["b"], path[-1] if path else None)):
            cands = pick(resolve(graph, short, gid), ref)
            if not cands:
                notes.append(f"UNRESOLVED {gid}")
            elif len(cands) > 1:
                notes.append(f"{gid}: {len(cands)} candidates drawn")
            ends.append(cands)
        cap = f"#{n:02d}  {c['a']} -> {c['b']}  {c.get('line_core', '')}  {c.get('confidence', '')}"
        for kind in ("pair", "path"):
            pts = [p for e in ends for _, p in e] + ([tuple(p) for p in path] if kind == "path" else [])
            note = "; ".join(notes + (["NO PATH YET"] if kind == "path" and not path else []))
            if not pts:  # nothing to place: blank tile
                tile, tf = np.full((MIN_H, MIN_W, 3), 255, np.uint8), None
            else:
                tile, tf = crop(image, pts)
            if tf:
                if kind == "pair":
                    for _, pa in ends[0]:
                        for _, pb in ends[1]:
                            dashed(tile, tf(pa), tf(pb), (80, 80, 80))
                elif path:
                    ov = tile.copy()
                    cv2.polylines(ov, [np.array([tf(p) for p in path], np.int32)], False, (0, 140, 255), 6)
                    tile = cv2.addWeighted(ov, 0.45, tile, 0.55, 0)
                    for p in path:
                        cv2.circle(tile, tf(p), 4, (0, 90, 200), -1)
                for e, lab, col in ((ends[0], "A: " + c["a"], RED), (ends[1], "B: " + c["b"], BLUE)):
                    for _, p in e:
                        ring(tile, tf(p), lab, col)
            cv2.imwrite(os.path.join(out, f"{kind}_{n:02d}.png"), with_caption(tile, cap, note))
        rows.append((cap, "; ".join(notes), n))
    tr = "".join(f"<tr><td style='width:20%'>{html.escape(cap)}<br><small>{html.escape(nt)}</small></td>"
                 f"<td><img src='pair_{n:02d}.png' style='width:100%'></td>"
                 f"<td><img src='path_{n:02d}.png' style='width:100%'></td></tr>" for cap, nt, n in rows)
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(f"<!doctype html><meta charset=utf-8><title>{golden['sheet']}</title>"
                f"<table border=1 style='width:100%;table-layout:fixed;border-collapse:collapse'>{tr}</table>")
    return len(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--golden")
    ap.add_argument("--graph")
    ap.add_argument("--image")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    s = a.stem
    golden = json.load(open(a.golden or f"garnet/tests/golden/{s}.draft.json"))
    graph = json.load(open(a.graph or f"output/run1/out/{s}/graph.json"))
    image = cv2.imread(a.image or f"output/run1/sheets/{s}/{s}.jpg")
    if image is None:
        raise SystemExit("cannot read image")
    out = a.out or f"garnet/tests/golden/review/{short_sheet(s)}"
    print(render(golden, graph, image, out), "connections ->", out)


if __name__ == "__main__":
    main()
