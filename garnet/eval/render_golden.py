"""Render one review image per golden connection (conn_NN.png: ends + path along the pipe + tracer path),
plus index.html and review.json.
python3 -m garnet.eval.render_golden --stem S [--golden F] [--graph G] [--image J] [--out DIR]"""
import argparse, html, json, os

import cv2
import numpy as np

from garnet.eval.score import _key, score
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


def tracer_path(graph, short, a, b):
    """Edges of the shortest graph path a -> b through tee/end nodes, or None.
    # ponytail: re-walks the graph (score.py keeps its BFS private); verdict itself comes from score()."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    adj = {}
    for e in graph["edges"]:
        if e.get("kind") in ("pipe", "cross_sheet_continues"):
            adj.setdefault(e["u"], []).append((e["v"], e))
            adj.setdefault(e["v"], []).append((e["u"], e))
    starts = [i for i, _ in resolve(graph, short, a)]
    seen = {i: [] for i in starts}
    q = list(starts)
    while q:
        u = q.pop(0)
        if u not in starts and nodes.get(u, {}).get("kind") not in ("tee", "end"):
            if _key(u) == f"{short}/{b}":
                return seen[u]
            continue
        for v, e in adj.get(u, []):
            if v not in seen:
                seen[v] = seen[u] + [e]
                q.append(v)
    return None


def verdicts(golden, graph):
    """-> {(a, b, line_core): (found, line_ok)} from score.score()'s own misses."""
    short = short_sheet(golden["sheet"])
    m = {x["golden"] for x in score(graph, [golden])["misses"] if x["metric"] in ("connection_recall", "line_assignment")}
    out = {}
    for c in golden.get("connections", []):
        a, b, core = f"{short}/{c['a']}", f"{short}/{c['b']}", c.get("line_core", "")
        found = f"{a} -- {b}" not in m
        out[(c["a"], c["b"], core)] = (found, found and f"{a} -- {b} core {core}" not in m)
    return out


def render(golden, graph, image, out):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):  # stale pair_/path_/conn_ images from earlier runs
        if f.endswith(".png") and f.split("_")[0] in ("pair", "path", "conn"):
            os.remove(os.path.join(out, f))
    short = short_sheet(golden["sheet"])
    vd = verdicts(golden, graph)
    recs = []
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
        found, line_ok = vd[(c["a"], c["b"], c.get("line_core", ""))]
        tp = tracer_path(graph, short, c["a"], c["b"]) if found else None
        cap = (f"#{n:02d}  {c['a']} -> {c['b']}  {c.get('line_core', '')}  {c.get('confidence', '')}")
        verdict = f"tracer: {'found' if found else 'MISSED'}, line: {'ok' if line_ok else 'WRONG'}"
        if not path:
            notes.append("NO PATH TRACED")
        pts = [p for e in ends for _, p in e] + [tuple(p) for p in path]
        tpl = [tuple(p) for e in (tp or []) for p in e.get("polyline", [])]
        pts += tpl
        tile, tf = crop(image, pts) if pts else (np.full((MIN_H, MIN_W, 3), 255, np.uint8), None)
        if tf:
            if path:
                ov = tile.copy()
                cv2.polylines(ov, [np.array([tf(p) for p in path], np.int32)], False, (0, 140, 255), 6)
                tile = cv2.addWeighted(ov, 0.45, tile, 0.55, 0)
                for p in path:
                    cv2.circle(tile, tf(p), 4, (0, 90, 200), -1)
            for e in tp or []:
                if e.get("polyline"):
                    cv2.polylines(tile, [np.array([tf(p) for p in e["polyline"]], np.int32)], False, (200, 0, 200), 2)
            for e, lab, col in ((ends[0], "A: " + c["a"], RED), (ends[1], "B: " + c["b"], BLUE)):
                for _, p in e:
                    ring(tile, tf(p), lab, col)
        img = f"conn_{n:02d}.png"
        cv2.imwrite(os.path.join(out, img), with_caption(tile, cap + "  " + verdict, "; ".join(notes)))
        recs.append({"nn": f"{n:02d}", "a": c["a"], "b": c["b"], "line_core": c.get("line_core", ""),
                     "confidence": c.get("confidence", ""), "why": c.get("why", c.get("note", "")), "image": img,
                     "tracer_found": found, "tracer_line_ok": line_ok, "has_path": bool(path), "_cap": cap,
                     "_v": verdict, "_n": "; ".join(notes)})
    tr = "".join(f"<tr><td style='width:22%'>{html.escape(r['_cap'])}<br><b>{html.escape(r['_v'])}</b><br>"
                 f"<small>{html.escape(r['_n'])}<br>{html.escape(r['why'])}</small></td>"
                 f"<td><img src='{r['image']}' style='width:100%'></td></tr>" for r in recs)
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(f"<!doctype html><meta charset=utf-8><title>{golden['sheet']}</title>"
                f"<table border=1 style='width:100%;table-layout:fixed;border-collapse:collapse'>{tr}</table>")
    for r in recs:
        for k in ("_cap", "_v", "_n"):
            del r[k]
    json.dump({"sheet": golden["sheet"], "connections": recs}, open(os.path.join(out, "review.json"), "w"),
              indent=1, ensure_ascii=False)
    return recs


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
    r = render(golden, graph, image, out)
    f = sum(x["tracer_found"] for x in r)
    ok = sum(x["tracer_line_ok"] for x in r)
    print(f"{len(r)} conns -> {out}: found {f} missed {len(r) - f}; line ok {ok} wrong {f - ok}")


if __name__ == "__main__":
    main()
