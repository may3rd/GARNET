"""Line-number candidate review crops: per label, nearby traces drawn in colour+letter for a vision agent."""
import argparse
import json
import os

import cv2
import numpy as np

MAX_DIST, MAX_CAND, PAD = 300, 5, 80
MINWH, MAXWH = (500, 400), (1400, 1000)
COLORS = [(0, 0, 230), (230, 80, 0), (0, 160, 0), (200, 0, 200), (0, 140, 255)]  # BGR: red blue green magenta orange
TILE_W, TILE_H, CAP_H = 900, 600, 56


def _dense(poly, step=4.0):
    """Polyline -> points every ~step px. ponytail: sampling, not exact segment-box distance (error <= step/2)."""
    p = np.asarray(poly, float)
    if len(p) < 2:
        return p.reshape(-1, 2)
    out = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(np.hypot(*(b - a)) / step), 1)
        out.append(a + (b - a) * (np.arange(n)[:, None] / n))
    out.append(p[-1:])
    return np.vstack(out)


def _box_dist(pts, b):
    dx = np.maximum(np.maximum(b[0] - pts[:, 0], pts[:, 0] - b[2]), 0)
    dy = np.maximum(np.maximum(b[1] - pts[:, 1], pts[:, 1] - b[3]), 0)
    return np.hypot(dx, dy)


def find_candidates(bbox, traces, pick_tid=None):
    """traces {tid: polyline} -> (candidates list, geometric pick xy or None)."""
    c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    rows = []
    for tid, poly in traces.items():
        pts = _dense(poly)
        if not len(pts):
            continue
        d = _box_dist(pts, bbox)
        if d.min() > MAX_DIST and tid != pick_tid:
            continue
        xy = pts[np.argmin(np.hypot(*(pts - c).T))]
        rows.append((float(d.min()), tid, [round(float(xy[0]), 1), round(float(xy[1]), 1)]))
    rows.sort(key=lambda r: r[0])
    keep = rows[:MAX_CAND]
    if pick_tid and pick_tid not in [r[1] for r in keep]:
        pick = [r for r in rows if r[1] == pick_tid]
        keep = keep[:MAX_CAND - 1] + pick
    return [{"letter": chr(65 + i), "trace_id": t, "xy": xy, "dist_px": round(d, 1)}
            for i, (d, t, xy) in enumerate(keep)]


def _window(bbox, cands, shape):
    """Crop window x1,y1,x2,y2; drops farthest candidates (returns kept) until it fits MAXWH."""
    cands = list(cands)
    while True:
        xs = [bbox[0], bbox[2]] + [c["xy"][0] for c in cands]
        ys = [bbox[1], bbox[3]] + [c["xy"][1] for c in cands]
        x1, x2, y1, y2 = min(xs) - PAD, max(xs) + PAD, min(ys) - PAD, max(ys) + PAD
        if (x2 - x1 <= MAXWH[0] and y2 - y1 <= MAXWH[1]) or not cands:
            break
        cands.pop()  # candidates are nearest-first
    w, h = min(max(x2 - x1, MINWH[0]), MAXWH[0]), min(max(y2 - y1, MINWH[1]), MAXWH[1])
    w, h = min(w, shape[1]), min(h, shape[0])
    x1 = int(round(min(max((x1 + x2) / 2 - w / 2, 0), shape[1] - w)))
    y1 = int(round(min(max((y1 + y2) / 2 - h / 2, 0), shape[0] - h)))
    return (x1, y1, x1 + int(w), y1 + int(h)), cands


def _txt(s):
    return s.replace("\u00bd", "1/2").encode("ascii", "replace").decode()  # ponytail: Hershey font is ascii-only


def render_crop(img, lab, traces, out_path):
    bbox, cands = lab["bbox"], lab["candidates"]
    (x1, y1, x2, y2), shown = _window(bbox, cands, img.shape)
    crop = img[y1:y2, x1:x2].copy()
    crop = cv2.addWeighted(crop, 0.6, np.full_like(crop, 255), 0.4, 0)  # lighten
    layer = crop.copy()
    off = np.array([x1, y1])
    for c in shown:
        col = COLORS[ord(c["letter"]) - 65]
        pts = np.round(np.asarray(traces[c["trace_id"]], float) - off).astype(np.int32)
        cv2.polylines(layer, [pts.reshape(-1, 1, 2)], False, col, 5, cv2.LINE_AA)  # cv2 clips
    crop = cv2.addWeighted(layer, 0.5, crop, 0.5, 0)
    # original ink back on top where it is dark, so arrows/text stay crisp
    ink = img[y1:y2, x1:x2].min(axis=2) < 110
    crop[ink] = img[y1:y2, x1:x2][ink]
    cv2.rectangle(crop, (int(bbox[0] - x1), int(bbox[1] - y1)), (int(bbox[2] - x1), int(bbox[3] - y1)), (0, 0, 0), 2)
    for c in shown:
        col = COLORS[ord(c["letter"]) - 65]
        p = (int(c["xy"][0] - x1), int(c["xy"][1] - y1))
        cv2.circle(crop, p, 15, col, -1, cv2.LINE_AA)
        cv2.circle(crop, p, 15, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(crop, c["letter"], (p[0] - 8, p[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cap = np.full((CAP_H, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(cap, _txt(lab["id"]), (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(cap, _txt(lab["text"]), (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    out = np.vstack([cap, crop])
    cv2.imwrite(out_path, out)
    return out


def contact_sheets(items, out_dir, per=6):
    """items [(crop_img, label_dict)] -> sheet paths; 2 cols x 3 rows, id + letters under each tile."""
    paths = []
    for n in range(0, len(items), per):
        sheet = np.full((3 * (TILE_H + 70), 2 * TILE_W, 3), 255, np.uint8)
        for k, (im, lab) in enumerate(items[n:n + per]):
            s = min((TILE_W - 10) / im.shape[1], TILE_H / im.shape[0])
            im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            ox, oy = (k % 2) * TILE_W + 5, (k // 2) * (TILE_H + 70)
            sheet[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
            cv2.rectangle(sheet, (ox, oy), (ox + im.shape[1], oy + im.shape[0]), (120, 120, 120), 1)
            cv2.putText(sheet, _txt(lab["id"]), (ox, oy + TILE_H + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            for j, c in enumerate(lab["candidates"]):
                t = f'{c["letter"]}={c["trace_id"]}' + (" *" if lab["geometric_pick"] and lab["geometric_pick"]["trace_id"] == c["trace_id"] else "")
                cv2.putText(sheet, _txt(t), (ox + (j % 3) * 290, oy + TILE_H + 44 + (j // 3) * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS[j], 1, cv2.LINE_AA)
        p = os.path.join(out_dir, f"contact_{n // per + 1:02d}.png")
        cv2.imwrite(p, sheet)
        paths.append(p)
    return paths


def load_labels(indir, stem):
    """Ids as Stage 6 assigns them: line_number_NNNN in upload order, skipping empty-text rows (host.merge_line_numbers)."""
    objs = json.load(open(os.path.join(indir, f"{stem}_line_numbers.json")))["objects"]
    labs = []
    for o in objs:
        text = str(o.get("Text") or "").strip()
        if not text:
            continue
        labs.append({"id": f"line_number_{len(labs) + 1:04d}", "text": text,
                     "bbox": [int(o["Left"]), int(o["Top"]), int(o["Left"] + o["Width"]), int(o["Top"] + o["Height"])]})
    return labs


def run(stem, dir_, indir):
    s6 = json.load(open(os.path.join(dir_, "stage6_trace_associations.json")))
    traces = {e["trace_id"]: e["polyline"] for e in s6["trace_edges"] if e.get("polyline")}
    picks = {a["id"]: (e["trace_id"], a.get("projected_xy"))
             for e in s6["trace_edges"] for a in e.get("attachments", {}).get("line_numbers", [])}
    img = cv2.imread(os.path.join(indir, f"{stem}.jpg"))
    rev = os.path.join(dir_, "ln_review")
    os.makedirs(rev, exist_ok=True)
    labels, items = [], []
    for lab in load_labels(indir, stem):
        ptid, pxy = picks.get(lab["id"], (None, None))
        lab["geometric_pick"] = {"trace_id": ptid, "xy": [round(float(v), 1) for v in pxy]} if ptid and pxy else None
        lab["candidates"] = find_candidates(lab["bbox"], traces, ptid if ptid in traces else None)
        lab["crop"] = os.path.join(rev, lab["id"] + ".png")
        items.append((render_crop(img, lab, traces, lab["crop"]), lab))
        labels.append(lab)
    out = {"sheet": stem, "labels": [{k: l[k] for k in ("id", "text", "bbox", "geometric_pick", "candidates", "crop")} for l in labels],
           "contact_sheets": contact_sheets(items, rev)}
    json.dump(out, open(os.path.join(dir_, "ln_candidates.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--indir", required=True)
    a = ap.parse_args()
    r = run(a.stem, a.dir, a.indir)
    print(a.stem, len(r["labels"]), "labels,", sum(len(l["candidates"]) > 1 for l in r["labels"]), "with >1 candidate")
