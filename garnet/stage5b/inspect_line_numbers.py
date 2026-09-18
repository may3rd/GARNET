#!/usr/bin/env python
"""Visual inspection of Stage 6 line-number assignment.

Renders one review tile per assigned label so a human can confirm the association is right.

    python -m garnet.stage5b.inspect_line_numbers --stem Test-00001

Why the tile shows more than the chosen path: Stage 6 attaches a label to the path it is written
ALONG, not merely the nearest one — when the closest segment runs perpendicular to the label's
long axis, an orientation-matching segment wins instead. That rule is invisible if you only draw
the winner, and it is the rule most likely to be wrong. So each tile draws:

    GREEN    the path the label was assigned to
    BLUE     the path that was geometrically CLOSER but lost on orientation (if any)
    RED box  the label's own bbox
    MAGENTA  where the label projected onto the assigned path
    YELLOW   where it would have projected onto the closer path

If blue+green look right, the assignment is right. If the label visibly annotates the blue path,
the orientation preference made a mistake.

Outputs (into <stage5b dir>/line_number_review/):
    tile_NNN_<label>.png     one tile per assignment
    contact_sheet.png        all tiles, numbered, for scanning
    inspection.html          clickable gallery + verdict form    <- open in a browser
    assignments.json         machine-readable index of every tile
    verdicts.json            pre-filled stub to record correct/wrong per tile
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("stage6.inspect")

# colours (BGR)
C_ASSIGNED = (0, 200, 0)        # green
C_COMPETING = (230, 120, 0)     # blue
C_LABEL = (0, 0, 235)           # red
C_PROJ = (255, 0, 255)          # magenta
C_PROJ_ALT = (0, 220, 255)      # yellow


def _pt_seg(px: float, py: float, ax: float, ay: float, bx: float, by: float):
    """Point-to-segment: returns (qx, qy, t, distance)."""
    abx, aby = bx - ax, by - ay
    L = abx * abx + aby * aby
    if L <= 0:
        return ax, ay, 0.0, float(np.hypot(px - ax, py - ay))
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / L))
    qx, qy = ax + t * abx, ay + t * aby
    return qx, qy, t, float(np.hypot(px - qx, py - qy))


def label_corners(bbox: dict) -> list[tuple[float, float]]:
    xs = (bbox["x_min"], bbox["x_max"])
    ys = (bbox["y_min"], bbox["y_max"])
    cx = (bbox["x_min"] + bbox["x_max"]) / 2.0
    cy = (bbox["y_min"] + bbox["y_max"]) / 2.0
    return [(cx, cy), (xs[0], ys[0]), (xs[1], ys[0]), (xs[0], ys[1]),
            (xs[1], ys[1]), (cx, ys[0]), (cx, ys[1]), (xs[0], cy), (xs[1], cy)]


def nearest_on_edge(bbox: dict, edge: dict):
    """Nearest point from a label to an edge, mirroring the gate's corner-sampling metric."""
    best = None
    for px, py in label_corners(bbox):
        for si, s in enumerate(edge.get("segments", [])):
            qx, qy, t, d = _pt_seg(px, py, s["x1"], s["y1"], s["x2"], s["y2"])
            if best is None or d < best[3]:
                best = (qx, qy, t, d, si)
    if best is None:
        return None
    return {"projected_xy": [best[0], best[1]], "distance_px": best[3], "segment_index": best[4]}


def label_orientation(bbox: dict) -> str | None:
    w = bbox["x_max"] - bbox["x_min"]
    h = bbox["y_max"] - bbox["y_min"]
    if w >= h * 1.5:
        return "horizontal"
    if h >= w * 1.5:
        return "vertical"
    return None


def segment_orientation(s: dict) -> str:
    return "horizontal" if abs(s["x2"] - s["x1"]) > abs(s["y2"] - s["y1"]) else "vertical"


def crop_region(pts: list[tuple[float, float]], bbox: dict, shape, pad: int):
    xs = [p[0] for p in pts] + [bbox["x_min"], bbox["x_max"]]
    ys = [p[1] for p in pts] + [bbox["y_min"], bbox["y_max"]]
    x0 = max(0, int(min(xs)) - pad)
    y0 = max(0, int(min(ys)) - pad)
    x1 = min(shape[1], int(max(xs)) + pad)
    y1 = min(shape[0], int(max(ys)) + pad)
    return x0, y0, x1, y1


def draw_edge(vis, edge, ox, oy, color, width_halo=7, width_line=2, dashed=False):
    for s in edge.get("segments", []):
        a = (int(s["x1"]) - ox, int(s["y1"]) - oy)
        b = (int(s["x2"]) - ox, int(s["y2"]) - oy)
        if dashed:
            # draw every other 8px so a competitor path is visibly distinct from the winner
            n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
            for k in range(0, n, 16):
                f0, f1 = k / max(n, 1), min(1.0, (k + 8) / max(n, 1))
                p0 = (int(a[0] + (b[0] - a[0]) * f0), int(a[1] + (b[1] - a[1]) * f0))
                p1 = (int(a[0] + (b[0] - a[0]) * f1), int(a[1] + (b[1] - a[1]) * f1))
                cv2.line(vis, p0, p1, color, width_line + 1)
        else:
            cv2.line(vis, a, b, color, width_line)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", default="Test-00001")
    p.add_argument("--indir", default="garnet/tests/input")
    p.add_argument("--dir", default=None, help="stage5b dir (default output/stage5b_<stem>)")
    p.add_argument("--out", default=None, help="output dir (default <dir>/line_number_review)")
    p.add_argument("--pad", type=int, default=110, help="context padding around each tile")
    p.add_argument("--focus-px", type=int, default=650,
                   help="max half-extent around the label. A competitor path can run the length "
                        "of the sheet; without a cap the crop shrinks to unreadable. Paths are "
                        "still drawn across the full crop, just windowed around the label.")
    p.add_argument("--tile-w", type=int, default=900)
    p.add_argument("--tile-h", type=int, default=520)
    p.add_argument("--cols", type=int, default=3, help="columns in the contact sheet")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    indir = Path(args.indir)
    if not indir.is_absolute():
        indir = REPO_ROOT / indir
    sdir = Path(args.dir) if args.dir else Path(f"output/stage5b_{args.stem}")
    if not sdir.is_absolute():
        sdir = REPO_ROOT / sdir
    out_dir = Path(args.out) if args.out else sdir / "line_number_review"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(indir / f"{args.stem}.jpg"))
    if image is None:
        print(f"error: cannot read {indir / f'{args.stem}.jpg'}", file=sys.stderr)
        return 1
    sa = sdir / "stage6_trace_associations.json"
    if not sa.is_file():
        print(f"error: missing {sa}\n  run: python -m garnet.stage5b.run_associations "
              f"--stem {args.stem}", file=sys.stderr)
        return 1
    d = json.loads(sa.read_text())
    edges = {e["trace_id"]: e for e in d["trace_edges"]}
    ln = d["associations"]["line_numbers"]

    # every label that got an association, plus the ones that found nothing
    rows = []
    for item in ln.get("accepted", []):
        rows.append({**item, "_outcome": "accepted"})
    for item in ln.get("rejected", []):
        rows.append({**item, "_outcome": "unattached"})

    if not rows:
        print("no line-number associations to inspect")
        return 0

    # stable order: top-to-bottom, left-to-right on the sheet reads like the drawing.
    # `bbox` is absent entirely for `missing_bbox` rejections and explicitly null for
    # `no_trace_edges`, so neither `.get("bbox", {})` nor `x["bbox"]["y_min"]` is safe.
    def _order(item: dict) -> tuple[float, float]:
        bbox = item.get("bbox") or {}
        return (bbox.get("y_min", 0), bbox.get("x_min", 0))

    rows.sort(key=_order)

    print(f"{args.stem}: {len(ln.get('accepted', []))} attached, "
          f"{len(ln.get('rejected', []))} unattached -> {len(rows)} tiles")

    tiles = []
    index = []
    for i, item in enumerate(rows, start=1):
        bbox = item.get("bbox")
        assigned_id = item.get("trace_id")
        assigned = edges.get(assigned_id)

        # The nearest OTHER path, recomputed here so the tile is independent of whatever
        # ranking the stage did. It is only a "closer, lost on orientation" path when its
        # distance is genuinely below the assigned one's — otherwise it is merely the
        # next-nearest candidate, and labelling it "closer" would mislead a reviewer.
        competing_id, competing, competing_is_closer = None, None, False
        if bbox:
            ranked = []
            for eid, e in edges.items():
                near = nearest_on_edge(bbox, e)
                if near is not None:
                    ranked.append((near["distance_px"], eid, near))
            ranked.sort()
            for dist, eid, near in ranked:
                if eid != assigned_id:
                    competing_id, competing = eid, {**near, "trace_id": eid}
                    ad = item.get("distance_px")
                    competing_is_closer = (
                        ad is not None and competing["distance_px"] < float(ad) - 0.5)
                    break

        ref = assigned if assigned is not None else (edges.get(competing_id) if competing_id else None)
        if ref is None:
            print(f"  [{i}] {item.get('id')}: no path to draw, skipping")
            continue

        pts = []
        for s in ref.get("segments", []):
            pts += [(s["x1"], s["y1"]), (s["x2"], s["y2"])]
        # include the competitor's geometry so the crop shows both paths
        competing_edge = edges.get(competing_id) if competing_id else None
        if assigned is not None and competing_edge is not None:
            for s in competing_edge.get("segments", []):
                pts += [(s["x1"], s["y1"]), (s["x2"], s["y2"])]
        x0, y0, x1, y1 = crop_region(pts, bbox, image.shape, args.pad)

        # Window the crop around the LABEL. A competing path can run the length of the sheet
        # (a branch trace), and fitting the whole thing shrinks the tile until the label and
        # both paths are unreadable. Cap the half-extent so the label stays legible; the paths
        # are still drawn across the full window, just clipped to it.
        cap = args.focus_px
        lcx = (bbox["x_min"] + bbox["x_max"]) // 2
        lcy = (bbox["y_min"] + bbox["y_max"]) // 2
        x0 = max(x0, lcx - cap)
        x1 = min(x1, lcx + cap)
        y0 = max(y0, lcy - cap)
        y1 = min(y1, lcy + cap)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
        if x1 - x0 < 80 or y1 - y0 < 80:
            print(f"  [{i}] {item.get('id')}: crop too small, skipping")
            continue
        crop = image[y0:y1, x0:x1].copy()

        halo = crop.copy()
        if assigned is not None:
            draw_edge(halo, assigned, x0, y0, C_ASSIGNED, width_halo=9)
        if competing is not None:
            draw_edge(halo, edges[competing_id], x0, y0, C_COMPETING, width_halo=9)
        vis = cv2.addWeighted(halo, 0.40, crop, 0.60, 0)
        # solid centreline = assigned, dashed = competitor
        if assigned is not None:
            draw_edge(vis, assigned, x0, y0, C_ASSIGNED, width_line=2)
        if competing is not None:
            draw_edge(vis, edges[competing_id], x0, y0, C_COMPETING, width_line=2, dashed=True)

        cv2.rectangle(vis, (bbox["x_min"] - x0, bbox["y_min"] - y0),
                      (bbox["x_max"] - x0, bbox["y_max"] - y0), C_LABEL, 2)
        proj = item.get("projected_xy")
        if proj:
            cv2.drawMarker(vis, (int(proj[0]) - x0, int(proj[1]) - y0), C_PROJ,
                           cv2.MARKER_CROSS, 30, 3)
        if competing is not None:
            cp = competing["projected_xy"]
            cv2.drawMarker(vis, (int(cp[0]) - x0, int(cp[1]) - y0), C_PROJ_ALT,
                           cv2.MARKER_TILTED_CROSS, 26, 3)

        # tile frame with a header
        tw, th = args.tile_w, args.tile_h
        sc = min(tw / vis.shape[1], th / vis.shape[0])
        vis = cv2.resize(vis, (max(1, int(vis.shape[1] * sc)), max(1, int(vis.shape[0] * sc))))
        head = 104
        tile = np.full((th + head, tw, 3), 248, np.uint8)
        tile[head:head + vis.shape[0], 0:vis.shape[1]] = vis
        cv2.rectangle(tile, (0, 0), (tw - 1, head - 1), (255, 255, 255), -1)
        cv2.rectangle(tile, (0, 0), (tw - 1, head - 1), (200, 200, 200), 1)

        text = str(item.get("text") or item.get("normalized_text") or "").strip()
        shown = text if text else "(no text — detector box only)"
        cv2.putText(tile, f"#{i}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
        cv2.putText(tile, shown[:40], (104, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 2)
        assigned_line = (f"assigned -> {assigned_id}  dist={item.get('distance_px')}px"
                         if assigned is not None else "assigned -> NONE")
        cv2.putText(tile, assigned_line, (104, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 120, 0), 2)
        if competing is not None:
            c_or = segment_orientation(edges[competing_id]["segments"][competing["segment_index"]])
            if competing_is_closer:
                note = (f"closer: {competing_id}  {competing['distance_px']:.0f}px "
                        f"({c_or} vs label {label_orientation(bbox) or '?'}) -- lost on orientation")
                col = (150, 70, 0)
            else:
                note = (f"next-nearest (not closer): {competing_id} "
                        f"{competing['distance_px']:.0f}px ({c_or})")
                col = (110, 110, 110)
            cv2.putText(tile, note[:82], (104, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.54, col, 1)
        else:
            cv2.putText(tile, "no competing path", (104, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                        (120, 120, 120), 1)

        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (shown or "notext"))[:40]
        fname = f"tile_{i:03d}_{safe}.png"
        cv2.imwrite(str(out_dir / fname), tile)
        tiles.append(tile)

        index.append({
            "tile": i, "file": fname,
            "label_id": item.get("id"),
            "source_object_id": item.get("source_object_id"),
            "text": text,
            "outcome": item["_outcome"],
            "assigned_trace_id": assigned_id,
            "assigned_distance_px": item.get("distance_px"),
            "assigned_segment_index": item.get("segment_index"),
            "label_orientation": label_orientation(bbox) if bbox else None,
            "assigned_segment_orientation": (
                segment_orientation(assigned["segments"][item["segment_index"]])
                if assigned and isinstance(item.get("segment_index"), int)
                and item["segment_index"] < len(assigned.get("segments", [])) else None),
            "closer_trace_id": competing_id,
            "competing_is_closer": competing_is_closer,
            "closer_distance_px": round(competing["distance_px"], 2) if competing else None,
            "closer_segment_orientation": (
                segment_orientation(edges[competing_id]["segments"][competing["segment_index"]])
                if competing and competing["segment_index"] < len(
                    edges[competing_id].get("segments", [])) else None),
            "bbox": bbox,
        })
        print(f"  [{i}/{len(rows)}] {shown[:34]:34} -> {assigned_id}"
              + (f"  (closer: {competing_id} {competing['distance_px']:.0f}px)"
                 if competing else ""))

    # contact sheet. Every row can skip (bbox-less rejections carry no path to draw), and
    # np.vstack([]) raises, so report that instead of crashing after the tiles loop.
    if not tiles:
        print(f"no drawable tiles (all {len(rows)} rows skipped) -> nothing written")
        return 0
    cols = max(1, args.cols)
    grid = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.full_like(tiles[0], 248))
        grid.append(np.hstack(row))
    sheet = np.vstack(grid)
    cv2.imwrite(str(out_dir / "contact_sheet.png"), sheet)
    print(f"\n  contact_sheet.png  {sheet.shape[1]}x{sheet.shape[0]}  ({len(tiles)} tiles)")

    (out_dir / "assignments.json").write_text(json.dumps({
        "stem": args.stem,
        "legend": {
            "green_solid": "path the label was assigned to",
            "blue_dashed": "nearest competing path (only 'closer' when competing_is_closer)",
            "red_box": "the label's own bbox",
            "magenta_cross": "label projected onto the assigned path",
            "yellow_cross": "where it would have projected onto the closer path",
        },
        "counts": {"accepted": len(ln.get("accepted", [])),
                   "unattached": len(ln.get("rejected", []))},
        "assignments": index,
    }, indent=2), encoding="utf-8")

    # pre-filled verdict stub
    (out_dir / "verdicts.json").write_text(json.dumps({
        "stem": args.stem,
        "instructions": "Set verdict to 'correct' or 'wrong' per tile. For 'wrong', put the "
                        "trace_id the label SHOULD attach to in correct_trace_id (or null if "
                        "it should attach to nothing). Leave 'unreviewed' to skip.",
        "verdicts": [{"tile": r["tile"], "label_id": r["label_id"], "text": r["text"],
                      "assigned_trace_id": r["assigned_trace_id"],
                      "verdict": "unreviewed", "correct_trace_id": None, "note": ""}
                     for r in index],
    }, indent=2), encoding="utf-8")

    # ---- clickable gallery ----
    cards = []
    for r in index:
        if r["closer_trace_id"] and r.get("competing_is_closer"):
            closer = (f"<div class='closer'>closer path "
                      f"<b>{html.escape(str(r['closer_trace_id']))}</b> at "
                      f"{r['closer_distance_px']}px "
                      f"({html.escape(str(r['closer_segment_orientation']))}) — lost on orientation"
                      f"</div>")
        elif r["closer_trace_id"]:
            closer = (f"<div class='closer none'>next-nearest (not closer): "
                      f"{html.escape(str(r['closer_trace_id']))} at "
                      f"{r['closer_distance_px']}px</div>")
        else:
            closer = "<div class='closer none'>no competing path</div>"
        txt = html.escape(r["text"] or "(no text)")
        cards.append(f"""
    <figure class="card" data-tile="{r['tile']}">
      <img src="{html.escape(r['file'])}" alt="tile {r['tile']}" loading="lazy">
      <figcaption>
        <div class="row"><span class="num">#{r['tile']}</span>
          <span class="txt">{txt}</span></div>
        <div class="meta">assigned <b>{html.escape(str(r['assigned_trace_id']))}</b>
          · {r['assigned_distance_px']}px
          · label {html.escape(str(r['label_orientation']))}
          vs segment {html.escape(str(r['assigned_segment_orientation']))}</div>
        {closer}
        <div class="btns">
          <button data-v="correct">correct</button>
          <button data-v="wrong">wrong</button>
          <input placeholder="correct trace_id (if wrong)" class="ft">
          <input placeholder="note" class="fn">
        </div>
        <div class="state">unreviewed</div>
      </figcaption>
    </figure>""")

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>line-number assignment review — {html.escape(args.stem)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
         margin: 0; padding: 18px 20px 60px; background: var(--card, #fff);
         color: var(--foreground, #111); }}
  h1 {{ font-size: 17px; margin: 0 0 4px; }}
  .sub {{ color: var(--muted-foreground, #666); margin-bottom: 14px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin: 0 0 18px; font-size: 13px; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
  .sw {{ width: 14px; height: 14px; border-radius: 3px; display: inline-block; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }}
  .card {{ margin: 0; border: 1px solid var(--border, #ddd); border-radius: 10px; overflow: hidden;
          background: var(--card, #fff); }}
  .card img {{ width: 100%; display: block; background: #f4f4f4; }}
  figcaption {{ padding: 9px 11px 12px; }}
  .row {{ display: flex; gap: 8px; align-items: baseline; }}
  .num {{ font-weight: 700; }}
  .txt {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px;
          word-break: break-all; }}
  .meta {{ color: var(--muted-foreground, #666); font-size: 12px; margin-top: 3px; }}
  .closer {{ font-size: 12px; margin-top: 4px; color: #a35a00; }}
  .closer.none {{ color: var(--muted-foreground, #888); }}
  .btns {{ display: flex; gap: 6px; margin-top: 9px; flex-wrap: wrap; }}
  button {{ font: inherit; font-size: 12.5px; padding: 3px 11px; border-radius: 6px;
            border: 1px solid var(--border, #ccc); background: transparent; cursor: pointer;
            color: inherit; }}
  button:hover {{ border-color: #888; }}
  button.on[data-v="correct"] {{ background: #1a7f37; border-color: #1a7f37; color: #fff; }}
  button.on[data-v="wrong"] {{ background: #b62324; border-color: #b62324; color: #fff; }}
  input {{ font: inherit; font-size: 12px; padding: 3px 7px; border-radius: 6px;
           border: 1px solid var(--border, #ccc); background: transparent; color: inherit;
           min-width: 0; flex: 1 1 120px; }}
  .state {{ font-size: 11.5px; margin-top: 6px; color: var(--muted-foreground, #888); }}
  .bar {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 10px; align-items: center;
          padding: 9px 0 11px; background: var(--card, #fff); border-bottom: 1px solid var(--border, #eee);
          margin-bottom: 14px; flex-wrap: wrap; }}
  .bar b {{ font-variant-numeric: tabular-nums; }}
  .bar button {{ padding: 4px 12px; }}
</style></head>
<body>
  <h1>Line-number assignment review</h1>
  <div class="sub">{html.escape(args.stem)} — {len(index)} tiles.
    Confirm each label is attached to the pipe it is actually written along.</div>

  <div class="legend">
    <span><i class="sw" style="background:#00c800"></i>green solid = assigned path</span>
    <span><i class="sw" style="background:#e67800"></i>blue dashed = closer path (lost on orientation)</span>
    <span><i class="sw" style="background:#eb0000"></i>red box = the label</span>
    <span><i class="sw" style="background:#ff00ff"></i>magenta = projection onto assigned</span>
    <span><i class="sw" style="background:#ffdc00"></i>yellow = projection onto closer</span>
  </div>

  <div class="bar">
    <span>reviewed <b id="n">0</b> / {len(index)}</span>
    <span>wrong <b id="w">0</b></span>
    <button id="exp">download verdicts.json</button>
    <button id="clr">clear all</button>
  </div>

  <div class="grid">{''.join(cards)}
  </div>

<script>
const KEY = 'lnreview:{html.escape(args.stem)}';
const state = JSON.parse(localStorage.getItem(KEY) || '{{}}');
const cards = [...document.querySelectorAll('.card')];

function persist() {{ localStorage.setItem(KEY, JSON.stringify(state)); }}

function render() {{
  let n = 0, w = 0;
  cards.forEach(c => {{
    const t = c.dataset.tile, s = state[t] || {{}},
          st = c.querySelector('.state');
    c.querySelectorAll('button[data-v]').forEach(b =>
      b.classList.toggle('on', s.verdict === b.dataset.v));
    if (s.verdict === 'correct' || s.verdict === 'wrong') {{
      n++; if (s.verdict === 'wrong') w++;
      st.textContent = s.verdict + (s.correct_trace_id ? ' → ' + s.correct_trace_id : '')
                     + (s.note ? ' · ' + s.note : '');
      st.style.color = s.verdict === 'wrong' ? '#b62324' : '#1a7f37';
    }} else {{
      st.textContent = 'unreviewed'; st.style.color = '';
    }}
    const ft = c.querySelector('.ft'), fn = c.querySelector('.fn');
    if (document.activeElement !== ft) ft.value = s.correct_trace_id || '';
    if (document.activeElement !== fn) fn.value = s.note || '';
  }});
  document.getElementById('n').textContent = n;
  document.getElementById('w').textContent = w;
}}

cards.forEach(c => {{
  const t = c.dataset.tile;
  c.querySelectorAll('button[data-v]').forEach(b => b.onclick = () => {{
    state[t] = state[t] || {{}};
    state[t].verdict = state[t].verdict === b.dataset.v ? 'unreviewed' : b.dataset.v;
    persist(); render();
  }});
  c.querySelector('.ft').oninput = e => {{
    state[t] = state[t] || {{}}; state[t].correct_trace_id = e.target.value.trim() || null;
    persist(); render();
  }};
  c.querySelector('.fn').oninput = e => {{
    state[t] = state[t] || {{}}; state[t].note = e.target.value.trim();
    persist(); render();
  }};
}});

document.getElementById('clr').onclick = () => {{
  if (!confirm('Clear all verdicts for this sheet?')) return;
  for (const k in state) delete state[k];
  persist(); render();
}};

document.getElementById('exp').onclick = () => {{
  const out = cards.map(c => {{
    const t = c.dataset.tile, s = state[t] || {{}};
    return {{ tile: Number(t),
             label_id: null,
             assigned_trace_id: null,
             verdict: s.verdict || 'unreviewed',
             correct_trace_id: s.correct_trace_id || null,
             note: s.note || '' }};
  }});
  // fill ids from the embedded index so the export is self-contained
  const IDX = {json.dumps([{"tile": r["tile"], "label_id": r["label_id"],
                            "assigned_trace_id": r["assigned_trace_id"]} for r in index])};
  const byTile = Object.fromEntries(IDX.map(r => [r.tile, r]));
  out.forEach(o => {{ const m = byTile[o.tile]; if (m) {{ o.label_id = m.label_id;
                       o.assigned_trace_id = m.assigned_trace_id; }} }});
  const blob = new Blob([JSON.stringify({{ stem: {json.dumps(args.stem)}, verdicts: out }}, null, 2)],
                        {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'verdicts.json';
  a.click();
  URL.revokeObjectURL(a.href);
}};

render();
</script>
</body></html>"""
    (out_dir / "inspection.html").write_text(page, encoding="utf-8")

    print(f"  inspection.html    clickable gallery + verdict form")
    print(f"  assignments.json   machine-readable index")
    print(f"  verdicts.json      pre-filled stub")
    print(f"\nOK -> {out_dir}")
    print(f"open: {out_dir / 'inspection.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
