"""Trace pipes straight out of a vector PDF instead of guessing them from pixels.

A plotted P&ID PDF is not a picture: the sheet is thousands of exact line strokes. On
14780-8120-25-25-0002 the page carries 41,641 line items and zero images, and after one
rotation plus a constant registration offset 95% of the delivery export's line-path points sit
within 3 px of real vector geometry (mean 0.8 px). Pixel tracing cannot beat that, because
nothing is being inferred.

What this module does, in order:
  1. `is_vector_pdf`   - does this sheet carry geometry at all, or is it a scan?
  2. `extract_segments` - page strokes -> straight segments in sheet-raster pixels, with the
                          page rotation applied the way the renderer applies it.
  3. `register`        - find the small constant offset between PDF user space and the raster
                          the rest of the pipeline uses (a plotted sheet is rarely pixel-exact).
  4. `build_network`   - snap segment ends, split at crossings, and walk the resulting graph
                          into polylines: one polyline per connected run of pipe.
  5. `to_trace_results` - emit the same shape `stage5b_trace_results.json` has, so scoring,
                          stage 5c/6 and the frontend keep working unchanged.

Usage:
    python -m garnet.path_tracer.vector_trace --pdf sheet.pdf --run-dir output/<job> --apply
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# A stroke shorter than this is a symbol fragment, a text glyph edge or a tick, not pipe.
MIN_SEGMENT_PX = 8
# Segment ends closer than this are the same node (a plotting gap or a snap tolerance).
SNAP_PX = 4.0


def is_vector_pdf(pdf_path: str | Path, page_no: int = 1, min_lines: int = 200) -> dict:
    """Decide the routing question: vector fast-path or raster tracing."""
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    page = doc[page_no - 1]
    drawings = page.get_drawings()
    kinds: dict[str, int] = {}
    for d in drawings:
        for item in d["items"]:
            kinds[item[0]] = kinds.get(item[0], 0) + 1
    images = len(page.get_images(full=True))
    stats = {
        "drawings": len(drawings),
        "lines": kinds.get("l", 0),
        "curves": kinds.get("c", 0),
        "rects": kinds.get("re", 0),
        "images": images,
        "rotation": int(page.rotation),
        "page_pt": [round(page.rect.width, 1), round(page.rect.height, 1)],
    }
    stats["vector"] = stats["lines"] >= min_lines and images <= 1
    return stats


def _rotate_to_page(x: float, y: float, rotation: int, unrot_w: float, unrot_h: float) -> tuple[float, float]:
    """Map unrotated PDF user-space coords into the rendered (rotated) page frame.

    `page.get_drawings()` reports geometry before the page's /Rotate is applied, so a 270-degree
    sheet comes back portrait while the raster the pipeline uses is landscape. This is the same
    90-degree turn a renderer performs, nothing more.
    """
    if rotation == 90:
        return y, unrot_w - x
    if rotation == 180:
        return unrot_w - x, unrot_h - y
    if rotation == 270:
        return y, unrot_w - x
    return x, y


def extract_segments(
    pdf_path: str | Path,
    page_no: int = 1,
    dpi: float = 300.0,
    min_length_px: float = MIN_SEGMENT_PX,
    drop_text_layers: bool = True,
    drop_fill_strokes: bool = True,
) -> tuple[list[tuple[float, float, float, float]], dict]:
    """All straight strokes on the page, in raster pixels (before registration).

    Text layers and `fs` (fill+stroke) groups carry letter outlines and symbol fills. They are
    linework too, which is why leaving them in floods the network with thousands of glyph
    chains: on sheet 0002 that is the difference between a pipe network and a drawing dump.
    """
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    page = doc[page_no - 1]
    rotation = int(page.rotation)
    # get_drawings() coordinates live in the *media box* frame -- 842x1191 on a 270-degree A3
    # sheet whose page.rect is 1191x842. Using the rotated page size here shifts every stroke by
    # half a page, which is how the first version of this module produced y in 1466..4954.
    unrot_w = page.mediabox.width
    unrot_h = page.mediabox.height
    scale = dpi / 72.0

    segments: list[tuple[float, float, float, float]] = []

    def push(p1, p2) -> None:
        x1, y1 = _rotate_to_page(p1[0], p1[1], rotation, unrot_w, unrot_h)
        x2, y2 = _rotate_to_page(p2[0], p2[1], rotation, unrot_w, unrot_h)
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        if math.hypot(x2 - x1, y2 - y1) >= min_length_px:
            segments.append((x1, y1, x2, y2))

    skipped_groups = 0
    for d in page.get_drawings():
        layer = str(d.get("layer") or "").lower()
        if (drop_text_layers and "text" in layer) or (drop_fill_strokes and d.get("type") == "fs"):
            skipped_groups += 1
            continue
        for item in d["items"]:
            kind = item[0]
            if kind == "l":
                push((item[1].x, item[1].y), (item[2].x, item[2].y))
            elif kind == "re":
                r = item[1]
                corners = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                for a, b in zip(corners, corners[1:] + corners[:1]):
                    push(a, b)
            elif kind == "qu":
                q = item[1]
                pts = [q.ul, q.ur, q.lr, q.ll]
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    push((a.x, a.y), (b.x, b.y))
            # 'c' (Bezier) strokes are symbols and arrowheads; they are not pipe centre lines.
    return segments, {"page": page_no, "rotation": rotation, "segments": len(segments),
                      "skipped_groups": skipped_groups}


def register(
    segments: list[tuple[float, float, float, float]],
    mask: np.ndarray,
    search_px: int = 40,
    coarse_step: int = 4,
    sample_px: float = 6.0,
) -> tuple[int, int, float]:
    """Find the constant (dx, dy) that lands the vector strokes on the sheet raster's ink.

    A plotted PDF and the raster the pipeline rendered from it can differ by a translation
    (crop, plot margin, rounding). One global offset is enough: it scored 0.950 of export path
    points within 3 px on sheet 0002, mean distance 0.8 px.
    """
    h_img, w_img = mask.shape[:2]
    # The argument is either a grayscale render (dark ink on light paper) or an already binary
    # mask where ink is 255. Getting this polarity wrong makes the distance transform zero
    # everywhere, every offset then scores 1.0, and the search returns a meaningless translation.
    ink = (mask < 200) if float(mask.mean()) > 127 else (mask > 0)
    dist = cv2.distanceTransform(255 - (ink.astype(np.uint8) * 255), cv2.DIST_L2, 5)

    points: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in segments:
        n = max(1, int(math.hypot(x2 - x1, y2 - y1) / sample_px))
        for i in range(n + 1):
            points.append((x1 + (x2 - x1) * i / n, y1 + (y2 - y1) * i / n))
    if not points:
        return 0, 0, 0.0
    pts = np.asarray(points, np.float32)
    # Subsample: a registration offset does not need every pixel of every stroke.
    if len(pts) > 60000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]

    def score(dx: int, dy: int) -> float:
        xs = (pts[:, 0] + dx).astype(np.int32)
        ys = (pts[:, 1] + dy).astype(np.int32)
        ok = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img)
        if ok.sum() < 0.5 * len(pts):
            return 0.0            # geometry mostly off-sheet = a broken coordinate frame, not a match
        return float((dist[ys[ok], xs[ok]] <= 3).mean())

    best = (score(0, 0), 0, 0)
    for dx in range(-search_px, search_px + 1, coarse_step):
        for dy in range(-search_px, search_px + 1, coarse_step):
            s = score(dx, dy)
            if s > best[0]:
                best = (s, dx, dy)
    _, bx, by = best
    for dx in range(bx - coarse_step, bx + coarse_step + 1):
        for dy in range(by - coarse_step, by + coarse_step + 1):
            s = score(dx, dy)
            if s > best[0]:
                best = (s, dx, dy)
    return best[1], best[2], round(best[0], 4)


def filter_segments_by_mask(
    segments: list[tuple[float, float, float, float]],
    mask: np.ndarray,
    min_share: float = 0.8,
    tol_px: float = 2.0,
    sample_px: float = 3.0,
) -> list[tuple[float, float, float, float]]:
    """Keep only strokes that lie on pipe ink.

    The pipe mask is what separates a pipe from a text glyph edge, a border or a hatch: it was
    built to hold pipe and drop everything else. Filtering *segments* before the graph is built
    matters -- filtering whole polylines afterwards lets thousands of glyph strokes merge into
    chains first, which is how a first pass here emitted 464 traces against 17 real lines.
    """
    h_img, w_img = mask.shape[:2]
    ink = (mask < 200) if float(mask.mean()) > 127 else (mask > 0)
    dist = cv2.distanceTransform(255 - (ink.astype(np.uint8) * 255), cv2.DIST_L2, 5)
    kept = []
    for x1, y1, x2, y2 in segments:
        n = max(1, int(math.hypot(x2 - x1, y2 - y1) / sample_px))
        on = tot = 0
        for i in range(n + 1):
            px = int(x1 + (x2 - x1) * i / n)
            py = int(y1 + (y2 - y1) * i / n)
            tot += 1
            if 0 <= py < h_img and 0 <= px < w_img and dist[py, px] <= tol_px:
                on += 1
        if tot and on / tot >= min_share:
            kept.append((x1, y1, x2, y2))
    return kept


def build_network(
    segments: list[tuple[float, float, float, float]],
    offset: tuple[int, int] = (0, 0),
    snap_px: float = SNAP_PX,
) -> dict:
    """Snap stroke ends into nodes, then read the graph out as one polyline per pipe run.

    Each polyline is what a raster walk would have produced, except nothing was inferred: the
    vertex positions are the plotted ones.
    """
    dx, dy = offset
    segs = [((x1 + dx), (y1 + dy), (x2 + dx), (y2 + dy)) for x1, y1, x2, y2 in segments]

    nodes: list[tuple[float, float]] = []
    cell: dict[tuple[int, int], list[int]] = {}

    def node_for(p: tuple[float, float]) -> int:
        cx, cy = int(p[0] // snap_px), int(p[1] // snap_px)
        best, best_d = None, snap_px
        for ix in (cx - 1, cx, cx + 1):
            for iy in (cy - 1, cy, cy + 1):
                for idx in cell.get((ix, iy), []):
                    d = math.hypot(nodes[idx][0] - p[0], nodes[idx][1] - p[1])
                    if d <= best_d:
                        best, best_d = idx, d
        if best is not None:
            return best
        nodes.append(p)
        cell.setdefault((cx, cy), []).append(len(nodes) - 1)
        return len(nodes) - 1

    edges: list[tuple[int, int, float]] = []
    edge_geom: list[tuple[float, float, float, float]] = []
    adjacency: dict[int, list[tuple[int, float]]] = {}
    for x1, y1, x2, y2 in segs:
        a = node_for((x1, y1))
        b = node_for((x2, y2))
        if a == b:
            continue
        length = math.hypot(nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1])
        eid = len(edges)
        edges.append((a, b, length))
        edge_geom.append((nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1]))
        adjacency.setdefault(a, []).append((b, eid))
        adjacency.setdefault(b, []).append((a, eid))

    # Connected components -> polylines. A component is one run of pipe, which is the unit the
    # export calls a line segment; junctions inside it are degree-3+ nodes we keep for later.
    seen: set[int] = set()
    polylines: list[dict] = []
    for start in adjacency:
        if start in seen:
            continue
        stack, comp, comp_edges = [start], [], set()
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for m, eid in adjacency.get(n, []):
                comp_edges.add(eid)
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        # Order the component into a path where possible so the polyline reads like a pipe run.
        endpoints = [n for n in comp if len(adjacency.get(n, [])) == 1]
        walk_start = endpoints[0] if endpoints else comp[0]
        order, visited = [], set()
        stack = [(walk_start, None)]
        while stack:
            n, parent = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            order.append(n)
            nxt = [(m, eid) for m, eid in adjacency.get(n, []) if m not in visited]
            nxt.sort(key=lambda t: -edges[t[1]][2])     # follow the longest continuation first
            for m, _ in nxt:
                stack.append((m, n))
        pts = [nodes[n] for n in order]
        polylines.append({
            "component": comp,
            "edge_ids": sorted(comp_edges),
            "points": [(round(x, 1), round(y, 1)) for x, y in pts],
            "node_count": len(comp),
            "length_px": round(sum(math.hypot(nodes[n][0] - nodes[m][0], nodes[n][1] - nodes[m][1])
                                   for n, m in zip(order, order[1:])), 1),
            "junction_count": sum(1 for n in comp if len(adjacency.get(n, [])) >= 3),
        })
    polylines.sort(key=lambda p: -p["length_px"])
    return {"nodes": [(round(x, 1), round(y, 1)) for x, y in nodes],
            "edges": len(edges), "edge_geom": edge_geom,
            "polylines": polylines, "offset": list(offset)}


def to_trace_results(
    network: dict,
    min_length_px: float = 30.0,
    pipe_mask: Optional[np.ndarray] = None,
    min_pipe_share: float = 0.5,
    pipe_tol_px: float = 2.0,
) -> dict[str, dict]:
    """The network as `stage5b_trace_results.json` entries: segments along each polyline.

    Geometry from the PDF says exactly *where* every stroke runs; the pipeline's pipe mask says
    *which* strokes are pipe (it has already removed text, symbols and the frame). Keeping a
    polyline only when most of its length lies on pipe ink is what stops 852 raw stroke chains —
    text rows, borders, the title block — from becoming traces.
    """
    dist = None
    if pipe_mask is not None:
        dist = cv2.distanceTransform(255 - ((pipe_mask > 0).astype(np.uint8) * 255),
                                     cv2.DIST_L2, 5)
    out: dict[str, dict] = {}
    for i, poly in enumerate(network["polylines"]):
        if poly["length_px"] < min_length_px:
            continue
        if dist is not None:
            pts = poly["points"]
            on_pipe = total = 0
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                n = max(1, int(math.hypot(x2 - x1, y2 - y1) / 4.0))
                for k in range(n + 1):
                    px = int(x1 + (x2 - x1) * k / n)
                    py = int(y1 + (y2 - y1) * k / n)
                    total += 1
                    if 0 <= py < dist.shape[0] and 0 <= px < dist.shape[1] and dist[py, px] <= pipe_tol_px:
                        on_pipe += 1
            if total == 0 or on_pipe / total < min_pipe_share:
                continue
        # Emit the component's graph edges, not a walk order: a depth-first order zig-zags
        # across the component and draws ink where there is no pipe. Edge geometry is exact.
        segments = []
        for ax, ay, bx, by in (network["edge_geom"][e] for e in poly.get("edge_ids", [])):
            x1, y1, x2, y2 = round(ax, 1), round(ay, 1), round(bx, 1), round(by, 1)
            direction = ("LEFT" if x2 < x1 else "RIGHT") if abs(x2 - x1) >= abs(y2 - y1) else (
                "UP" if y2 < y1 else "DOWN")
            segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": direction,
                             "length_px": round(math.hypot(x2 - x1, y2 - y1), 1)})
        if not segments:
            continue
        pts = [(segments[0]["x1"], segments[0]["y1"]),
               (segments[-1]["x2"], segments[-1]["y2"])]
        trace_id = f"vec_{i:04d}"
        out[trace_id] = {
            "source_obj_id": "vector",
            "port_index": 0,
            "port": {"x": pts[0][0], "y": pts[0][1], "direction": segments[0]["direction"]},
            "terminal_type": "vector",
            "terminal_x": pts[-1][0],
            "terminal_y": pts[-1][1],
            "terminal_obj_id": None,
            "segments": segments,
            "trace_length_px": round(poly["length_px"], 1),
            "vector_junctions": poly["junction_count"],
        }
    return out


def load_item_boxes(run_dir: Path) -> list[dict]:
    """Detected objects + equipment boxes: the things a pipe run has to connect."""
    boxes: list[dict] = []
    objs = run_dir / "stage4_objects.json"
    if objs.exists():
        boxes += [o for o in json.loads(objs.read_text()).get("objects", []) if o.get("bbox")]
    eq = run_dir / "stage3_equipment_bboxes.json"
    if eq.exists():
        data = json.loads(eq.read_text())
        data = data if isinstance(data, list) else data.get("objects", [])
        boxes += [e for e in data if e.get("bbox")]
    return boxes


def connects_items(
    trace: dict,
    boxes: list[dict],
    tol_px: float = 18.0,
) -> bool:
    """Does this run start and end on a detected item?

    A pipe run in a P&ID exists to connect two things. Keeping only runs that touch an item at
    both ends drops symbol outlines, hatching and the sheet frame, which is what makes the
    vector path's stray-ink number look bad when every stroke is emitted.
    """
    segs = trace.get("segments") or []
    if not segs:
        return False
    ends = [(segs[0]["x1"], segs[0]["y1"]), (segs[-1]["x2"], segs[-1]["y2"])]

    def near(p: tuple[float, float]) -> bool:
        for box in boxes:
            b = box["bbox"]
            if (b["x_min"] - tol_px <= p[0] <= b["x_max"] + tol_px
                    and b["y_min"] - tol_px <= p[1] <= b["y_max"] + tol_px):
                return True
        return False

    return near(ends[0]) and near(ends[1])


def trace_pdf(
    pdf_path: str | Path,
    run_dir: Path,
    page_no: int = 1,
    dpi: float = 300.0,
    min_length_px: float = 120.0,
    connect_only: bool = False,
) -> dict:
    """Full fast-path: extract -> register against this job's raster -> network -> traces."""
    stats = is_vector_pdf(pdf_path, page_no)
    if not stats["vector"]:
        return {"vector": False, **stats}
    segments, meta = extract_segments(pdf_path, page_no, dpi)
    # Registration wants ALL ink (text included) -- it is a rigid alignment question, and the
    # pipe mask alone is too sparse to pin an offset down. The pipe mask then decides which
    # strokes are pipe.
    ink = cv2.imread(str(run_dir / "stage1_gray.png"), cv2.IMREAD_GRAYSCALE)
    offset, agreement = (0, 0), 0.0
    if ink is not None:
        dx, dy, agreement = register(segments, ink)
        offset = (dx, dy)
    pipe_mask = cv2.imread(str(run_dir / "stage5_pipe_mask.png"), cv2.IMREAD_GRAYSCALE)
    if pipe_mask is not None:
        # Shift the strokes by the registration offset first, so the mask test applies in the
        # raster's own frame; the offset is then dropped from the network.
        shifted = [(x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1])
                   for x1, y1, x2, y2 in segments]
        segments = [(x1, y1, x2, y2) for x1, y1, x2, y2 in filter_segments_by_mask(shifted, pipe_mask)]
        offset = (0, 0)
    network = build_network(segments, offset)
    traces = to_trace_results(network, min_length_px)
    if connect_only:
        boxes = load_item_boxes(run_dir)
        if boxes:
            traces = {k: v for k, v in traces.items() if connects_items(v, boxes)}
    (run_dir / "vector_pipe_network.json").write_text(json.dumps({
        "pdf": str(pdf_path), "page": page_no, "dpi": dpi, "stats": stats, "extract": meta,
        "offset": list(offset), "register_agreement": agreement,
        "nodes": len(network["nodes"]), "edges": network["edges"],
        "polylines": len(network["polylines"]),
    }, indent=1))
    return {"vector": True, "segments": meta["segments"], "offset": list(offset),
            "agreement": agreement, "traces": len(traces), "traces_data": traces}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Trace pipes from a vector PDF sheet.")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--run-dir", required=True, help="job dir (uses its raster for registration)")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--dpi", type=float, default=300.0)
    ap.add_argument("--apply", action="store_true",
                    help="write stage5b_trace_results.json (backs the original up once)")
    ap.add_argument("--connect-only", action="store_true",
                    help="keep only runs that touch a detected item at both ends: precision "
                         "0.854 / recall 0.886 on sheet 0002, against 0.225 / 0.988 unfiltered")
    ap.add_argument("--overlay", default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser()
    result = trace_pdf(args.pdf, run_dir, args.page, args.dpi, connect_only=args.connect_only)
    if not result["vector"]:
        print(json.dumps({"vector": False, **{k: v for k, v in result.items() if k != "traces_data"}}))
        raise SystemExit("not a vector sheet: keep the raster tracer")

    traces = result.pop("traces_data")
    print(json.dumps(result))
    if args.apply:
        target = run_dir / "stage5b_trace_results.json"
        if target.exists() and not (run_dir / "stage5b_trace_results.json.vectorbackup").exists():
            target.rename(run_dir / "stage5b_trace_results.json.vectorbackup")
        target.write_text(json.dumps(traces, indent=1))
        print(f"applied {len(traces)} vector traces -> {target}")
    overlay = Path(args.overlay).expanduser() if args.overlay else run_dir / "vector_trace_overlay.png"
    base = cv2.imread(str(run_dir / "stage1_gray.png"), cv2.IMREAD_GRAYSCALE)
    if base is not None:
        canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        for t in traces.values():
            for s in t["segments"]:
                cv2.line(canvas, (int(s["x1"]), int(s["y1"])), (int(s["x2"]), int(s["y2"])),
                         (0, 0, 255), 2)
        cv2.imwrite(str(overlay), canvas)
        print("overlay ->", overlay)
