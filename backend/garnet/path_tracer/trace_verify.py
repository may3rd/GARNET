"""Verification pass over Stage 5b traces: classify every walk's endpoint, repair what can be
repaired, and put the rest on an explicit HOLD list.

Why this exists: 16-27% of traces on the reference sheets end as `dead_end` — the walk stops in
empty space, not on an item. Nothing downstream can tell that apart from "there is no pipe
here". This pass makes the difference visible and recovers the cheap cases:

  repair, in order  1. continue straight past the endpoint (relaxed probe gap: a jumpover,
                       a dashed gap or a crossing that the tracer's own lookahead missed)
                    2. turn left / right off the endpoint (a corner it stopped short of)
  accept            the new walk adds >= min_add_px AND ends on an item, a node or the sheet
                    edge. Anything else keeps the original endpoint and is reported HOLD.

It runs on artifacts, not inside the tracer's loops, so it cannot change existing traces
unless `apply=True` — repair is additive and every change is recorded per trace.

Read-only use:
    python garnet/verify_trace.py --run-dir output/trace_suite/0003
Apply repairs to the run dir (backs the originals up once):
    python garnet/verify_trace.py --run-dir output/trace_suite/0003 --apply
"""
from __future__ import annotations

import csv
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from garnet.path_tracer.cv_pipe_tracer import TURN_LEFT, TURN_RIGHT, CVPipeTracer

DELTAS = {
    "UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0),
}
INLINE_CLASSES = {
    "gate_valve", "globe_valve", "check_valve", "ball_valve", "butterfly_valve",
    "control_valve", "pressure_relief_valve", "reducer", "spectacle_blind", "strainer",
    "gate valve", "globe valve", "check valve", "ball valve", "butterfly valve",
    "control valve", "pressure relief valve", "spectacle blind",
}
PAGE_CONNECTION_CLASSES = {
    "page_connection", "page connection", "connection", "utility connection",
    "page connection symbol",
}
INSTRUMENT_CLASSES = {"instrument tag", "instrument dcs", "instrument logic"}
EQUIPMENT_CLASSES = {
    "vessel", "column", "pump", "compressor", "blower", "heat_exchanger", "tank",
    "reactor", "knockout_drum", "filter", "cooler", "heater",
}
GOOD_TERMINALS = {
    "equipment", "instrument_tag", "page_connection", "tee_junction",
    "branch_connection", "sheet_edge",
}


def _json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def _bbox_contains(bbox: dict, x: float, y: float, margin: int = 0) -> bool:
    return (
        bbox["x_min"] - margin <= x <= bbox["x_max"] + margin
        and bbox["y_min"] - margin <= y <= bbox["y_max"] + margin
    )


def _bbox_distance(bbox: dict, x: float, y: float) -> float:
    dx = max(bbox["x_min"] - x, 0, x - bbox["x_max"])
    dy = max(bbox["y_min"] - y, 0, y - bbox["y_max"])
    return float(np.hypot(dx, dy))


def trace_items(run_dir: Path) -> list[dict]:
    """Every object a walk should be able to end on, for scoring endpoint proximity."""
    objects = _json(run_dir / "stage4_objects.json", {}).get("objects", [])
    items = [o for o in objects if o.get("bbox")]
    equipment = _json(run_dir / "stage3_equipment_bboxes.json", {})
    equipment = equipment if isinstance(equipment, list) else equipment.get("objects", [])
    known = {str(i.get("id")) for i in items}
    items += [e for e in equipment if e.get("bbox") and str(e.get("id")) not in known]
    return items


def _buckets(items: list[dict]) -> dict[str, list[dict]]:
    def pick(classes: set[str]) -> list[dict]:
        return [o for o in items if str(o.get("class_name")) in classes]

    return {
        "equipment": pick(EQUIPMENT_CLASSES),
        "instrument": pick(INSTRUMENT_CLASSES),
        "page": pick(PAGE_CONNECTION_CLASSES),
        "inline": pick(INLINE_CLASSES),
        "nodes": [o for o in items if o.get("class_name") == "node"],
    }


def _tracer_params() -> dict:
    """The tracer knobs the pipeline itself would use, so verification walks identically.

    Reuses Stage5bPipelineMixin._trace_cv_params rather than keeping a second copy of the
    PipelineConfig -> CVPipeTracer name map: that map is not 1:1 (``lookahead`` comes from
    ``trace_lookahead_px``), and a duplicated copy would silently drift.
    """
    from garnet.path_tracer.stage5b_pipeline import Stage5bPipelineMixin
    from garnet.pid_extractor import PipelineConfig

    class _ConfigHolder:
        cfg = PipelineConfig()

    return Stage5bPipelineMixin._trace_cv_params(_ConfigHolder())


def _walk_traces(run_dir: Path) -> list[tuple[str, dict]]:
    """(trace_id, trace) for port traces and traced branches."""
    traces = [(k, v) for k, v in _json(run_dir / "stage5b_trace_results.json", {}).items()]
    branches = _json(run_dir / "stage5b_branch_trace_results.json", {}).get("branches", {})
    items = branches.values() if isinstance(branches, dict) else branches
    for b in items:
        if isinstance(b, dict) and b.get("status") == "traced":
            traces.append((str(b.get("candidate", {}).get("id", "branch")), b))
    return traces


def _endpoint(trace: dict) -> Optional[tuple[int, int, str]]:
    segs = trace.get("segments") or []
    if not segs:
        return None
    last = segs[-1]
    return int(last["x2"]), int(last["y2"]), str(last.get("direction") or "").upper()


def _added_length(result) -> int:
    return int(sum(s.length_px for s in (result.segments or [])))


def _try_repair(
    endpoint: tuple[int, int, str],
    *,
    pipe_mask: np.ndarray,
    image: np.ndarray,
    buckets: dict[str, list[dict]],
    markers: list[dict],
    params: dict,
    source_id: str,
    straight_only: bool,
    min_add_px: int = 20,
    max_add_px: int = 300,
    probe_px: int = 24,
    backtrack_px: int = 12,
) -> Optional[dict]:
    """Walk on from an endpoint. Returns the winning walk as a dict, or None."""
    x, y, direction = endpoint
    if direction not in DELTAS:
        return None
    h_img, w_img = pipe_mask.shape[:2]
    sdx, sdy = DELTAS[direction]
    moves = [(direction, 0)] if straight_only else [
        (direction, 0),
        (TURN_LEFT.get(direction), 3),       # corners are probed a few px back off the end
        (TURN_RIGHT.get(direction), 3),
    ]
    for move_dir, back_step in moves:
        if not move_dir or move_dir not in DELTAS:
            continue
        mdx, mdy = DELTAS[move_dir]
        backtracks = range(0, backtrack_px + 1, back_step or 1) if back_step else [0]
        for back in backtracks:
            bx = x - sdx * back
            by = y - sdy * back
            for probe in range(2, probe_px + 1):
                sx = bx + mdx * probe
                sy = by + mdy * probe
                if not (0 <= sx < w_img and 0 <= sy < h_img) or not pipe_mask[sy, sx]:
                    continue
                tracer = CVPipeTracer(
                    pipe_mask=pipe_mask,
                    image=image,
                    page_connections=buckets["page"],
                    instrument_tags=buckets["instrument"],
                    equipment_objects=buckets["equipment"],
                    junction_markers=markers,
                    visited_mask=np.zeros_like(pipe_mask),   # verifying, not claiming ink
                    **params,
                )
                tracer.set_inline_symbols(buckets["inline"])
                result = tracer.trace(sx, sy, move_dir, source_obj_id=source_id)
                if not result.segments or result.terminal_type not in GOOD_TERMINALS:
                    continue
                added = _added_length(result)
                if not (min_add_px <= added <= max_add_px):
                    continue          # too short to matter, or a new line rather than a repair
                return {
                    "kind": "straight" if move_dir == direction else "turn",
                    "direction": move_dir,
                    "terminal_type": result.terminal_type,
                    "terminal_x": int(result.terminal_x),
                    "terminal_y": int(result.terminal_y),
                    "terminal_obj_id": result.terminal_obj_id,
                    "added_px": added,
                    "segments": [
                        {"x1": s.x1, "y1": s.y1, "x2": s.x2, "y2": s.y2,
                         "direction": s.direction, "length_px": s.length_px}
                        for s in result.segments
                    ],
                }
    return None


def _graph_route_repair(
    endpoint: tuple[int, int, str],
    mask: np.ndarray,
    graph,
    item_nodes: list[int],
    snap_px: float = 25.0,
    max_add_px: float = 1500.0,
) -> Optional[dict]:
    """Repair a dead-ended walk by *routing* to an item instead of walking again.

    The walker's own retry (`_try_repair`) is still greedy: it re-probes a straight run and two
    corners from the same pixel that just failed. This asks a stronger question — which items are
    reachable from the ink at this endpoint, and what is the shortest path to one — so a walk that
    stopped short of a glyph is resolved globally. It only works inside one connected blob of ink,
    which is exactly the case the walker's local retry was failing in.
    """
    from .skeleton_graph import _nearest_node, route

    x, y, _direction = endpoint
    start = _nearest_node(graph, x, y, snap_px)
    if start is None:
        return None
    best_points, best_len = None, None
    for goal in item_nodes:
        if goal == start:
            continue
        points = route(graph, start, goal)
        if not points:
            continue
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(points, points[1:]))
        if length > max_add_px:
            continue
        if best_len is None or length < best_len:
            best_points, best_len = points, length
    if not best_points:
        return None
    simplified = cv2.approxPolyDP(
        np.array(best_points, np.int32).reshape(-1, 1, 2), 1.5, False
    ).reshape(-1, 2)
    corners = [(int(px), int(py)) for px, py in simplified]
    segments = []
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        if (x1, y1) == (x2, y2):
            continue
        direction = ("LEFT" if x2 < x1 else "RIGHT") if abs(x2 - x1) >= abs(y2 - y1) else (
            "UP" if y2 < y1 else "DOWN")
        segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "direction": direction,
                         "length_px": round(math.hypot(x2 - x1, y2 - y1), 1)})
    if not segments:
        return None
    return {"kind": "graph_route", "added_px": int(round(best_len or 0)),
            "terminal_type": "skeleton_route",
            "terminal_x": segments[-1]["x2"], "terminal_y": segments[-1]["y2"],
            "terminal_obj_id": None, "segments": segments}


def verify_run(
    run_dir: Path,
    apply: bool = False,
    overlay: Optional[Path] = None,
    repair: bool = False,
    graph_repair: bool = False,
    params: Optional[dict] = None,
    stem: str = "trace_verification",
) -> dict:
    mask = cv2.imread(str(run_dir / "stage5_pipe_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"no stage5_pipe_mask.png in {run_dir}")
    gray = cv2.imread(str(run_dir / "stage1_gray.png"), cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray is not None else cv2.cvtColor(
        mask, cv2.COLOR_GRAY2BGR)

    items = trace_items(run_dir)
    buckets = _buckets(items)
    markers = _json(run_dir / "stage4_topology_markers.json", {})
    markers = markers if isinstance(markers, list) else markers.get("markers", [])
    params = params or _tracer_params()
    traces = _walk_traces(run_dir)
    h_img, w_img = mask.shape[:2]
    edge_tol = 20
    item_tol = 12

    graph = item_nodes = None
    if graph_repair:
        from .skeleton_graph import _nearest_node, skeleton_graph

        graph = skeleton_graph(mask)
        item_nodes = []
        for item in items:
            b = item["bbox"]
            node = _nearest_node(graph, (b["x_min"] + b["x_max"]) / 2,
                                 (b["y_min"] + b["y_max"]) / 2, 40.0)
            if node is not None:
                item_nodes.append(node)

    rows: list[dict] = []
    repaired = holds = 0
    for trace_id, trace in traces:
        endpoint = _endpoint(trace)
        if endpoint is None:
            continue
        x, y, direction = endpoint
        terminal_type = str(trace.get("terminal_type") or "")
        nearest, nearest_class, nearest_dist = "", "", float("inf")
        for item in items:
            d = _bbox_distance(item["bbox"], x, y)
            if d < nearest_dist:
                nearest, nearest_class, nearest_dist = str(item.get("id")), str(item.get("class_name")), d

        on_item = nearest_dist <= item_tol
        on_edge = min(x, y, w_img - x, h_img - y) <= edge_tol
        on_node = terminal_type in ("tee_junction", "branch_connection")
        verdict = "ok_item" if on_item else "ok_edge" if on_edge else "ok_node" if on_node else "suspect"

        row = {
            "trace_id": trace_id, "terminal_type": terminal_type,
            "end_x": x, "end_y": y, "direction": direction,
            "nearest_item": nearest, "nearest_class": nearest_class,
            "nearest_dist_px": round(nearest_dist, 1),
            "verdict": verdict, "repair_kind": "", "added_px": 0, "new_terminal": "",
        }
        # Only a walk that ended in empty space is a repair candidate. A node, item or edge
        # terminal is already a real answer: continuing past it produced 800-1000 px walks
        # along neighbouring runs (precision 0.866 -> 0.799 on 0003), which is inventing a
        # line, not verifying one.
        if verdict == "suspect" and graph_repair and graph is not None and item_nodes:
            attempt = _graph_route_repair(endpoint, mask, graph, item_nodes)
            if attempt:
                row.update({"verdict": "repaired", "repair_kind": attempt["kind"],
                            "added_px": attempt["added_px"],
                            "new_terminal": attempt["terminal_type"]})
                repaired += 1
                if apply:
                    trace.setdefault("segments", []).extend(attempt["segments"])
                    trace["terminal_type"] = attempt["terminal_type"]
                    trace["terminal_x"] = attempt["terminal_x"]
                    trace["terminal_y"] = attempt["terminal_y"]
                    trace["verified"] = {"repair": "graph_route",
                                         "added_px": attempt["added_px"]}
        if verdict == "suspect" and repair and row["verdict"] != "repaired":
            attempt = _try_repair(
                endpoint, pipe_mask=mask, image=image, buckets=buckets, markers=markers,
                params=params, source_id=trace_id, straight_only=False,
            )
            if attempt:
                row.update({"verdict": "repaired", "repair_kind": attempt["kind"],
                            "added_px": attempt["added_px"],
                            "new_terminal": attempt["terminal_type"]})
                repaired += 1
                if apply:
                    trace.setdefault("segments", []).extend(attempt["segments"])
                    trace["terminal_type"] = attempt["terminal_type"]
                    trace["terminal_x"] = attempt["terminal_x"]
                    trace["terminal_y"] = attempt["terminal_y"]
                    trace["terminal_obj_id"] = attempt["terminal_obj_id"]
                    trace["verified"] = {"repair": attempt["kind"], "added_px": attempt["added_px"]}
        if verdict == "suspect" and row["verdict"] != "repaired":
            row["verdict"] = "hold"
            holds += 1
        rows.append(row)

    if apply:
        for name in ("stage5b_trace_results.json", "stage5b_branch_trace_results.json"):
            src = run_dir / name
            if src.exists() and not (run_dir / (name + ".orig")).exists():
                shutil.copy(src, run_dir / (name + ".orig"))
        (run_dir / "stage5b_trace_results.json").write_text(
            json.dumps({tid: t for tid, t in traces if not tid.startswith("branch_")}, indent=1)
        )
        branches = _json(run_dir / "stage5b_branch_trace_results.json", {})
        raw = branches.get("branches", {})
        branch_map = dict(raw) if isinstance(raw, dict) else {b.get("id"): b for b in raw}
        for tid, t in traces:
            if tid.startswith("branch_") and tid in branch_map:
                branch_map[tid] = t
        branches["branches"] = branch_map
        (run_dir / "stage5b_branch_trace_results.json").write_text(json.dumps(branches, indent=1))

    summary = {
        "traces": len(rows),
        "ok_item": sum(1 for r in rows if r["verdict"] == "ok_item"),
        "ok_edge": sum(1 for r in rows if r["verdict"] == "ok_edge"),
        "ok_node": sum(1 for r in rows if r["verdict"] == "ok_node"),
        "repaired": repaired,
        "hold": holds,
        "applied": bool(apply),
        "repair_enabled": bool(repair),
        "graph_repair_enabled": bool(graph_repair),
    }
    (run_dir / f"{stem}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    with (run_dir / f"{stem}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["trace_id"])
        writer.writeheader()
        writer.writerows(rows)

    if overlay is not None:
        canvas = image.copy()
        for r in rows:
            colour = {"hold": (0, 140, 255), "repaired": (255, 120, 0)}.get(r["verdict"])
            if colour:
                cv2.circle(canvas, (r["end_x"], r["end_y"]), 10, colour, 2)
        cv2.imwrite(str(overlay), canvas)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Verify Stage 5b trace endpoints: repair or HOLD.")
    ap.add_argument("--run-dir", required=True, help="output/<job> dir with stage5b results")
    ap.add_argument("--repair", action="store_true",
                    help="also walk on from a dead-ended endpoint and adopt the continuation. Off by "
                         "default: on 0002A/0002/0003 it buys no recall and costs ~0.005 precision, "
                         "so the classifier alone is the useful output.")
    ap.add_argument("--graph-repair", action="store_true",
                    help="repair a dead-ended walk by routing it to the nearest reachable item on "
                         "the skeleton graph, instead of re-probing from the same pixel")
    ap.add_argument("--apply", action="store_true", help="imply --repair and write results back")
    ap.add_argument("--overlay", default="", help="write a PNG marking HOLD orange / repaired blue")
    a = ap.parse_args(argv)

    run_dir = Path(a.run_dir).expanduser()
    overlay = Path(a.overlay).expanduser() if a.overlay else run_dir / "trace_verification_overlay.png"
    summary = verify_run(run_dir, apply=a.apply, overlay=overlay,
                         repair=a.repair or a.apply, graph_repair=a.graph_repair)
    print(json.dumps(summary))
    print(f"rows -> {run_dir/'trace_verification.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
