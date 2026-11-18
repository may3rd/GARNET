"""Trace pipe runs as a graph problem instead of a greedy pixel walk.

The directional walker in Stage 5b decides one step at a time: at every pixel it asks "does the
ink continue, turn, or stop". That is why 13-17 walks per sheet end in empty space — a local
decision cannot see that a run continues 8 px past a glyph, and a wrong turn is unrecoverable.
This module reads the whole mask into a graph first, then routes between terminals:

    skeletonise  ->  pixel graph  ->  shortest path between the two terminals

A dead end becomes a degree-1 node you can count *before* routing anything, and a route is chosen
globally, so it cannot paint itself into a corner. Same inputs as Stage 5b (pipe mask + seeded
ports), same output schema (`stage5b_trace_results.json`), so scoring and stage 5c/6 are unchanged.

Usage:
    python -m garnet.path_tracer.skeleton_graph --run-dir output/<job> --apply
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def skeleton_graph(mask: np.ndarray) -> nx.Graph:
    """Skeleton pixels -> graph, one node per skeleton pixel.

    A pixel graph rather than a chain-contracted one on purpose: chain walking has to guess
    which neighbour continues the line, and one wrong guess silently drops the rest of that
    chain, which resurfaces as a disconnected pipe. Per-pixel edges cannot lose ink. Degree
    still says what a pixel is -- 1 is a dead end, 2 is a run, 3+ is a junction -- so the failure
    the walker hides (a walk that stopped in empty space) is a countable node.
    """
    skel = skeletonize(mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(skel)
    index = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    g = nx.Graph()
    for (y, x), i in index.items():
        g.add_node(i, y=y, x=x)
    for (y, x), i in index.items():
        for dy, dx in NEIGHBOURS:
            j = index.get((y + dy, x + dx))
            if j is not None and i < j:
                g.add_edge(i, j, weight=math.hypot(dx, dy))
    return g


def _nearest_node(g: nx.Graph, x: float, y: float, max_dist: float) -> Optional[int]:
    best, best_d = None, max_dist
    for idx, data in g.nodes(data=True):
        d = math.hypot(data["x"] - x, data["y"] - y)
        if d <= best_d:
            best, best_d = idx, d
    return best


def route(g: nx.Graph, start: int, goal: int) -> Optional[list[tuple[int, int]]]:
    """Shortest path over the skeleton graph, returned as one pixel polyline."""
    try:
        chain = nx.shortest_path(g, start, goal, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return [(int(g.nodes[n]["x"]), int(g.nodes[n]["y"])) for n in chain]


def trace_ports(
    mask: np.ndarray,
    ports: dict,
    item_centres: list[tuple[float, float]],
    snap_px: float = 30.0,
    min_length_px: float = 60.0,
) -> dict[str, dict]:
    """One trace per seeded port: route from the port to the nearest item it can reach.

    `ports` is the pipeline's own `stage5_connection_ports.json` — the same seeds the walker
    uses — so a difference here is the routing, not the seeding.
    """
    g = skeleton_graph(mask)
    if g.number_of_nodes() == 0:
        return {}
    goal_nodes = []
    for cx, cy in item_centres:
        n = _nearest_node(g, cx, cy, snap_px)
        if n is not None:
            goal_nodes.append(n)

    traces: dict[str, dict] = {}
    for obj_id, port_list in ports.items():
        for i, port in enumerate(port_list or []):
            start = _nearest_node(g, float(port[0]), float(port[1]), snap_px)
            if start is None:
                continue
            best_points, best_len = None, None
            for goal in goal_nodes:
                if goal == start:
                    continue
                points = route(g, start, goal)
                if not points:
                    continue
                length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                             for a, b in zip(points, points[1:]))
                if best_len is None or length < best_len:
                    best_points, best_len = points, length
            if not best_points or (best_len or 0) < min_length_px:
                continue
            # A route is a per-pixel polyline; simplify to its corners before it becomes segments.
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
                segments.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                 "direction": direction,
                                 "length_px": round(math.hypot(x2 - x1, y2 - y1), 1)})
            if not segments:
                continue
            traces[f"{obj_id}:port_{i + 1:02d}"] = {
                "source_obj_id": obj_id,
                "port_index": i + 1,
                "port": {"x": segments[0]["x1"], "y": segments[0]["y1"],
                         "direction": segments[0]["direction"]},
                "terminal_type": "skeleton_route",
                "terminal_x": segments[-1]["x2"],
                "terminal_y": segments[-1]["y2"],
                "terminal_obj_id": None,
                "segments": segments,
                "trace_length_px": round(best_len or 0, 1),
            }
    return traces


def item_centres_from_run(run_dir: Path) -> list[tuple[float, float]]:
    boxes: list[dict] = []
    objects = run_dir / "stage4_objects.json"
    if objects.exists():
        boxes += [o for o in json.loads(objects.read_text()).get("objects", []) if o.get("bbox")]
    equipment = run_dir / "stage3_equipment_bboxes.json"
    if equipment.exists():
        data = json.loads(equipment.read_text())
        boxes += [e for e in (data if isinstance(data, list) else data.get("objects", []))
                  if e.get("bbox")]
    return [((b["bbox"]["x_min"] + b["bbox"]["x_max"]) / 2,
             (b["bbox"]["y_min"] + b["bbox"]["y_max"]) / 2) for b in boxes]


def run(run_dir: Path, apply: bool = False, snap_px: float = 30.0) -> dict:
    mask = cv2.imread(str(run_dir / "stage5_pipe_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"no stage5_pipe_mask.png in {run_dir}")
    ports_path = run_dir / "stage5_connection_ports.json"
    if not ports_path.exists():
        raise SystemExit(f"no stage5_connection_ports.json in {run_dir}")
    ports = json.loads(ports_path.read_text())
    centres = item_centres_from_run(run_dir)

    traces = trace_ports(mask, ports, centres, snap_px=snap_px)
    summary = {
        "traces": len(traces),
        "ports": sum(len(v or []) for v in ports.values()),
        "items": len(centres),
        "apply": bool(apply),
    }
    (run_dir / "skeleton_route_summary.json").write_text(json.dumps(summary, indent=1))
    if apply:
        target = run_dir / "stage5b_trace_results.json"
        if target.exists() and not (run_dir / "stage5b_trace_results.json.walkbackup").exists():
            target.rename(run_dir / "stage5b_trace_results.json.walkbackup")
        target.write_text(json.dumps(traces, indent=1))
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Route pipe runs over a skeleton graph.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--snap-px", type=float, default=30.0,
                    help="how far a terminal may sit from the skeleton and still be routed")
    ap.add_argument("--apply", action="store_true",
                    help="write stage5b_trace_results.json (backs the walker's output up once)")
    args = ap.parse_args()
    print(json.dumps(run(Path(args.run_dir).expanduser(), apply=args.apply,
                         snap_px=args.snap_px)))
