"""
Stage 16: Connection + Pipeline Overlay Visualization

Draws a combined overlay showing:
1. Page connection markers (blue boxes + anchor dots) from stage12_connection_attachments.json (accepted only)
2. FULL pipe paths (red) from each connection anchor to the nearest terminal(s) in both directions along the pipe network — not just the edge_id stub
3. Inline element connectors (orange) from stage12_edge_connections.json

Background: original P&ID image
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (BGR, OpenCV convention)
# ---------------------------------------------------------------------------
BLUE_MARKER = (255, 100, 0)          # page-connection marker boxes + anchor dots
RED_HIGHLIGHT = (0, 0, 255)         # pipeline segments connected to anchor points
ORANGE_CONNECTOR = (0, 165, 255)    # inline element connectors
STUB_COLOR = (180, 180, 180)        # dashed stub line from anchor dot → pipe entry
WHITE_TEXT = (255, 255, 255)

THICKNESS_HIGHLIGHT = 3
THICKNESS_BOX = 2
ANCHOR_RADIUS = 8
CONNECTOR_RADIUS = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS_TEXT = 1

# BFS limit when tracing full paths from connection anchors to terminals
PATH_TRACE_MAX_HOPS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_dashed_line(
    canvas: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    *,
    gap: int = 8,
) -> None:
    """Draw a dashed line segment (line-gap-line-gap...) between p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return

    udx, udy = dx / length, dy / length
    drawn = 0.0
    while drawn < length:
        seg_len = min(gap, length - drawn)
        start = (int(p1[0] + udx * drawn), int(p1[1] + udy * drawn))
        end = (int(p1[0] + udx * (drawn + seg_len)), int(p1[1] + udy * (drawn + seg_len)))
        cv2.line(canvas, start, end, color, thickness)
        drawn += gap * 2


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_edge_lookup(edges: list[dict]) -> dict[str, dict]:
    return {e["id"]: e for e in edges}


def polyline_to_cv2_points(polyline: list[dict]) -> np.ndarray:
    """Convert [{row, col}, ...] → np.array of (col, row) for cv2.polylines."""
    pts = np.array([[pt["col"], pt["row"]] for pt in polyline], dtype=np.int32)
    return pts


def polyline_midpoint(polyline: list[dict]) -> tuple[float, float]:
    """Average of first and last point as (col, row)."""
    first = polyline[0]
    last = polyline[-1]
    cx = (first["col"] + last["col"]) / 2.0
    cy = (first["row"] + last["row"]) / 2.0
    return cx, cy


def draw_polyline(img: np.ndarray, polyline: list[dict], color: tuple[int, int, int],
                  thickness: int) -> None:
    if len(polyline) < 2:
        return
    pts = polyline_to_cv2_points(polyline)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)


def draw_anchor_label(img: np.ndarray, anchor_name: str, anchor_xy: list[float],
                       color: tuple[int, int, int]) -> None:
    """Write anchor side label (top/bottom/left/right) near the anchor dot."""
    x, y = anchor_xy
    label = anchor_name
    text_x = int(x) + 12
    text_y = int(y) - 8
    cv2.putText(img, label, (text_x, text_y), FONT, FONT_SCALE, color, THICKNESS_TEXT)


def _build_junction_adjacency(edges: list[dict]) -> dict[str, list[tuple[str, dict]]]:
    """Build adjacency list from non-attachment edges (junction ↔ junction only)."""
    adj: dict[str, list[tuple[str, dict]]] = {}
    for e in edges:
        if "attachment" not in e["id"]:
            src, tgt = e["source"], e["target"]
            if src not in adj:
                adj[src] = []
            if tgt not in adj:
                adj[tgt] = []
            adj[src].append((tgt, e))
            adj[tgt].append((src, e))
    return adj


def _bfs_find_best_path_to_terminal(
    start_node: str,
    adj: dict[str, list[tuple[str, dict]]],
    node_lookup: dict,
    max_hops: int = PATH_TRACE_MAX_HOPS,
) -> tuple[list[dict], int]:
    """
    BFS from start_node to nearest terminal (endpoint or pump).
    Returns (path_edges, total_pixel_length) for the best (longest) path found.
    """
    visited = {start_node}
    queue = deque([(start_node, 0, [])])
    best = ([], 0)
    while queue:
        curr, depth, path = queue.popleft()
        if depth >= max_hops:
            continue
        for neighbor, edge in adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                ntype = node_lookup.get(neighbor, {}).get("type", "")
                new_path = path + [edge]
                if ntype in ("endpoint", "pump"):
                    total_px = sum(ee["pixel_length"] for ee in new_path)
                    if total_px > best[1]:
                        best = (new_path, total_px)
                else:
                    queue.append((neighbor, depth + 1, new_path))
    return best


def _find_full_connection_paths(
    conn: dict,
    graph_edges: list[dict],
    node_lookup: dict,
    edge_lookup: dict,
) -> list[list[dict]]:
    """
    For one connection, trace complete paths in BOTH directions from the
    attachment node to the nearest terminals (endpoint/pump).

    Returns a list of path edge lists. Each path is a list of edge dicts.
    """
    det_id = conn["det_id"]
    attach_id = f"attach::{det_id}"
    edge_id = conn.get("edge_id", "")

    adj = _build_junction_adjacency(graph_edges)

    def bfs_to_terminals(start_node: str, max_hops: int = PATH_TRACE_MAX_HOPS) -> list:
        visited = {start_node}
        queue = deque([(start_node, 0, [])])
        results = []
        while queue:
            curr, depth, path = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor, edge in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    ntype = node_lookup.get(neighbor, {}).get("type", "")
                    new_path = path + [edge]
                    if ntype in ("endpoint", "pump"):
                        total_px = sum(ee["pixel_length"] for ee in new_path)
                        results.append((neighbor, ntype, depth + 1, total_px, new_path))
                    else:
                        queue.append((neighbor, depth + 1, new_path))
        return results

    paths = []

    if edge_id and edge_id in edge_lookup:
        attached_edge = edge_lookup[edge_id]
        src_node = attached_edge["source"]
        tgt_node = attached_edge["target"]

        src_results = bfs_to_terminals(src_node)
        tgt_results = bfs_to_terminals(tgt_node)

        # Best path from source direction
        if src_results:
            best = max(src_results, key=lambda x: x[3])
            paths.append(best[4])

        # Best path from target direction (avoid duplicate if same)
        if tgt_results:
            best = max(tgt_results, key=lambda x: x[3])
            if not paths or best[4] != paths[-1]:
                paths.append(best[4])

    return paths


def render_overlay(
    connection_attachments_path: str,
    edge_connections_path: str,
    edge_terminals_path: str,
    graph_path: str,
    objects_path: str,
    output_path: str,
    image_base_path: str | None = None,
) -> dict:
    # Load data
    attachments_data = load_json(connection_attachments_path)
    edge_conn_data = load_json(edge_connections_path)
    edge_terminals_data = load_json(edge_terminals_path)
    graph_data = load_json(graph_path)
    objects_data = load_json(objects_path)

    # Determine background image path
    background_path: str | None = None
    if image_base_path:
        background_path = str(image_base_path)
    else:
        img_path = objects_data.get("image_path")
        if img_path and Path(img_path).exists():
            background_path = img_path

    if background_path and Path(background_path).exists():
        background_img = cv2.imread(background_path)
        if background_img is None:
            print(f"[WARN] Could not read background image: {background_path}", file=sys.stderr)
            height = 3500
            width = 2700
            background_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        else:
            print(f"[INFO] Using background: {background_path}")
    else:
        fallback = Path(objects_path).parent / "stage4_objects_overlay.png"
        if fallback.exists():
            background_img = cv2.imread(str(fallback))
            print(f"[INFO] Using fallback background: {fallback}")
        else:
            height = 3500
            width = 2700
            background_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            print(f"[WARN] No background image found; using blank canvas", file=sys.stderr)

    canvas = background_img.copy()

    # Load accepted connections early so all layers can reference it
    accepted = attachments_data.get("accepted", [])

    # Build edge lookup from graph
    edges_list = graph_data.get("edges", [])
    edge_lookup = build_edge_lookup(edges_list)
    node_lookup = {n["id"]: n for n in graph_data.get("nodes", [])}

    stats = {
        "connections_drawn": 0,
        "highlighted_edges": 0,
        "inline_connectors_drawn": 0,
        "marker_boxes_drawn": 0,
    }

    # --------------------------------------------------------------------------
    # Layer 1: FULL pipe paths from connection anchors (red) — drawn FIRST
    #
    # For each accepted connection:
    #   anchor → equipment_attachment node → pipe junction → ... → nearest terminal
    #
    # Not just the short edge_id stub — traces the complete path in both
    # directions from the connection's attachment point through the pipe network.
    # --------------------------------------------------------------------------
    for conn in accepted:
        det_id = conn["det_id"]
        attach_id = f"attach::{det_id}"
        anchor_xy = conn.get("anchor_xy", [])
        if not anchor_xy:
            continue

        attach_node = node_lookup.get(attach_id)
        if not attach_node:
            continue

        # Draw anchor → attachment node stub (RED_HIGHLIGHT thick line)
        ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
        ex = int(attach_node["position"]["x"])
        ey = int(attach_node["position"]["y"])
        cv2.line(canvas, (ax, ay), (ex, ey), RED_HIGHLIGHT, THICKNESS_HIGHLIGHT)

        # Draw full pipe paths in both directions from the attachment node
        paths = _find_full_connection_paths(conn, edges_list, node_lookup, edge_lookup)
        for path_edges in paths:
            for e in path_edges:
                poly = e.get("polyline", [])
                if len(poly) >= 2:
                    draw_polyline(canvas, poly, RED_HIGHLIGHT, THICKNESS_HIGHLIGHT)
                    stats["highlighted_edges"] += 1

    # --------------------------------------------------------------------------
    # Layer 2: Inline element connectors (orange dotted lines + circles)
    # --------------------------------------------------------------------------
    edge_connections = edge_conn_data.get("edge_connections", [])
    for ec in edge_connections:
        if ec.get("kind") != "inline_element":
            continue
        src_edge_id = ec.get("source_edge_id")
        tgt_edge_id = ec.get("target_edge_id")
        src_edge = edge_lookup.get(src_edge_id)
        tgt_edge = edge_lookup.get(tgt_edge_id)
        if src_edge and tgt_edge:
            src_poly = src_edge.get("polyline", [])
            tgt_poly = tgt_edge.get("polyline", [])
            if len(src_poly) >= 2 and len(tgt_poly) >= 2:
                sx, sy = polyline_midpoint(src_poly)
                tx, ty = polyline_midpoint(tgt_poly)
                cv2.circle(canvas, (int(sx), int(sy)), CONNECTOR_RADIUS, ORANGE_CONNECTOR, -1)
                cv2.circle(canvas, (int(tx), int(ty)), CONNECTOR_RADIUS, ORANGE_CONNECTOR, -1)
                cv2.line(canvas, (int(sx), int(sy)), (int(tx), int(ty)),
                         ORANGE_CONNECTOR, 1, lineType=cv2.LINE_4)
                stats["inline_connectors_drawn"] += 1

    # --------------------------------------------------------------------------
    # Layer 3: Page connection marker boxes (blue) — drawn LAST so they are
    # on top of pipes and pipeline highlights, and never overwritten
    # -------------------------------------------------------------------------
    for conn in accepted:
        bbox = conn.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                          BLUE_MARKER, THICKNESS_BOX)
            stats["marker_boxes_drawn"] += 1

    #
    # Stub line: anchor dot → pipe entry (L-shaped: horizontal + vertical).
    # The corner is at (anchor_x, pipe_y) — i.e. go horizontal to the pipe's
    # y-level first, then across to the nearest pipe point.  This correctly
    # represents a pipe running at pipe_y and a box connection on its side.
    # -------------------------------------------------------------------------
    for conn in accepted:
        anchor_xy = conn.get("anchor_xy")
        nearest_xy = conn.get("nearest_point_xy")
        if anchor_xy and nearest_xy:
            ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
            px, py = int(nearest_xy[0]), int(nearest_xy[1])
            pipe_y = py  # y-level of the horizontal pipe
            _draw_dashed_line(canvas, (ax, ay), (ax, pipe_y), STUB_COLOR, 1, gap=6)
            _draw_dashed_line(canvas, (ax, pipe_y), (px, pipe_y), STUB_COLOR, 1, gap=6)

    # --------------------------------------------------------------------------
    # Layer 4: Anchor dots + labels (blue) — always on TOP so they are visible
    # even if a pipeline polyline passes through the anchor point
    # -------------------------------------------------------------------------
    for conn in accepted:
        anchor_xy = conn.get("anchor_xy")
        anchor_name = conn.get("anchor_name", "unknown")
        if anchor_xy:
            cx, cy = int(anchor_xy[0]), int(anchor_xy[1])
            cv2.circle(canvas, (cx, cy), ANCHOR_RADIUS + 2, WHITE_TEXT, 1)
            cv2.circle(canvas, (cx, cy), ANCHOR_RADIUS, BLUE_MARKER, -1)
            draw_anchor_label(canvas, anchor_name, anchor_xy, WHITE_TEXT)
        stats["connections_drawn"] += 1

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path_obj), canvas)
    print(f"[INFO] Overlay saved → {output_path}")

    return {
        "overlay_path": str(output_path_obj.resolve()),
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render connection + pipeline overlay for GARNET stage 16"
    )
    parser.add_argument(
        "--connection-attachments",
        required=True,
        help="Path to stage12_connection_attachments.json",
    )
    parser.add_argument(
        "--edge-connections",
        required=True,
        help="Path to stage12_edge_connections.json",
    )
    parser.add_argument(
        "--edge-terminals",
        required=True,
        help="Path to stage12_edge_terminals.json",
    )
    parser.add_argument(
        "--graph",
        required=True,
        help="Path to stage12_graph.json",
    )
    parser.add_argument(
        "--objects",
        required=True,
        help="Path to stage4_objects.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path for the overlay",
    )
    parser.add_argument(
        "--image-base-path",
        default=None,
        help="Path to the original P&ID image (optional; stage4_objects_overlay.png used as fallback)",
    )
    args = parser.parse_args()

    result = render_overlay(
        connection_attachments_path=args.connection_attachments,
        edge_connections_path=args.edge_connections,
        edge_terminals_path=args.edge_terminals,
        graph_path=args.graph,
        objects_path=args.objects,
        output_path=args.output,
        image_base_path=args.image_base_path,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()