"""
Stage 14: Connection + Pipeline Overlay Visualization

Draws a combined overlay showing:
1. Page connection markers (blue boxes + anchor dots) from stage12_connection_attachments.json (accepted only)
2. Connected pipeline segments (red) for each accepted attachment's edge_id
3. Inline element connectors (orange) from stage12_edge_connections.json

Background: original P&ID image
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (BGR, OpenCV convention)
# ---------------------------------------------------------------------------
BLUE_MARKER = (255, 100, 0)          # page-connection marker boxes + anchor dots
RED_HIGHLIGHT = (0, 0, 255)           # pipeline segments connected to anchor points
ORANGE_CONNECTOR = (0, 165, 255)    # inline element connectors
WHITE_TEXT = (255, 255, 255)

THICKNESS_HIGHLIGHT = 3
THICKNESS_BOX = 2
ANCHOR_RADIUS = 8
CONNECTOR_RADIUS = 5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS_TEXT = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        # Fallback: try stage4_objects_overlay.png in same directory as objects JSON
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

    stats = {
        "connections_drawn": 0,
        "highlighted_edges": 0,
        "inline_connectors_drawn": 0,
        "marker_boxes_drawn": 0,
    }

    # --------------------------------------------------------------------------
    # Layer 1: Connected pipeline segments (red) drawn FIRST — below everything
    # --------------------------------------------------------------------------
    highlighted_edge_ids = set()
    for conn in accepted:
        edge_id = conn.get("edge_id")
        if edge_id and edge_id in edge_lookup:
            edge = edge_lookup[edge_id]
            polyline = edge.get("polyline", [])
            if len(polyline) >= 2:
                draw_polyline(canvas, polyline, RED_HIGHLIGHT, THICKNESS_HIGHLIGHT)
                highlighted_edge_ids.add(edge_id)
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
    # --------------------------------------------------------------------------
    for conn in accepted:
        bbox = conn.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                          BLUE_MARKER, THICKNESS_BOX)
            stats["marker_boxes_drawn"] += 1

    # --------------------------------------------------------------------------
    # Layer 4: Anchor dots + labels (blue) — always on TOP so they are visible
    # even if a pipeline polyline passes through the anchor point
    # --------------------------------------------------------------------------
    for conn in accepted:
        anchor_xy = conn.get("anchor_xy")
        anchor_name = conn.get("anchor_name", "unknown")
        if anchor_xy:
            cx, cy = int(anchor_xy[0]), int(anchor_xy[1])
            # Draw a white outline first (anti-aliasing ring), then filled dot
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
        description="Render connection + pipeline overlay for GARNET stage 14"
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