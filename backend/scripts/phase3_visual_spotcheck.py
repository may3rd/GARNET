#!/usr/bin/env python3
"""
Phase 3 Visual Spot-Check — overlay Phase 3 graph on original P&ID images.

Generates:
  - phase3_edge_overlay.png   (edges + junction nodes on original image)
  - phase3_graph_overlay.png  (full graph: edges, junctions, terminals, attachments)

Run manually on 3 representative images to verify topology looks correct.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    b, g, r = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Color palette (BGR for cv2)
PALETTE = {
    "edge_h": (40, 180, 40),      # green — horizontal edges
    "edge_v": (180, 80, 40),      # orange-red — vertical edges
    "junction_L": (0, 255, 255),  # yellow — L-junction
    "junction_T": (0, 128, 255),  # orange — T-junction
    "junction_X": (128, 0, 255),  # purple — X-crossing
    "junction_multi": (0, 0, 255),  # red — multi-way
    "terminal": (200, 200, 50),   # cyan-ish — terminal nodes
    "equipment": (80, 0, 200),    # violet — equipment attachments
    "text": (0, 220, 220),        # teal — text labels
    "arrow": (0, 100, 255),       # bright orange — flow arrows
}


def draw_polyline(bg: np.ndarray, polyline: list[dict], color: tuple[int, int, int], thickness: int = 2) -> None:
    if not polyline:
        return
    pts = np.array([[p["col"], p["row"]] for p in polyline], dtype=np.int32)
    cv2.polylines(bg, [pts], isClosed=False, color=color, thickness=thickness)


def overlay_edges(
    image: np.ndarray,
    edges: list[dict],
    node_positions: dict[str, tuple[float, float]],
    node_types: dict[str, str],
    arrows: list[dict],
    text_attachments: list[dict],
    instrument_tags: list[dict],
    show_labels: bool = False,
) -> np.ndarray:
    bg = image.copy()
    if bg.ndim == 2:
        bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

    # Draw edges
    for edge in edges:
        polyline = edge.get("polyline", [])
        if not polyline:
            continue

        # Color by dominant orientation
        if len(polyline) >= 2:
            p1 = polyline[0]
            p2 = polyline[-1]
            dy = abs(p2["row"] - p1["row"])
            dx = abs(p2["col"] - p1["col"])
            color = PALETTE["edge_h"] if dx >= dy else PALETTE["edge_v"]
        else:
            color = PALETTE["edge_h"]

        draw_polyline(bg, polyline, color, thickness=2)

        # Draw arrow on edge if flow direction assigned
        if edge.get("assigned_arrow_id") and len(polyline) >= 3:
            mid = len(polyline) // 2
            pt = polyline[mid]
            arrow_dir = edge.get("flow_direction", "forward")
            offset = 5 if arrow_dir == "forward" else -5
            if mid + offset < len(polyline) and mid + offset >= 0:
                pt2 = polyline[mid + offset]
                pt_arr = (int(pt["col"]), int(pt["row"]))
                pt_arr2 = (int(pt2["col"]), int(pt2["row"]))
                cv2.arrowedLine(bg, pt_arr, pt_arr2, PALETTE["arrow"], 2, tipLength=0.3)

    # Draw junction nodes
    for node_id, (x, y) in node_positions.items():
        ntype = node_types.get(node_id, "unknown")
        if ntype == "junction" or ntype.startswith("junction_"):
            subtype = ntype.replace("junction_", "").replace("geo_junction_", "")
            color = PALETTE.get(f"junction_{subtype}", PALETTE["junction_L"])
        elif ntype == "terminal":
            color = PALETTE["terminal"]
        else:
            continue
        cx, cy = int(x), int(y)
        cv2.circle(bg, (cx, cy), radius=5, color=color, thickness=-1)
        cv2.circle(bg, (cx, cy), radius=7, color=color, thickness=2)
        if show_labels:
            cv2.putText(bg, node_id[:12], (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Draw text attachments
    for ta in text_attachments:
        pos = ta.get("position", ta.get("polyline", [{}])[0] if ta.get("polyline") else {})
        if "col" in pos and "row" in pos:
            cv2.putText(bg, f"'{ta.get('text', '')[:8]}'", (int(pos["col"]), int(pos["row"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, PALETTE["text"], 1)

    # Draw instrument tags
    for it in instrument_tags:
        pos = it.get("position", {})
        if "col" in pos and "row" in pos:
            tag_text = it.get("instrument_tag_id", it.get("tag_id", ""))[:10]
            cv2.putText(bg, f"[{tag_text}]", (int(pos["col"]), int(pos["row"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 100, 255), 1)

    return bg


def main():
    parser = argparse.ArgumentParser(description="Phase 3 visual spot-check overlay")
    parser.add_argument("--image", required=True, help="Original P&ID image path")
    parser.add_argument("--phase3-dir", required=True, help="Phase 3 output directory")
    parser.add_argument("--out", default=None, help="Output image path (default: <phase3-dir>/phase3_graph_overlay.png)")
    parser.add_argument("--compare-rect", default=None, help="Rectangular overlay directory to compare side-by-side")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    phase3_dir = Path(args.phase3_dir)

    # Load Phase 3 artifacts
    edges = []
    with open(phase3_dir / "phase3_pipe_edges.json") as f:
        edges_data = json.load(f)
        edges = edges_data.get("edges", [])

    junctions = []
    if (phase3_dir / "phase3_junctions.json").exists():
        with open(phase3_dir / "phase3_junctions.json") as f:
            junctions_data = json.load(f)
            junctions = junctions_data.get("junctions", [])

    graph = {}
    # Prefer stage12_graph.json (post-attachment) over phase3_graph.json (raw geometric)
    for fname in ["stage12_graph.json", "phase3_graph.json"]:
        if (phase3_dir / fname).exists():
            with open(phase3_dir / fname) as f:
                graph = json.load(f)
            break

    graph_summary = {}
    for fname in ["stage12_graph_summary.json", "phase3_graph_summary.json"]:
        if (phase3_dir / fname).exists():
            with open(phase3_dir / fname) as f:
                graph_summary = json.load(f)
            break

    # Build node position map from graph nodes
    node_positions = {}
    node_types = {}
    for node in graph.get("nodes", []):
        nid = node.get("id", "")
        pos = node.get("position", {})
        if "x" in pos and "y" in pos:
            node_positions[nid] = (pos["x"], pos["y"])
        elif "col" in pos and "row" in pos:
            node_positions[nid] = (pos["col"], pos["row"])
        node_types[nid] = node.get("type", "unknown")

    # Load attachments
    text_attachments = []
    for item in graph.get("nodes", []):
        if item.get("type") == "text_attachment":
            text_attachments.append(item)

    instrument_tags = []
    for item in graph.get("nodes", []):
        if item.get("type") in ("instrument_tag_attachment", "instrument_tag"):
            instrument_tags.append(item)

    # Generate overlay
    overlay = overlay_edges(
        image, edges, node_positions, node_types,
        arrows=[],  # could load from phase3_edge_direction.json
        text_attachments=text_attachments,
        instrument_tags=instrument_tags,
        show_labels=False,
    )

    out_path = args.out or str(phase3_dir / "phase3_graph_overlay.png")
    cv2.imwrite(out_path, overlay)
    print(f"Saved: {out_path}")

    # Print summary
    geo_bp = graph_summary.get("geometric_bypass", {})
    print(f"\n=== Phase 3 Summary ===")
    print(f"  Segments: {geo_bp.get('segment_count', 'N/A')}")
    print(f"  Runs: {geo_bp.get('run_count', 'N/A')}")
    print(f"  Junctions: {geo_bp.get('junction_count', 'N/A')}")
    print(f"  Nodes: {graph_summary.get('node_count', 'N/A')}")
    print(f"  Edges: {graph_summary.get('edge_count', 'N/A')}")
    print(f"  Components: {graph_summary.get('connected_component_count', 'N/A')}")
    print(f"  Unresolved: {graph_summary.get('unresolved_junction_count', 'N/A')}")


if __name__ == "__main__":
    main()
