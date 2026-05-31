#!/usr/bin/env python3
"""
patchify.py
- Takes a full-plan image + full-plan graph_v1 JSON (global coordinates)
- Tiles the image with SAHI-compatible tiling
- Emits per-tile images + per-tile patch graphs (local coordinates)
- Adds border nodes + border edges for connections that exit the tile

Assumptions:
- Full-plan graph is already clean: edges connect only existing nodes; no self-loops.
- Graph uses bbox xywh in global coords.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import cv2


# -------------------------
# Utilities
# -------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False,
                 indent=2), encoding="utf-8")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_xywh_to_xyxy(b: Dict[str, float]) -> Tuple[float, float, float, float]:
    x1 = b["x"]
    y1 = b["y"]
    x2 = b["x"] + b["w"]
    y2 = b["y"] + b["h"]
    return x1, y1, x2, y2


def bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Dict[str, float]:
    return {"x": x1, "y": y1, "w": max(0.0, x2 - x1), "h": max(0.0, y2 - y1)}


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def bbox_intersects_tile(bxyxy: Tuple[float, float, float, float],
                         tile_xyxy: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bxyxy
    tx1, ty1, tx2, ty2 = tile_xyxy
    return not (x2 <= tx1 or x1 >= tx2 or y2 <= ty1 or y1 >= ty2)


def project_bbox_to_tile_xywh(b: Dict[str, float], ox: int, oy: int) -> Dict[str, float]:
    # local bbox = global bbox - offset
    return {"x": b["x"] - ox, "y": b["y"] - oy, "w": b["w"], "h": b["h"]}


def node_center(b: Dict[str, float]) -> Tuple[float, float]:
    return (b["x"] + 0.5 * b["w"], b["y"] + 0.5 * b["h"])


# -------------------------
# Tiling (SAHI-like)
# -------------------------

@dataclass(frozen=True)
class Tile:
    tile_id: str
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    overlap_x: int
    overlap_y: int


def generate_tiles(img_w: int, img_h: int, tile_w: int, tile_h: int,
                   overlap_ratio: float) -> List[Tile]:
    """
    SAHI tiling concept: fixed tile size + overlap.
    stride = tile_size - overlap
    overlap = round(tile_size * overlap_ratio)
    """
    overlap_x = int(round(tile_w * overlap_ratio))
    overlap_y = int(round(tile_h * overlap_ratio))
    stride_x = max(1, tile_w - overlap_x)
    stride_y = max(1, tile_h - overlap_y)

    tiles: List[Tile] = []
    row = 0
    y = 0
    while y < img_h:
        col = 0
        x = 0
        y2 = min(y + tile_h, img_h)
        y1 = max(0, y2 - tile_h)  # keep size stable near bottom
        while x < img_w:
            x2 = min(x + tile_w, img_w)
            x1 = max(0, x2 - tile_w)  # keep size stable near right
            tile_id = f"tile_r{row:03d}_c{col:03d}"
            tiles.append(Tile(tile_id, row, col, x1, y1,
                         tile_w, tile_h, overlap_x, overlap_y))
            col += 1
            x += stride_x
        row += 1
        y += stride_y
    return tiles


# -------------------------
# Border node creation
# -------------------------

def segment_intersects_tile_border(p1: Tuple[float, float], p2: Tuple[float, float],
                                   tile_xyxy: Tuple[int, int, int, int]) -> Optional[Tuple[float, float]]:
    """
    Return one intersection point where segment crosses tile border, if it exits.
    This is a pragmatic clip: if one endpoint inside and one outside => compute intersection.
    """
    tx1, ty1, tx2, ty2 = tile_xyxy

    def inside(p: Tuple[float, float]) -> bool:
        return (tx1 <= p[0] <= tx2) and (ty1 <= p[1] <= ty2)

    in1, in2 = inside(p1), inside(p2)
    if in1 == in2:
        return None

    # Liang-Barsky line clipping to rectangle
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1

    p = [-dx, dx, -dy, dy]
    q = [x1 - tx1, tx2 - x1, y1 - ty1, ty2 - y1]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None
        else:
            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
    if u1 > u2:
        return None

    # Choose the boundary crossing point (the one closer to the inside point)
    u = u1 if in1 else u2
    ix, iy = x1 + u * dx, y1 + u * dy
    ix = clamp(ix, tx1, tx2)
    iy = clamp(iy, ty1, ty2)
    return (ix, iy)


def make_border_node(node_id: str, ix: float, iy: float,
                     box_size: int, ox: int, oy: int,
                     annotated_by: str) -> dict:
    # border node bbox stored in *local* coordinates in patch graph
    half = box_size / 2.0
    local_x = ix - ox - half
    local_y = iy - oy - half
    return {
        "id": node_id,
        "type": "border",
        "bbox": {"x": float(local_x), "y": float(local_y), "w": float(box_size), "h": float(box_size)},
        "confidence": 1.0,
        "text": {"raw": "", "normalized": "", "confidence": 0.0},
        "role": {"is_symbol": False, "is_topology": True},
        "provenance": {
            "annotated_by": annotated_by,
            "annotated_at": now_iso(),
            "source": "auto",
            "notes": "Auto-generated border node at tile boundary intersection."
        },
        "geometry": {"center": {"x": float(ix - ox), "y": float(iy - oy)}},
        "patch_link": {
            "global_bbox_xywh": {"x": float(ix - half), "y": float(iy - half), "w": float(box_size), "h": float(box_size)},
            "tile_id": ""
        },
        "tags": {"pid_tag": "", "line_tag": "", "service": ""}
    }


# -------------------------
# Patchification
# -------------------------

def patchify(
    image_path: Path,
    graph_path: Path,
    out_dir: Path,
    tile_w: int,
    tile_h: int,
    overlap: float,
    border_box: int,
    annotated_by: str,
    min_iou_keep: float = 0.0
) -> None:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    img_h, img_w = img.shape[:2]

    g = load_json(graph_path)
    doc = g.get("document", {})
    doc_id = doc.get("doc_id") or image_path.stem

    nodes = g.get("nodes", [])
    edges = g.get("edges", [])

    # Build node lookup (global)
    node_by_id: Dict[str, dict] = {n["id"]: n for n in nodes}
    center_by_id: Dict[str, Tuple[float, float]] = {
        nid: node_center(node_by_id[nid]["bbox"]) for nid in node_by_id
    }

    tiles = generate_tiles(img_w, img_h, tile_w, tile_h, overlap)

    base_out = out_dir / doc_id
    img_out_dir = base_out / "images"
    patch_out_dir = base_out / "graphs_patch"
    ensure_dir(img_out_dir)
    ensure_dir(patch_out_dir)

    for t in tiles:
        ox, oy = t.x, t.y
        tile_xyxy = (t.x, t.y, t.x + t.w, t.y + t.h)

        # crop tile image
        tile_img = img[oy:oy + t.h, ox:ox + t.w].copy()
        tile_img_path = img_out_dir / f"{t.tile_id}.png"
        cv2.imwrite(str(tile_img_path), tile_img)

        # select nodes that intersect tile
        kept_node_ids: List[str] = []
        patch_nodes: List[dict] = []

        for n in nodes:
            bxyxy = bbox_xywh_to_xyxy(n["bbox"])
            if bbox_intersects_tile(bxyxy, tile_xyxy):
                # optionally enforce some minimum overlap
                if min_iou_keep > 0:
                    # compute IoU with tile box to ensure non-trivial intersection
                    tile_box = (tile_xyxy[0], tile_xyxy[1],
                                tile_xyxy[2], tile_xyxy[3])
                    if iou_xyxy(bxyxy, tile_box) < min_iou_keep:
                        continue

                nn = json.loads(json.dumps(n))  # deep copy
                nn["bbox"] = project_bbox_to_tile_xywh(n["bbox"], ox, oy)

                # patch_link
                nn.setdefault("patch_link", {})
                nn["patch_link"]["global_bbox_xywh"] = n["bbox"]
                nn["patch_link"]["tile_id"] = t.tile_id

                patch_nodes.append(nn)
                kept_node_ids.append(n["id"])

        kept_set = set(kept_node_ids)

        # create patch edges
        patch_edges: List[dict] = []
        # quantized intersection -> border node id
        border_nodes: Dict[Tuple[int, int], str] = {}
        new_border_nodes: List[dict] = []

        def get_or_create_border(ix: float, iy: float) -> str:
            # quantize to reduce duplicates
            key = (int(round(ix)), int(round(iy)))
            if key in border_nodes:
                return border_nodes[key]
            bid = f"B_{t.tile_id}_{len(border_nodes):04d}"
            border_nodes[key] = bid
            bn = make_border_node(bid, ix, iy, border_box,
                                  ox, oy, annotated_by)
            bn["patch_link"]["tile_id"] = t.tile_id
            new_border_nodes.append(bn)
            return bid

        for e in edges:
            src = e["src"]
            dst = e["dst"]
            if src not in node_by_id or dst not in node_by_id:
                continue

            p1 = center_by_id[src]
            p2 = center_by_id[dst]

            src_in = src in kept_set
            dst_in = dst in kept_set

            if src_in and dst_in:
                ee = json.loads(json.dumps(e))
                ee["provenance"] = ee.get("provenance") or {}
                patch_edges.append(ee)
                continue

            # If one endpoint inside and the other outside, create a border node and connect.
            if src_in != dst_in:
                ixiy = segment_intersects_tile_border(p1, p2, tile_xyxy)
                if ixiy is None:
                    continue
                ix, iy = ixiy
                b_id = get_or_create_border(ix, iy)

                inside_id = src if src_in else dst
                ee = {
                    "id": f"E_{t.tile_id}_{len(patch_edges):06d}",
                    "src": inside_id,
                    "dst": b_id,
                    "type": e.get("type", "solid"),
                    "confidence": 1.0,
                    "directed": bool(e.get("directed", False)),
                    "provenance": {
                        "annotated_by": annotated_by,
                        "annotated_at": now_iso(),
                        "source": "auto",
                        "notes": f"Auto-generated border edge from {inside_id} to border for original edge {e.get('id', '')}."
                    }
                }
                patch_edges.append(ee)

        # append border nodes
        patch_nodes.extend(new_border_nodes)

        # finalize patch graph
        patch_graph = {
            "schema_version": "graph_v1",
            "description": "Patch graph exported by patchify.py",
            "coordinate_system": g.get("coordinate_system", {}),
            "document": {
                "doc_id": doc_id,
                "source": doc.get("source", {}),
                "image": {"width": t.w, "height": t.h}
            },
            "tiling": {
                "is_patch": True,
                "tile_engine": "sahi",
                "tile": {
                    "tile_id": t.tile_id,
                    "tile_row": t.row,
                    "tile_col": t.col,
                    "tile_width": t.w,
                    "tile_height": t.h,
                    "overlap_x": t.overlap_x,
                    "overlap_y": t.overlap_y,
                    "offset_x": ox,
                    "offset_y": oy
                },
                "global_image": {"width": img_w, "height": img_h}
            },
            "classes": g.get("classes", {}),
            "nodes": patch_nodes,
            "edges": patch_edges,
            "constraints": g.get("constraints", {}),
            "recommended_defaults": g.get("recommended_defaults", {})
        }

        patch_graph_path = patch_out_dir / f"{t.tile_id}.json"
        save_json(patch_graph_path, patch_graph)

    # write a manifest
    manifest = {
        "doc_id": doc_id,
        "image": {"path": str(image_path), "width": img_w, "height": img_h},
        "graph_full": str(graph_path),
        "tiling": {"tile_w": tile_w, "tile_h": tile_h, "overlap": overlap},
        "outputs": {
            "tiles_dir": str(img_out_dir),
            "patch_graphs_dir": str(patch_out_dir),
            "num_tiles": len(tiles)
        },
        "created_at": now_iso()
    }
    save_json(base_out / "manifest.json", manifest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path,
                    help="Full-plan image (png/jpg).")
    ap.add_argument("--graph", required=True, type=Path,
                    help="Full-plan graph_v1 JSON (global coords).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory.")
    ap.add_argument("--tile-w", type=int, default=1500)
    ap.add_argument("--tile-h", type=int, default=1500)
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Overlap ratio (0..0.9).")
    ap.add_argument("--border-box", type=int, default=10,
                    help="Border node bbox size in pixels (local).")
    ap.add_argument("--annotated-by", type=str, default="patchify.py")
    ap.add_argument("--min-iou-keep", type=float, default=0.0,
                    help="Optional: keep nodes only if IoU(bbox,tile) >= value.")
    args = ap.parse_args()

    patchify(
        image_path=args.image,
        graph_path=args.graph,
        out_dir=args.out,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        overlap=args.overlap,
        border_box=args.border_box,
        annotated_by=args.annotated_by,
        min_iou_keep=args.min_iou_keep
    )


if __name__ == "__main__":
    main()
