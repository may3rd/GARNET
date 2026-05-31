#!/usr/bin/env python3
"""
merge_predictions.py

Merge patch-level prediction graphs (Relationformer outputs) into a full-plan graph.

Inputs
- manifest.json created by patchify.py (recommended) OR explicit global size
- directory of patch prediction JSONs, one per tile (same tile_id naming)
- optional: original tiling metadata (offsets) from manifest or per-patch file

Core steps (aligned to the paper)
1) Map patch node bboxes to GLOBAL coords using tile offsets
2) Border confidence decay for nodes near patch boundaries
3) Filter low-confidence nodes/edges
4) Node merging:
   - NMS (high IoU) to remove duplicates
   - WBF (lower IoU) to fuse boxes and confidence
5) Build mapping from patch nodes -> merged nodes (IoU assignment)
6) Map edges to merged nodes, dedupe, drop self-loops, drop isolated junk

Notes
- This script assumes prediction JSONs use graph_v1-like structure:
  nodes: [{id,type,bbox,confidence,...}]
  edges: [{id,src,dst,type,confidence,...}]
- It will also work if fields exist but have extra keys (ignored).

"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, DefaultDict
from collections import defaultdict

# -------------------------
# IO helpers
# -------------------------


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False,
                 indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# -------------------------
# Geometry helpers
# -------------------------


def bbox_xywh_to_xyxy(b: Dict[str, float]) -> Tuple[float, float, float, float]:
    x1 = float(b["x"])
    y1 = float(b["y"])
    x2 = x1 + float(b["w"])
    y2 = y1 + float(b["h"])
    return x1, y1, x2, y2


def bbox_xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Dict[str, float]:
    return {"x": float(x1), "y": float(y1), "w": float(max(0.0, x2 - x1)), "h": float(max(0.0, y2 - y1))}


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


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def box_center_xyxy(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

# -------------------------
# Merge primitives
# -------------------------


@dataclass
class NodePred:
    src_node_id: str          # original node id (patch-local)
    tile_id: str
    node_type: str
    bbox_xyxy: Tuple[float, float, float, float]  # GLOBAL coords
    conf: float


@dataclass
class EdgePred:
    tile_id: str
    src_node_id: str
    dst_node_id: str
    edge_type: str
    conf: float


@dataclass
class TileMeta:
    tile_id: str
    offset_x: int
    offset_y: int
    tile_w: int
    tile_h: int


def confidence_decay_near_border(conf: float, bbox_xyxy: Tuple[float, float, float, float],
                                 tile: TileMeta, alpha: float = 0.4) -> float:
    """
    Paper-style idea: reduce confidence for boxes near patch borders to avoid duplicates.
    We approximate:
      d = min distance from bbox center to tile border in LOCAL coords
      decay = alpha * exp(-3 * |d^2 / S|)
      conf' = conf - decay
    """
    cx, cy = box_center_xyxy(bbox_xyxy)
    # convert center to LOCAL coords
    lx = cx - tile.offset_x
    ly = cy - tile.offset_y
    d_left = lx
    d_right = tile.tile_w - lx
    d_top = ly
    d_bottom = tile.tile_h - ly
    d = min(d_left, d_right, d_top, d_bottom)
    S = float(tile.tile_w * tile.tile_h)
    if S <= 0:
        return conf
    decay = alpha * math.exp(-3.0 * abs((d * d) / S))
    return float(clamp(conf - decay, 0.0, 1.0))


def nms_indices(boxes: List[Tuple[float, float, float, float]],
                scores: List[float],
                iou_thresh: float) -> List[int]:
    """
    Standard greedy NMS. Returns kept indices.
    """
    idxs = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept: List[int] = []
    while idxs:
        i = idxs.pop(0)
        kept.append(i)
        rest = []
        for j in idxs:
            if iou_xyxy(boxes[i], boxes[j]) < iou_thresh:
                rest.append(j)
        idxs = rest
    return kept


def weighted_box_fusion(cluster: List[int],
                        boxes: List[Tuple[float, float, float, float]],
                        scores: List[float]) -> Tuple[Tuple[float, float, float, float], float]:
    """
    Simple WBF: weighted average of x1,y1,x2,y2 by score.
    Output score = max score in cluster.
    """
    if not cluster:
        raise ValueError("Empty cluster in WBF")
    wsum = 0.0
    x1 = y1 = x2 = y2 = 0.0
    maxs = 0.0
    for idx in cluster:
        w = float(scores[idx])
        bx = boxes[idx]
        x1 += bx[0] * w
        y1 += bx[1] * w
        x2 += bx[2] * w
        y2 += bx[3] * w
        wsum += w
        maxs = max(maxs, w)
    if wsum <= 0:
        # fallback: take best
        best = max(cluster, key=lambda i: scores[i])
        return boxes[best], float(scores[best])
    fused = (x1 / wsum, y1 / wsum, x2 / wsum, y2 / wsum)
    return fused, float(maxs)


def wbf_merge(boxes: List[Tuple[float, float, float, float]],
              scores: List[float],
              iou_thresh: float) -> Tuple[List[Tuple[float, float, float, float]], List[float], List[List[int]]]:
    """
    Cluster boxes by IoU >= iou_thresh, then fuse each cluster with WBF.
    Returns fused_boxes, fused_scores, clusters(original indices).
    """
    remaining = set(range(len(boxes)))
    clusters: List[List[int]] = []
    while remaining:
        seed = max(remaining, key=lambda i: scores[i])
        remaining.remove(seed)
        cluster = [seed]
        to_check = list(remaining)
        for j in to_check:
            if iou_xyxy(boxes[seed], boxes[j]) >= iou_thresh:
                cluster.append(j)
                remaining.remove(j)
        clusters.append(cluster)

    fused_boxes: List[Tuple[float, float, float, float]] = []
    fused_scores: List[float] = []
    for cl in clusters:
        fb, fs = weighted_box_fusion(cl, boxes, scores)
        fused_boxes.append(fb)
        fused_scores.append(fs)
    return fused_boxes, fused_scores, clusters

# -------------------------
# Core merge logic
# -------------------------


def read_tile_meta_from_patch_graph(g: dict) -> TileMeta:
    t = g["tiling"]["tile"]
    return TileMeta(
        tile_id=str(t["tile_id"]),
        offset_x=int(t["offset_x"]),
        offset_y=int(t["offset_y"]),
        tile_w=int(t["tile_width"]),
        tile_h=int(t["tile_height"]),
    )


def load_predictions(pred_dir: Path) -> Tuple[Dict[str, dict], Dict[str, TileMeta]]:
    """
    Load all patch prediction graphs in pred_dir.
    Return: graphs_by_tile_id, tile_meta_by_tile_id
    """
    graphs: Dict[str, dict] = {}
    metas: Dict[str, TileMeta] = {}
    for p in sorted(pred_dir.glob("*.json")):
        g = load_json(p)
        tile_id = g.get("tiling", {}).get("tile", {}).get("tile_id")
        if not tile_id:
            # fallback: filename stem
            tile_id = p.stem
        tile_id = str(tile_id)
        graphs[tile_id] = g
        if "tiling" in g and "tile" in g["tiling"]:
            metas[tile_id] = read_tile_meta_from_patch_graph(g)
    return graphs, metas


def merge_predictions(
    manifest_path: Optional[Path],
    pred_dir: Path,
    out_path: Path,
    global_w: Optional[int],
    global_h: Optional[int],
    min_node_conf: float,
    min_edge_conf: float,
    nms_iou: float,
    wbf_iou: float,
    decay_alpha: float,
    assign_iou: float,
) -> None:
    manifest = load_json(manifest_path) if manifest_path else None

    graphs_by_tile, metas = load_predictions(pred_dir)

    if manifest:
        doc_id = manifest.get("doc_id", "doc")
        img_meta = manifest.get("image", {})
        GW = int(img_meta.get("width"))
        GH = int(img_meta.get("height"))
    else:
        doc_id = "doc"
        if global_w is None or global_h is None:
            raise ValueError(
                "If no manifest is provided, --global-w and --global-h are required.")
        GW, GH = int(global_w), int(global_h)

    # Collect all nodes/edges in global coords
    all_nodes: List[NodePred] = []
    all_edges: List[EdgePred] = []

    for tile_id, g in graphs_by_tile.items():
        if tile_id not in metas:
            # If missing tile meta, try to read it anyway
            if "tiling" in g and "tile" in g["tiling"]:
                metas[tile_id] = read_tile_meta_from_patch_graph(g)
            else:
                raise ValueError(
                    f"Missing tiling metadata for tile_id={tile_id}")

        tm = metas[tile_id]

        # nodes
        for n in g.get("nodes", []):
            b = n.get("bbox")
            if not b:
                continue
            # local -> global
            x1l, y1l, x2l, y2l = bbox_xywh_to_xyxy(b)
            x1g = x1l + tm.offset_x
            y1g = y1l + tm.offset_y
            x2g = x2l + tm.offset_x
            y2g = y2l + tm.offset_y

            conf = float(n.get("confidence", 0.0))
            # apply confidence decay near border
            conf = confidence_decay_near_border(
                conf, (x1g, y1g, x2g, y2g), tm, alpha=decay_alpha)

            if conf < min_node_conf:
                continue

            all_nodes.append(NodePred(
                src_node_id=str(n.get("id")),
                tile_id=tile_id,
                node_type=str(n.get("type", "equipment_general")),
                bbox_xyxy=(x1g, y1g, x2g, y2g),
                conf=conf
            ))

        # edges
        for e in g.get("edges", []):
            conf = float(e.get("confidence", 0.0))
            if conf < min_edge_conf:
                continue
            all_edges.append(EdgePred(
                tile_id=tile_id,
                src_node_id=str(e.get("src")),
                dst_node_id=str(e.get("dst")),
                edge_type=str(e.get("type", "solid")),
                conf=conf
            ))

    # Merge nodes per class (type) to avoid fusing different classes
    merged_nodes: List[dict] = []
    merged_boxes: List[Tuple[float, float, float, float]] = []
    merged_types: List[str] = []
    merged_scores: List[float] = []

    # Keep track of which original nodes contributed to which merged node (for diagnostics)
    # [(tile_id, src_node_id), ...]
    contributors: List[List[Tuple[str, str]]] = []

    nodes_by_type: DefaultDict[str, List[int]] = defaultdict(list)
    for idx, nd in enumerate(all_nodes):
        nodes_by_type[nd.node_type].append(idx)

    for node_type, idxs in nodes_by_type.items():
        boxes = [all_nodes[i].bbox_xyxy for i in idxs]
        scores = [all_nodes[i].conf for i in idxs]

        # 1) NMS (high IoU) to remove obvious duplicates
        keep_local = nms_indices(boxes, scores, iou_thresh=nms_iou)
        kept_idxs = [idxs[i] for i in keep_local]
        boxes_k = [all_nodes[i].bbox_xyxy for i in kept_idxs]
        scores_k = [all_nodes[i].conf for i in kept_idxs]

        # 2) WBF (lower IoU) to fuse remaining close boxes
        fused_boxes, fused_scores, clusters = wbf_merge(
            boxes_k, scores_k, iou_thresh=wbf_iou)

        for fb, fs, cl in zip(fused_boxes, fused_scores, clusters):
            mid = f"N{len(merged_nodes):06d}"
            merged_nodes.append({
                "id": mid,
                "type": node_type,
                "bbox": bbox_xyxy_to_xywh(*fb),
                "confidence": float(fs),
                "text": {"raw": "", "normalized": "", "confidence": 0.0},
                "role": {
                    "is_symbol": node_type not in {"crossing", "ankle", "border"},
                    "is_topology": node_type in {"crossing", "ankle", "border"}
                },
                "provenance": {
                    "annotated_by": "merge_predictions.py",
                    "annotated_at": now_iso(),
                    "source": "auto",
                    "notes": f"Merged from {len(cl)} candidates (NMS IoU={nms_iou}, WBF IoU={wbf_iou})."
                },
                "geometry": {
                    "center": {"x": float(0.5*(fb[0]+fb[2])), "y": float(0.5*(fb[1]+fb[3]))}
                },
                "patch_link": {"global_bbox_xywh": bbox_xyxy_to_xywh(*fb), "tile_id": ""},
                "tags": {"pid_tag": "", "line_tag": "", "service": ""}
            })
            merged_boxes.append(fb)
            merged_types.append(node_type)
            merged_scores.append(float(fs))

            # contributors
            contrib: List[Tuple[str, str]] = []
            for k in cl:
                src_global_idx = kept_idxs[k]
                contrib.append(
                    (all_nodes[src_global_idx].tile_id, all_nodes[src_global_idx].src_node_id))
            contributors.append(contrib)

    # Build mapping from patch node -> merged node using IoU assignment within same type
    # Patch node keys: (tile_id, src_node_id)
    patch_to_merged: Dict[Tuple[str, str], str] = {}

    # Index merged nodes by type for speed
    merged_idx_by_type: DefaultDict[str, List[int]] = defaultdict(list)
    for mi, t in enumerate(merged_types):
        merged_idx_by_type[t].append(mi)

    for nd in all_nodes:
        cand = merged_idx_by_type.get(nd.node_type, [])
        best_iou = -1.0
        best_mi = None
        for mi in cand:
            iou = iou_xyxy(nd.bbox_xyxy, merged_boxes[mi])
            if iou > best_iou:
                best_iou = iou
                best_mi = mi
        if best_mi is not None and best_iou >= assign_iou:
            patch_to_merged[(nd.tile_id, nd.src_node_id)
                            ] = merged_nodes[best_mi]["id"]

    # Merge edges: map endpoints to merged node ids, dedupe by undirected pair + edge type
    # If directed edges are later introduced, update the key accordingly.
    edge_best: Dict[Tuple[str, str, str], float] = {}  # (u,v,type) -> conf
    edge_src_tiles: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    for e in all_edges:
        u = patch_to_merged.get((e.tile_id, e.src_node_id))
        v = patch_to_merged.get((e.tile_id, e.dst_node_id))
        if not u or not v:
            continue
        if u == v:
            continue

        # normalize undirected pair
        a, b = (u, v) if u < v else (v, u)
        key = (a, b, e.edge_type)
        prev = edge_best.get(key)
        if prev is None or e.conf > prev:
            edge_best[key] = float(e.conf)
        edge_src_tiles[key].append(e.tile_id)

    merged_edges: List[dict] = []
    for (a, b, et), conf in edge_best.items():
        merged_edges.append({
            "id": f"E{len(merged_edges):06d}",
            "src": a,
            "dst": b,
            "type": et,
            "confidence": float(conf),
            "directed": False,
            "provenance": {
                "annotated_by": "merge_predictions.py",
                "annotated_at": now_iso(),
                "source": "auto",
                "notes": f"Merged from tiles: {sorted(set(edge_src_tiles[(a, b, et)]))[:10]} (truncated)."
            }
        })

    # Optional cleanup: remove isolated non-border nodes (common false positives)
    deg: DefaultDict[str, int] = defaultdict(int)
    for e in merged_edges:
        deg[e["src"]] += 1
        deg[e["dst"]] += 1

    cleaned_nodes: List[dict] = []
    kept_node_ids: set[str] = set()
    for n in merged_nodes:
        nid = n["id"]
        ntype = n.get("type")
        if deg.get(nid, 0) == 0 and ntype not in {"border"}:
            continue
        cleaned_nodes.append(n)
        kept_node_ids.add(nid)

    cleaned_edges: List[dict] = []
    for e in merged_edges:
        if e["src"] in kept_node_ids and e["dst"] in kept_node_ids:
            cleaned_edges.append(e)

    # Output graph
    out_graph = {
        "schema_version": "graph_v1",
        "description": "Merged full-plan prediction graph",
        "coordinate_system": {
            "image_origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "units": "pixels",
            "bbox_format": "xywh"
        },
        "document": {
            "doc_id": doc_id,
            "source": {"file_name": "", "file_type": "png", "page_index": 0, "render_dpi": 0, "notes": ""},
            "image": {"width": GW, "height": GH}
        },
        "tiling": {
            "is_patch": False,
            "tile_engine": "sahi",
            "tile": {},
            "global_image": {"width": GW, "height": GH}
        },
        "classes": {
            "node_types": sorted(set(merged_types)),
            "edge_types": sorted(set(e["type"] for e in cleaned_edges)) if cleaned_edges else []
        },
        "nodes": cleaned_nodes,
        "edges": cleaned_edges,
        "provenance": {
            "created_by": "merge_predictions.py",
            "created_at": now_iso(),
            "inputs": {
                "pred_dir": str(pred_dir),
                "manifest": str(manifest_path) if manifest_path else "",
            },
            "params": {
                "min_node_conf": min_node_conf,
                "min_edge_conf": min_edge_conf,
                "nms_iou": nms_iou,
                "wbf_iou": wbf_iou,
                "decay_alpha": decay_alpha,
                "assign_iou": assign_iou
            }
        }
    }

    save_json(out_path, out_graph)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path,
                    help="Directory of patch prediction JSONs (*.json).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output merged full-plan graph JSON.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest.json from patchify.py (recommended).")
    ap.add_argument("--global-w", type=int, default=None,
                    help="Global width if no manifest.")
    ap.add_argument("--global-h", type=int, default=None,
                    help="Global height if no manifest.")

    ap.add_argument("--min-node-conf", type=float, default=0.20)
    ap.add_argument("--min-edge-conf", type=float, default=0.20)

    ap.add_argument("--nms-iou", type=float, default=0.85,
                    help="High IoU for duplicate suppression.")
    ap.add_argument("--wbf-iou", type=float, default=0.55,
                    help="Lower IoU for WBF fusion.")

    ap.add_argument("--decay-alpha", type=float, default=0.40,
                    help="Border confidence decay magnitude.")
    ap.add_argument("--assign-iou", type=float, default=0.30,
                    help="IoU threshold to map patch nodes to merged nodes.")

    args = ap.parse_args()

    merge_predictions(
        manifest_path=args.manifest,
        pred_dir=args.pred_dir,
        out_path=args.out,
        global_w=args.global_w,
        global_h=args.global_h,
        min_node_conf=args.min_node_conf,
        min_edge_conf=args.min_edge_conf,
        nms_iou=args.nms_iou,
        wbf_iou=args.wbf_iou,
        decay_alpha=args.decay_alpha,
        assign_iou=args.assign_iou,
    )


if __name__ == "__main__":
    main()
