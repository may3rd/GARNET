"""
Recreated P&ID pipeline orchestrator (from scratch) with staged execution.

Stages (iterative):
  1) Ingest: load image, detections (COCO), OCR JSON
  2) Preprocess: grayscale + binarize (adaptive), de-skew (todo)
  3) Symbols/Text: import detections, normalize labels, basic overlay
  4) Linework: binarize → skeletonize; extract endpoints/junctions
  5) Graph: build nodes/edges; attach symbols/text (minimal)
  6) From/To: emit endpoint pairs per segment/line
  7) Export: GraphML, CSV/JSON, DEXPI XML (minimal)

This file is intentionally compact and dependency-tolerant:
  - Uses OpenCV if available; falls back to PIL/numpy/skimage where possible.
  - Saves intermediate artifacts to `output/` for quick inspection.
"""

from __future__ import annotations

import os
import json
import time
import logging
from dataclasses import dataclass, field
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import pandas as pd

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # Optional

try:
    from skimage.morphology import skeletonize
except Exception:  # pragma: no cover
    skeletonize = None  # Optional

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover
    Image = None
    ImageDraw = None

from garnet.dexpi_exporter import export_dexpi


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pid")


DEFAULT_OUT = Path("output")
DEFAULT_OUT.mkdir(parents=True, exist_ok=True)

GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
ORANGE_COLOR = (0, 165, 255)
YELLOW_COLOR = (0, 255, 255)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)


class NodeType(Enum):
    VALVE = "valve"
    INSTRUMENT = "instrument"
    EQUIPMENT = "equipment"
    JUNCTION = "junction"
    ENDPOINT = "endpoint"
    OFFPAGE = "offpage"
    PORT = "port"
    UNKNOWN = "unknown"


@dataclass
class Symbol:
    id: int
    type: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    center: Tuple[float, float]
    text: Optional[str] = None


@dataclass
class TextItem:
    id: int
    text: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    center: Tuple[float, float]


@dataclass
class Node:
    id: int
    position: Tuple[float, float]
    type: NodeType
    label: Optional[str] = None
    symbol_ids: List[int] = field(default_factory=list)
    port_of: Optional[int] = None  # if this node is a connection port of another node (symbol)


@dataclass
class Edge:
    id: int
    source: int
    target: int
    path: List[Tuple[float, float]]
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    device: str = "auto"  # auto|cuda|mps|cpu (placeholder; affects detectors later)
    dpi: int = 400
    canny_low: int = 50
    canny_high: int = 150
    binarize_block: int = 21
    binarize_c: int = 5
    min_segment_len: int = 5
    merge_node_dist: int = 10
    deskew: bool = True
    max_deskew_deg: float = 3.0
    morph_kernel: int = 3
    close_hv: bool = True
    use_open: bool = False
    min_blob_area: int = 8
    # Stage 3 (detections/text)
    default_conf_thresh: float = 0.25
    class_conf_thresh: Dict[str, float] = field(default_factory=dict)  # e.g., {"arrow":0.4}
    # Map detection class name → role used for node typing
    class_role_map: Dict[str, str] = field(default_factory=lambda: {
        "utility connection": "offpage",
        "page connection": "offpage",
        "off-page": "offpage",
        "off page": "offpage",
        "connection": "offpage",
    })
    text_assoc_multiplier: float = 2.0  # bbox diagonal * multiplier
    # Stage 4 (skeleton masking)
    mask_symbols: bool = True
    mask_text: bool = True
    mask_inflate: int = 3
    # Stage 5 (graph snapping)
    connect_symbol_max_dist: float = 25.0  # legacy
    connect_radius: float = 100.0          # search radius for symbol→skeleton
    connect_symbol_max_links: int = 4      # default ports per inline symbol
    angle_sep_min_deg: float = 25.0        # minimal angular separation between picked links
    continue_straight_deg: float = 90.0    # allow tracing through junction if near-straight
    trace_all_after_stage4: bool = True    # also trace loops/segments with no endpoints
    # Ports per node type for symbol linking and bridging
    ports_per_type: Dict[str, int] = field(default_factory=lambda: {
        "arrow": 2,
        "valve": 2,
        "instrument": 2,
        "equipment": 2,
        "offpage": 1,
        "reducer": 2,
        "unknown": 1,
    })
    bridge_through_symbol: bool = True  # create A—symbol—B edge path for 2‑port inline symbols
    # Control which classes/roles create pipeline links
    link_exclude_roles: List[str] = field(default_factory=lambda: [
        "instrument",
        "instrument tag",
        "instrument logic",
        "instrument dcs",
    ])  # roles or class-like names treated as non-inline
    link_exclude_classes: List[str] = field(default_factory=lambda: [
        "line number",
        "line_no",
        "line tag",
        "line number tag",
        "instrument tag",
        "instrument logic",
        "instrument dcs",
    ]) 
    link_include_classes: List[str] = field(default_factory=list)  # overrides exclusion per class name
    # Explicit connection tracing for connection-like symbols
    connection_search_radius: float = 100.0
    connection_raycast_angles: int = 36
    connection_dir_window: int = 60  # px window to estimate dominant local line direction
    # Gap bridging and coverage
    bridge_max_dist: float = 20.0
    bridge_angle_max_deg: float = 15.0
    coverage_report: bool = True
    # Hough-based recovery for long straight runs
    hough_recover: bool = True
    hough_rho: float = 1.0
    hough_theta_deg: float = 1.0
    hough_threshold: int = 120
    hough_min_line_length: int = 80
    hough_max_line_gap: int = 12
    # Valve-specific linking
    valve_link_strategy: str = "directional"  # directional|generic|hybrid
    valve_edge_offset: int = 3                # px beyond bbox to start raycast
    valve_raycast_step: int = 2               # px per raycast step
    valve_max_search: float = 120.0           # search distance along ray
    valve_symbol_link_max_dist: float = 36.0  # if no skeleton, allow symbol-to-symbol link within this distance
    # Ports placement
    ports_on_bbox_edge: bool = True           # place a port node on bbox edge for symbol links
    valve_edge_sample_step: int = 1           # px sampling step along bbox edge for valve port search
    valve_edge_outward_search: float = 100.0  # px outward search distance to find skeleton crossing edge
    valve_inward_check_px: int = 3            # px inward distance to verify line exists inside bbox
    valve_inward_min_frac: float = 0.1        # min fraction of foreground pixels along inward probe
    # Tracing options
    trace_from_connections_only: bool = False # if true, keep only pipelines connected to connection/offpage nodes
    valve_directional_exclude_classes: List[str] = field(default_factory=lambda: [
        "control valve",
        "control",
    ])
    # Template matching for valve orientation
    template_root: str = "matching_templates"
    template_use_edges: bool = True           # use Canny edges for matching
    template_min_score: float = 0.5           # minimal score to trust orientation
    # Junction validation against binary (Stage 5)
    junction_validate: bool = True
    junction_validate_rays: int = 16
    junction_validate_radius: int = 8
    junction_validate_min_hits: int = 2
    junction_validate_angle_merge_deg: float = 25.0
    junction_min_branches: int = 3
    # Port search fallback when skeleton near bbox edge is missing
    port_binary_outward_search: int = 30      # px to search along outward ray in binary image
    port_binary_min_hits: int = 2            # minimal consecutive binary hits to consider a line
    port_skeleton_snap_radius: int = 10      # px radius around binary hit to snap to skeleton


class PIDPipeline:
    def __init__(self, image_path: str, coco_path: str, ocr_path: str, out_dir: str | Path = DEFAULT_OUT,
                 cfg: PipelineConfig | None = None, arrow_coco_path: Optional[str] = None) -> None:
        self.image_path = str(image_path)
        self.coco_path = str(coco_path)
        self.ocr_path = str(ocr_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or PipelineConfig()
        self.arrow_coco_path = arrow_coco_path

        # In-memory artifacts
        self.image_bgr: Optional[np.ndarray] = None
        self.gray: Optional[np.ndarray] = None
        self.binary: Optional[np.ndarray] = None
        self.skeleton: Optional[np.ndarray] = None

        self.symbols: List[Symbol] = []
        self.texts: List[TextItem] = []
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.graph = nx.Graph()
        # Debug: symbol connection candidates recorded during Stage 5
        self._symbol_pick_debug: Dict[int, List[Tuple[float, float]]] = {}

    # ---------- Utilities ----------
    def _save_img(self, name: str, img: np.ndarray) -> None:
        path = self.out_dir / f"{name}.png"
        if cv2 is not None:
            if img.dtype == bool:
                img = (img.astype(np.uint8) * 255)
            cv2.imwrite(str(path), img)
        elif Image is not None:
            mode = "L" if img.ndim == 2 else "RGB"
            im = Image.fromarray(img if img.dtype == np.uint8 else img.astype(np.uint8), mode=mode)
            im.save(str(path))
        logger.info(f"saved {path}")

    def _save_json(self, name: str, data: Any) -> None:
        path = self.out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"saved {path}")

    # ---------- Stage 1: Ingest ----------
    def stage1_ingest(self) -> None:
        t0 = time.time()
        # Image
        if cv2 is not None:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {self.image_path}")
            self.image_bgr = img
        elif Image is not None:
            im = Image.open(self.image_path).convert("RGB")
            self.image_bgr = np.array(im)[:, :, ::-1]  # RGB->BGR
        else:
            raise RuntimeError("No image backend available (cv2 or PIL)")

        # COCO detections
        with open(self.coco_path, "r") as f:
            coco = json.load(f)
        cats = coco.get("categories", [])
        anns = coco.get("annotations", [])
        cat_map: Dict[Any, str] = {}
        for c in cats:
            cid = c.get("id")
            name = c.get("name", "unknown")
            cat_map[cid] = name
            try:
                cat_map[int(cid)] = name
            except Exception:
                pass

        # OCR (supports list or COCO-like dict with annotations)
        with open(self.ocr_path, "r") as f:
            ocr_raw = json.load(f)

        def _normalize_ocr(obj: Any) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if isinstance(obj, list):
                for it in obj:
                    if not isinstance(it, dict):
                        continue
                    # support flat or nested attributes
                    txt = it.get("text") or (it.get("attributes", {}) or {}).get("text") or ""
                    conf = it.get("confidence")
                    if conf is None:
                        conf = (it.get("attributes", {}) or {}).get("score", 1.0)
                    bbox = it.get("bbox", [0, 0, 0, 0])
                    out.append({"text": txt, "confidence": float(conf or 1.0), "bbox": bbox})
                return out
            if isinstance(obj, dict):
                anns = obj.get("annotations", [])
                for a in anns:
                    if not isinstance(a, dict):
                        continue
                    attrs = a.get("attributes", {}) or {}
                    txt = attrs.get("text") or a.get("text", "")
                    conf = attrs.get("score", a.get("score", 1.0))
                    bbox = a.get("bbox", [0, 0, 0, 0])
                    out.append({"text": txt, "confidence": float(conf or 1.0), "bbox": bbox})
                return out
            return out

        ocr = _normalize_ocr(ocr_raw)

        # Optional arrow COCO
        arrow_anns_count = 0
        if self.arrow_coco_path:
            with open(self.arrow_coco_path, "r") as f:
                arrow_coco = json.load(f)
            arrow_anns = arrow_coco.get("annotations", [])
            anns.extend(arrow_anns)
            arrow_anns_count = len(arrow_anns)
            # Merge categories
            arrow_cats = arrow_coco.get("categories", [])
            for c in arrow_cats:
                if c not in cats:
                    cats.append(c)

        summary = {
            "image": os.path.basename(self.image_path),
            "shape": list(self.image_bgr.shape),
            "detections": len(anns),
            "categories": len(cats),
            "texts": len(ocr),
            "arrow_detections": arrow_anns_count,
        }
        self._save_json("stage1_summary", summary)
        logger.info("Stage 1 done in %.2fs", time.time() - t0)

        # Persist maps for next stages
        self._coco = coco
        self._cat_map = cat_map
        self._ann = anns
        self._ocr = ocr

    # ---------- Stage 2: Preprocess ----------
    def stage2_preprocess(self) -> None:
        t0 = time.time()
        if self.image_bgr is None:
            raise RuntimeError("Run stage1_ingest first")

        if cv2 is not None:
            gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
            # Optional de-skew using Hough line orientation
            deskew_applied = False
            estimated_angle = 0.0
            if self.cfg.deskew:
                estimated_angle = self._estimate_skew_angle(gray)
                if abs(estimated_angle) <= self.cfg.max_deskew_deg and abs(estimated_angle) > 0.1:
                    self.image_bgr = self._rotate_image(self.image_bgr, estimated_angle)
                    gray = self._rotate_image(gray, estimated_angle)
                    deskew_applied = True
            # Always save a deskewed image file (even if no rotation happened)
            try:
                self._save_img("stage2_gray_deskewed", gray)
                self._save_json("stage2_deskew_meta", {"estimated_angle_deg": estimated_angle, "applied": deskew_applied})
            except Exception:
                pass
            # Adaptive threshold for varied thickness/contrast
            blk = max(3, self.cfg.binarize_block | 1)
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, self.cfg.binarize_c
            )
            self._save_img("stage2_binary_init", binary)

            # Morphological cleanup tuned to preserve thin lines
            # 1) Directional closing (horizontal then vertical) to bridge tiny gaps
            if self.cfg.close_hv:
                kh = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
                kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kh, iterations=1)
                binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kv, iterations=1)
                self._save_img("stage2_binary_closed_hv", binary)

            # 2) Optional opening (disabled by default as it can erase thin lines)
            if self.cfg.use_open:
                k = max(1, int(self.cfg.morph_kernel))
                k = k + 1 if k % 2 == 0 else k
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                self._save_img("stage2_binary_after_open", binary)

            # 3) Remove tiny specks via connected component filtering (preserves long thin lines)
            try:
                mask = (binary > 0).astype(np.uint8)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= self.cfg.min_blob_area]
                if keep:
                    keep_mask = np.isin(labels, keep).astype(np.uint8) * 255
                    binary = keep_mask
                    self._save_img("stage2_binary_ccfiltered", binary)
            except Exception as _:
                pass

            self._save_img("stage2_binary_clean", binary)
        else:
            # Fallback: simple percentile threshold
            rgb = self.image_bgr[:, :, ::-1]
            gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
            thr = np.percentile(gray, 60)
            binary = (gray < thr).astype(np.uint8) * 255

        self.gray = gray
        self.binary = binary
        self._save_img("stage2_gray", gray)
        self._save_img("stage2_binary", binary)
        logger.info("Stage 2 done in %.2fs", time.time() - t0)

    def _estimate_skew_angle(self, gray: np.ndarray) -> float:
        """Estimate dominant skew angle near 0/90 deg using Hough lines (degrees)."""
        if cv2 is None:
            return 0.0
        edges = cv2.Canny(gray, self.cfg.canny_low, self.cfg.canny_high)
        self._save_img("stage2_edges", edges)
        lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=200)
        if lines is None:
            return 0.0
        angles = []
        for rho_theta in lines[:512]:
            rho, theta = rho_theta[0]
            ang = (theta * 180.0 / np.pi) - 90.0  # make 0 ~ horizontal-ish
            # Wrap to [-90, 90]
            if ang > 90:
                ang -= 180
            if ang < -90:
                ang += 180
            # Fold near multiples of 90
            if abs(ang) > 45:
                ang = (90 - abs(ang)) * (1 if ang > 0 else -1)
            angles.append(ang)
        if not angles:
            return 0.0
        median = float(np.median(angles))
        return median

    def _rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        if cv2 is None:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        border = 255 if img.ndim == 2 else (255, 255, 255)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border) # type: ignore

    def _dominant_angle_in_window(self, cx: float, cy: float, win: int, default: Optional[float] = None) -> Optional[float]:
        """Estimate dominant line angle (radians) near (cx,cy) using PCA on binary foreground.
        Uses the unmasked binary (stage2) to capture thick line direction.
        """
        # skeleton pixels to force tracing from (e.g., connection objects), with symbol id
        force_trace_starts: List[Tuple[int, int, int]] = []
        try:
            if self.binary is None:
                return default
            y0 = max(0, int(cy - win))
            y1 = min(self.binary.shape[0], int(cy + win))
            x0 = max(0, int(cx - win))
            x1 = min(self.binary.shape[1], int(cx + win))
            patch = self.binary[y0:y1, x0:x1]
            ys, xs = np.where(patch > 0)
            if len(xs) < 20:
                return default
            # Center coordinates
            X = np.column_stack((xs.astype(float), ys.astype(float)))
            X -= X.mean(axis=0, keepdims=True)
            C = np.cov(X.T)
            vals, vecs = np.linalg.eig(C)
            i = int(np.argmax(vals))
            vx, vy = vecs[0, i], vecs[1, i]
            ang = math.atan2(vy, vx)
            return ang
        except Exception:
            return default

    def _line_rect_intersection(self, p0: Tuple[float, float], p1: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Intersect line segment p0->p1 with rectangle boundary defined by bbox (x,y,w,h).
        Returns the closest intersection point to p0 that lies on the rectangle edges.
        Falls back to p1 if no intersection is found.
        """
        x, y, w, h = map(float, bbox)
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx, dy = x1 - x0, y1 - y0
        eps = 1e-9
        ts = []
        # Vertical edges x = x or x+w
        if abs(dx) > eps:
            t = (x - x0) / dx
            yy = y0 + t * dy
            if t >= 0.0 - eps and y - eps <= yy <= y + h + eps:
                ts.append((t, (x, yy)))
            t = (x + w - x0) / dx
            yy = y0 + t * dy
            if t >= 0.0 - eps and y - eps <= yy <= y + h + eps:
                ts.append((t, (x + w, yy)))
        # Horizontal edges y = y or y+h
        if abs(dy) > eps:
            t = (y - y0) / dy
            xx = x0 + t * dx
            if t >= 0.0 - eps and x - eps <= xx <= x + w + eps:
                ts.append((t, (xx, y)))
            t = (y + h - y0) / dy
            xx = x0 + t * dx
            if t >= 0.0 - eps and x - eps <= xx <= x + w + eps:
                ts.append((t, (xx, y + h)))
        if not ts:
            return (x1, y1)
        tmin, pt = min(ts, key=lambda v: v[0])
        return (float(pt[0]), float(pt[1]))

    def _through_bbox_line(self, center: Tuple[float, float], direction: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the two intersection points of an infinite line through 'center' with 'direction' and the rectangle bbox."""
        dx, dy = float(direction[0]), float(direction[1])
        nrm = math.hypot(dx, dy)
        if nrm < 1e-6:
            # Degenerate; choose horizontal or vertical based on bbox
            x, y, w, h = bbox
            if w >= h:
                direction = (1.0, 0.0)
            else:
                direction = (0.0, 1.0)
            dx, dy = direction
            nrm = 1.0
        ux, uy = dx / nrm, dy / nrm
        # Far points along +/- directions
        p_fwd = (center[0] + ux * 10000.0, center[1] + uy * 10000.0)
        p_bwd = (center[0] - ux * 10000.0, center[1] - uy * 10000.0)
        a = self._line_rect_intersection(center, p_fwd, bbox)
        b = self._line_rect_intersection(center, p_bwd, bbox)
        return a, b

    # ---------- Template helpers ----------
    def _load_templates_for_bbox(self, bbox: Tuple[float, float, float, float]) -> List[np.ndarray]:
        """Load candidate templates for valve types (ball, gate, check). Returns list of images (grayscale)."""
        troot = Path(self.cfg.template_root)
        out: List[np.ndarray] = []
        if cv2 is None:
            return out
        for sub in ["ball valve", "gate valve", "check valve"]:
            d = troot / sub
            if not d.exists():
                continue
            for p in sorted(d.glob("*.png")):
                try:
                    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        continue
                    out.append(im)
                except Exception:
                    continue
        return out

    def _shrink_text_bbox(self, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """Shrink text bbox by removing edge rows/columns that are all empty or all lines."""
        if self.binary is None or w <= 0 or h <= 0:
            return x, y, w, h
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(self.binary.shape[1], int(x + w)), min(self.binary.shape[0], int(y + h))
        patch = self.binary[y0:y1, x0:x1]
        if patch.size == 0:
            return x, y, w, h
        h_patch, w_patch = patch.shape
        # Shrink rows from top
        top = 0
        for i in range(h_patch):
            row = patch[i, :]
            if np.all(row == 0) or np.all(row == 255):
                top += 1
            else:
                break
        # Shrink rows from bottom
        bottom = 0
        for i in range(h_patch - 1, -1, -1):
            row = patch[i, :]
            if np.all(row == 0) or np.all(row == 255):
                bottom += 1
            else:
                break
        # Shrink columns from left
        left = 0
        for j in range(w_patch):
            col = patch[:, j]
            if np.all(col == 0) or np.all(col == 255):
                left += 1
            else:
                break
        # Shrink columns from right
        right = 0
        for j in range(w_patch - 1, -1, -1):
            col = patch[:, j]
            if np.all(col == 0) or np.all(col == 255):
                right += 1
            else:
                break
        # New bbox
        new_x = x0 + left
        new_y = y0 + top
        new_w = w_patch - left - right
        new_h = h_patch - top - bottom
        if new_w <= 0 or new_h <= 0:
            return x, y, w, h
        return float(new_x), float(new_y), float(new_w), float(new_h)

    def _valve_orientation_by_template(self, bbox: Tuple[float, float, float, float]) -> Optional[str]:
        """Return 'h' for horizontal, 'v' for vertical if template match is confident, else None."""
        if cv2 is None or self.gray is None:
            return None
        x, y, w, h = map(int, bbox)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.gray.shape[1], x + w)
        y1 = min(self.gray.shape[0], y + h)
        patch = self.gray[y0:y1, x0:x1]
        if patch.size == 0:
            return None
        # Preprocess: optionally use edges
        patch_use = patch
        if self.cfg.template_use_edges:
            patch_use = cv2.Canny(patch, self.cfg.canny_low, self.cfg.canny_high)
        tmpls = self._load_templates_for_bbox(bbox)
        if not tmpls:
            return None
        best_h, best_v = -1.0, -1.0
        for t in tmpls:
            t_use = t
            if self.cfg.template_use_edges:
                t_use = cv2.Canny(t, self.cfg.canny_low, self.cfg.canny_high)
            # Build two orientations: as-is (assume horizontal-ish) and rotated 90 deg (vertical-ish)
            for ori, img in [("h", t_use), ("v", cv2.rotate(t_use, cv2.ROTATE_90_CLOCKWISE))]:
                # Resize template to patch size (simple normalization)
                try:
                    tpl = cv2.resize(img, (patch_use.shape[1], patch_use.shape[0]), interpolation=cv2.INTER_AREA)
                except Exception:
                    continue
                res = cv2.matchTemplate(patch_use, tpl, cv2.TM_CCOEFF_NORMED)
                score = float(res[0, 0]) if res.size == 1 else float(res.max())
                if ori == "h":
                    best_h = max(best_h, score)
                else:
                    best_v = max(best_v, score)
        if max(best_h, best_v) < float(self.cfg.template_min_score):
            return None
        return 'h' if best_h >= best_v else 'v'

    # ---------- Stage 3: Symbols/Text ----------
    def stage3_symbols_text(self) -> None:
        t0 = time.time()
        if not hasattr(self, "_ann"):
            raise RuntimeError("Run stage1_ingest first")

        self.symbols = []
        kept = 0
        for i, a in enumerate(self._ann):
            x, y, w, h = a.get("bbox", [0, 0, 0, 0])
            cx, cy = x + w / 2.0, y + h / 2.0
            typ = a.get("category_name") or self._cat_map.get(a.get("category_id"), "unknown")
            conf = float(a.get("score", 1.0))
            tkey = str(typ).lower()
            min_thr = self.cfg.class_conf_thresh.get(tkey, self.cfg.default_conf_thresh)
            if conf < min_thr:
                continue
            # Shrink bbox for 'line number' class
            if "line number" in str(typ).lower() and self.binary is not None and w > 0 and h > 0:
                x, y, w, h = self._shrink_text_bbox(x, y, w, h)
                cx, cy = x + w / 2.0, y + h / 2.0
            self.symbols.append(Symbol(
                id=i,
                type=str(typ),
                bbox=(float(x), float(y), float(w), float(h)),
                confidence=conf,
                center=(float(cx), float(cy)),
                text=None,
            ))
            kept += 1

        self.texts = []
        for i, t in enumerate(self._ocr):
            bx = t.get("bbox", [0, 0, 0, 0])
            if len(bx) == 4:
                x, y, w, h = bx
            elif len(bx) == 2:
                x, y = bx
                w, h = 10, 10
            else:
                x = y = w = h = 0
            # Shrink bbox if edges are all empty or all lines
            if self.binary is not None and w > 0 and h > 0:
                x, y, w, h = self._shrink_text_bbox(x, y, w, h)
            cx, cy = x + w / 2.0, y + h / 2.0
            self.texts.append(TextItem(
                id=i,
                text=str(t.get("text", "")),
                bbox=(float(x), float(y), float(w), float(h)),
                confidence=float(t.get("confidence", 1.0)),
                center=(float(cx), float(cy)),
            ))

        # Associate nearest text to symbols (simple proximity)
        if self.symbols and self.texts:
            txt_centers = np.array([t.center for t in self.texts], dtype=float)
            for s in self.symbols:
                diag = math.hypot(s.bbox[2], s.bbox[3])
                max_d = diag * max(0.5, float(self.cfg.text_assoc_multiplier))
                sc = np.array(s.center, dtype=float)
                dists = np.sqrt(((txt_centers - sc) ** 2).sum(axis=1))
                j = int(np.argmin(dists)) if len(dists) else -1
                if j >= 0 and dists[j] <= max_d:
                    s.text = self.texts[j].text

        # Adjust center for reducers to center of mass
        if self.binary is not None:
            for s in self.symbols:
                if "reducer" in s.type.lower():
                    x, y, w, h = map(int, s.bbox)
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(self.binary.shape[1], x + w), min(self.binary.shape[0], y + h)
                    patch = self.binary[y0:y1, x0:x1]
                    ys, xs = np.where(patch > 0)
                    if len(xs) > 0:
                        cx = x0 + np.mean(xs)
                        cy = y0 + np.mean(ys)
                        s.center = (float(cx), float(cy))

        # Quick overlay for QA
        try:
            vis = self.image_bgr.copy() # type: ignore
            if cv2 is not None:
                for s in self.symbols:
                    x, y, w, h = map(int, s.bbox)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    lbl = s.type[:18] if not s.text else f"{s.type[:12]}:{s.text[:12]}"
                    cv2.putText(vis, lbl, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                for t in self.texts:
                    x, y, w, h = map(int, t.bbox)
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 0, 200), 1)
            elif Image is not None:
                rgb = self.image_bgr[:, :, ::-1] # type: ignore
                im = Image.fromarray(rgb)
                dr = ImageDraw.Draw(im) # type: ignore
                for s in self.symbols:
                    x, y, w, h = s.bbox
                    dr.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
                for t in self.texts:
                    x, y, w, h = t.bbox
                    dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 255), width=1)
                vis = np.array(im)[:, :, ::-1]
            self._save_img("stage3_overlay", vis)
        except Exception as e:
            logger.warning(f"Overlay failed: {e}")

        self._save_json("stage3_counts", {
            "symbols": len(self.symbols),
            "texts": len(self.texts),
            "filtered": kept,
        })
        logger.info("Stage 3 done in %.2fs", time.time() - t0)

    # ---------- Stage 4: Linework (skeleton) ----------
    def stage4_linework(self) -> None:
        t0 = time.time()
        if self.binary is None:
            raise RuntimeError("Run stage2_preprocess first")

        # Start from binary and optionally mask symbol/text regions to create breaks
        bin_img = (self.binary > 0).astype(np.uint8)
        if (self.cfg.mask_symbols and self.symbols) or (self.cfg.mask_text and self.texts):
            H, W = bin_img.shape
            mask = np.zeros_like(bin_img)
            inf = int(max(0, self.cfg.mask_inflate))
            def clip(v, lo, hi):
                return max(lo, min(int(v), hi))
            if self.cfg.mask_symbols:
                for s in self.symbols:
                    x, y, w, h = s.bbox
                    x0, y0 = clip(x - inf, 0, W - 1), clip(y - inf, 0, H - 1)
                    x1, y1 = clip(x + w + inf, 0, W - 1), clip(y + h + inf, 0, H - 1)
                    mask[y0:y1, x0:x1] = 255
            if self.cfg.mask_text:
                for t in self.texts:
                    x, y, w, h = t.bbox
                    x0, y0 = clip(x - inf, 0, W - 1), clip(y - inf, 0, H - 1)
                    x1, y1 = clip(x + w + inf, 0, W - 1), clip(y + h + inf, 0, H - 1)
                    mask[y0:y1, x0:x1] = 255
            bin_img = (bin_img * ((mask == 0).astype(np.uint8)))
            self._save_img("stage4_mask", mask)
            self._save_img("stage4_binary_masked", bin_img * 255)
        if skeletonize is not None:
            skel = skeletonize(bin_img > 0)
            skel = skel.astype(np.uint8)
        else:
            # Very naive fallback: thin by one-pixel erosion (placeholder)
            skel = bin_img.copy()

        self.skeleton = skel
        self._save_img("stage4_skeleton", skel * 255)
        logger.info("Stage 4 done in %.2fs", time.time() - t0)

    # ---------- Stage 5: Graph construction ----------
    def stage5_graph(self) -> None: # type: ignore
        t0 = time.time()
        if self.skeleton is None:
            raise RuntimeError("Run stage4_linework first")

        sk = self.skeleton
        h, w = sk.shape
        # Precompute a quick lookup mask for text regions to speed edge scans
        text_mask = np.zeros((h, w), dtype=bool)
        try:
            for t in self.texts:
                x, y, bw, bh = map(int, t.bbox)
                x0 = max(0, x); y0 = max(0, y)
                x1 = min(w, x + max(0, bw)); y1 = min(h, y + max(0, bh))
                if x0 < x1 and y0 < y1:
                    text_mask[y0:y1, x0:x1] = True
        except Exception:
            pass

        def neighbors(x: int, y: int) -> List[Tuple[int, int]]:
            pts = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < w and 0 <= yy < h and sk[yy, xx] > 0:
                        pts.append((xx, yy))
            return pts

        # Identify critical points
        endpoints = []
        junctions = []
        pts = np.argwhere(sk > 0)
        for y, x in pts:
            deg = len(neighbors(x, y))
            if deg == 1:
                endpoints.append((x, y))
            elif deg >= 3:
                junctions.append((x, y))

        # Remove endpoints that are very close to any junction (cleanup spurs near intersections)
        try:
            if endpoints and junctions:
                jun_arr = np.array(junctions, dtype=float)
                keep_eps: List[Tuple[int, int]] = []
                removed = 0
                thr = float(self.cfg.merge_node_dist)
                for (ex, ey) in endpoints:
                    d = np.hypot(jun_arr[:, 0] - float(ex), jun_arr[:, 1] - float(ey))
                    if d.min() <= thr:
                        removed += 1
                        continue
                    keep_eps.append((ex, ey))
                if removed:
                    logger.info(f"Removed {removed} endpoints near junctions (<= {thr}px)")
                endpoints = keep_eps
        except Exception:
            pass

        # Validate junctions using the binary image to avoid false junctions from skeleton artifacts
        try:
            if self.binary is not None and self.cfg.junction_validate and junctions:
                bin_im = (self.binary > 0).astype(np.uint8)
                H2, W2 = bin_im.shape
                rays = max(8, int(self.cfg.junction_validate_rays))
                R: int = max(2, int(self.cfg.junction_validate_radius))
                min_hits = max(1, int(self.cfg.junction_validate_min_hits))
                ang_merge = float(self.cfg.junction_validate_angle_merge_deg)
                min_br = int(self.cfg.junction_min_branches)

                def branch_count(cx: int, cy: int) -> int:
                    # Sample along multiple angles; record axes with sufficient foreground runs
                    angs: List[float] = []
                    for k in range(rays):
                        ang = 2.0 * math.pi * (k / float(rays))
                        dx, dy = math.cos(ang), math.sin(ang)
                        hits = 0
                        for t in range(1, R + 1):
                            x = int(round(cx + dx * t))
                            y = int(round(cy + dy * t))
                            if x < 0 or y < 0 or x >= W2 or y >= H2:
                                break
                            if bin_im[y, x] > 0:
                                hits += 1
                                if hits >= min_hits:
                                    # Use axis angle in [0, 180)
                                    a_deg = (math.degrees(ang) + 360.0) % 180.0
                                    angs.append(a_deg)
                                    break
                        # if not enough hits, this direction is ignored
                    if not angs:
                        return 0
                    # Merge angles that represent the same axis within tolerance
                    angs.sort()
                    clusters: List[float] = []
                    for a in angs:
                        if not clusters:
                            clusters.append(a)
                        else:
                            if min(abs(a - clusters[-1]), 180.0 - abs(a - clusters[-1])) <= ang_merge:
                                # same cluster
                                continue
                            clusters.append(a)
                    return len(clusters)

                keep_juncs: List[Tuple[int, int]] = []
                removed = 0
                for (jx, jy) in junctions:
                    br = branch_count(int(jx), int(jy))
                    if br >= min_br:
                        keep_juncs.append((jx, jy))
                    else:
                        removed += 1
                if removed:
                    logger.info(f"Removed {removed} false junctions by binary validation (branches < {min_br})")
                junctions = keep_juncs
        except Exception as ex:
            logger.warning(f"junction validation failed: {ex}")
            
        # ---------------------------------------------------------------------------------------------------
        # TODO

        # Start nodes: endpoints + junctions (we defer symbol handling to connection ports)
        self.nodes = []
        id_map: Dict[Tuple[int, int], int] = {}

        def add_node(pos: Tuple[float, float], ntype: NodeType, label: Optional[str] = None) -> int:
            # Enforce invariant: node.id == index in self.nodes
            node_id = len(self.nodes)
            n = Node(id=node_id, position=pos, type=ntype, label=label)
            self.nodes.append(n)
            return n.id

        for (x, y) in endpoints:
            id_map[(x, y)] = add_node((float(x), float(y)), NodeType.ENDPOINT)
        for (x, y) in junctions:
            id_map[(x, y)] = add_node((float(x), float(y)), NodeType.JUNCTION)

        # Attach symbol centers as nodes (typed via mapping then coarse fallback)
        for s in self.symbols:
            stype = s.type.lower()
            # Skip creation for explicitly excluded classes
            if stype in [c.lower() for c in self.cfg.link_exclude_classes] and stype not in [c.lower() for c in self.cfg.link_include_classes]:
                continue

            role = self.cfg.class_role_map.get(stype)
            if role == "valve":
                ntype = NodeType.VALVE
            elif role == "instrument":
                ntype = NodeType.INSTRUMENT
            elif role == "equipment":
                ntype = NodeType.EQUIPMENT
            elif role == "offpage":
                ntype = NodeType.OFFPAGE
            elif "reducer" in stype:
                ntype = NodeType.EQUIPMENT
            elif "pump" in stype:
                ntype = NodeType.EQUIPMENT
            elif "page connection" in stype or "off-page" in stype or "off page" in stype or "utility connection" in stype:
                ntype = NodeType.OFFPAGE
            elif "valve" in stype:
                ntype = NodeType.VALVE
            elif "instrument" in stype or "flow" in stype:
                ntype = NodeType.INSTRUMENT
            elif "connection" in stype or "off" in stype:
                ntype = NodeType.OFFPAGE
            else:
                ntype = NodeType.UNKNOWN

            # Skip linking for excluded roles unless explicitly included by class
            if ntype == NodeType.INSTRUMENT and ("instrument" in [r.lower() for r in self.cfg.link_exclude_roles]) and stype not in [c.lower() for c in self.cfg.link_include_classes]:
                continue

            sid = add_node(s.center, ntype, label=s.text)
            self.nodes[sid].symbol_ids.append(s.id)

        # Early port discovery: create connection ports from symbols before tracing
        # Precompute skeleton nodes/pixels for connection
        sk_ids = [n.id for n in self.nodes if n.type in (NodeType.ENDPOINT, NodeType.JUNCTION)]
        sk_pos = np.array([self.nodes[i].position for i in sk_ids], dtype=float) if sk_ids else np.zeros((0, 2))

        def ensure_node_for_pixel(px: Tuple[int, int]) -> int:
            if px in id_map:
                return id_map[px]
            x, y = px
            deg = len(neighbors(x, y))
            ntype = NodeType.JUNCTION if deg >= 2 else NodeType.ENDPOINT
            node_id = add_node((float(x), float(y)), ntype)
            id_map[px] = node_id
            return node_id

        # Helper: intersect line between center and a target with bbox to place port on edge
        def _port_on_bbox(center: Tuple[float, float], target: Tuple[float, float], bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
            return self._line_rect_intersection(center, target, bbox)

        # Build connection nodes (ports) for selected symbol classes
        # Only ports + traced skeleton nodes will exist as graph nodes in Stage 5
        try:
            # Choose which symbols produce connection ports
            for s in self.symbols:
                stype = s.type.lower()
                role = self.cfg.class_role_map.get(stype, stype)
                # Skip non-inline/link-excluded classes
                if stype in [c.lower() for c in self.cfg.link_exclude_classes] and stype not in [c.lower() for c in self.cfg.link_include_classes]:
                    continue
                if any(r in [rl.lower() for rl in self.cfg.link_exclude_roles] for r in [role]) and stype not in [c.lower() for c in self.cfg.link_include_classes]:
                    continue

                # Determine desired links per type
                if "valve" in stype:
                    desired_links = int(self.cfg.ports_per_type.get("valve", 2))
                elif ("connection" in stype) or ("page" in stype) or ("utility" in stype) or ("off" in stype):
                    desired_links = int(self.cfg.ports_per_type.get("offpage", 1))
                elif "instrument" in stype:
                    desired_links = int(self.cfg.ports_per_type.get("instrument", 1))
                else:
                    desired_links = int(self.cfg.ports_per_type.get("unknown", 1))

                # Pick candidate skeleton targets near the symbol
                center = np.array(s.center, dtype=float)
                picks: List[Tuple[int, float, Tuple[float, float]]] = []  # (node_id, dist, pos)

                # Directional pick for valve/connection-like: scan bbox edges and confirm outward skeleton
                def _symbol_directional_picks(sym: Symbol) -> List[Tuple[int, float, Tuple[float, float]]]:
                    picks_local: List[Tuple[int, float, Tuple[float, float]]] = []
                    bb = sym.bbox
                    x, y, bw, bh = bb
                    cx, cy = sym.center
                    # Choose edges based on template/narrow side or all for connection-like
                    edges: Optional[List[str]] = None
                    # For page connection: exactly one port on the narrow side edges only
                    if "page connection" in stype:
                        edges = ["left", "right"] if bw < bh else ["top", "bottom"]
                    elif ("connection" in stype) or ("page" in stype) or ("utility" in stype) or ("off" in stype):
                        edges = ["left", "right", "top", "bottom"]
                    else:
                        try:
                            orient = self._valve_orientation_by_template(bb)
                            if orient == 'h':
                                edges = ["left", "right"]
                            elif orient == 'v':
                                edges = ["top", "bottom"]
                        except Exception:
                            pass
                        if edges is None:
                            edges = ["top", "bottom"] if bw < bh else ["left", "right"]

                    # Use adaptive sampling along edges to avoid O(perimeter * texts) slowness
                    step = max(1, int(max(self.cfg.valve_edge_sample_step, min(bw, bh) / 10.0)))

                    def add_pick_from_edge(pt_edge: Tuple[float, float], outward: Tuple[float, float], inward: Tuple[float, float]):
                        # Verify inside the bbox there is a line in Stage-2 binary along inward direction
                        if self.binary is None:
                            return False
                        in_len = max(1, int(self.cfg.valve_inward_check_px))
                        ix, iy = float(pt_edge[0]) - outward[0]*0.5, float(pt_edge[1]) - outward[1]*0.5
                        hit_cnt = 0
                        tot = 0
                        H2, W2 = self.binary.shape
                        for t in range(in_len):
                            xx = int(round(ix + inward[0] * t))
                            yy = int(round(iy + inward[1] * t))
                            if 0 <= xx < W2 and 0 <= yy < H2:
                                if (not text_mask[yy, xx]) and self.binary[yy, xx] > 0:
                                    hit_cnt += 1
                                tot += 1
                        frac = (hit_cnt / float(tot)) if tot > 0 else 0.0
                        if frac < float(self.cfg.valve_inward_min_frac):
                            return False
                        # Check for skeleton adjacent to the edge point outward
                        found = False
                        # Use larger outward search for connection-like symbols
                        check_dist = int(self.cfg.connection_search_radius) if (("connection" in stype) or ("page" in stype) or ("utility" in stype) or ("off" in stype)) else int(max(5, self.cfg.valve_edge_outward_search))
                        best_hit: Optional[Tuple[int,int]] = None
                        for dstep in range(1, check_dist + 1):
                            check_x = int(pt_edge[0] + outward[0] * dstep)
                            check_y = int(pt_edge[1] + outward[1] * dstep)
                            if 0 <= check_x < w and 0 <= check_y < h and sk[check_y, check_x] > 0:
                                found = True
                                best_hit = (check_x, check_y)
                                break
                        if not found or best_hit is None:
                            # Fallback: look for a line in the Stage-2 binary along the same outward ray
                            try:
                                max_bin = int(self.cfg.port_binary_outward_search)
                                min_run = max(1, int(self.cfg.port_binary_min_hits))
                                run = 0
                                last_bin_xy = None
                                for dstep in range(1, max_bin + 1):
                                    bx = int(round(pt_edge[0] + outward[0] * dstep))
                                    by = int(round(pt_edge[1] + outward[1] * dstep))
                                    if bx < 0 or by < 0 or bx >= W2 or by >= H2:
                                        break
                                    if self.binary[by, bx] > 0 and not text_mask[min(by, h-1), min(bx, w-1)]:
                                        run += 1
                                        last_bin_xy = (bx, by)
                                        if run >= min_run:
                                            break
                                    else:
                                        run = 0
                                if run >= min_run and last_bin_xy is not None:
                                    # Snap to nearest skeleton pixel in a local disk
                                    snap_r = max(1, int(self.cfg.port_skeleton_snap_radius))
                                    hit_xy = None
                                    for rr in range(0, snap_r + 1):
                                        # square ring search for simplicity
                                        for dx in range(-rr, rr + 1):
                                            for dy in (-rr, rr) if rr > 0 else (0,):
                                                sx = last_bin_xy[0] + dx
                                                sy = last_bin_xy[1] + dy
                                                if 0 <= sx < w and 0 <= sy < h and sk[sy, sx] > 0:
                                                    hit_xy = (sx, sy)
                                                    break
                                            if hit_xy is not None:
                                                break
                                        if hit_xy is not None:
                                            break
                                    if hit_xy is not None:
                                        px_node = ensure_node_for_pixel(hit_xy)
                                        distv = float(math.hypot(pt_edge[0] - cx, pt_edge[1] - cy))
                                        picks_local.append((px_node, distv, (float(pt_edge[0]), float(pt_edge[1]))))
                                        return True
                            except Exception:
                                pass
                            return False
                        # Ensure pixel has a node id
                        px_node = ensure_node_for_pixel(best_hit)
                        distv = float(math.hypot(pt_edge[0] - cx, pt_edge[1] - cy))
                        picks_local.append((px_node, distv, (float(pt_edge[0]), float(pt_edge[1]))))
                        return True

                    for ekey in edges:
                        if ekey == "left":
                            iy0, iy1 = int(y), int(y + bh)
                            for iy in range(iy0, iy1, step):
                                if add_pick_from_edge((x, iy), (-1.0, 0.0), (1.0, 0.0)):
                                    break
                        elif ekey == "right":
                            iy0, iy1 = int(y), int(y + bh)
                            for iy in range(iy0, iy1, step):
                                if add_pick_from_edge((x + bw, iy), (1.0, 0.0), (-1.0, 0.0)):
                                    break
                        elif ekey == "top":
                            ix0, ix1 = int(x), int(x + bw)
                            for ix in range(ix0, ix1, step):
                                if add_pick_from_edge((ix, y), (0.0, -1.0), (0.0, 1.0)):
                                    break
                        elif ekey == "bottom":
                            ix0, ix1 = int(x), int(x + bw)
                            for ix in range(ix0, ix1, step):
                                if add_pick_from_edge((ix, y + bh), (0.0, 1.0), (0.0, -1.0)):
                                    break
                    # Fallback: scan other edges if none were found
                    if not picks_local:
                        other = ["left", "right"] if edges == ["top", "bottom"] else ["top", "bottom"]
                        for ekey in other:
                            if ekey == "left":
                                iy0, iy1 = int(y), int(y + bh)
                                for iy in range(iy0, iy1, step):
                                    if add_pick_from_edge((x, iy), (-1.0, 0.0), (1.0, 0.0)):
                                        break
                            elif ekey == "right":
                                iy0, iy1 = int(y), int(y + bh)
                                for iy in range(iy0, iy1, step):
                                    if add_pick_from_edge((x + bw, iy), (1.0, 0.0), (-1.0, 0.0)):
                                        break
                            elif ekey == "top":
                                ix0, ix1 = int(x), int(x + bw)
                                for ix in range(ix0, ix1, step):
                                    if add_pick_from_edge((ix, y), (0.0, -1.0), (0.0, 1.0)):
                                        break
                            elif ekey == "bottom":
                                ix0, ix1 = int(x), int(x + bw)
                                for ix in range(ix0, ix1, step):
                                    if add_pick_from_edge((ix, y + bh), (0.0, 1.0), (0.0, -1.0)):
                                        break
                    # Last resort for connection-like symbols: 360° raycast from center to find nearest skeleton
                    if not picks_local and (("connection" in stype) or ("page" in stype) or ("utility" in stype) or ("off" in stype)):
                        try:
                            cx0, cy0 = int(round(cx)), int(round(cy))
                            R = int(self.cfg.connection_search_radius)
                            rays = max(8, int(self.cfg.connection_raycast_angles))
                            # Prefer rays aligned with dominant local direction
                            dom_ang = self._dominant_angle_in_window(cx0, cy0, self.cfg.connection_dir_window, default=None)
                            angs = [2.0 * math.pi * (k / rays) for k in range(rays)]
                            if dom_ang is not None:
                                def ang_dist(a, b):
                                    d = abs(a - b) % (2 * math.pi)
                                    return min(d, 2 * math.pi - d)
                                angs = sorted(angs, key=lambda a: min(ang_dist(a, dom_ang), ang_dist(a, (dom_ang + math.pi) % (2 * math.pi)))) # type: ignore
                            best_hit = None
                            best_dist = 1e9
                            best_dir = (1.0, 0.0)
                            for ang in angs:
                                dx, dy = math.cos(ang), math.sin(ang)
                                for step in range(2, R + 1):
                                    xh = int(round(cx0 + dx * step))
                                    yh = int(round(cy0 + dy * step))
                                    if xh < 0 or yh < 0 or xh >= w or yh >= h:
                                        break
                                    if sk[yh, xh] > 0:
                                        dcur = math.hypot(xh - cx0, yh - cy0)
                                        if dcur < best_dist:
                                            best_hit = (xh, yh)
                                            best_dist = dcur
                                            best_dir = (dx, dy)
                                        break
                            if best_hit is not None:
                                # Place port at bbox edge in the direction of the hit
                                edge_pt = self._line_rect_intersection((cx, cy), (cx + best_dir[0]*1000.0, cy + best_dir[1]*1000.0), bb)
                                px_node = ensure_node_for_pixel(best_hit)
                                picks_local.append((px_node, float(best_dist), (float(edge_pt[0]), float(edge_pt[1]))))
                        except Exception:
                            pass
                    # Enforce single port for page connection
                    if "page connection" in stype and len(picks_local) > 1:
                        picks_local = picks_local[:1]
                    return picks_local

                # Directional picks for valves/offpage; otherwise generic nearest skeleton nodes
                if ("valve" in stype) or ("connection" in stype) or ("page" in stype) or ("utility" in stype) or ("off" in stype):
                    try:
                        picks.extend(_symbol_directional_picks(s))
                    except Exception:
                        pass

                if len(picks) < desired_links and sk_pos.shape[0] > 0:
                    d = np.sqrt(((sk_pos - center) ** 2).sum(axis=1))
                    within = np.where(d <= float(self.cfg.connect_radius))[0]
                    order = within[np.argsort(d[within])] if within.size > 0 else np.array([], dtype=int)
                    picked_angles: List[float] = []
                    for idx in order:
                        vec = sk_pos[idx] - center
                        ang = float((math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0)
                        if any(min(abs(ang - a), 360.0 - abs(ang - a)) < float(self.cfg.angle_sep_min_deg) for a in picked_angles):
                            continue
                        picks.append((sk_ids[idx], float(d[idx]), tuple(sk_pos[idx])))
                        picked_angles.append(ang)
                        if len(picks) >= desired_links:
                            break

                # Ensure two ports for reducer-like equipment by adding fallback edge points on opposite sides
                if "reducer" in stype and len(picks) < 2:
                    try:
                        # Estimate dominant direction around the reducer
                        dom = self._dominant_angle_in_window(s.center[0], s.center[1], self.cfg.connection_dir_window, default=None)
                        if dom is None:
                            # fallback: choose along longer bbox side
                            x, y, bw, bh = s.bbox
                            dom = 0.0 if bw >= bh else math.pi / 2.0
                        dirv = (math.cos(dom), math.sin(dom))
                        a_pt, b_pt = self._through_bbox_line(s.center, dirv, s.bbox)
                        for q in [a_pt, b_pt]:
                            # find nearest skeleton node or raycast outward
                            node_id = None
                            if sk_pos.shape[0] > 0:
                                cq = np.array(q, dtype=float)
                                d = np.sqrt(((sk_pos - cq) ** 2).sum(axis=1))
                                order = np.argsort(d)
                                if order.size > 0 and d[order[0]] <= float(self.cfg.connect_radius):
                                    node_id = sk_ids[int(order[0])]
                            if node_id is None:
                                # Raycast from edge outward
                                vx, vy = q[0] - s.center[0], q[1] - s.center[1]
                                nrm = math.hypot(vx, vy) + 1e-6
                                ux, uy = vx / nrm, vy / nrm
                                R = int(self.cfg.connection_search_radius)
                                hit = None
                                for step in range(2, R + 1):
                                    xh = int(round(q[0] + ux * step))
                                    yh = int(round(q[1] + uy * step))
                                    if xh < 0 or yh < 0 or xh >= w or yh >= h:
                                        break
                                    if sk[yh, xh] > 0:
                                        hit = (xh, yh)
                                        break
                                if hit is not None:
                                    node_id = ensure_node_for_pixel(hit)
                            if node_id is not None:
                                distv = float(math.hypot(q[0] - s.center[0], q[1] - s.center[1]))
                                picks.append((node_id, distv, (float(q[0]), float(q[1]))))
                            if len(picks) >= 2:
                                break
                    except Exception:
                        pass

                # Record debug picks
                try:
                    self._symbol_pick_debug[s.id] = [(float(pos[0]), float(pos[1])) for (_, __, pos) in picks]
                except Exception:
                    self._symbol_pick_debug[s.id] = []

                # Create PORT nodes at bbox edge along vector to pick, then connect PORT -> skeleton pick
                for node_id, distv, pos in picks:
                    port_pos = _port_on_bbox(s.center, (float(pos[0]), float(pos[1])), s.bbox)
                    port_id = add_node(port_pos, NodeType.PORT)
                    # Link port to skeleton target
                    self.edges.append(Edge(
                        id=len(self.edges),
                        source=port_id,
                        target=node_id,
                        path=[port_pos, (float(pos[0]), float(pos[1]))],
                        attributes={"length": float(distv), "inferred": True, "from_port": True, "symbol_id": int(s.id)},
                    ))
                    # If this is a connection-like symbol, force tracing to start from the skeleton hit
                    if ("connection" in stype) or ("page" in stype) or ("utility" in stype) or ("off" in stype):
                        try:
                            sxn, syn = self.nodes[node_id].position
                            force_trace_starts.append((int(round(sxn)), int(round(syn)), int(s.id)))
                        except Exception:
                            pass
        except Exception:
            pass

        # Trace segments from critical pixels
        visited = np.zeros_like(sk, dtype=bool)
        edges: List[Edge] = []
        eid = 0

        def classify_junction(px: Tuple[int, int], tol_deg: float = 20.0) -> Tuple[bool, List[Tuple[int,int]], List[Tuple[int,int]]]:
            """Classify a pixel as junction with an approx straight pair.
            Returns (has_straight_pair, straight_neighbors, branch_neighbors).
            """
            x, y = px
            nbrs = neighbors(x, y)
            if len(nbrs) < 3:
                return (False, [], nbrs)
            # Compute angles from center to neighbors
            angs = []  # (deg, (nx,ny)) in [0,360)
            for nx_, ny_ in nbrs:
                ang = (math.degrees(math.atan2(ny_ - y, nx_ - x)) + 360.0) % 360.0
                angs.append((ang, (nx_, ny_)))
            # Find best opposite pair
            best_pair = None
            best_dev = 1e9
            for i in range(len(angs)):
                for j in range(i+1, len(angs)):
                    a1, p1 = angs[i]
                    a2, p2 = angs[j]
                    d = abs(((a1 - a2 + 180.0) % 360.0) - 180.0)
                    if d < best_dev:
                        best_dev = d
                        best_pair = (p1, p2)
            if best_pair is not None and best_dev <= tol_deg:
                straight_set = [best_pair[0], best_pair[1]]
                branch = [p for (_, p) in angs if p not in straight_set]
                return (True, straight_set, branch)
            return (False, [], [p for (_, p) in angs])

        def trace_from(start: Tuple[int, int]) -> None:
            nonlocal eid
            sx, sy = start
            # explore immediate neighbors
            start_nbrs = neighbors(sx, sy)
            # Rule 4: when starting at a T-junction, only start tracing in branch directions
            has_straight, straight_list, branch_list = classify_junction((sx, sy))
            if has_straight and branch_list:
                start_nbrs = branch_list
            for nx0, ny0 in start_nbrs:
                if visited[ny0, nx0]:
                    continue
                path = [(sx, sy), (nx0, ny0)]
                gap_jump_hits: List[Tuple[int,int]] = []
                endpoint_turn_hits: List[Tuple[int,int]] = []
                # Do not mark junction pixels as visited to allow multiple passes (Rule 3)
                if len(neighbors(nx0, ny0)) < 3:
                    visited[ny0, nx0] = True
                cx, cy = nx0, ny0
                while True:
                    nbrs = [(xx, yy) for (xx, yy) in neighbors(cx, cy) if not visited[yy, xx]]
                    if len(nbrs) == 0:
                        # Try to jump a small gap in the current direction
                        if len(path) >= 2:
                            prevx, prevy = path[-2]
                            vx, vy = cx - prevx, cy - prevy
                            nrm = math.hypot(vx, vy) + 1e-6
                            ux, uy = vx / nrm, vy / nrm
                            maxd = int(max(1.0, float(self.cfg.bridge_max_dist)))
                            jumped = False
                            for tgap in range(2, maxd + 1):
                                gx = int(round(cx + ux * tgap))
                                gy = int(round(cy + uy * tgap))
                                if gx < 0 or gy < 0 or gx >= w or gy >= h:
                                    break
                                if sk[gy, gx] > 0 and not visited[gy, gx]:
                                    # Found a hit ahead; continue tracing from there
                                    path.append((gx, gy))
                                    gap_jump_hits.append((gx, gy))
                                    if len(neighbors(gx, gy)) < 3:
                                        visited[gy, gx] = True
                                    cx, cy = gx, gy
                                    jumped = True
                                    break
                        if jumped:
                            continue
                        # Try 90-degree turn from endpoint (left/right) within small gap window
                        # Only applies at dead-ends (not junction choices)
                        turned = False
                        # compute forward unit vector even if no jump
                        if len(path) >= 2:
                            prevx, prevy = path[-2]
                            vx, vy = cx - prevx, cy - prevy
                            nrm = math.hypot(vx, vy) + 1e-6
                            ux, uy = vx / nrm, vy / nrm
                            # left and right perpendiculars
                            dirs = [(-uy, ux), (uy, -ux)]
                            maxd = int(max(1.0, float(self.cfg.bridge_max_dist)))
                            for dx, dy in dirs:
                                for tgap in range(2, maxd + 1):
                                    gx = int(round(cx + dx * tgap))
                                    gy = int(round(cy + dy * tgap))
                                    if gx < 0 or gy < 0 or gx >= w or gy >= h:
                                        break
                                    if sk[gy, gx] > 0 and not visited[gy, gx]:
                                        path.append((gx, gy))
                                        endpoint_turn_hits.append((gx, gy))
                                        if len(neighbors(gx, gy)) < 3:
                                            visited[gy, gx] = True
                                        cx, cy = gx, gy
                                        turned = True
                                        break
                                if turned:
                                    break
                        if turned:
                            continue
                        break
                    if len(nbrs) == 1:
                        nx1, ny1 = nbrs[0]
                        path.append((nx1, ny1))
                        if len(neighbors(nx1, ny1)) < 3:
                            visited[ny1, nx1] = True
                        cx, cy = nx1, ny1
                        continue
                    # Multiple choices (junction). Try to continue straight through
                    prevx, prevy = path[-2]
                    vx, vy = cx - prevx, cy - prevy
                    best = None
                    best_ang = 1e9
                    for (tx, ty) in nbrs:
                        vx2, vy2 = tx - cx, ty - cy
                        dot = vx * vx2 + vy * vy2
                        n1 = math.hypot(vx, vy) + 1e-6
                        n2 = math.hypot(vx2, vy2) + 1e-6
                        ang = math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))
                        if ang < best_ang:
                            best_ang = ang
                            best = (tx, ty)
                    # Rule 2 and 5: only continue if nearly straight; otherwise stop at T (no turning)
                    straight_thresh = float(min(self.cfg.continue_straight_deg, 20.0))
                    if best is not None and best_ang <= straight_thresh:
                        nx1, ny1 = best
                        path.append((nx1, ny1))
                        if len(neighbors(nx1, ny1)) < 3:
                            visited[ny1, nx1] = True
                        cx, cy = nx1, ny1
                        continue
                    # Otherwise try a gap jump in the straight direction before stopping
                    nrm = math.hypot(vx, vy) + 1e-6
                    ux, uy = vx / nrm, vy / nrm
                    maxd = int(max(1.0, float(self.cfg.bridge_max_dist)))
                    jumped = False
                    for tgap in range(2, maxd + 1):
                        gx = int(round(cx + ux * tgap))
                        gy = int(round(cy + uy * tgap))
                        if gx < 0 or gy < 0 or gx >= w or gy >= h:
                            break
                        if sk[gy, gx] > 0 and not visited[gy, gx]:
                            path.append((gx, gy))
                            gap_jump_hits.append((gx, gy))
                            if len(neighbors(gx, gy)) < 3:
                                visited[gy, gx] = True
                            cx, cy = gx, gy
                            jumped = True
                            break
                    if jumped:
                        continue
                    # Otherwise stop the segment here
                    break

                # Determine endpoints of the segment; ensure we have graph nodes at both ends
                a = path[0]
                b = path[-1]
                a_id = ensure_node_for_pixel(a)
                b_id = ensure_node_for_pixel(b)
                if a_id != b_id:
                    edges.append(Edge(
                        id=eid,
                        source=a_id,
                        target=b_id,
                        path=[(float(px), float(py)) for (px, py) in path],
                        attributes={
                            "length": float(len(path)),
                            "gap_jumps": int(len(gap_jump_hits)),
                            "gap_jump": bool(len(gap_jump_hits) > 0),
                            "endpoint_turns": int(len(endpoint_turn_hits)),
                            "endpoint_turn": bool(len(endpoint_turn_hits) > 0)
                        },
                    ))
                    eid += 1

        # Force tracing from connection-hit pixels first (deduped)
        try:
            force_trace_starts  # type: ignore[name-defined]
        except NameError:
            force_trace_starts = []  # fallback if early block failed
        if force_trace_starts:
            seen_force: set[Tuple[int, int]] = set()
            for px, py, sid in force_trace_starts:
                if (px, py) in seen_force:
                    continue
                seen_force.add((px, py))
                before = len(edges)
                trace_from((px, py))
                after = len(edges)
                if after > before:
                    logger.info(f"Traced pipeline from connection symbol id {sid} at ({px},{py}); +{after - before} edge(s)")
        for p in endpoints + junctions:
            trace_from(p)

        # Combine traced edges with previously created connection edges
        self.edges = edges + self.edges
        # Normalize edge IDs to be unique and sequential
        for i, e in enumerate(self.edges):
            e.id = i

        # Save overlay with full traced paths before consolidation
        try:
            vis_traced_original = self.image_bgr.copy() # type: ignore
            if cv2 is not None:
                # Map symbol id -> bbox for quick access
                sym_bbox = {s.id: s.bbox for s in self.symbols}
                # Draw symbol bboxes in red and centers in orange on both
                for node in self.nodes:
                    if not node.symbol_ids:
                        continue
                    sid = node.symbol_ids[0]
                    bb = sym_bbox.get(sid)
                    if bb is not None:
                        x, y, w, h = map(int, bb)
                        cv2.rectangle(vis_traced_original, (x, y), (x + w, y + h), RED_COLOR, 2)
                    cx, cy = map(int, node.position)
                    cv2.circle(vis_traced_original, (cx, cy), 4, ORANGE_COLOR, -1)  # orange center
                for n in self.nodes:
                    x, y = map(int, n.position)
                    if n.type == NodeType.PORT:
                        cv2.circle(vis_traced_original, (x, y), 6, BLUE_COLOR, -1)  # blue for ports
                    else:
                        color = YELLOW_COLOR if n.type == NodeType.JUNCTION else RED_COLOR if n.type == NodeType.ENDPOINT else BLUE_COLOR
                        cv2.circle(vis_traced_original, (x, y), 4, color, -1)
                # Draw edges with full paths
                for e in self.edges:
                    if e.attributes.get("bridged_gap"):
                        continue  # Skip bridged edges
                    pts = e.path
                    for i in range(len(pts) - 1):
                        x1, y1 = map(int, pts[i])
                        x2, y2 = map(int, pts[i + 1])
                        if e.attributes.get("symbol_port") or e.attributes.get("inferred"):
                            # Symbol links
                            if e.attributes.get("symbol_port"):
                                color = RED_COLOR # red
                            else:
                                color = BLUE_COLOR  # blue
                            cv2.line(vis_traced_original, (x1, y1), (x2, y2), color, 2)
                        else:
                            # Traced
                            cv2.line(vis_traced_original, (x1, y1), (x2, y2), GREEN_COLOR, 2)
            self._save_img("stage5_graph_overlay_traced_original", vis_traced_original)
        except Exception as ex:
            logger.warning(f"graph overlay original failed: {ex}")

        # Consolidate traced pipelines to optimum points (assume straight lines for all)
        for e in self.edges:
            if len(e.path) > 2:
                start = e.path[0]
                end = e.path[-1]
                e.path = [start, end]

        # Optional pass: trace any remaining skeleton pixels (loops or isolated segments)
        if self.cfg.trace_all_after_stage4:
            remaining = np.argwhere((sk > 0) & (~visited))
            def pick_next(curr: Tuple[int,int], prev: Tuple[int,int] | None) -> Optional[Tuple[int,int]]:
                nbrs = [q for q in neighbors(curr[0], curr[1]) if not visited[q[1], q[0]]]
                if not nbrs:
                    return None
                if prev is None:
                    # pick arbitrary but stable
                    return nbrs[0]
                # prefer near-straight continuation
                vx, vy = curr[0] - prev[0], curr[1] - prev[1]
                best = None
                best_ang = 1e9
                for (tx, ty) in nbrs:
                    vx2, vy2 = tx - curr[0], ty - curr[1]
                    dot = vx * vx2 + vy * vy2
                    n1 = math.hypot(vx, vy) + 1e-6
                    n2 = math.hypot(vx2, vy2) + 1e-6
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, dot / (n1 * n2)))))
                    if ang < best_ang:
                        best_ang = ang
                        best = (tx, ty)
                return best

            for y, x in remaining:
                if visited[y, x]:
                    continue
                # Start a walk
                start = (int(x), int(y))
                path = [start]
                prev = None
                curr = start
                # mark current as visited to avoid re-entry
                visited[curr[1], curr[0]] = True
                loop_detect_guard = 0
                while True:
                    nxt = pick_next(curr, prev)
                    if nxt is None:
                        break
                    path.append(nxt)
                    visited[nxt[1], nxt[0]] = True
                    prev, curr = curr, nxt
                    loop_detect_guard += 1
                    if loop_detect_guard > 100000:
                        break

                if len(path) < 2:
                    continue

                # Create nodes at ends (or two positions along loop)
                a = path[0]
                b = path[-1]
                if a == b or (len(path) > 4 and math.hypot(b[0]-a[0], b[1]-a[1]) < 2):
                    # Likely loop: create two nodes far apart on the path
                    mid = len(path) // 2
                    a = path[0]
                    b = path[mid]
                a_id = ensure_node_for_pixel(a)
                b_id = ensure_node_for_pixel(b)
                if a_id == b_id:
                    continue
                edges.append(Edge(
                    id=len(edges),
                    source=a_id,
                    target=b_id,
                    path=[(float(px), float(py)) for (px, py) in path],
                    attributes={"length": float(len(path)), "full_trace": True},
                ))

        # Precompute skeleton nodes/pixels for connection (post-tracing may add nodes; refresh later if needed)
        sk_ids = [n.id for n in self.nodes if n.type in (NodeType.ENDPOINT, NodeType.JUNCTION)]
        sk_pos = np.array([self.nodes[i].position for i in sk_ids], dtype=float) if sk_ids else np.zeros((0, 2))
        sk_pixels_xy = np.array([(int(xy[1]), int(xy[0])) for xy in pts])  # store as (y,x) for indexing
        sk_pixels_xy_float = np.array([(int(p[1]), int(p[0])) for p in pts], dtype=float)

        # Note: symbol center nodes are no longer created; only ports were added earlier
        if self.nodes:
            eid_local = eid

            # Helper: cast a ray from a start point in a direction and return first skeleton hit
            def _ray_hit(start_xy: Tuple[float, float], dir_xy: Tuple[float, float], max_R: float, step: int = 2) -> Optional[Tuple[int, int]]:
                if max_R <= 0:
                    return None
                dx, dy = float(dir_xy[0]), float(dir_xy[1])
                nrm = math.hypot(dx, dy) + 1e-6
                ux, uy = dx / nrm, dy / nrm
                x0, y0 = float(start_xy[0]), float(start_xy[1])
                for t in range(0, int(max_R) + 1, max(1, int(step))):
                    x = int(round(x0 + ux * t))
                    y = int(round(y0 + uy * t))
                    if 0 <= x < w and 0 <= y < h and sk[y, x] > 0:
                        return (x, y)
                return None

            def _bbox_for_symbol_id(sid: int) -> Optional[Tuple[float, float, float, float]]:
                try:
                    s = next((ss for ss in self.symbols if ss.id == sid), None)
                    if s is None:
                        return None
                    return s.bbox
                except Exception:
                    return None

            def _valve_directional_picks(node: Node) -> List[Tuple[int, float, Tuple[float, float]]]:
                """Find valve connections only where a line passes through the bbox edge.
                We scan along the narrower sides of the bbox (width<height => top/bottom; else left/right),
                and cast a short outward ray to hit the skeleton exactly outside the edge.
                Returns list of (node_id, distance_from_center, hit_position).
                """
                picks_local: List[Tuple[int, float, Tuple[float, float]]] = []
                if not node.symbol_ids:
                    return picks_local
                sid = node.symbol_ids[0]
                bb = _bbox_for_symbol_id(sid)
                if bb is None:
                    return picks_local
                x, y, bw, bh = bb
                cx, cy = node.position
                # Determine preferred orientation via templates (fallback to narrow-side)
                edges = None
                if node.type == NodeType.OFFPAGE:
                    # For connections, scan all edges since lines can come from any side
                    edges = ["left", "right", "top", "bottom"]
                else:
                    try:
                        orient = self._valve_orientation_by_template(bb)
                        if orient == 'h':
                            edges = ["left", "right"]
                        elif orient == 'v':
                            edges = ["top", "bottom"]
                    except Exception:
                        pass
                    if edges is None:
                        if bw < bh:
                            edges = ["top", "bottom"]
                        else:
                            edges = ["left", "right"]
                step = max(1, int(self.cfg.valve_edge_sample_step))

                def add_pick_from_edge(pt_edge: Tuple[float, float], outward: Tuple[float, float], inward: Tuple[float, float]):
                    # Verify inside the bbox there is a line in the Stage-2 binary (foreground) along inward direction
                    if self.binary is None:
                        return False
                    in_len = max(1, int(self.cfg.valve_inward_check_px))
                    ix, iy = float(pt_edge[0]) - outward[0]*0.5, float(pt_edge[1]) - outward[1]*0.5  # nudge slightly inside
                    hit_cnt = 0
                    tot = 0
                    H2, W2 = self.binary.shape
                    for t in range(in_len):
                        xx = int(round(ix + inward[0] * t))
                        yy = int(round(iy + inward[1] * t))
                        if 0 <= xx < W2 and 0 <= yy < H2:
                            # Check if inside any text bbox to avoid counting text as foreground
                            inside_text = False
                            for txt in self.texts:
                                tx, ty, tw, th = txt.bbox
                                if tx <= xx < tx + tw and ty <= yy < ty + th:
                                    inside_text = True
                                    break
                            if not inside_text and self.binary[yy, xx] > 0:
                                hit_cnt += 1
                            tot += 1
                    frac = (hit_cnt / float(tot)) if tot > 0 else 0.0
                    if frac < float(self.cfg.valve_inward_min_frac):
                        return False
                    # Check for skeleton adjacent to the edge point outward
                    found = False
                    check_dist = 5  # px outward to check
                    for d in range(1, check_dist + 1):
                        check_x = int(pt_edge[0] + outward[0] * d)
                        check_y = int(pt_edge[1] + outward[1] * d)
                        if 0 <= check_x < w and 0 <= check_y < h and sk[check_y, check_x] > 0:
                            found = True
                            break
                    if not found:
                        return False
                    # Use the edge point as the hit, since skeleton is broken
                    px_node = ensure_node_for_pixel((int(pt_edge[0]), int(pt_edge[1])))
                    distv = float(math.hypot(pt_edge[0] - cx, pt_edge[1] - cy))
                    picks_local.append((px_node, distv, pt_edge))
                    return True

                # Scan the selected edges
                for ekey in edges:
                    if ekey == "left":
                        iy0, iy1 = int(y), int(y + bh)
                        for iy in range(iy0, iy1, step):
                            if add_pick_from_edge((x, iy), (-1.0, 0.0), (1.0, 0.0)):
                                break
                    elif ekey == "right":
                        iy0, iy1 = int(y), int(y + bh)
                        for iy in range(iy0, iy1, step):
                            if add_pick_from_edge((x + bw, iy), (1.0, 0.0), (-1.0, 0.0)):
                                break
                    elif ekey == "top":
                        ix0, ix1 = int(x), int(x + bw)
                        for ix in range(ix0, ix1, step):
                            if add_pick_from_edge((ix, y), (0.0, -1.0), (0.0, 1.0)):
                                break
                    elif ekey == "bottom":
                        ix0, ix1 = int(x), int(x + bw)
                        for ix in range(ix0, ix1, step):
                            if add_pick_from_edge((ix, y + bh), (0.0, 1.0), (0.0, -1.0)):
                                break
                # Fallback: if no picks found, scan the other edges (in case orientation is wrong)
                if not picks_local:
                    other_edges = ["left", "right"] if edges == ["top", "bottom"] else ["top", "bottom"]
                    for ekey in other_edges:
                        if ekey == "left":
                            iy0, iy1 = int(y), int(y + bh)
                            for iy in range(iy0, iy1, step):
                                if add_pick_from_edge((x, iy), (-1.0, 0.0), (1.0, 0.0)):
                                    break
                        elif ekey == "right":
                            iy0, iy1 = int(y), int(y + bh)
                            for iy in range(iy0, iy1, step):
                                if add_pick_from_edge((x + bw, iy), (1.0, 0.0), (-1.0, 0.0)):
                                    break
                        elif ekey == "top":
                            ix0, ix1 = int(x), int(x + bw)
                            for ix in range(ix0, ix1, step):
                                if add_pick_from_edge((ix, y), (0.0, -1.0), (0.0, 1.0)):
                                    break
                        elif ekey == "bottom":
                            ix0, ix1 = int(x), int(x + bw)
                            for ix in range(ix0, ix1, step):
                                if add_pick_from_edge((ix, y + bh), (0.0, 1.0), (0.0, -1.0)):
                                    break
                return picks_local

            # Legacy symbol linking block retained but mostly skipped since we no longer create symbol center nodes
            for n in self.nodes:
                if n.type in (NodeType.VALVE, NodeType.INSTRUMENT, NodeType.EQUIPMENT, NodeType.OFFPAGE, NodeType.UNKNOWN):
                    # Respect exclusion rules at linking stage too
                    sym_class = ""
                    if n.symbol_ids:
                        try:
                            sym_class = self.symbols[n.symbol_ids[0]].type.lower()
                        except Exception:
                            sym_class = ""
                    if sym_class in [c.lower() for c in self.cfg.link_exclude_classes] and sym_class not in [c.lower() for c in self.cfg.link_include_classes]:
                        continue
                    if n.type == NodeType.INSTRUMENT and ("instrument" in [r.lower() for r in self.cfg.link_exclude_roles]) and sym_class not in [c.lower() for c in self.cfg.link_include_classes]:
                        continue
                    center = np.array(n.position, dtype=float)
                    desired_links = int(self.cfg.ports_per_type.get(n.type.value, 1))
                    desired_links = max(1, min(desired_links, int(self.cfg.connect_symbol_max_links)))
                    picks: List[Tuple[int, float, Tuple[float, float]]] = []  # (node_id, dist, pos)

                    # Valve-specialized directional linking (exclude control valves)
                    is_control = False
                    if n.type == NodeType.VALVE:
                        try:
                            low = sym_class.lower()
                            is_control = ("control" in low) or any(c in low for c in [s.lower() for s in self.cfg.valve_directional_exclude_classes])
                        except Exception:
                            is_control = False
                    if (n.type == NodeType.VALVE and (self.cfg.valve_link_strategy in ("directional", "hybrid")) and not is_control) or n.type == NodeType.OFFPAGE or (n.type == NodeType.EQUIPMENT and ("reducer" in sym_class.lower() or "pump" in sym_class.lower() or "strainer" in sym_class.lower())):
                        try:
                            picks.extend(_valve_directional_picks(n))
                        except Exception:
                            pass

                    # First try: existing skeleton nodes (endpoints/junctions)
                    if sk_pos.shape[0] > 0 and ((self.cfg.valve_link_strategy in ("generic", "hybrid")) or (n.type == NodeType.VALVE and is_control)):
                        d = np.sqrt(((sk_pos - center) ** 2).sum(axis=1))
                        within = np.where(d <= float(self.cfg.connect_radius))[0]
                        order = within[np.argsort(d[within])] if within.size > 0 else np.array([], dtype=int)
                        picked_angles: List[float] = []
                        for idx in order:
                            vec = sk_pos[idx] - center
                            ang = float((math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0)
                            if any(min(abs(ang - a), 360.0 - abs(ang - a)) < float(self.cfg.angle_sep_min_deg) for a in picked_angles):
                                continue
                            picks.append((sk_ids[idx], float(d[idx]), tuple(sk_pos[idx])))
                            picked_angles.append(ang)
                            if len(picks) >= desired_links:
                                break

                    # Fallback: nearest skeleton pixels, promote to nodes
                    if len(picks) < desired_links and len(pts) > 0 and ((self.cfg.valve_link_strategy in ("generic", "hybrid")) or (n.type == NodeType.VALVE and is_control)):
                        pix_xy = np.array([[p[1], p[0]] for p in pts], dtype=float)  # (x,y)
                        dpx = np.sqrt(((pix_xy - center) ** 2).sum(axis=1))
                        within_px = np.where(dpx <= float(self.cfg.connect_radius))[0]
                        ord_px = within_px[np.argsort(dpx[within_px])] if within_px.size > 0 else np.array([], dtype=int)
                        picked_angles_px: List[float] = [
                            (math.degrees(math.atan2((np.array(p[2])[1] - center[1]), (np.array(p[2])[0] - center[0]))) + 360.0) % 360.0
                            for p in picks
                        ]
                        for j in ord_px:
                            pos = pix_xy[j]
                            vec = pos - center
                            ang = float((math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0)
                            if any(min(abs(ang - a), 360.0 - abs(ang - a)) < float(self.cfg.angle_sep_min_deg) for a in picked_angles_px):
                                continue
                            px_node = ensure_node_for_pixel((int(pos[0]), int(pos[1])))
                            picks.append((px_node, float(dpx[j]), (pos[0], pos[1])))
                            picked_angles_px.append(ang)
                            if len(picks) >= desired_links:
                                break

                    # Record debug picks for visualization
                    try:
                        self._symbol_pick_debug[n.id] = [(float(pos[0]), float(pos[1])) for (_, __, pos) in picks]
                    except Exception:
                        self._symbol_pick_debug[n.id] = []

                    created_ports: List[Tuple[int, Tuple[float, float]]] = []  # (port_node_id, port_pos)
                    for node_id, distv, pos in picks:
                        # Optionally create a PORT node on the bbox edge along the vector from symbol center to 'pos'
                        port_pos = None
                        if self.cfg.ports_on_bbox_edge and n.symbol_ids:
                            try:
                                sref = next((s for s in self.symbols if s.id == n.symbol_ids[0]), None)
                                if sref is not None:
                                    port_pos = self._line_rect_intersection(n.position, (float(pos[0]), float(pos[1])), sref.bbox)
                            except Exception:
                                port_pos = None
                        if port_pos is None:
                            port_pos = (float(pos[0]), float(pos[1]))

                        # Create a port node if using bbox port; otherwise directly to skeleton
                        if self.cfg.ports_on_bbox_edge:
                            port_id = add_node(port_pos, NodeType.PORT)
                            created_ports.append((port_id, port_pos))
                            # edge: symbol center -> port on bbox
                            self.edges.append(Edge(
                                id=eid_local,
                                source=n.id,
                                target=port_id,
                                path=[n.position, port_pos],
                                attributes={"inferred": True, "symbol_port": True},
                            ))
                            eid_local += 1
                            # edge: port -> skeleton/node pick
                            self.edges.append(Edge(
                                id=eid_local,
                                source=port_id,
                                target=node_id,
                                path=[port_pos, (float(pos[0]), float(pos[1]))],
                                attributes={"length": float(distv), "inferred": True, "from_port": True},
                            ))
                            eid_local += 1
                        else:
                            self.edges.append(Edge(
                                id=eid_local,
                                source=n.id,
                                target=node_id,
                                path=[n.position, (float(pos[0]), float(pos[1]))],
                                attributes={"length": float(distv), "inferred": True},
                            ))
                            eid_local += 1

                    # Optional: bridge across inline 2-port symbols so the line passes through
                    if self.cfg.bridge_through_symbol and desired_links >= 2 and n.type == NodeType.VALVE:
                        # Prefer using created PORT nodes if available
                        if len(created_ports) >= 2:
                            a_port_id, a_pos = created_ports[0]
                            b_port_id, b_pos = created_ports[1]
                            self.edges.append(Edge(
                                id=eid_local,
                                source=a_port_id,
                                target=b_port_id,
                                path=[(float(a_pos[0]), float(a_pos[1])), n.position, (float(b_pos[0]), float(b_pos[1]))],
                                attributes={"inferred": True, "via_symbol": int(n.id), "port_bridge": True},
                            ))
                            eid_local += 1

                    # Explicit raycast for connection-like classes to ensure at least 1 outward link
                    try:
                        sym_class = ""
                        if n.symbol_ids:
                            sym_class = self.symbols[n.symbol_ids[0]].type.lower()
                        is_connection_like = any(k in sym_class for k in ["connection", "page", "utility"]) or n.type == NodeType.OFFPAGE
                        # Count links already added for this node
                        link_count = sum(1 for e in self.edges if e.source == n.id or e.target == n.id)
                        if is_connection_like and link_count == 0:
                            best_hit = None
                            best_dist = 1e9
                            cx, cy = n.position
                            R = float(self.cfg.connection_search_radius)
                            rays = max(8, int(self.cfg.connection_raycast_angles))
                            # Prefer rays aligned with dominant line direction around the symbol
                            dom_ang = self._dominant_angle_in_window(cx, cy, self.cfg.connection_dir_window, default=None)
                            angs = [2.0 * math.pi * (k / rays) for k in range(rays)]
                            if dom_ang is not None:
                                def ang_dist(a, b):
                                    d = abs(a - b) % (2 * math.pi)
                                    return min(d, 2 * math.pi - d)
                                angs = sorted(angs, key=lambda a: min(ang_dist(a, dom_ang), ang_dist(a, (dom_ang + math.pi) % (2 * math.pi)))) # type: ignore
                            for ang in angs:
                                dx, dy = math.cos(ang), math.sin(ang)
                                for step in range(3, int(R)):
                                    x = int(round(cx + dx * step))
                                    y = int(round(cy + dy * step))
                                    if x < 0 or y < 0 or x >= w or y >= h:
                                        break
                                    if sk[y, x] > 0:
                                        d = math.hypot(x - cx, y - cy)
                                        best_hit = (x, y)
                                        best_dist = d
                                        break
                                if best_hit is not None:
                                    break
                            if best_hit is not None:
                                px_node = ensure_node_for_pixel(best_hit)
                                self.edges.append(Edge(
                                    id=eid_local,
                                    source=n.id,
                                    target=px_node,
                                    path=[n.position, (float(best_hit[0]), float(best_hit[1]))],
                                    attributes={"length": float(best_dist), "inferred": True, "raycast": True},
                                ))
                                eid_local += 1
                    except Exception:
                        pass

        # Optional gap-bridging between close endpoints to improve continuity
        try:
            maxd = float(self.cfg.bridge_max_dist)
            maxang = float(self.cfg.bridge_angle_max_deg)
            ep_ids = [n.id for n in self.nodes if n.type == NodeType.ENDPOINT]
            ep_pos = np.array([self.nodes[i].position for i in ep_ids], dtype=float) if ep_ids else np.zeros((0,2))
            # Compute tangent direction for each endpoint (vector from its only skeleton neighbor)
            ep_dir: Dict[int, Tuple[float, float]] = {}
            for node_id in ep_ids:
                x, y = map(int, self.nodes[node_id].position)
                nbrs = neighbors(x, y)
                if len(nbrs) == 1:
                    nx_, ny_ = nbrs[0]
                    vx, vy = x - nx_, y - ny_
                    nrm = math.hypot(vx, vy) + 1e-6
                    ep_dir[node_id] = (vx / nrm, vy / nrm)
            def angle_between(v1, v2) -> float:
                dot = v1[0]*v2[0] + v1[1]*v2[1]
                dot = max(-1.0, min(1.0, dot))
                return math.degrees(math.acos(dot))
            def not_connected(a: int, b: int) -> bool:
                return not any((e.source == a and e.target == b) or (e.source == b and e.target == a) for e in self.edges)
            eid_local2 = len(self.edges)
            for i in range(len(ep_ids)):
                for j in range(i+1, len(ep_ids)):
                    d = float(np.hypot(ep_pos[i][0]-ep_pos[j][0], ep_pos[i][1]-ep_pos[j][1]))
                    if d <= maxd and not_connected(ep_ids[i], ep_ids[j]):
                        a = int(ep_ids[i]); b = int(ep_ids[j])
                        va = ep_dir.get(a); vb = ep_dir.get(b)
                        ok = True
                        if va and vb:
                            # Expect directions roughly facing each other (180°) and colinear
                            ang = angle_between(va, (-vb[0], -vb[1]))
                            if ang > maxang:
                                ok = False
                        if ok:
                            pa = tuple(self.nodes[a].position)
                            pb = tuple(self.nodes[b].position)
                            self.edges.append(Edge(
                                id=eid_local2,
                                source=a,
                                target=b,
                                path=[pa, pb], # type: ignore
                                attributes={"length": d, "bridged_gap": True, "inferred": True},
                            ))
                            eid_local2 += 1
        except Exception as _:
            pass

        # Bridge small gaps between ports and nearby endpoints
        try:
            maxd = float(self.cfg.bridge_max_dist)
            maxang = float(self.cfg.bridge_angle_max_deg)
            port_ids = [n.id for n in self.nodes if n.type == NodeType.PORT]
            ep_ids = [n.id for n in self.nodes if n.type == NodeType.ENDPOINT]
            if port_ids and ep_ids:
                port_pos = np.array([self.nodes[i].position for i in port_ids], dtype=float)
                ep_pos = np.array([self.nodes[i].position for i in ep_ids], dtype=float)
                # Approximate direction at ports from connected edge (port->skeleton)
                port_dir: Dict[int, Tuple[float, float]] = {}
                for pid in port_ids:
                    # find a neighbor via edges
                    nb = next((e.target for e in self.edges if e.source == pid), None)
                    if nb is None:
                        nb = next((e.source for e in self.edges if e.target == pid), None)
                    if nb is not None:
                        px, py = self.nodes[pid].position
                        qx, qy = self.nodes[nb].position
                        vx, vy = qx - px, qy - py
                        nrm = math.hypot(vx, vy) + 1e-6
                        port_dir[pid] = (vx / nrm, vy / nrm)
                # Tangent direction for endpoints
                ep_dir: Dict[int, Tuple[float, float]] = {}
                for eid in ep_ids:
                    x, y = map(int, self.nodes[eid].position)
                    nbrs = neighbors(x, y)
                    if len(nbrs) == 1:
                        nx_, ny_ = nbrs[0]
                        vx, vy = x - nx_, y - ny_
                        nrm = math.hypot(vx, vy) + 1e-6
                        ep_dir[eid] = (vx / nrm, vy / nrm)
                def angle_between(v1, v2) -> float:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    dot = max(-1.0, min(1.0, dot))
                    return math.degrees(math.acos(dot))
                def not_connected(a: int, b: int) -> bool:
                    return not any((e.source == a and e.target == b) or (e.source == b and e.target == a) for e in self.edges)
                eid_local3 = len(self.edges)
                for i, pid in enumerate(port_ids):
                    for j, eid_ in enumerate(ep_ids):
                        d = float(np.hypot(port_pos[i][0]-ep_pos[j][0], port_pos[i][1]-ep_pos[j][1]))
                        if d <= maxd and not_connected(pid, eid_):
                            ok = True
                            vpe = (ep_pos[j][0]-port_pos[i][0], ep_pos[j][1]-port_pos[i][1])
                            nrm = math.hypot(vpe[0], vpe[1]) + 1e-6
                            vpe_u = (vpe[0]/nrm, vpe[1]/nrm)
                            pd = port_dir.get(pid)
                            ed = ep_dir.get(eid_)
                            if pd is not None:
                                angp = angle_between(pd, vpe_u)
                                if angp > maxang:
                                    ok = False
                            if ok and ed is not None:
                                ange = angle_between(ed, (-vpe_u[0], -vpe_u[1]))
                                if ange > maxang:
                                    ok = False
                            if ok:
                                pa = tuple(self.nodes[pid].position)
                                pb = tuple(self.nodes[eid_].position)
                                self.edges.append(Edge(
                                    id=eid_local3,
                                    source=pid,
                                    target=eid_,
                                    path=[pa, pb],
                                    attributes={"length": d, "bridged_gap": True, "inferred": True, "port_endpoint_bridge": True},
                                ))
                                eid_local3 += 1
        except Exception:
            pass

        # Process arrows (create ports only; no separate symbol-center nodes)
        arrow_annotations = []
        arrow_categories = [
            {"id": 1, "name": "up_arrow"},
            {"id": 2, "name": "down_arrow"},
            {"id": 3, "name": "left_arrow"},
            {"id": 4, "name": "right_arrow"}
        ]
        for s in self.symbols:
            if "arrow" not in s.type.lower():
                continue
            direction = None
            if "up" in s.type.lower():
                direction = "up"
            elif "down" in s.type.lower():
                direction = "down"
            elif "left" in s.type.lower():
                direction = "left"
            elif "right" in s.type.lower():
                direction = "right"
            if direction is None:
                continue
            # Create input and output ports at arrow bbox edges
            x, y, w, h = s.bbox
            if direction == "up":
                out_pos = (x + w/2, y)  # top (output)
                in_pos = (x + w/2, y + h)  # bottom (input)
            elif direction == "down":
                out_pos = (x + w/2, y + h)  # bottom (output)
                in_pos = (x + w/2, y)  # top (input)
            elif direction == "left":
                out_pos = (x, y + h/2)  # left (output)
                in_pos = (x + w, y + h/2)  # right (input)
            elif direction == "right":
                out_pos = (x + w, y + h/2)  # right (output)
                in_pos = (x, y + h/2)  # left (input)
            # Find the arrow symbol node (center) to attach ports to
            arrow_center_node_id = None
            try:
                arrow_center_node_id = next((n.id for n in self.nodes if (n.symbol_ids and n.symbol_ids[0] == s.id)), None)
            except Exception:
                arrow_center_node_id = None
            # Create ports
            in_port_id = add_node(in_pos, NodeType.PORT)
            out_port_id = add_node(out_pos, NodeType.PORT)
            # Attach ports to the arrow center node if available (for bookkeeping/overlays)
            try:
                if arrow_center_node_id is not None:
                    self.nodes[in_port_id].port_of = arrow_center_node_id
                    self.nodes[out_port_id].port_of = arrow_center_node_id
            except Exception:
                pass
            # Link ports to nearest skeleton (if available)
            for port_id, ppos, ptype in [(in_port_id, in_pos, "input"), (out_port_id, out_pos, "output")]:
                center = np.array(ppos, dtype=float)
                if sk_pos.shape[0] > 0:
                    d = np.sqrt(((sk_pos - center) ** 2).sum(axis=1))
                    within = np.where(d <= float(self.cfg.connect_radius))[0]
                    if within.size > 0:
                        order = within[np.argsort(d[within])]
                        sk_id = sk_ids[order[0]]
                        self.edges.append(Edge(
                            id=len(self.edges),
                            source=port_id,
                            target=sk_id,
                            path=[ppos, tuple(sk_pos[order[0]])],
                            attributes={"length": float(d[order[0]]), "inferred": True, "arrow_port": True, "port_type": ptype, "symbol_id": int(s.id)},
                        ))
            # Ensure both arrow ports are connected: if any port has no link, raycast outward
            def _port_has_link(pid: int) -> bool:
                return any(e.source == pid or e.target == pid for e in self.edges)
            # Determine preferred directions for raycast
            if direction == "up":
                pref_dirs = [(0.0, -1.0)]
            elif direction == "down":
                pref_dirs = [(0.0, 1.0)]
            elif direction == "left":
                pref_dirs = [(-1.0, 0.0)]
            elif direction == "right":
                pref_dirs = [(1.0, 0.0)]
            else:
                pref_dirs = []
            R = int(self.cfg.connection_search_radius)
            for pid, ppos in [(in_port_id, in_pos), (out_port_id, out_pos)]:
                if _port_has_link(pid):
                    continue
                hit = None
                # Try preferred directions first
                for dx, dy in pref_dirs:
                    for step in range(2, R + 1):
                        xh = int(round(ppos[0] + dx * step))
                        yh = int(round(ppos[1] + dy * step))
                        if xh < 0 or yh < 0 or xh >= w or yh >= h:
                            break
                        if sk[yh, xh] > 0:
                            hit = (xh, yh)
                            break
                    if hit is not None:
                        break
                # Fallback: 360° raycast
                if hit is None:
                    rays = max(8, int(self.cfg.connection_raycast_angles))
                    for k in range(rays):
                        ang = 2.0 * math.pi * (k / float(rays))
                        dx, dy = math.cos(ang), math.sin(ang)
                        for step in range(2, R + 1):
                            xh = int(round(ppos[0] + dx * step))
                            yh = int(round(ppos[1] + dy * step))
                            if xh < 0 or yh < 0 or xh >= w or yh >= h:
                                break
                            if sk[yh, xh] > 0:
                                hit = (xh, yh)
                                break
                        if hit is not None:
                            break
                if hit is not None:
                    px_node = ensure_node_for_pixel(hit)
                    distv = float(math.hypot(hit[0] - ppos[0], hit[1] - ppos[1]))
                    self.edges.append(Edge(
                        id=len(self.edges),
                        source=pid,
                        target=px_node,
                        path=[ppos, (float(hit[0]), float(hit[1]))],
                        attributes={"length": distv, "inferred": True, "arrow_port": True, "raycast": True, "symbol_id": int(s.id)},
                    ))
            # Log how many ports were created for this arrow
            try:
                num_ports = sum(1 for n in self.nodes if n.type == NodeType.PORT and n.port_of == arrow_center_node_id)
                logger.info(f"Arrow symbol id {s.id} ports: {num_ports} (expected 2)")
            except Exception:
                pass
            # Add to annotations
            cat_id = {"up":1, "down":2, "left":3, "right":4}[direction]
            arrow_annotations.append({
                "id": s.id,
                "image_id": 0,  # assume single image
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "score": s.confidence
            })
        # Save COCO
        arrow_coco = {
            "images": [{"id": 0, "file_name": os.path.basename(self.image_path)}],
            "categories": arrow_categories,
            "annotations": arrow_annotations
        }
        self._save_json("arrow_coco", arrow_coco)

        # Heuristic: for valves with one port, check opposite edge for adjacent symbols and add second port
        try:
            valve_nodes = [n for n in self.nodes if n.type == NodeType.VALVE]
            for valve in valve_nodes:
                if not valve.symbol_ids:
                    continue
                sid = valve.symbol_ids[0]
                bb = _bbox_for_symbol_id(sid)
                if bb is None:
                    continue
                x, y, w, h = bb
                # Get ports for this valve
                valve_ports = [n for n in self.nodes if n.type == NodeType.PORT and n.port_of == valve.id]
                if len(valve_ports) != 1:
                    continue  # Only for valves with exactly one port
                port = valve_ports[0]
                px, py = port.position
                # Determine edge of the port
                edge = None
                if abs(px - x) < 1e-3:
                    edge = "left"
                elif abs(px - (x + w)) < 1e-3:
                    edge = "right"
                elif abs(py - y) < 1e-3:
                    edge = "top"
                elif abs(py - (y + h)) < 1e-3:
                    edge = "bottom"
                if edge is None:
                    continue
                # Opposite edge
                if edge == "left":
                    opp_edge = "right"
                    opp_x = x + w
                    opp_y = y + h / 2
                elif edge == "right":
                    opp_edge = "left"
                    opp_x = x
                    opp_y = y + h / 2
                elif edge == "top":
                    opp_edge = "bottom"
                    opp_x = x + w / 2
                    opp_y = y + h
                elif edge == "bottom":
                    opp_edge = "top"
                    opp_x = x + w / 2
                    opp_y = y
                else:
                    continue
                # Check for adjacent symbols on opposite edge
                adjacent_found = False
                adj_thresh = 20.0  # px threshold for adjacency
                for s in self.symbols:
                    if s.id == sid:
                        continue  # Skip self
                    sx, sy, sw, sh = s.bbox
                    if opp_edge == "right" and abs(sx - (x + w)) < adj_thresh and max(sy, y) < min(sy + sh, y + h):
                        adjacent_found = True
                        break
                    elif opp_edge == "left" and abs((sx + sw) - x) < adj_thresh and max(sy, y) < min(sy + sh, y + h):
                        adjacent_found = True
                        break
                    elif opp_edge == "bottom" and abs(sy - (y + h)) < adj_thresh and max(sx, x) < min(sx + sw, x + w):
                        adjacent_found = True
                        break
                    elif opp_edge == "top" and abs((sy + sh) - y) < adj_thresh and max(sx, x) < min(sx + sw, x + w):
                        adjacent_found = True
                        break
                if adjacent_found:
                    # Add second port on opposite edge
                    opp_pos = (opp_x, opp_y)
                    opp_port_id = add_node(opp_pos, NodeType.PORT)
                    # Link to nearest skeleton
                    center = np.array(opp_pos, dtype=float)
                    if sk_pos.shape[0] > 0:
                        d = np.sqrt(((sk_pos - center) ** 2).sum(axis=1))
                        within = np.where(d <= float(self.cfg.connect_radius))[0]
                        if within.size > 0:
                            order = within[np.argsort(d[within])]
                            sk_id = sk_ids[order[0]]
                            self.edges.append(Edge(
                                id=len(self.edges),
                                source=opp_port_id,
                                target=sk_id,
                                path=[opp_pos, tuple(sk_pos[order[0]])],
                                attributes={"length": float(d[order[0]]), "inferred": True, "heuristic_second_port": True},
                            ))
                    # Also link valve to port
                    self.edges.append(Edge(
                        id=len(self.edges),
                        source=valve.id,
                        target=opp_port_id,
                        path=[valve.position, opp_pos],
                        attributes={"inferred": True, "symbol_port": True},
                    ))
        except Exception as _:
            pass
        
        # Valve connecting nodes only: images focusing solely on PORT nodes linked to valves
        try:
            if cv2 is not None and self.image_bgr is not None and self.skeleton is not None:
                H, W = self.skeleton.shape
                valve_ids = {n.id for n in self.nodes if n.type == NodeType.VALVE}
                connect_ids = set()
                for e in self.edges:
                    if e.source in valve_ids and self.nodes[e.target].type == NodeType.PORT and self.nodes[e.target].port_of in valve_ids:
                        connect_ids.add(e.target)
                    if e.target in valve_ids and self.nodes[e.source].type == NodeType.PORT and self.nodes[e.source].port_of in valve_ids:
                        connect_ids.add(e.source)
                # Build visuals: overlay, white-only points, and binary mask
                overlay = self.image_bgr.copy()
                white = np.full_like(self.image_bgr, 255)
                mask = np.zeros((H, W), dtype=np.uint8)
                # Draw red bboxes for valve symbols
                try:
                    for n in self.nodes:
                        if n.type != NodeType.VALVE or not n.symbol_ids:
                            continue
                        sid = n.symbol_ids[0]
                        srec = next((s for s in self.symbols if s.id == sid), None)
                        if srec is None:
                            continue
                        x, y, w, h = map(int, srec.bbox)
                        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.rectangle(white, (x, y), (x + w, y + h), (0, 0, 255), 2)
                except Exception:
                    pass
                for nid in connect_ids:
                    x, y = map(int, self.nodes[nid].position)
                    cv2.circle(overlay, (x, y), 4, (255, 0, 0), -1)
                    cv2.circle(white, (x, y), 4, (0, 0, 255), -1)
                    cv2.circle(mask, (x, y), 3, (255, 255, 255), -1)
                self._save_img("stage5_valve_connections_overlay", overlay)
                self._save_img("stage5_valve_connections_points", white)
                self._save_img("stage5_valve_connections_mask", mask)
        except Exception as ex:
            logger.warning(f"valve connection nodes overlay failed: {ex}")

        # Imagination line overlay for valves: straight path through valve aligned with incoming pipe
        try:
            if cv2 is not None and self.image_bgr is not None and self.skeleton is not None:
                vis3 = self.image_bgr.copy()
                out_meta = []
                # Build adjacency by node id for quick lookups
                adj_by_node: Dict[int, List[Edge]] = {}
                for e in self.edges:
                    adj_by_node.setdefault(e.source, []).append(e)
                    adj_by_node.setdefault(e.target, []).append(e)
                for v in [n for n in self.nodes if n.type == NodeType.VALVE]:
                    if not v.symbol_ids:
                        continue
                    sid = v.symbol_ids[0]
                    srec = next((s for s in self.symbols if s.id == sid), None)
                    if srec is None:
                        continue
                    bbox = srec.bbox
                    cx, cy = v.position
                    # Gather port nodes and estimate incoming direction(s)
                    port_nodes = [n for n in self.nodes if n.type == NodeType.PORT and n.port_of == v.id]
                    dirs = []
                    for p in port_nodes:
                        # find external edge from this port (from_port=True)
                        edges_p = [e for e in adj_by_node.get(p.id, []) if e.attributes.get("from_port")]
                        if not edges_p:
                            continue
                        # neighbor node id
                        e0 = edges_p[0]
                        nb_id = e0.target if e0.source == p.id else e0.source
                        nb_pos = self.nodes[nb_id].position
                        vec = (p.position[0] - nb_pos[0], p.position[1] - nb_pos[1])  # pointing into valve
                        nrm = math.hypot(vec[0], vec[1]) + 1e-6
                        dirs.append((vec[0] / nrm, vec[1] / nrm))
                    source = "ports" if dirs else "fallback"
                    if not dirs:
                        # Fallback orientation by template or narrow-side
                        ori = None
                        try:
                            ori = self._valve_orientation_by_template(bbox)
                        except Exception:
                            pass
                        if ori == 'h':
                            dirs = [(1.0, 0.0)]
                        elif ori == 'v':
                            dirs = [(0.0, 1.0)]
                        else:
                            x, y, w, h = bbox
                            dirs = [(1.0, 0.0)] if w >= h else [(0.0, 1.0)]
                    # Average directions (ensure direction is unit)
                    ux = sum(d[0] for d in dirs) / max(len(dirs), 1)
                    uy = sum(d[1] for d in dirs) / max(len(dirs), 1)
                    nrm = math.hypot(ux, uy) + 1e-6
                    dir_final = (ux / nrm, uy / nrm)
                    a, b = self._through_bbox_line((cx, cy), dir_final, bbox)
                    # Draw bbox and line
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(vis3, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    ax, ay = map(int, a)
                    bx, by = map(int, b)
                    # extend 8px beyond
                    ex = dir_final[0]
                    ey = dir_final[1]
                    a_ext = (int(ax - ex * 8), int(ay - ey * 8))
                    b_ext = (int(bx + ex * 8), int(by + ey * 8))
                    cv2.line(vis3, a_ext, b_ext, (0, 255, 255), 2)
                    cv2.circle(vis3, (ax, ay), 3, (0, 255, 255), -1)
                    cv2.circle(vis3, (bx, by), 3, (0, 255, 255), -1)
                    ang_deg = (math.degrees(math.atan2(dir_final[1], dir_final[0])) + 360.0) % 360.0
                    out_meta.append({
                        "valve_node_id": int(v.id),
                        "bbox": [float(b) for b in bbox],
                        "endpoints": [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]],
                        "angle_deg": float(ang_deg),
                        "source": source,
                    })
                self._save_img("stage5_valve_imageline", vis3)
                if out_meta:
                    self._save_json("stage5_valve_imageline", out_meta)
        except Exception as ex:
            logger.warning(f"valve imageline overlay failed: {ex}")
            
        # Build nx graph
        self.graph.clear()
        for n in self.nodes:
            self.graph.add_node(n.id, type=n.type.value, position=n.position, label=n.label)
        for e in self.edges:
            self.graph.add_edge(e.source, e.target, id=e.id, length=e.attributes.get("length", 0.0))

        # Optional: filter to only pipelines connected to connection/offpage nodes
        if self.cfg.trace_from_connections_only:
            offpage_nodes = [n.id for n in self.nodes if n.type == NodeType.OFFPAGE]
            if offpage_nodes:
                reachable = set(offpage_nodes)
                # For undirected graphs, use BFS to get connected nodes
                for start in offpage_nodes:
                    try:
                        for comp in nx.connected_components(self.graph):
                            if start in comp:
                                reachable.update(comp)
                    except Exception:
                        # Fallback simple BFS
                        queue = [start]
                        seen = {start}
                        while queue:
                            cur = queue.pop(0)
                            for nb in self.graph.neighbors(cur):
                                if nb not in seen:
                                    seen.add(nb)
                                    queue.append(nb)
                        reachable.update(seen)
                # Keep only nodes in reachable
                self.nodes = [n for n in self.nodes if n.id in reachable]
                # Keep only edges where both ends are in reachable
                self.edges = [e for e in self.edges if e.source in reachable and e.target in reachable]
                # Rebuild graph
                self.graph.clear()
                for n in self.nodes:
                    self.graph.add_node(n.id, type=n.type.value, position=n.position, label=n.label)
                for e in self.edges:
                    self.graph.add_edge(e.source, e.target, id=e.id, length=e.attributes.get("length", 0.0))

        self._save_json("stage5_stats", {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "endpoints": len(endpoints),
            "junctions": len(junctions),
        })

        # Coverage report and overlays
        cov_stats = {}
        cov_mask = None
        sk_uint = None
        if (self.cfg.coverage_report or self.cfg.hough_recover) and cv2 is not None:
            try:
                sk_uint = (self.skeleton.astype(np.uint8) * 255)
                cov = np.zeros_like(sk_uint)
                for e in self.edges:
                    pts = e.path
                    for i in range(len(pts) - 1):
                        x1, y1 = map(int, pts[i])
                        x2, y2 = map(int, pts[i + 1])
                        cv2.line(cov, (x1, y1), (x2, y2), 255, 1) # type: ignore
                cov_mask = cov
                if self.cfg.coverage_report:
                    uncovered = ((sk_uint > 0) & (cov == 0)).astype(np.uint8) * 255
                    self._save_img("stage5_uncovered", uncovered)
                    total = int((sk_uint > 0).sum())
                    covered = int(((sk_uint > 0) & (cov > 0)).sum())
                    cov_stats = {"skeleton_pixels": total, "covered_pixels": covered, "coverage": (covered/total if total>0 else 0.0)}
                    self._save_json("stage5_coverage", cov_stats)
            except Exception as _:
                pass

        # Hough-based recovery for long straight segments missing from graph coverage
        if self.cfg.hough_recover and cv2 is not None:
            try:
                # Use edges from Canny on stage2 binary/gray for strong long lines
                img_for_hough = self.gray if self.gray is not None else (self.skeleton * 255)
                edges_img = cv2.Canny(img_for_hough, self.cfg.canny_low, self.cfg.canny_high)
                theta = np.deg2rad(float(self.cfg.hough_theta_deg))
                linesP = cv2.HoughLinesP(edges_img, rho=float(self.cfg.hough_rho), theta=theta,
                                         threshold=int(self.cfg.hough_threshold),
                                         minLineLength=int(self.cfg.hough_min_line_length),
                                         maxLineGap=int(self.cfg.hough_max_line_gap))
                added = 0
                if linesP is not None:
                    for l in linesP[:4096]:
                        x1, y1, x2, y2 = map(int, l[0])
                        # Skip if coverage already exists at midpoint (to avoid duplicates)
                        if cov_mask is not None:
                            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            if 0 <= my < cov_mask.shape[0] and 0 <= mx < cov_mask.shape[1]:
                                if cov_mask[my, mx] > 0:
                                    continue
                        a_id = ensure_node_for_pixel((x1, y1))
                        b_id = ensure_node_for_pixel((x2, y2))
                        if a_id == b_id:
                            continue
                        self.edges.append(Edge(
                            id=len(self.edges),
                            source=a_id,
                            target=b_id,
                            path=[(float(x1), float(y1)), (float(x2), float(y2))],
                            attributes={"length": float(np.hypot(x2-x1, y2-y1)), "hough": True, "inferred": True},
                        ))
                        added += 1
                if added:
                    logger.info(f"Hough recovery added {added} edges")
            except Exception as ex:
                logger.warning(f"Hough recovery failed: {ex}")

        # Graph overlay (edges + nodes) - separate traced and links
        try:
            vis_traced = self.image_bgr.copy() # type: ignore
            vis_links = self.image_bgr.copy() # type: ignore
            if cv2 is not None:
                # Map symbol id -> bbox for quick access
                sym_bbox = {s.id: s.bbox for s in self.symbols}
                # Draw symbol bboxes in red and centers in orange on both
                for vis in [vis_traced]:
                    for node in self.nodes:
                        if not node.symbol_ids:
                            continue
                        sid = node.symbol_ids[0]
                        bb = sym_bbox.get(sid)
                        if bb is not None:
                            x, y, w, h = map(int, bb)
                            cv2.rectangle(vis, (x, y), (x + w, y + h), RED_COLOR, 2)
                        cx, cy = map(int, node.position)
                        cv2.circle(vis, (cx, cy), 4, ORANGE_COLOR, -1)  # orange center
                    for n in self.nodes:
                        x, y = map(int, n.position)
                        color = BLUE_COLOR
                        size = 4
                        if n.type == NodeType.PORT:
                            color = BLUE_COLOR
                            size = 6
                        elif n.type == NodeType.JUNCTION:
                            color = YELLOW_COLOR
                        elif n.type == NodeType.ENDPOINT:
                            color = RED_COLOR
                        else:
                            color = BLUE_COLOR
                        cv2.circle(vis, (x, y), size, color, -1)
                        
                # Draw edges
                for e in self.edges:
                    if e.attributes.get("bridged_gap"):
                        continue  # Skip bridged edges
                    pts = e.path
                    for i in range(len(pts) - 1):
                        x1, y1 = map(int, pts[i])
                        x2, y2 = map(int, pts[i + 1])
                        if e.attributes.get("symbol_port") or e.attributes.get("inferred"):
                            # Symbol links
                            if e.attributes.get("symbol_port"):
                                color = RED_COLOR # red
                            else:
                                color = BLUE_COLOR  # blue
                            cv2.line(vis_links, (x1, y1), (x2, y2), color, 2)
                        else:
                            # Traced
                            cv2.line(vis_traced, (x1, y1), (x2, y2), GREEN_COLOR, 2)
            self._save_img("stage5_graph_overlay_traced", vis_traced)
        except Exception as ex:
            logger.warning(f"graph overlay failed: {ex}")

        # # Symbol connections overlay (detailed): show symbol bboxes, centers, picks, and links
        # try:
        #     if cv2 is not None and self.image_bgr is not None:
        #         vis2 = self.image_bgr.copy()
        #         # Map symbol id -> bbox for quick access
        #         sym_bbox = {s.id: s.bbox for s in self.symbols}
        #         sym_type = {s.id: s.type for s in self.symbols}
        #         sym_text = {s.id: s.text for s in self.symbols}
        #         # Build adjacency from edges
        #         adj: Dict[int, List[int]] = {}
        #         for e in self.edges:
        #             adj.setdefault(e.source, []).append(e.target)
        #             adj.setdefault(e.target, []).append(e.source)
        #         for node in self.nodes:
        #             if not node.symbol_ids:
        #                 continue
        #             # draw symbol bbox and center
        #             sid = node.symbol_ids[0]
        #             bb = sym_bbox.get(sid)
        #             if bb is not None:
        #                 x, y, w, h = map(int, bb)
        #                 cv2.rectangle(vis2, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #                 label = (sym_type.get(sid, "") or "")[:14]
        #                 if sym_text.get(sid):
        #                     label = f"{label}:{str(sym_text[sid])[:12]}"
        #                 cv2.putText(vis2, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        #             cx, cy = map(int, node.position)
        #             cv2.circle(vis2, (cx, cy), 4, (0, 165, 255), -1)  # orange center
        #             # candidate picks
        #             for pos in self._symbol_pick_debug.get(node.id, []):
        #                 px, py = map(int, pos)
        #                 cv2.circle(vis2, (px, py), 3, (0, 255, 0), -1)
        #                 cv2.line(vis2, (cx, cy), (px, py), (0, 255, 0), 1)
        #             # final links (neighbors in graph)
        #             for nb in adj.get(node.id, []):
        #                 pos = self.nodes[nb].position
        #                 px, py = map(int, pos)
        #                 cv2.circle(vis2, (px, py), 3, (255, 0, 0), -1)
        #                 cv2.line(vis2, (cx, cy), (px, py), (255, 0, 0), 2)
        #         self._save_img("stage5_symbol_connections", vis2)
        # except Exception as ex:
        #     logger.warning(f"symbol connections overlay failed: {ex}")

        # Emit JSON for symbol links
        try:
            sym_links: List[Dict[str, Any]] = []
            for node in self.nodes:
                if not node.symbol_ids:
                    continue
                sid = node.symbol_ids[0]
                srec = next((s for s in self.symbols if s.id == sid), None)
                links = []
                for e in self.edges:
                    if e.source == node.id:
                        links.append({"to": int(e.target), "pos": list(map(float, self.nodes[e.target].position)), "length": float(e.attributes.get("length", 0.0))})
                    elif e.target == node.id:
                        links.append({"to": int(e.source), "pos": list(map(float, self.nodes[e.source].position)), "length": float(e.attributes.get("length", 0.0))})
                sym_links.append({
                    "node_id": int(node.id),
                    "symbol_id": int(sid),
                    "symbol_type": srec.type if srec else "",
                    "symbol_text": srec.text if srec else None,
                    "symbol_bbox": list(map(float, srec.bbox)) if srec else None,
                    "center": list(map(float, node.position)),
                    "candidate_picks": [list(map(float, p)) for p in self._symbol_pick_debug.get(node.id, [])],
                    "links": links,
                })
            if sym_links:
                self._save_json("stage5_symbol_links", sym_links)

            # Export traced pipelines to JSON (always emit)
            traced_edges = [
                    {
                        "id": e.id,
                        "source": e.source,
                        "target": e.target,
                        "path": [[float(p[0]), float(p[1])] for p in e.path],
                        "length": e.attributes.get("length", 0.0),
                        "attributes": e.attributes
                    }
                    for e in self.edges
                    if not e.attributes.get("inferred") and not e.attributes.get("bridged_gap")
                ]
            self._save_json("stage5_traced_pipelines", traced_edges)
        except Exception:
            pass
        
        # Overlay: draw symbol bounding boxes and ports/endpoints/junctions on the skeleton image
        try:
            if cv2 is not None and self.skeleton is not None:
                sk_vis = (self.skeleton.astype(np.uint8) * 255)
                if sk_vis.ndim == 2:
                    sk_vis = cv2.cvtColor(sk_vis, cv2.COLOR_GRAY2BGR)
                # Colors
                red = (0, 0, 255)      # symbols, endpoints
                green = (0, 255, 0)    # ports
                yellow = (0, 255, 255) # junctions
                # Draw symbol bounding boxes
                for s in self.symbols:
                    x, y, w, h = map(int, s.bbox)
                    cv2.rectangle(sk_vis, (x, y), (x + w, y + h), red, 2)
                # Draw nodes
                for n in self.nodes:
                    x, y = map(int, n.position)
                    if n.type == NodeType.PORT:
                        cv2.circle(sk_vis, (x, y), 5, green, -1)
                    elif n.type == NodeType.JUNCTION:
                        cv2.circle(sk_vis, (x, y), 4, yellow, -1)
                    elif n.type == NodeType.ENDPOINT:
                        cv2.circle(sk_vis, (x, y), 4, red, -1)
                self._save_img("stage5_skeleton_overlay", sk_vis)
        except Exception as ex:
            logger.warning(f"skeleton overlay failed: {ex}")
            
        # end if stage 5
        logger.info("Stage 5 done in %.2fs", time.time() - t0)

    # ---------- Stage 6: From/To ----------
    def stage6_fromto(self) -> List[Dict[str, Any]]:
        t0 = time.time()
        res: List[Dict[str, Any]] = []
        for e in self.edges:
            src = next((n for n in self.nodes if n.id == e.source), None)
            dst = next((n for n in self.nodes if n.id == e.target), None)
            if not src or not dst:
                continue
            res.append({
                "edge_id": e.id,
                "from_id": src.id,
                "from_type": src.type.value,
                "to_id": dst.id,
                "to_type": dst.type.value,
                "length": e.attributes.get("length", 0.0),
            })
        self._save_json("stage6_fromto", res)
        logger.info("Stage 6 done in %.2fs", time.time() - t0)
        return res

    # ---------- Stage 7: Export ----------
    def stage7_export(self) -> None:
        t0 = time.time()
        # GraphML
        nx.write_graphml(self.graph, self.out_dir / "graph.graphml")
        # From/To CSV
        df = pd.DataFrame(self.stage6_fromto())
        df.to_csv(self.out_dir / "from_to.csv", index=False)
        # DEXPI stub
        export_dexpi(self.graph, self.nodes, self.edges, out_path=str(self.out_dir / "dexpi.xml"),
                     doc_meta={"title": "P&ID Extraction", "source": os.path.basename(self.image_path)})
        logger.info("Stage 7 done in %.2fs", time.time() - t0)


def main():
    import argparse

    p = argparse.ArgumentParser("P&ID pipeline (staged)")
    p.add_argument("--image", required=True)
    p.add_argument("--coco", required=True)
    p.add_argument("--ocr", required=True)
    p.add_argument("--arrow-coco", help="Optional COCO file for arrow detections")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--stop-after", type=int, default=7, help="Run up to this stage (1-7)")
    args = p.parse_args()

    pipe = PIDPipeline(args.image, args.coco, args.ocr, out_dir=args.out, arrow_coco_path=getattr(args, 'arrow_coco', None))

    # Execute stages progressively
    pipe.stage1_ingest()
    if args.stop_after <= 1:
        return
    pipe.stage2_preprocess()
    if args.stop_after <= 2:
        return
    pipe.stage3_symbols_text()
    if args.stop_after <= 3:
        return
    pipe.stage4_linework()
    if args.stop_after <= 4:
        return
    pipe.stage5_graph()
    if args.stop_after <= 5:
        return
    pipe.stage6_fromto()
    if args.stop_after <= 6:
        return
    pipe.stage7_export()


if __name__ == "__main__":
    main()
