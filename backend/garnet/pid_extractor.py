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
from typing import Any, Dict, List, Optional, Tuple, Callable, Set

import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial import KDTree

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
from garnet.utils.deeplsd_utils import define_torch_device, load_deeplsd_model, detect_lines, draw_and_save_lines, export_lines_to_json, combine_close_lines
from garnet.connectivity_graph import ConnectivityEngine


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
    SYMBOL = "symbol" # Added
    TEXT = "text"     # Added
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

    def _refine_bbox(self, bbox: Tuple[float, float, float, float], padding: int = 2) -> Tuple[float, float, float, float]:
        """
        Shrinks the bounding box to tightly fit the foreground pixels in the binary image.
        Uses the binary mask to find the minimal rectangle containing non-zero pixels within the original bbox.
        """
        if self.binary is None:
            return bbox
            
        x, y, w, h = map(int, bbox)
        H, W = self.binary.shape
        
        # Clamp to image bounds
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        
        # Extract Region of Interest (ROI)
        if x1 <= x0 or y1 <= y0:
            return bbox
            
        roi = self.binary[y0:y1, x0:x1]
        
        # Find all foreground pixels
        if cv2 is not None:
            points = cv2.findNonZero(roi)
            if points is None:
                return bbox # Empty box, return original
            
            # Get bounding rect of the pixels
            rx, ry, rw, rh = cv2.boundingRect(points)
        else:
            # Numpy fallback
            ys, xs = np.where(roi > 0)
            if len(xs) == 0:
                return bbox
            rx, ry = np.min(xs), np.min(ys)
            rw = np.max(xs) - rx + 1
            rh = np.max(ys) - ry + 1

        # Calculate new global coordinates with padding
        # Ensure we don't expand beyond the original crop if not needed, but allow padding up to image bounds
        new_x = max(0, x0 + rx - padding)
        new_y = max(0, y0 + ry - padding)
        
        # Width/Height should accommodate the found content + padding
        new_w = rw + 2 * padding
        new_h = rh + 2 * padding
        
        # Optional: Ensure we don't shrink drastically if it looks like a mistake (e.g., < 10% area)?
        # For now, trust the binary mask.
        
        return (float(new_x), float(new_y), float(new_w), float(new_h))

    def _refine_bbox(self, bbox: Tuple[float, float, float, float], padding: int = 2, skip_trim: bool = False) -> Tuple[float, float, float, float]:
        """
        Shrinks the bounding box to tightly fit the symbol body, trimming both empty space
        and connecting pipelines (tails).
        """
        if self.binary is None or skip_trim:
            return bbox
            
        x, y, w, h = map(int, bbox)
        H, W = self.binary.shape
        
        # 1. Clamp and Extract ROI
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return bbox
            
        roi = self.binary[y0:y1, x0:x1]
        
        # 2. Find Tight Bounding Box (Trim Empty Space)
        if cv2 is not None:
            points = cv2.findNonZero(roi)
            if points is None: return bbox
            rx, ry, rw, rh = cv2.boundingRect(points)
        else:
            ys, xs = np.where(roi > 0)
            if len(xs) == 0: return bbox
            rx, ry = np.min(xs), np.min(ys)
            rw = np.max(xs) - rx + 1
            rh = np.max(ys) - ry + 1
            
        # Refined ROI coordinates relative to original ROI
        roi_sub = roi[ry:ry+rh, rx:rx+rw]
        
        # 3. Trim Tails (Pipelines)
        # Heuristic: A pipeline connection usually has a small cross-section (e.g. < 6px).
        # We shrink from edges inwards as long as the pixel count is small (but > 0).
        
        tail_thresh = 6  # max pixels to consider as "just a line"
        h_sub, w_sub = roi_sub.shape
        
        # Offsets relative to the tight box (rx, ry)
        trim_l, trim_r = 0, 0
        trim_t, trim_b = 0, 0
        
        # Trim Left
        for i in range(w_sub // 2):
            count = np.count_nonzero(roi_sub[:, i])
            if 0 < count <= tail_thresh: trim_l += 1
            elif count > tail_thresh: break
            
        # Trim Right
        for i in range(w_sub - 1, w_sub // 2, -1):
            count = np.count_nonzero(roi_sub[:, i])
            if 0 < count <= tail_thresh: trim_r += 1
            elif count > tail_thresh: break
            
        # Trim Top
        for i in range(h_sub // 2):
            count = np.count_nonzero(roi_sub[i, :])
            if 0 < count <= tail_thresh: trim_t += 1
            elif count > tail_thresh: break
            
        # Trim Bottom
        for i in range(h_sub - 1, h_sub // 2, -1):
            count = np.count_nonzero(roi_sub[i, :])
            if 0 < count <= tail_thresh: trim_b += 1
            elif count > tail_thresh: break

        # Calculate final coordinates
        # Start from original top-left (x0, y0) -> add tight box offset (rx, ry) -> add trim offsets
        final_x = x0 + rx + trim_l
        final_y = y0 + ry + trim_t
        final_w = rw - trim_l - trim_r
        final_h = rh - trim_t - trim_b
        
        # Safety: If we trimmed everything away (e.g. symbol is just lines), revert to tight box
        if final_w <= 0 or final_h <= 0:
            final_x, final_y, final_w, final_h = x0 + rx, y0 + ry, rw, rh

        # Apply padding
        final_x = max(0, final_x - padding)
        final_y = max(0, final_y - padding)
        final_w += 2 * padding
        final_h += 2 * padding
        
        return (float(final_x), float(final_y), float(final_w), float(final_h))

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
            
            # Check for no-shrink classes
            no_shrink_list = ["page connection", "off page", "connection", "utility connection"]
            skip_trim = any(ns in tkey for ns in no_shrink_list)

            # Refine bbox using binary mask for all symbols
            final_bbox = self._refine_bbox((x, y, w, h), skip_trim=skip_trim)
            
            # Recalculate center based on new bbox
            fx, fy, fw, fh = final_bbox
            cx, cy = fx + fw / 2.0, fy + fh / 2.0

            self.symbols.append(Symbol(
                id=i,
                type=str(typ),
                bbox=final_bbox,
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
            
            # Refine text bbox as well
            final_bbox = self._refine_bbox((x, y, w, h))
            fx, fy, fw, fh = final_bbox
            cx, cy = fx + fw / 2.0, fy + fh / 2.0
            
            self.texts.append(TextItem(
                id=i,
                text=str(t.get("text", "")),
                bbox=final_bbox,
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
    def _create_skeleton_overlay_with_boxes(self) -> None:
        """Create and save initial overlay of skeleton with symbol bounding boxes."""
        try:
            if self.skeleton is None:
                raise RuntimeError("Skeleton is None; run stage4_linework first")
            sk_vis = (self.skeleton.astype(np.uint8) * 255)
            if cv2 is not None:
                if sk_vis.ndim == 2:
                    sk_vis = cv2.cvtColor(sk_vis, cv2.COLOR_GRAY2BGR)
                RED_COLOR = (0, 0, 255)
                for s in self.symbols:
                    x, y, w, h = map(int, s.bbox)
                    cv2.rectangle(sk_vis, (x, y), (x + w, y + h), RED_COLOR, 2)
            elif Image is not None:
                pil = Image.fromarray(sk_vis if sk_vis.ndim == 2 else sk_vis[:, :, ::-1])
                dr = ImageDraw.Draw(pil)  # type: ignore
                for s in self.symbols:
                    x, y, w, h = map(int, s.bbox)
                    dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
                sk_vis = np.array(pil)
                if sk_vis.ndim == 3:
                    sk_vis = sk_vis[:, :, ::-1]
            self._save_img("stage5_step1_skeleton_boxes", sk_vis)
        except Exception as ex:
            logger.warning(f"stage5 overlay (skeleton+boxes) failed: {ex}")

    def _setup_graph_helpers(self) -> Tuple[Callable[[int, int], int], Callable[[Tuple[float, float], NodeType], int], Callable[[Tuple[int, int]], int], Optional[np.ndarray], Optional[np.ndarray], Callable[[np.ndarray, int, int, int], Optional[Tuple[int, int]]], Callable[[Tuple[float, float], int], Tuple[Optional[Tuple[int, int]], str]]]:
        """Initialize graph building helpers: neighbors_xy, add_node, ensure_node_for_pixel, bin_im, canny_im, _ring_search, find_connection_hit."""
        sk = self.skeleton
        if sk is None:
            raise RuntimeError("Skeleton is None; run stage4_linework first")
        h, w = sk.shape

        def neighbors_xy(x: int, y: int) -> int:
            deg = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < w and 0 <= yy < h and sk[yy, xx] > 0:
                        deg += 1
            return deg

        def add_node(pos: Tuple[float, float], ntype: NodeType) -> int:
            nid = len(self.nodes)
            self.nodes.append(Node(id=nid, position=pos, type=ntype))
            return nid

        def ensure_node_for_pixel(px: Tuple[int, int]) -> int:
            # If a node already exists at this exact pixel, reuse it
            for n in self.nodes:
                if int(round(n.position[0])) == px[0] and int(round(n.position[1])) == px[1] and n.type in (NodeType.ENDPOINT, NodeType.JUNCTION):
                    return n.id
            deg = neighbors_xy(px[0], px[1])
            ntype = NodeType.JUNCTION if deg >= 2 else NodeType.ENDPOINT
            return add_node((float(px[0]), float(px[1])), ntype)

        # Build auxiliary maps from original image to recover lines missing in skeleton
        bin_im = None
        canny_im = None
        try:
            if self.binary is not None:
                bin_im = (self.binary > 0).astype(np.uint8)
        except Exception:
            bin_im = None
        try:
            if cv2 is not None and self.gray is not None:
                canny_im = cv2.Canny(self.gray, self.cfg.canny_low, self.cfg.canny_high)
        except Exception:
            canny_im = None

        def _ring_search(img: np.ndarray, cx: int, cy: int, R: int) -> Optional[Tuple[int, int]]:
            # Search outward square rings for a nonzero pixel
            for r in range(0, int(R) + 1):
                x0, x1 = max(0, cx - r), min(w - 1, cx + r)
                y0, y1 = max(0, cy - r), min(h - 1, cy + r)
                # rows
                for x in range(x0, x1 + 1):
                    if img[y0, x] > 0:
                        return (x, y0)
                    if img[y1, x] > 0:
                        return (x, y1)
                # cols
                for y in range(y0, y1 + 1):
                    if img[y, x0] > 0:
                        return (x0, y)
                    if img[y, x1] > 0:
                        return (x1, y)
            return None

        def find_connection_hit(ppos: Tuple[float, float], R: int) -> Tuple[Optional[Tuple[int, int]], str]:
            """Find a nearby connection pixel, preferring skeleton, then binary, then Canny.
            Returns ((x,y), source) where source in {"skeleton","binary","canny","none"}.
            """
            cx, cy = int(round(ppos[0])), int(round(ppos[1]))
            # 1) Skeleton
            if 0 <= cx < w and 0 <= cy < h and sk[cy, cx] > 0:
                return (cx, cy), "skeleton"
            hit = _ring_search(sk, cx, cy, R)
            if hit is not None:
                return hit, "skeleton"
            # 2) Binary
            if bin_im is not None:
                hit = _ring_search(bin_im, cx, cy, R)
                if hit is not None:
                    return hit, "binary"
            # 3) Canny on original image
            if canny_im is not None:
                hit = _ring_search(canny_im, cx, cy, R)
                if hit is not None:
                    return hit, "canny"
            return None, "none"

        def _ring_search_original(self, img: Optional[np.ndarray], cx: int, cy: int, R: int) -> Optional[Tuple[int, int]]:
            if img is None:
                return None
            h_img, w_img = img.shape
            for r in range(0, int(R) + 1):
                x0, x1 = max(0, cx - r), min(w_img - 1, cx + r)
                y0, y1 = max(0, cy - r), min(h_img - 1, cy + r)
                # rows
                for x in range(x0, x1 + 1):
                    if img[y0, x] > 0:
                        return (x, y0)
                    if img[y1, x] > 0:
                        return (x, y1)
                # cols
                for y in range(y0, y1 + 1):
                    if img[y, x0] > 0:
                        return (x0, y)
                    if img[y, x1] > 0:
                        return (x1, y)
            return None

        return neighbors_xy, add_node, ensure_node_for_pixel, bin_im, canny_im, _ring_search, find_connection_hit

    def _process_arrows(self, add_node: Callable[[Tuple[float, float], NodeType], int], canny_im: Optional[np.ndarray], bin_im: Optional[np.ndarray]) -> int:
        """Process arrow symbols using original-image port detection:
        - Do not use skeleton to place ports.
        - Create 2 ports on opposite bbox edges that align in a straight line.
        - Validate via raycasting back into the bbox, then connect to nearest original-image line.
        """
        created_edges = 0
        # Ensure Canny available
        if canny_im is None and cv2 is not None and self.gray is not None:
            canny_im = cv2.Canny(self.gray, self.cfg.canny_low, self.cfg.canny_high)
        bin_img = bin_im if bin_im is not None else ((self.binary > 0).astype(np.uint8) if self.binary is not None else None)

        def connect_arrow_port(ppos: Tuple[float, float]) -> Optional[int]:
            R = int(max(10, getattr(self.cfg, 'connect_radius', 100)))
            hit, src = self._find_connection_hit_original(ppos, R, canny_im, bin_img)
            if hit is None:
                return None
            pid = add_node(ppos, NodeType.PORT)
            tgt = add_node((float(hit[0]), float(hit[1])), NodeType.ENDPOINT)
            self.edges.append(Edge(
                id=len(self.edges), source=pid, target=tgt,
                path=[ppos, (float(hit[0]), float(hit[1]))],
                attributes={"arrow_port": True, "hit_src": src},
            ))
            return pid

        for s in self.symbols:
            low = s.type.lower()
            if "arrow" not in low:
                continue
            x, y, bw, bh = s.bbox
            # Detect crossings on original image
            crossings_left = self._detect_line_crossings_on_border((x, y, bw, bh), 'left', canny_im, bin_img)
            crossings_right = self._detect_line_crossings_on_border((x, y, bw, bh), 'right', canny_im, bin_img)
            crossings_top = self._detect_line_crossings_on_border((x, y, bw, bh), 'top', canny_im, bin_img)
            crossings_bottom = self._detect_line_crossings_on_border((x, y, bw, bh), 'bottom', canny_im, bin_img)

            ports: List[Tuple[float, float]] = []
            # Prefer straight opposite-edge alignment
            ports = self._find_straight_line_ports((x, y, bw, bh), crossings_left, crossings_right, crossings_top, crossings_bottom, canny_im, bin_img)

            if not ports:
                # Fallback: midpoints of opposite edges along longer dimension, if validated
                if bw >= bh:
                    c_top = (x + bw / 2.0, y)
                    c_bot = (x + bw / 2.0, y + bh)
                    if self._raycast_back_into_bbox(c_top, (0, +bh), canny_im, bin_img, length=12, min_hits=2) and \
                       self._raycast_back_into_bbox(c_bot, (0, -bh), canny_im, bin_img, length=12, min_hits=2):
                        ports = [c_top, c_bot]
                else:
                    c_left = (x, y + bh / 2.0)
                    c_right = (x + bw, y + bh / 2.0)
                    if self._raycast_back_into_bbox(c_left, (+bw, 0), canny_im, bin_img, length=12, min_hits=2) and \
                       self._raycast_back_into_bbox(c_right, (-bw, 0), canny_im, bin_img, length=12, min_hits=2):
                        ports = [c_left, c_right]

            for p in ports:
                if connect_arrow_port(p) is not None:
                    created_edges += 1

        return created_edges

    def _save_arrow_overlay(self) -> None:
        """Save overlay showing arrow ports over skeleton."""
        try:
            if self.skeleton is None:
                return
            sk_vis2 = (self.skeleton.astype(np.uint8) * 255)
            if cv2 is not None:
                if sk_vis2.ndim == 2:
                    sk_vis2 = cv2.cvtColor(sk_vis2, cv2.COLOR_GRAY2BGR)
                RED = (0,0,255); BLUE = (255,0,0)
                for s in self.symbols:
                    x, y, w2, h2 = map(int, s.bbox)
                    cv2.rectangle(sk_vis2, (x, y), (x + w2, y + h2), RED, 1)
                for n in self.nodes:
                    if n.type == NodeType.PORT:
                        x, y = map(int, n.position)
                        cv2.circle(sk_vis2, (x, y), 5, BLUE, -1)
                for e in self.edges:
                    x1, y1 = map(int, e.path[0]); x2, y2 = map(int, e.path[-1])
                    cv2.line(sk_vis2, (x1, y1), (x2, y2), BLUE, 2)
            self._save_img("stage5_step2_arrows", sk_vis2)
        except Exception:
            pass

    # ---------- Original-image hit helpers ----------
    def _ring_search_original(self, img: Optional[np.ndarray], cx: int, cy: int, R: int) -> Optional[Tuple[int, int]]:
        if img is None:
            return None
        h_img, w_img = img.shape
        for r in range(0, int(R) + 1):
            x0, x1 = max(0, cx - r), min(w_img - 1, cx + r)
            y0, y1 = max(0, cy - r), min(h_img - 1, cy + r)
            # rows
            for x in range(x0, x1 + 1):
                if img[y0, x] > 0:
                    return (x, y0)
                if img[y1, x] > 0:
                    return (x, y1)
            # cols
            for y in range(y0, y1 + 1):
                if img[y, x0] > 0:
                    return (x0, y)
                if img[y, x1] > 0:
                    return (x1, y)
        return None

    def _find_connection_hit_original(self, ppos: Tuple[float, float], R: int, canny_img: Optional[np.ndarray], bin_img: Optional[np.ndarray]) -> Tuple[Optional[Tuple[int, int]], str]:
        """Find nearby connection hit on original image (Canny preferred, binary fallback)."""
        cx, cy = int(round(ppos[0])), int(round(ppos[1]))
        if canny_img is not None:
            h_img, w_img = canny_img.shape
            if 0 <= cx < w_img and 0 <= cy < h_img and canny_img[cy, cx] > 0:
                return (cx, cy), "canny"
            hit = self._ring_search_original(canny_img, cx, cy, R)
            if hit is not None:
                return hit, "canny"
        if bin_img is not None:
            hit = self._ring_search_original(bin_img, cx, cy, R)
            if hit is not None:
                return hit, "binary"
        return None, "none"

    # ---------- Objects (non-arrows) port scanning on original image ----------
    def _process_other_symbols(self, add_node: Callable[[Tuple[float, float], NodeType], int], ensure_node_for_pixel: Callable[[Tuple[int, int]], int], find_connection_hit: Callable[[Tuple[float, float], int], Tuple[Optional[Tuple[int, int]], str]], bin_im: Optional[np.ndarray], canny_im: Optional[np.ndarray]) -> int:
        """Scan ports for non-arrow objects on original image per rules and connect to nearby lines.
        - Uses original Canny/binary for border crossings and hit finding.
        - Ports lie on bbox border (not at corners), with inward ray validation.
        - Inline (valve/reducer): two opposite-side ports aligned and near center.
        - Single-port classes: page connection/connection/utility connection.
        """
        created = 0
        # Ensure Canny available
        if canny_im is None and cv2 is not None and self.gray is not None:
            canny_im = cv2.Canny(self.gray, self.cfg.canny_low, self.cfg.canny_high)
        bin_img = bin_im if bin_im is not None else ((self.binary > 0).astype(np.uint8) if self.binary is not None else None)

        def try_connect_port(ppos: Tuple[float, float]) -> Optional[int]:
            R = int(max(10, getattr(self.cfg, 'connect_radius', 100)))
            hit, src = self._find_connection_hit_original(ppos, R, canny_im, bin_img)
            if hit is None:
                return None
            port_id = add_node(ppos, NodeType.PORT)
            tgt = add_node((float(hit[0]), float(hit[1])), NodeType.ENDPOINT)
            self.edges.append(Edge(
                id=len(self.edges), source=port_id, target=tgt,
                path=[ppos, (float(hit[0]), float(hit[1]))],
                attributes={"object_port": True, "hit_src": src},
            ))
            return port_id

        for s in self.symbols:
            low = s.type.lower()
            # Exclude arrows explicitly, and skip obvious non-inline text-like classes
            if "arrow" in low or "line number" in low:
                continue
            x, y, bw, bh = s.bbox

            # Compute crossings per edge on original image
            crossings_left = self._detect_line_crossings_on_border((x, y, bw, bh), 'left', canny_im, bin_img)
            crossings_right = self._detect_line_crossings_on_border((x, y, bw, bh), 'right', canny_im, bin_img)
            crossings_top = self._detect_line_crossings_on_border((x, y, bw, bh), 'top', canny_im, bin_img)
            crossings_bottom = self._detect_line_crossings_on_border((x, y, bw, bh), 'bottom', canny_im, bin_img)
            all_crossings = crossings_left + crossings_right + crossings_top + crossings_bottom

            ports: List[Tuple[float, float]] = []
            if ("valve" in low) or ("reducer" in low):
                # Inline two-port, prefer opposite-side and center aligned
                ports = self._find_straight_line_ports((x, y, bw, bh), crossings_left, crossings_right, crossings_top, crossings_bottom, canny_im, bin_img)
                if not ports and len(all_crossings) >= 2:
                    # Fallback: choose two crossings on opposite sides closest to center
                    cx, cy = x + bw / 2.0, y + bh / 2.0
                    def center_dist(p):
                        return (p[0] - cx) ** 2 + (p[1] - cy) ** 2
                    all_crossings.sort(key=center_dist)
                    ports = all_crossings[:2]
            elif ("page connection" in low):
                # Single-port on narrow side only
                if bw < bh:
                    allowed_edges = ['top', 'bottom']
                elif bh < bw:
                    allowed_edges = ['left', 'right']
                else:
                    allowed_edges = None  # square-ish; no strict narrow side
                sp = self._find_single_port((x, y, bw, bh), all_crossings, canny_im, bin_img, allowed_edges=allowed_edges)
                if sp is not None:
                    ports = [sp]
            elif ("utility connection" in low) or ("connection" in low):
                # Single-port objects (no narrow-side restriction)
                sp = self._find_single_port((x, y, bw, bh), all_crossings, canny_im, bin_img)
                if sp is not None:
                    ports = [sp]
            else:
                # Generic: if two good crossings exist on opposite edges and align, use them
                cand = self._find_straight_line_ports((x, y, bw, bh), crossings_left, crossings_right, crossings_top, crossings_bottom, canny_im, bin_img)
                if cand:
                    ports = cand

            # Connect discovered ports
            for p in ports:
                pid = try_connect_port(p)
                if pid is not None:
                    created += 1

        return created

    def _detect_line_crossings_on_border(self, bbox: Tuple[float, float, float, float], edge: str, canny_img: np.ndarray, binary_img: Optional[np.ndarray] = None, step: int = 2, corner_tol: int = 4, outward_px: int = 8, inward_px: int = 6, min_hits: int = 2) -> List[Tuple[float, float]]:
        """Detect points on a specific bbox edge where a straight line from the original image crosses the border.
        Rules satisfied:
         - Uses original image (Canny preferred; binary fallback) — not skeleton.
         - Port point lies exactly on bbox border; never near corners.
         - Crossing validated by short outward and inward raycasts perpendicular to the edge; when raycasting, run back to bbox for the port.
        Returns list of candidate crossing points on the border (px, py).
        """
        x, y, bw, bh = bbox
        crossings: List[Tuple[float, float]] = []
        if canny_img is None:
            return crossings

        H, W = canny_img.shape

        def in_bounds(xx: int, yy: int) -> bool:
            return 0 <= xx < W and 0 <= yy < H

        def is_corner(px: float, py: float, tol: float = corner_tol) -> bool:
            corners = [(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)]
            for cx, cy in corners:
                if math.hypot(px - cx, py - cy) <= tol:
                    return True
            return False

        def has_hits_along_ray(px: float, py: float, dx: int, dy: int, length: int) -> bool:
            hits = 0
            for t in range(1, length + 1):
                xi = int(round(px + dx * t))
                yi = int(round(py + dy * t))
                if not in_bounds(xi, yi):
                    break
                val = (canny_img[yi, xi] > 0) or (binary_img is not None and binary_img[yi, xi] > 0)
                if val:
                    hits += 1
                    if hits >= min_hits:
                        return True
                else:
                    hits = 0  # require consecutive hits
            return False

        # Perpendicular directions for each edge (outward = away from bbox, inward = into bbox)
        if edge == 'left':
            px = x
            for iy in range(int(y + corner_tol), int(y + bh - corner_tol) + 1, step):
                if is_corner(px, float(iy)):
                    continue
                # outward left (-1, 0), inward right (+1, 0)
                if has_hits_along_ray(px, iy, -1, 0, outward_px) and has_hits_along_ray(px, iy, +1, 0, inward_px):
                    crossings.append((float(px), float(iy)))
        elif edge == 'right':
            px = x + bw
            for iy in range(int(y + corner_tol), int(y + bh - corner_tol) + 1, step):
                if is_corner(px, float(iy)):
                    continue
                if has_hits_along_ray(px, iy, +1, 0, outward_px) and has_hits_along_ray(px, iy, -1, 0, inward_px):
                    crossings.append((float(px), float(iy)))
        elif edge == 'top':
            py = y
            for ix in range(int(x + corner_tol), int(x + bw - corner_tol) + 1, step):
                if is_corner(float(ix), py):
                    continue
                if has_hits_along_ray(ix, py, 0, -1, outward_px) and has_hits_along_ray(ix, py, 0, +1, inward_px):
                    crossings.append((float(ix), float(py)))
        elif edge == 'bottom':
            py = y + bh
            for ix in range(int(x + corner_tol), int(x + bw - corner_tol) + 1, step):
                if is_corner(float(ix), py):
                    continue
                if has_hits_along_ray(ix, py, 0, +1, outward_px) and has_hits_along_ray(ix, py, 0, -1, inward_px):
                    crossings.append((float(ix), float(py)))
        return crossings

    def _find_straight_line_ports(self, bbox: Tuple[float, float, float, float], crossings_left: List[Tuple[float, float]], crossings_right: List[Tuple[float, float]], crossings_top: List[Tuple[float, float]], crossings_bottom: List[Tuple[float, float]], canny_img: np.ndarray, binary_img: Optional[np.ndarray], max_align_dev: float = 4.0) -> List[Tuple[float, float]]:
        """Pick two ports on opposite sides that lie on a straight line through the bbox.
        - Try both horizontal (L-R) and vertical (T-B) alignments.
        - Prefer the pair whose midpoint is closest to bbox center ("usually middle of edges").
        - Validate each candidate by short raycasts back into the bbox along the connecting direction.
        Returns up to two port points (on the border).
        """
        x, y, bw, bh = bbox

        def valid_pair(p0: Tuple[float, float], p1: Tuple[float, float]) -> bool:
            # Direction from p0 to p1, then raycast a few pixels back into bbox from each end
            dx, dy = (p1[0] - p0[0], p1[1] - p0[1])
            n = math.hypot(dx, dy)
            if n == 0:
                return False
            ux, uy = dx / n, dy / n
            # back directions into bbox from each port
            return (
                self._raycast_back_into_bbox(p0, (ux, uy), canny_img, binary_img, length=12, min_hits=2) and
                self._raycast_back_into_bbox(p1, (-ux, -uy), canny_img, binary_img, length=12, min_hits=2)
            )

        center = (x + bw / 2.0, y + bh / 2.0)
        best: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None

        # Horizontal candidates
        for pL in crossings_left:
            for pR in crossings_right:
                if abs(pL[1] - pR[1]) <= max_align_dev:
                    mid = ((pL[0] + pR[0]) / 2.0, (pL[1] + pR[1]) / 2.0)
                    score = math.hypot(mid[0] - center[0], mid[1] - center[1])
                    if valid_pair(pL, pR):
                        if best is None or score < best[2]:
                            best = (pL, pR, score)

        # Vertical candidates
        for pT in crossings_top:
            for pB in crossings_bottom:
                if abs(pT[0] - pB[0]) <= max_align_dev:
                    mid = ((pT[0] + pB[0]) / 2.0, (pT[1] + pB[1]) / 2.0)
                    score = math.hypot(mid[0] - center[0], mid[1] - center[1])
                    if valid_pair(pT, pB):
                        if best is None or score < best[2]:
                            best = (pT, pB, score)

        if best is not None:
            return [best[0], best[1]]
        return []

    def _find_single_port(self, bbox: Tuple[float, float, float, float], all_crossings: List[Tuple[float, float]], canny_img: np.ndarray, binary_img: Optional[np.ndarray], allowed_edges: Optional[List[str]] = None) -> Optional[Tuple[float, float]]:
        """Pick a single port for 1-port objects.
        - If allowed_edges provided (e.g., for "page connection"), restrict to those edges only.
        - Prefer the middle-most crossing on the longest eligible edge with validated crossings.
        - Validate by raycasting back into the bbox from the border point.
        """
        x, y, bw, bh = bbox
        edges: Dict[str, List[Tuple[float, float]]] = {'left': [], 'right': [], 'top': [], 'bottom': []}

        for px, py in all_crossings:
            if abs(px - x) <= 1:
                edges['left'].append((px, py))
            elif abs(px - (x + bw)) <= 1:
                edges['right'].append((px, py))
            elif abs(py - y) <= 1:
                edges['top'].append((px, py))
            elif abs(py - (y + bh)) <= 1:
                edges['bottom'].append((px, py))

        # choose edge with most crossings; tie-break by edge length
        edge_lengths = {'left': bh, 'right': bh, 'top': bw, 'bottom': bw}
        candidates = ['left', 'right', 'top', 'bottom']
        if allowed_edges is not None:
            allowed_set = set(allowed_edges)
            candidates = [e for e in candidates if e in allowed_set]
            if not candidates:
                candidates = allowed_edges  # fallback to whatever passed
        ranked = sorted(candidates, key=lambda e: (len(edges[e]), edge_lengths[e]), reverse=True)
        for e in ranked:
            if not edges[e]:
                continue
            if e in ('left', 'right'):
                mid = y + bh / 2.0
                cand = min(edges[e], key=lambda p: abs(p[1] - mid))
                direction = (bw if e == 'left' else -bw, 0)
            else:
                mid = x + bw / 2.0
                cand = min(edges[e], key=lambda p: abs(p[0] - mid))
                direction = (0, bh if e == 'top' else -bh)
            if self._raycast_back_into_bbox(cand, direction, canny_img, binary_img, length=12, min_hits=2):
                return cand
        return None

    def _raycast_back_into_bbox(self, start: Tuple[float, float], direction: Tuple[float, float], canny_img: np.ndarray, binary_img: Optional[np.ndarray], length: int = 20, min_hits: int = 3) -> bool:
        """Raycast back into bbox from start in direction; validate if sufficient foreground pixels."""
        dx, dy = direction
        norm = math.hypot(dx, dy)
        if norm == 0:
            return False
        ux, uy = dx / norm, dy / norm
        h_img, w_img = canny_img.shape
        hits = 0
        for d in range(1, length + 1):
            xx = int(round(start[0] + ux * d))  # Into bbox (note: + for forward)
            yy = int(round(start[1] + uy * d))
            if not (0 <= xx < w_img and 0 <= yy < h_img):
                break
            is_foreground = False
            if canny_img[yy, xx] > 0:
                is_foreground = True
            elif binary_img is not None and binary_img[yy, xx] > 0:
                is_foreground = True
            if is_foreground:
                hits += 1
                if hits >= min_hits:
                    return True
            else:
                hits = 0
        return False

        def try_connect_port(ppos: Tuple[float, float]) -> Optional[int]:
            # Updated to use original image for hit detection (Canny preferred)
            hit, src = self._find_connection_hit_original(ppos, int(max(10, getattr(self.cfg, 'connect_radius', 100))), canny_im, bin_im)
            if hit is None:
                return None
            port_id = add_node(ppos, NodeType.PORT)
            # For original image hit, create endpoint node at hit
            tgt = add_node((float(hit[0]), float(hit[1])), NodeType.ENDPOINT)
            self.edges.append(Edge(
                id=len(self.edges),
                source=port_id,
                target=tgt,
                path=[ppos, (float(hit[0]), float(hit[1]))],
                attributes={"object_port": True, "hit_src": src},
            ))
            return port_id

        other_edges = 0
        # Prepare Canny on original gray if not present
        if canny_im is None and cv2 is not None and self.gray is not None:
            canny_im = cv2.Canny(self.gray, self.cfg.canny_low, self.cfg.canny_high)
        bin_img = bin_im if bin_im is not None else (self.binary > 0).astype(np.uint8) if self.binary is not None else None

        for s in self.symbols:
            low = s.type.lower()
            if "arrow" in low or "line number" in low or "instrument" in low:
                continue
            x, y, bw, bh = s.bbox

            # Detect crossings on all borders using original Canny
            crossings_left = self._detect_line_crossings_on_border((x, y, bw, bh), 'left', canny_img, bin_img)
            crossings_right = self._detect_line_crossings_on_border((x, y, bw, bh), 'right', canny_img, bin_img)
            crossings_top = self._detect_line_crossings_on_border((x, y, bw, bh), 'top', canny_img, bin_img)
            crossings_bottom = self._detect_line_crossings_on_border((x, y, bw, bh), 'bottom', canny_img, bin_img)
            all_crossings = crossings_left + crossings_right + crossings_top + crossings_bottom

            ports = []
            if "valve" in low or "reducer" in low:
                # 2-port inline: find straight line across opposite sides
                # Prefer horizontal if wider
                if bw >= bh:
                    ports = self._find_straight_line_ports((x, y, bw, bh), crossings_left, crossings_right, crossings_top, crossings_bottom, canny_img, bin_img)
                else:
                    ports = self._find_straight_line_ports((x, y, bw, bh), crossings_top, crossings_bottom, crossings_left, crossings_right, canny_img, bin_img)  # Vertical
                if not ports and len(all_crossings) >= 2:
                    # Fallback: pair closest crossings on opposite sides
                    ports = all_crossings[:2]  # Simple fallback
            elif any(cls in low for cls in ["page connection", "connection", "utility connection"]):
                # 1-port
                single_port = self._find_single_port((x, y, bw, bh), all_crossings, canny_img, bin_img)
                if single_port:
                    ports = [single_port]
            else:
                # Generic: up to 2 ports from crossings, prefer aligned
                if len(all_crossings) >= 2:
                    ports = all_crossings[:2]

            # Connect found ports
            for ppos in ports:
                pid = try_connect_port(ppos)
                if pid is not None:
                    other_edges += 1

        return other_edges

    def _save_all_ports_overlay(self) -> None:
        """Save combined overlay: skeleton + all boxes + all ports/edges."""
        try:
            if self.skeleton is None:
                return
            sk_vis_all = (self.skeleton.astype(np.uint8) * 255)
            if cv2 is not None:
                if sk_vis_all.ndim == 2:
                    sk_vis_all = cv2.cvtColor(sk_vis_all, cv2.COLOR_GRAY2BGR)
                RED = (0,0,255); BLUE = (255,0,0)
                for s in self.symbols:
                    x, y, w2, h2 = map(int, s.bbox)
                    cv2.rectangle(sk_vis_all, (x, y), (x + w2, y + h2), RED, 1)
                for n in self.nodes:
                    if n.type == NodeType.PORT:
                        xx, yy = map(int, n.position)
                        cv2.circle(sk_vis_all, (xx, yy), 5, BLUE, -1)
                for e in self.edges:
                    x1, y1 = map(int, e.path[0]); x2, y2 = map(int, e.path[-1])
                    cv2.line(sk_vis_all, (x1, y1), (x2, y2), BLUE, 2)
            self._save_img("stage5_step3_ports_all", sk_vis_all)
        except Exception:
            pass

    def _save_image_overlay_with_ports(self) -> None:
        """Save overlay on original image with boxes, ports, and links."""
        try:
            if cv2 is not None and self.image_bgr is not None:
                img_vis = self.image_bgr.copy()
                RED = (0,0,255); BLUE = (255,0,0)
                for s in self.symbols:
                    x, y, w2, h2 = map(int, s.bbox)
                    cv2.rectangle(img_vis, (x, y), (x + w2, y + h2), RED, 2)
                # draw ports
                for n in self.nodes:
                    if n.type == NodeType.PORT:
                        x, y = map(int, n.position)
                        cv2.circle(img_vis, (x, y), 5, BLUE, -1)
                # draw short links from ports to hits
                for e in self.edges:
                    x1, y1 = map(int, e.path[0]); x2, y2 = map(int, e.path[-1])
                    cv2.line(img_vis, (x1, y1), (x2, y2), BLUE, 2)
                
                # Draw DeepLSD lines
                if hasattr(self, 'combined_deeplsd_lines') and self.combined_deeplsd_lines:
                    for line in self.combined_deeplsd_lines:
                        x1, y1 = map(int, line[0])
                        x2, y2 = map(int, line[1])
                        cv2.line(img_vis, (x1, y1), (x2, y2), GREEN_COLOR, 1) # Green color for DeepLSD lines

                self._save_img("stage5_step4_boxes_on_image", img_vis)
            elif Image is not None and self.image_bgr is not None:
                rgb = self.image_bgr[:, :, ::-1]
                pil = Image.fromarray(rgb)
                dr = ImageDraw.Draw(pil)  # type: ignore
                for s in self.symbols:
                    x, y, w2, h2 = map(int, s.bbox)
                    dr.rectangle([x, y, x + w2, y + h2], outline=(255,0,0), width=2)
                # draw ports
                try:
                    for n in self.nodes:
                        if n.type == NodeType.PORT:
                            x, y = map(int, n.position)
                            r = 4
                            dr.ellipse([x - r, y - r, x + r, y + r], outline=(0,0,255), fill=(0,0,255))
                    # draw links
                    for e in self.edges:
                        x1, y1 = map(int, e.path[0]); x2, y2 = map(int, e.path[-1])
                        dr.line([x1, y1, x2, y2], fill=(0,0,255), width=2)
                    
                    # Draw DeepLSD lines
                    if hasattr(self, 'combined_deeplsd_lines') and self.combined_deeplsd_lines:
                        for line in self.combined_deeplsd_lines:
                            x1, y1 = map(int, line[0])
                            x2, y2 = map(int, line[1])
                            dr.line([x1, y1, x2, y2], fill=(0,255,0), width=1) # Green color for DeepLSD lines

                except Exception:
                    pass
                out = np.array(pil)[:, :, ::-1]
                self._save_img("stage5_step4_boxes_on_image", out)
        except Exception:
            pass

    def _save_graph_overlay(self) -> None:
        """Save overlay of the constructed graph on the original image."""
        try:
            if cv2 is not None and self.image_bgr is not None:
                img_vis = self.image_bgr.copy()
                
                # Draw graph edges (lines)
                for edge in self.edges:
                    if edge.path:
                        pts = np.array(edge.path, np.int32).reshape((-1, 1, 2))
                        
                        line_color = ORANGE_COLOR
                        if edge.attributes.get("type") == "deeplsd_line":
                            line_color = BLUE_COLOR
                        
                        cv2.polylines(img_vis, [pts], False, line_color, 2)

                # Draw object bounding boxes (symbols and text) in red, 1px
                for s in getattr(self, 'symbols', []):
                    try:
                        x, y, w, h = map(int, s.bbox)
                        cv2.rectangle(img_vis, (x, y), (x + w, y + h), RED_COLOR, 1)
                    except Exception:
                        pass
                for t in getattr(self, 'texts', []):
                    try:
                        x, y, w, h = map(int, t.bbox)
                        cv2.rectangle(img_vis, (x, y), (x + w, y + h), RED_COLOR, 1)
                    except Exception:
                        pass
                
                # Draw graph nodes (symbols, text, ports, endpoints, junctions)
                for node in self.nodes:
                    x, y = map(int, node.position)
                    if node.type == NodeType.SYMBOL:
                        cv2.circle(img_vis, (x, y), 8, RED_COLOR, -1)
                    elif node.type == NodeType.TEXT:
                        cv2.circle(img_vis, (x, y), 6, YELLOW_COLOR, -1)
                    elif node.type == NodeType.PORT:
                        cv2.circle(img_vis, (x, y), 4, BLUE_COLOR, -1)
                    elif node.type == NodeType.ENDPOINT:
                        cv2.circle(img_vis, (x, y), 3, WHITE_COLOR, -1)
                    elif node.type == NodeType.JUNCTION:
                        cv2.circle(img_vis, (x, y), 5, BLACK_COLOR, -1)
                
                self._save_img("stage6_graph_overlay", img_vis)
            elif Image is not None and self.image_bgr is not None:
                rgb = self.image_bgr[:, :, ::-1]
                pil = Image.fromarray(rgb)
                dr = ImageDraw.Draw(pil)
                
                # Draw graph edges (lines)
                for edge in self.edges:
                    if edge.path:
                        dr.line(edge.path, fill=ORANGE_COLOR, width=2)

                # Draw object bounding boxes
                try:
                    for s in getattr(self, 'symbols', []):
                        x, y, w, h = map(int, s.bbox)
                        dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
                    for t in getattr(self, 'texts', []):
                        x, y, w, h = map(int, t.bbox)
                        dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=1)
                except Exception:
                    pass
                
                # Draw graph nodes
                for node in self.nodes:
                    x, y = map(int, node.position)
                    r = 0
                    color = (0,0,0)
                    if node.type == NodeType.SYMBOL:
                        r = 8; color = RED_COLOR
                    elif node.type == NodeType.TEXT:
                        r = 6; color = YELLOW_COLOR
                    elif node.type == NodeType.PORT:
                        r = 4; color = BLUE_COLOR
                    elif node.type == NodeType.ENDPOINT:
                        r = 3; color = WHITE_COLOR
                    elif node.type == NodeType.JUNCTION:
                        r = 5; color = BLACK_COLOR
                    
                    if r > 0:
                        dr.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
                
                out = np.array(pil)[:, :, ::-1]
                self._save_img("stage6_graph_overlay", out)
        except Exception as e:
            logger.warning(f"Graph overlay failed: {e}")

    def stage5_graph(self) -> None: # type: ignore
        t0 = time.time()
        if self.skeleton is None:
            raise RuntimeError("Run stage4_linework first")

        # Invert stage4_skeleton.png and save it
        skeleton_path = self.out_dir / "stage4_skeleton.png"
        if cv2 is not None:
            skel_img = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
            if skel_img is not None:
                inverted = cv2.bitwise_not(skel_img)
                self._save_img("stage5_skeleton_inverted", inverted)
        elif Image is not None:
            try:
                im = Image.open(skeleton_path).convert("L")
                skel_array = np.array(im)
                inverted = 255 - skel_array
                self._save_img("stage5_skeleton_inverted", inverted)
            except Exception:
                pass
        # Initialize graph structures
        self.nodes = []
        self.edges = []
        self.graph.clear()

        # DeepLSD Integration
        logger.info("Running DeepLSD line detection...")
        device = define_torch_device()
        deeplsd_model, deeplsd_conf = load_deeplsd_model(device)
        
        if self.gray is None:
            raise RuntimeError("Gray image is None; run stage2_preprocess first")
        
        _, detected_deeplsd_lines = detect_lines(deeplsd_model, inverted, device)
        self.combined_deeplsd_lines = combine_close_lines(detected_deeplsd_lines.tolist())
        logger.info(f"DeepLSD detected {len(detected_deeplsd_lines)} lines, combined into {len(self.combined_deeplsd_lines)} lines.")

        # Step 1: Initial overlay
        self._create_skeleton_overlay_with_boxes()

        # Step 2: Setup helpers
        neighbors_xy, add_node, ensure_node_for_pixel, bin_im, canny_im, _ring_search, find_connection_hit = self._setup_graph_helpers()

        # Step 3: Process arrows
        arrow_edges = self._process_arrows(add_node, canny_im, bin_im)
        self._save_arrow_overlay()

        # Step 4: Process other symbols
        other_edges = self._process_other_symbols(add_node, ensure_node_for_pixel, find_connection_hit, bin_im, canny_im)

        # Step 5: Final overlays
        self._save_all_ports_overlay()
        self._save_image_overlay_with_ports()

        # Save stats
        self._save_json("stage5_stats", {"nodes": len(self.nodes), "edges": len(self.edges), "arrow_edges": arrow_edges, "object_edges": other_edges})
        
        # Export DeepLSD-detected lines to JSON (raw and combined)
        try:
            if 'detected_deeplsd_lines' in locals() and detected_deeplsd_lines is not None:
                export_lines_to_json(np.array(detected_deeplsd_lines), str(self.out_dir / "stage5_deeplsd_lines.json"))
            if hasattr(self, 'combined_deeplsd_lines') and self.combined_deeplsd_lines:
                export_lines_to_json(np.array(self.combined_deeplsd_lines), str(self.out_dir / "stage5_deeplsd_lines_combined.json"))
        except Exception as e:
            logger.warning(f"Failed to export DeepLSD lines JSON: {e}")
        logger.info("Stage 5 (ports: arrows + objects) done in %.2fs", time.time() - t0)
        return

    def _collect_ports_for_graph(self) -> List[Dict]:
        """Collects all PORT nodes from Stage 5 to be used in the new graph."""
        ports = []
        for n in self.nodes:
            if n.type == NodeType.PORT:
                ports.append({
                    'id': n.id,
                    'pos': n.position,
                    'parent_id': n.port_of if n.port_of is not None else -1,
                    'type': 'port'
                })
        return ports

    def stage6_line_graph(self) -> None:
        t0 = time.time()
        logger.info("Stage 6: Building line graph with ConnectivityEngine...")

        self.graph.clear()
        self.nodes = []
        self.edges = []
        
        # 0. Collect Line Segments
        if not hasattr(self, 'combined_deeplsd_lines') or not self.combined_deeplsd_lines:
            logger.warning("No DeepLSD lines found for graph construction.")
            return
            
        lines = [
            ((float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1])))
            for line in self.combined_deeplsd_lines
        ]

        # 1. Collect Ports
        ports = self._collect_ports_for_graph()

        # 2. Run Connectivity Engine
        engine = ConnectivityEngine(
            merge_dist=float(self.cfg.merge_node_dist), 
            snap_dist=float(self.cfg.connect_radius)
        )
        
        nx_graph, graph_nodes = engine.build_graph(lines, ports)
        
        # 3. Convert back to internal structures (Node, Edge)
        new_nodes = []
        new_edges = []
        node_id_map = {} 
        
        # 3a. Add Pipeline Nodes
        for gn_id, gn in graph_nodes.items():
            new_id = len(new_nodes)
            node_id_map[gn_id] = new_id
            
            ntype = NodeType.ENDPOINT
            if gn.type == "junction": ntype = NodeType.JUNCTION
            elif gn.type == "port": ntype = NodeType.PORT
            
            new_nodes.append(Node(id=new_id, position=(gn.x, gn.y), type=ntype))
        
        # 3b. Add Edges (Pipelines)
        for u, v, data in nx_graph.edges(data=True):
            if u not in node_id_map or v not in node_id_map: continue
            
            nu, nv = node_id_map[u], node_id_map[v]
            path = [(float(p[0]), float(p[1])) for p in data.get('path', [])]
            
            new_edges.append(Edge(
                id=len(new_edges), source=nu, target=nv, 
                path=path, attributes=data
            ))
            
        # 3c. Re-integrate Symbols and Text
        for s in self.symbols:
            sid = len(new_nodes)
            new_nodes.append(Node(id=sid, position=s.center, type=NodeType.SYMBOL, label=s.type, symbol_ids=[s.id]))
            
            # Connect Symbol -> Port (Strict: only if linked port exists)
            linked_ports = [node_id_map[gn_id] for gn_id, gn in graph_nodes.items() if gn.ref_id == s.id]
            
            for pid in linked_ports:
                port_node = new_nodes[pid]
                new_edges.append(Edge(
                    id=len(new_edges), source=sid, target=pid,
                    path=[s.center, port_node.position],
                    attributes={"type": "symbol_connection"}
                ))

    def _repair_inline_connections(self, nx_graph: nx.Graph, graph_nodes: Dict[int, GraphNode], node_id_map: Dict[int, int], new_nodes: List[Node], new_edges: List[Edge]) -> None:
        """
        Attempts to repair connectivity for inline 2-port symbols (valves/reducers) 
        that failed to connect to 2 lines.
        """
        # Identify inline candidates from symbols
        inline_types = ["valve", "reducer", "flange"]
        
        # Build spatial index of graph endpoints for fast query
        endpoint_ids = []
        endpoint_coords = []
        for gn_id, gn in graph_nodes.items():
            if gn.type == "endpoint" or gn.type == "junction":
                endpoint_ids.append(gn_id)
                endpoint_coords.append((gn.x, gn.y))
        
        if not endpoint_coords:
            return
            
        endpoint_tree = KDTree(np.array(endpoint_coords))
        search_radius = float(self.cfg.connect_radius) * 1.5 # slightly larger search for repair
        
        # Check each symbol in the new graph
        # Note: We need to map from Symbol ID -> Graph Node ID
        # The 'new_nodes' list contains Symbol nodes. We need to find them.
        
        symbol_node_indices = [i for i, n in enumerate(new_nodes) if n.type == NodeType.SYMBOL]
        
        for sid_idx in symbol_node_indices:
            s_node = new_nodes[sid_idx]
            s_type = s_node.label.lower() if s_node.label else ""
            
            if not any(t in s_type for t in inline_types):
                continue
                
            # Check current degree in the *reconstructed* graph logic
            # We must check 'new_edges' to see how many connect to 'sid_idx'
            current_degree = sum(1 for e in new_edges if e.source == sid_idx or e.target == sid_idx)
            
            if current_degree >= 2:
                continue
                
            # Needs repair. Find aligned endpoints.
            # 1. Find nearby endpoints
            dists, indices = endpoint_tree.query(s_node.position, k=10, distance_upper_bound=search_radius)
            if isinstance(indices, int): indices = [indices]
            
            # 2. Filter by alignment (horizontal/vertical relative to symbol)
            # Simple heuristic: if symbol is roughly 2-port, we expect connections on opposite sides.
            
            candidates = []
            sx, sy = s_node.position
            
            for i, d in zip(indices, dists):
                if d == float('inf'): continue
                
                ep_id = endpoint_ids[i]
                ep_node = graph_nodes[ep_id]
                ex, ey = ep_node.x, ep_node.y
                
                # Check alignment
                dx, dy = abs(ex - sx), abs(ey - sy)
                is_horz = dy < 10 and dx > 0
                is_vert = dx < 10 and dy > 0
                
                if is_horz or is_vert:
                    candidates.append((d, ep_id))
            
            # Sort by distance
            candidates.sort(key=lambda x: x[0])
            
            # Attempt to connect to up to (2 - current_degree) distinct endpoints
            needed = 2 - current_degree
            connected_count = 0
            
            for _, ep_graph_id in candidates:
                if connected_count >= needed:
                    break
                    
                target_node_idx = node_id_map[ep_graph_id]
                
                # Avoid duplicate edges
                exists = any(
                    (e.source == sid_idx and e.target == target_node_idx) or 
                    (e.source == target_node_idx and e.target == sid_idx) 
                    for e in new_edges
                )
                
                if not exists:
                    # Create Bridge Edge
                    new_edges.append(Edge(
                        id=len(new_edges),
                        source=sid_idx,
                        target=target_node_idx,
                        path=[s_node.position, new_nodes[target_node_idx].position],
                        attributes={"type": "inferred_connection", "confidence": "low"}
                    ))
                    connected_count += 1

    # 3d. Add Text Nodes
    def _add_text_nodes(self, new_nodes: List[Node]) -> None:
        for t in self.texts:
            new_nodes.append(Node(id=len(new_nodes), position=t.center, type=NodeType.TEXT, label=t.text))

    # 4. Finalize
    def _save_pipeline_only_overlay(self) -> None:
        """Save overlay of only the pipeline graph (no symbols/text circles) on original image."""
        try:
            if self.image_bgr is None: return
            img_vis = self.image_bgr.copy()
            
            if cv2 is not None:
                # Draw graph edges (pipelines)
                for edge in self.edges:
                    if edge.path:
                        pts = np.array(edge.path, np.int32).reshape((-1, 1, 2))
                        line_color = ORANGE_COLOR
                        if edge.attributes.get("type") == "deeplsd_line":
                            line_color = BLUE_COLOR
                        cv2.polylines(img_vis, [pts], False, line_color, 2)

                # Draw graph nodes (pipeline only: ports, endpoints, junctions)
                for node in self.nodes:
                    x, y = map(int, node.position)
                    if node.type == NodeType.PORT:
                        cv2.circle(img_vis, (x, y), 4, BLUE_COLOR, -1)
                    elif node.type == NodeType.ENDPOINT:
                        cv2.circle(img_vis, (x, y), 3, WHITE_COLOR, -1)
                    elif node.type == NodeType.JUNCTION:
                        cv2.circle(img_vis, (x, y), 5, BLACK_COLOR, -1)
                
                self._save_img("stage6_final_pipeline_only", img_vis)
            elif Image is not None:
                rgb = self.image_bgr[:, :, ::-1]
                pil = Image.fromarray(rgb)
                dr = ImageDraw.Draw(pil)
                for edge in self.edges:
                    if edge.path:
                        dr.line(edge.path, fill=ORANGE_COLOR, width=2)
                for node in self.nodes:
                    x, y = map(int, node.position)
                    r = 0; color = (0,0,0)
                    if node.type == NodeType.PORT:
                        r = 4; color = BLUE_COLOR
                    elif node.type == NodeType.ENDPOINT:
                        r = 3; color = WHITE_COLOR
                    elif node.type == NodeType.JUNCTION:
                        r = 5; color = BLACK_COLOR
                    if r > 0:
                        dr.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=color)
                out = np.array(pil)[:, :, ::-1]
                self._save_img("stage6_final_pipeline_only", out)
        except Exception as e:
            logger.warning(f"Pipeline-only overlay failed: {e}")

    def _finalize_graph(self, new_nodes: List[Node], new_edges: List[Edge]) -> None:
        self.nodes = new_nodes
        self.edges = new_edges
        
        self.graph.clear()
        for n in self.nodes:
            self.graph.add_node(n.id, type=n.type.value, position=n.position, label=n.label)
        for e in self.edges:
            attrs = {k: v for k, v in e.attributes.items() if k not in ('id', 'path')}
            self.graph.add_edge(e.source, e.target, id=e.id, path=e.path, **attrs)

        pipe_count = sum(1 for e in self.edges if e.attributes.get("type") in ("pipe", "inferred_connection", "deeplsd_line"))
        logger.info(f"Stage 6: Graph rebuilt. Nodes: {len(self.nodes)}, Edges: {len(self.edges)} (Pipelines: {pipe_count})")
        self._save_graph_overlay()
        self._save_pipeline_only_overlay()
        logger.info("Stage 6 done in %.2fs", time.time() - self._t0) # self._t0 needs to be passed or stored

    def stage6_line_graph(self) -> None:
        self._t0 = time.time() # Store t0 for finalize
        logger.info("Stage 6: Building line graph with ConnectivityEngine...")

        self.graph.clear()
        self.nodes = []
        self.edges = []
        
        # 0. Collect Line Segments
        if not hasattr(self, 'combined_deeplsd_lines') or not self.combined_deeplsd_lines:
            logger.warning("No DeepLSD lines found for graph construction.")
            return
            
        lines = [
            ((float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1])))
            for line in self.combined_deeplsd_lines
        ]

        # 1. Collect Ports
        ports = self._collect_ports_for_graph()

        # 2. Run Connectivity Engine
        engine = ConnectivityEngine(
            merge_dist=float(self.cfg.merge_node_dist), 
            snap_dist=25.0, # Relaxed from default
            ortho_tol=5.0   # Relaxed from 2.0
        )
        
        nx_graph, graph_nodes = engine.build_graph(lines, ports)
        
        # 3. Convert back to internal structures (Node, Edge)
        new_nodes = []
        new_edges = []
        node_id_map = {} 
        
        # 3a. Add Pipeline Nodes
        for gn_id, gn in graph_nodes.items():
            new_id = len(new_nodes)
            node_id_map[gn_id] = new_id
            
            ntype = NodeType.ENDPOINT
            if gn.type == "junction": ntype = NodeType.JUNCTION
            elif gn.type == "port": ntype = NodeType.PORT
            
            new_nodes.append(Node(id=new_id, position=(gn.x, gn.y), type=ntype))
        
        # 3b. Add Edges (Pipelines)
        for u, v, data in nx_graph.edges(data=True):
            if u not in node_id_map or v not in node_id_map: continue
            
            nu, nv = node_id_map[u], node_id_map[v]
            path = [(float(p[0]), float(p[1])) for p in data.get('path', [])]
            
            new_edges.append(Edge(
                id=len(new_edges), source=nu, target=nv, 
                path=path, attributes=data
            ))
            
        # 3c. Re-integrate Symbols and Text
        for s in self.symbols:
            sid = len(new_nodes)
            new_nodes.append(Node(id=sid, position=s.center, type=NodeType.SYMBOL, label=s.type, symbol_ids=[s.id]))
            
            # Connect Symbol -> Port (Strict: only if linked port exists)
            linked_ports = [node_id_map[gn_id] for gn_id, gn in graph_nodes.items() if gn.ref_id == s.id]
            
            for pid in linked_ports:
                port_node = new_nodes[pid]
                new_edges.append(Edge(
                    id=len(new_edges), source=sid, target=pid,
                    path=[s.center, port_node.position],
                    attributes={"type": "symbol_connection"}
                ))

        # ** Repair Pass: Try to bridge disconnected 2-port symbols **
        self._repair_inline_connections(nx_graph, graph_nodes, node_id_map, new_nodes, new_edges)

        # 3d. Add Text Nodes
        self._add_text_nodes(new_nodes)

        # 4. Finalize
        self._finalize_graph(new_nodes, new_edges)

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
    pipe.stage6_line_graph()
    if args.stop_after <= 6:
        return

if __name__ == "__main__":
    main()
