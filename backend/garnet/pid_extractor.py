"""
Stage-based P&ID pipeline rebuild.

The current implementation intentionally stays small and reviewable:
- Stage 1: input normalization
- Stage 2: selected OCR route discovery
- Stage 4: fixed-baseline object detection
- Stage 5: provisional pipe-mask generation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from garnet.easyocr_sahi import EasyOcrSahiConfig, run_easyocr_sahi
from garnet.edge_direction import run_edge_direction_stage
from garnet.edge_split import split_edges_at_inline_elements
from garnet.equipment_tag_fusion import run_equipment_tag_fusion_stage
from garnet.gemini_ocr_sahi import GeminiOcrSahiConfig, run_gemini_ocr_sahi
from garnet.geometric_graph_builder import (
    build_graph_from_runs_and_junctions,
    chain_geometric_segments,
    detect_junctions_from_runs,
)
from garnet.graph_export_adapter import build_graph_v1_payload
from garnet.instrument_tag_fusion import run_instrument_tag_fusion_stage
from garnet.line_number_fusion import run_line_number_fusion_stage
from garnet.model_defaults import pick_default_weight_file
from garnet.object_detection_sahi import DetectionSahiConfig, run_object_detection_sahi
from garnet.visual_primitives.agent2_hybrid import compute_port_vlm
from garnet.visual_primitives.cv_pipe_tracer import CVPipeTracer
from garnet.ocrmac_sahi import OcrMacSahiConfig, run_ocrmac_sahi
from garnet.pipe_edges import run_pipe_edge_stage
from garnet.pipe_continuity_helpers import GAP_THRESHOLD_PX
from garnet.pipe_equipment_attachment import run_pipe_equipment_attachment_stage
from garnet.pipe_graph import run_pipe_graph_stage
from garnet.pipe_graph_qa import run_pipe_graph_qa_stage
from garnet.run_continuity_checker_stage import run_continuity_checker_stage
from garnet.pipe_edge_connectivity import (
    build_pipe_edge_connectivity,
    render_candidate_link_overlay,
    render_junction_decision_overlay,
)
from garnet.render_connection_pipeline_overlay import render_overlay
from garnet.pipe_crossings import run_pipe_crossing_stage
from garnet.pipe_junctions import run_pipe_junction_stage
from garnet.pipe_text_attachment import (
    _filter_border_like_edges,
    render_connection_attachment_overlay,
    render_text_attachment_overlay,
    run_node_text_attachment_stage,
    run_pipe_text_attachment_stage,
)
from garnet.line_detection_inpaint import run_line_detection_inpaint, render_line_overlay
from garnet.paddle_ocr_sahi import PaddleOcrSahiConfig, run_paddle_ocr_sahi
from garnet.pipe_mask import generate_continuity_mask, run_pipe_mask_stage
from garnet.pipe_node_clusters import run_pipe_node_cluster_stage
from garnet.pipe_nodes import run_pipe_node_stage
from garnet.polyline_simplify import run_polyline_simplification_stage
from garnet.pipe_seal import run_pipe_seal_stage
from garnet.pipe_skeleton import run_pipe_skeleton_stage
from garnet.pipe_terminals import classify_pipe_edge_terminals
from garnet.topology_markers import run_topology_marker_router

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pid")

DEFAULT_OUT = Path("output")
DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent


def load_pipeline_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=False)
    load_dotenv(BACKEND_DIR / ".env", override=False)


load_pipeline_env()


def normalize_for_save(img: np.ndarray) -> np.ndarray:
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


@dataclass
class PipelineConfig:
    adaptive_block_size: int = 21
    adaptive_c: int = 5
    blur_kernel: int = 5
    ocr_route: str = "ocrmac"
    gemini_postprocess_match_threshold: float = 0.1
    port_detection_model: str = "anthropic/claude-haiku-4.5"
    port_detection_mode: str = "cv"  # "vlm", "cv", or "vlm+cv"
    ocr_slice_height: int = 1600
    ocr_slice_width: int = 1600
    ocr_overlap_height_ratio: float = 0.2
    ocr_overlap_width_ratio: float = 0.2
    ocr_min_score: float = 0.2
    ocr_min_text_len: int = 2
    ocr_low_text: float = 0.3
    ocr_link_threshold: float = 0.7
    ocr_line_merge_gap_px: int = 24
    ocr_line_merge_y_tolerance_px: int = 10
    ocr_enable_rotated: bool = True
    ocrmac_framework: str = "vision"
    ocrmac_recognition_level: str = "accurate"
    detection_weight_path: str = pick_default_weight_file("ultralytics") or "yolo_weights/yolo26n_PPCL_640_20260227.pt"
    detection_image_size: int = 640
    detection_overlap_ratio: float = 0.2
    detection_postprocess_type: str = "GREEDYNMM"
    detection_postprocess_match_metric: str = "IOS"
    detection_postprocess_match_threshold: float = 0.1
    line_number_fusion_max_distance_px: float = 80.0
    instrument_tag_fusion_max_distance_px: float = 60.0
    equipment_tag_fusion_max_distance_px: float = 60.0
    equipment_tag_attachment_max_distance_px: float = 80.0
    use_geometric_line_detection: bool = False
    pipe_mask_ocr_padding: int = 1
    pipe_mask_object_inset: int = 1
    pipe_mask_min_component_area: int = 16
    pipe_mask_preserve_ocr_classes: tuple[str, ...] = ()
    pipe_mask_preserve_object_classes: tuple[str, ...] = (
        "arrow",
        "node",
    )
    pipe_mask_continuity_ocr_padding: int = 1
    pipe_mask_continuity_min_component_area: int = 16
    pipe_seal_horizontal_close_kernel: int = 5
    pipe_seal_vertical_close_kernel: int = 5
    pipe_seal_min_component_area: int = 16
    node_cluster_eps: float = 6.0
    node_cluster_min_samples: int = 1
    min_edge_length_px: int = 2
    crossing_branch_stub_length_px: int = 8
    crossing_branch_merge_angle_tolerance_deg: float = 18.0
    crossing_opposite_angle_tolerance_deg: float = 50.0
    crossing_center_blob_radius_px: int = 4
    crossing_center_blob_threshold: float = 0.5
    crossing_stage4_marker_match_distance_px: float = 24.0
    polyline_simplify_epsilon: float = 2.0
    arrow_proximity_px: float = 40.0
    inline_split_confidence_threshold: float = 0.5
    equipment_attachment_classes: tuple[str, ...] = (
        "pump",
        "heat exchanger",
        "tank",
        "vessel",
        "column",
        "compressor",
        "blower",
        "fan",
    )
    equipment_attachment_max_distance_px: float = 48.0
    equipment_attachment_k_candidate_edges: int = 10
    connection_attachment_classes: tuple[str, ...] = (
        "connection",
        "page connection",
        "utility connection",
    )
    connection_attachment_max_distance_px: float = 48.0
    connection_attachment_k_candidate_edges: int = 10
    line_text_attachment_max_distance_px: float = 80.0
    terminal_equipment_classes: tuple[str, ...] = (
        "pump",
        "heat exchanger",
        "tank",
        "vessel",
        "column",
        "compressor",
        "blower",
        "fan",
    )
    terminal_connection_classes: tuple[str, ...] = (
        "connection",
        "page connection",
        "utility connection",
    )
    terminal_inline_passthrough_classes: tuple[str, ...] = (
        "arrow",
        "valve",
        "gate valve",
        "ball valve",
        "globe valve",
        "check valve",
        "butterfly valve",
        "control valve",
        "pressure relief valve",
        "reducer",
        "spectacle blind",
    )
    terminal_match_distance_px: float = 72.0
    graph_inline_connector_classes: tuple[str, ...] = (
        "arrow",
        "valve",
        "gate valve",
        "ball valve",
        "globe valve",
        "check valve",
        "butterfly valve",
        "control valve",
        "pressure relief valve",
        "reducer",
        "spectacle blind",
    )
    graph_inline_connector_match_distance_px: float = 36.0


class PIDPipeline:
    def __init__(
        self,
        image_path: str,
        output_dir: str | Path = DEFAULT_OUT,
        cfg: PipelineConfig | None = None,
        stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs: Any,
    ) -> None:
        self.image_path = str(image_path)
        self.out_dir = Path(out_dir)
        self.cfg = cfg or PipelineConfig()
        self.stage_callback = stage_callback
        if kwargs:
            logger.warning("Ignoring unexpected PIDPipeline kwargs: %s", sorted(kwargs))

        self.image_bgr: Optional[np.ndarray] = None
        self.stage_manifest: Dict[str, Any] = {}
        self._current_stage_artifacts: list[str] = []

    # ---------- Stage runner ----------
    def _stage_definitions(self) -> List[Tuple[int, str, Callable[[], None]]]:
        """Return the ordered stage list executed by the pipeline."""
        return [
            (1, "stage1_input_normalization", self.stage1_input_normalization),
            (2, "stage2_ocr_discovery", self.stage2_ocr_discovery),
            # Stage 4 sub-stages share the same number intentionally.
            # They all depend on Stage 4 object detection output and run together when stop_after=4.
            (4, "stage4_object_detection", self.stage4_object_detection),
            (4, "stage4_line_number_fusion", self.stage4_line_number_fusion),
            (4, "stage4_instrument_tag_fusion", self.stage4_instrument_tag_fusion),
            (5, "stage5_pipe_mask", self.stage5_pipe_mask),
            (6, "stage6_morphological_sealing", self.stage6_morphological_sealing),
            (7, "stage7_skeleton_generation", self.stage7_skeleton_generation),
            (8, "stage8_skeleton_node_detection", self.stage8_skeleton_node_detection),
            (9, "stage9_node_clustering", self.stage9_node_clustering),
            (10, "stage10_edge_tracing", self.stage10_edge_tracing),
            (11, "stage11_junction_review", self.stage11_junction_review),
            (12, "stage12_edge_topology", self.stage12_edge_topology),
            (13, "stage13_text_attachment", self.stage13_text_attachment),
            (14, "stage14_graph_assembly", self.stage14_graph_assembly),
            (15, "stage15_graph_qa", self.stage15_graph_qa),
        ]
        if self.cfg.use_geometric_line_detection:
            stages.extend(
                [
                    (5, "stage5b_pipe_trace", self.stage5b_pipe_trace),
                    (12, "stage12_geometric_graph_assembly", self.stage12_geometric_graph_assembly),
                    (12, "stage12c_page_connector_labeling", self.stage12c_page_connector_labeling),
                    (12, "stage12b_graph_export", self.stage12b_graph_export),
                    (13, "stage13_graph_qa", self.stage13_graph_qa),
                    (14, "stage14_continuity_check", self.stage14_continuity_check),
                    (15, "stage15_recovery_loop", self.stage15_recovery_loop),
                    (16, "stage16_connection_overlay", self.stage16_connection_overlay),
                ]
            )
            return stages
        stages.extend(
            [
                (6, "stage6_morphological_sealing", self.stage6_morphological_sealing),
                (7, "stage7_skeleton_generation", self.stage7_skeleton_generation),
                (8, "stage8_skeleton_node_detection", self.stage8_skeleton_node_detection),
                (9, "stage9_node_clustering", self.stage9_node_clustering),
                (10, "stage10_edge_tracing", self.stage10_edge_tracing),
                (10, "stage10b_polyline_simplification", self.stage10b_polyline_simplification),
                (10, "stage10c_edge_direction", self.stage10c_edge_direction),
                (10, "stage10d_edge_split", self.stage10d_edge_split),
                (11, "stage11_junction_review", self.stage11_junction_review),
                (12, "stage12_graph_assembly", self.stage12_graph_assembly),
                (12, "stage12c_page_connector_labeling", self.stage12c_page_connector_labeling),
                (12, "stage12b_graph_export", self.stage12b_graph_export),
                (13, "stage13_graph_qa", self.stage13_graph_qa),
                (14, "stage14_continuity_check", self.stage14_continuity_check),
                (15, "stage15_recovery_loop", self.stage15_recovery_loop),
                (16, "stage16_connection_overlay", self.stage16_connection_overlay),
            ]
        )
        return stages

    def _manifest_path(self) -> Path:
        return self.out_dir / "stage_manifest.json"

    def _write_stage_manifest(self) -> None:
        path = self._manifest_path()
        tmp_path = path.with_name(f".{path.name}.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.stage_manifest, f, indent=2)
        tmp_path.replace(path)
        logger.info(f"saved {path}")

    def _notify_stage_callback(self, payload: Dict[str, Any]) -> None:
        """Forward stage lifecycle events to the optional callback."""
        if self.stage_callback is not None:
            self.stage_callback(payload)

    def _reset_stage_manifest(self, stop_after: int) -> None:
        """Initialize a fresh stage manifest for the current run."""
        self.stage_manifest = {
            "image_path": self.image_path,
            "out_dir": str(self.out_dir),
            "stop_after": stop_after,
            "ocr_route": self.cfg.ocr_route,
            "detection_weight_path": self.cfg.detection_weight_path,
            "stage_numbering_note": "Stage numbering is intentionally sparse: Stage 3 is not implemented yet.",
            "stages": [],
        }
        self._write_stage_manifest()

    def _register_artifact(self, name: str) -> None:
        self._current_stage_artifacts.append(name)

    def _run_stage(self, stage_num: int, stage_name: str, stage_fn: Callable[[], None]) -> None:
        """Execute one stage and persist manifest status, timing, and artifacts."""
        started_at = time.time()
        entry = {
            "num": stage_num,
            "name": stage_name,
            "status": "started",
            "started_at": started_at,
            "artifacts": [],
        }
        self.stage_manifest["stages"].append(entry)
        self._write_stage_manifest()
        self._notify_stage_callback({"event": "stage_started", "stage": entry.copy(), "manifest": self.stage_manifest})
        self._current_stage_artifacts = []
        try:
            stage_fn()
        except Exception as exc:
            entry["status"] = "failed"
            entry["ended_at"] = time.time()
            entry["duration_sec"] = round(entry["ended_at"] - started_at, 6)
            entry["artifacts"] = list(self._current_stage_artifacts)
            entry["error"] = str(exc)
            self._write_stage_manifest()
            self._notify_stage_callback({"event": "stage_failed", "stage": entry.copy(), "manifest": self.stage_manifest})
            raise
        entry["status"] = "completed"
        entry["ended_at"] = time.time()
        entry["duration_sec"] = round(entry["ended_at"] - started_at, 6)
        entry["artifacts"] = list(self._current_stage_artifacts)
        self._write_stage_manifest()
        self._notify_stage_callback({"event": "stage_completed", "stage": entry.copy(), "manifest": self.stage_manifest})

    def run(self, stop_after: int = 1, resume: bool = False) -> None:
        stages = self._stage_definitions()
        valid_stop_after = {num for num, _, _ in stages}
        if stop_after not in valid_stop_after:
            raise ValueError(f"stop_after must be one of {sorted(valid_stop_after)}, got {stop_after}")

        if not os.path.isfile(self.image_path):
            raise FileNotFoundError(f"Input image does not exist or is not a file: {self.image_path}")
        if self.out_dir.exists() and not self.out_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {self.out_dir}")
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Output directory cannot be created: {self.out_dir}") from exc
        write_probe = self.out_dir / f".pid_pipeline_write_test_{time.time_ns()}"
        try:
            with open(write_probe, "w", encoding="utf-8"):
                pass
        except OSError as exc:
            raise PermissionError(f"Output directory is not writable: {self.out_dir}") from exc
        finally:
            if write_probe.exists():
                write_probe.unlink()

        completed_stage_names: set[str] = set()
        manifest_path = self._manifest_path()
        if resume and manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.stage_manifest = json.load(f)
            self.stage_manifest.setdefault("stages", [])
            manifest_image_path = self.stage_manifest.get("image_path")
            if manifest_image_path not in (None, self.image_path):
                raise ValueError(
                    f"Cannot resume from manifest for different image: {manifest_image_path} != {self.image_path}"
                )
            manifest_out_dir = self.stage_manifest.get("out_dir")
            if manifest_out_dir not in (None, str(self.out_dir)):
                raise ValueError(
                    f"Cannot resume from manifest for different output directory: {manifest_out_dir} != {self.out_dir}"
                )
            self.stage_manifest["image_path"] = self.image_path
            self.stage_manifest["out_dir"] = str(self.out_dir)
            self.stage_manifest["stop_after"] = stop_after
            self.stage_manifest["ocr_route"] = self.cfg.ocr_route
            self.stage_manifest["detection_weight_path"] = self.cfg.detection_weight_path
            self.stage_manifest["stage_numbering_note"] = (
                "Stage numbering is intentionally sparse: Stage 3 is not implemented yet."
            )
            last_completed_stage: str | None = None
            for entry in self.stage_manifest.get("stages", []):
                if entry.get("status") == "completed" and isinstance(entry.get("name"), str):
                    completed_stage_names.add(entry["name"])
                    last_completed_stage = entry["name"]
            if last_completed_stage is not None:
                logger.info("Resuming pipeline from %s after %s", manifest_path, last_completed_stage)
            self._write_stage_manifest()
        else:
            self._reset_stage_manifest(stop_after)

        for stage_num, stage_name, stage_fn in stages:
            if stage_num > stop_after:
                break
            if resume and stage_name in completed_stage_names:
                logger.info("Skipping completed stage during resume: %s", stage_name)
                continue
            self._run_stage(stage_num, stage_name, stage_fn)

    # ---------- Persistence ----------
    def _save_img(self, name: str, img: np.ndarray) -> None:
        """Persist an image artifact to the output directory and register it."""
        path = self.out_dir / f"{name}.png"
        out = normalize_for_save(img)
        if cv2 is not None:
            cv2.imwrite(str(path), out)
        elif Image is not None:
            Image.fromarray(out).save(str(path))
        else:  # pragma: no cover
            raise RuntimeError("No image backend available")
        self._register_artifact(path.name)
        logger.info(f"saved {path}")

    def _save_json(self, name: str, data: Any) -> None:
        """Persist a JSON artifact to the output directory and register it."""
        path = self.out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._register_artifact(path.name)
        logger.info(f"saved {path}")

    def _load_json_artifact(self, name: str) -> Any:
        """Load a required JSON artifact from the output directory."""
        path = self.out_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _load_json_artifact_or_default(self, name: str, default: Any) -> Any:
        path = self.out_dir / f"{name}.json"
        if not path.exists():
            return default
        with open(path, "r") as f:
            return json.load(f)

    # ---------- Stage 1 ----------
    def _ensure_image_loaded(self) -> np.ndarray:
        """Load and cache the source image as BGR for downstream stages."""
        if self.image_bgr is not None:
            return self.image_bgr
        if cv2 is not None:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {self.image_path}")
            self.image_bgr = img
            return img
        if Image is not None:
            img = Image.open(self.image_path).convert("RGB")
            self.image_bgr = np.array(img)[:, :, ::-1]
            return self.image_bgr
        raise RuntimeError("No image backend available")  # pragma: no cover

    @staticmethod
    def _extend_mask_to_terminals(
        mask: "np.ndarray",
        terminals: list[dict],
        max_gap: int = 80,
    ) -> "np.ndarray":
        """Fill terminal bbox entry regions so the CV tracer can walk in.

        Instead of drawing long bridge lines (which can create loops),
        this fills a small rectangular pad at the bbox edge closest to
        the nearest pipe pixel. The tracer walks into the pad, the
        position-inside-bbox check fires, and the terminal is classified.

        Args:
            mask: Binary pipe mask (0/255).
            terminals: Stage4 objects with bbox dicts.
            max_gap: Max pixel distance from mask to terminal center
                     to create a bridge.

        Returns:
            Modified mask (new array, original unchanged).
        """
        import cv2 as _cv2

        pipe_ys, pipe_xs = np.where(mask > 0)
        if len(pipe_xs) == 0:
            return mask

        result = mask.copy()
        h, w = result.shape
        pad_size = 20  # px to fill at the entry edge

        for obj in terminals:
            b = obj["bbox"]
            bx1 = max(0, b["x_min"])
            by1 = max(0, b["y_min"])
            bx2 = min(w, b["x_max"])
            by2 = min(h, b["y_max"])
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2

            dists = np.sqrt((pipe_xs - cx) ** 2 + (pipe_ys - cy) ** 2)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            if min_dist > max_gap:
                continue

            px = int(pipe_xs[min_idx])
            py = int(pipe_ys[min_idx])

            # Which bbox edge is closest to the pipe pixel?
            dist_left   = abs(px - bx1)
            dist_right  = abs(px - bx2)
            dist_top    = abs(py - by1)
            dist_bottom = abs(py - by2)
            min_edge = min(dist_left, dist_right, dist_top, dist_bottom)

            # Fill a pad at that edge so the tracer enters the bbox
            if min_edge == dist_left:
                fill_x1 = bx1
                fill_x2 = min(bx1 + pad_size, bx2)
                fill_y1 = max(0, cy - pad_size // 2)
                fill_y2 = min(h, cy + pad_size // 2)
            elif min_edge == dist_right:
                fill_x1 = max(bx1, bx2 - pad_size)
                fill_x2 = bx2
                fill_y1 = max(0, cy - pad_size // 2)
                fill_y2 = min(h, cy + pad_size // 2)
            elif min_edge == dist_top:
                fill_x1 = max(0, cx - pad_size // 2)
                fill_x2 = min(w, cx + pad_size // 2)
                fill_y1 = by1
                fill_y2 = min(by1 + pad_size, by2)
            else:  # dist_bottom
                fill_x1 = max(0, cx - pad_size // 2)
                fill_x2 = min(w, cx + pad_size // 2)
                fill_y1 = max(by1, by2 - pad_size)
                fill_y2 = by2

            # Also draw a short connector from pipe to the pad
            target_x = (fill_x1 + fill_x2) // 2
            target_y = (fill_y1 + fill_y2) // 2
            _cv2.line(result, (px, py), (target_x, target_y), 255, thickness=4)

            # Fill the pad
            result[fill_y1:fill_y2, fill_x1:fill_x2] = 255

        return result

    def _add_port_markers_to_overlay(
        self,
        ports: dict[str, list[tuple[int, int, str]]],
        radius: int = 8,
    ) -> None:
        """Draw port markers (cyan circles with labels) onto stage4_objects_overlay.

        Takes the existing overlay and draws on top of it.
        """
        import cv2 as cv2_local

        overlay_path = self.out_dir / "stage4_objects_overlay.png"
        if not overlay_path.exists():
            return  # nothing to annotate

        overlay = cv2_local.imread(str(overlay_path), cv2_local.IMREAD_COLOR)
        if overlay is None:
            return

        objects = self._load_json_artifact("stage4_objects").get("objects", [])
        id_to_obj = {obj["id"]: obj for obj in objects}

        port_count = 0
        for obj_id, port_list in ports.items():
            obj = id_to_obj.get(obj_id)
            if obj is None:
                continue

            bbox = obj["bbox"]
            x_min = bbox["x_min"]
            y_min = bbox["y_min"]
            x_max = bbox["x_max"]
            y_max = bbox["y_max"]

            # Green bbox for connection objects (visible on white)
            cv2_local.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 180, 0), 2)

            # Object ID label above bbox
            label = f"{obj_id}"
            if obj_id not in ports or not ports[obj_id]:
                label += " FAIL"
                label_color = (0, 0, 220)  # red for failed
            else:
                label_color = (0, 120, 0)  # dark green for success
            cv2_local.putText(
                overlay, label,
                (x_min, max(y_min - 8, 12)),
                cv2_local.FONT_HERSHEY_SIMPLEX, 0.4, label_color, 2,
            )

            for port_x, port_y, edge_name in port_list:
                # Filled dark teal circle at the actual pipe connection point
                cv2_local.circle(overlay, (port_x, port_y), radius, (180, 100, 0), -1)
                # White border for contrast
                cv2_local.circle(overlay, (port_x, port_y), radius, (255, 255, 255), 1)

                # Crosshair
                half = radius + 4
                cv2_local.line(overlay, (port_x - half, port_y), (port_x + half, port_y), (255, 255, 255), 1)
                cv2_local.line(overlay, (port_x, port_y - half), (port_x, port_y + half), (255, 255, 255), 1)

                # Edge label — dark green, visible on white
                cv2_local.putText(
                    overlay,
                    edge_name[:3].upper(),
                    (port_x + radius + 2, port_y - radius - 2),
                    cv2_local.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 120, 0),
                    2,
                )
                port_count += 1

        cv2_local.imwrite(str(overlay_path), overlay)

    def stage1_input_normalization(self) -> None:
        """Generate grayscale, adaptive/Otsu binary, and histogram-equalized views of the input image."""
        image = self._ensure_image_loaded()
        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (self.cfg.blur_kernel, self.cfg.blur_kernel), 0)
            adaptive = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.cfg.adaptive_block_size,
                self.cfg.adaptive_c,
            )
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            equalized = cv2.equalizeHist(gray)
        else:
            gray = np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
            blur = gray
            threshold = int(gray.mean())
            adaptive = (gray < threshold).astype(np.uint8) * 255
            otsu = adaptive.copy()
            equalized = gray

        self._save_img("stage1_gray", gray)
        self._save_img("stage1_gray_equalized", equalized)
        self._save_img("stage1_binary_adaptive", adaptive)
        self._save_img("stage1_binary_otsu", otsu)
        self._save_json(
            "stage1_normalization_summary",
            {
                "image_path": self.image_path,
                "dimensions": {"height": int(image.shape[0]), "width": int(image.shape[1])},
                "artifacts": [
                    "stage1_gray.png",
                    "stage1_gray_equalized.png",
                    "stage1_binary_adaptive.png",
                    "stage1_binary_otsu.png",
                ],
                "config": {
                    "adaptive_block_size": self.cfg.adaptive_block_size,
                    "adaptive_c": self.cfg.adaptive_c,
                    "blur_kernel": self.cfg.blur_kernel,
                },
            },
        )

    # ---------- Stage 2 ----------
    def stage2_ocr_discovery(self) -> None:
        """Run the configured OCR route on Stage 1 grayscale to discover text regions."""
        stage1_input = self.out_dir / "stage1_gray.png"
        if not stage1_input.exists():
            raise FileNotFoundError(f"Stage 2 requires Stage 1 artifact: {stage1_input}")

        if self.cfg.ocr_route == "easyocr":
            ocr_result = run_easyocr_sahi(
                stage1_input,
                image_id=Path(self.image_path).name,
                cfg=EasyOcrSahiConfig(
                    slice_height=self.cfg.ocr_slice_height,
                    slice_width=self.cfg.ocr_slice_width,
                    overlap_height_ratio=self.cfg.ocr_overlap_height_ratio,
                    overlap_width_ratio=self.cfg.ocr_overlap_width_ratio,
                    min_score=self.cfg.ocr_min_score,
                    min_text_len=self.cfg.ocr_min_text_len,
                    low_text=self.cfg.ocr_low_text,
                    link_threshold=self.cfg.ocr_link_threshold,
                    line_merge_gap_px=self.cfg.ocr_line_merge_gap_px,
                    line_merge_y_tolerance_px=self.cfg.ocr_line_merge_y_tolerance_px,
                    enable_rotated_ocr=self.cfg.ocr_enable_rotated,
                ),
            )
        elif self.cfg.ocr_route == "gemini":
            ocr_result = run_gemini_ocr_sahi(
                stage1_input,
                image_id=Path(self.image_path).name,
                cfg=GeminiOcrSahiConfig(
                    postprocess_match_threshold=self.cfg.gemini_postprocess_match_threshold,
                ),
            )
        elif self.cfg.ocr_route == "paddleocr":
            ocr_result = run_paddle_ocr_sahi(
                stage1_input,
                image_id=Path(self.image_path).name,
                cfg=PaddleOcrSahiConfig(
                    slice_height=self.cfg.ocr_slice_height,
                    slice_width=self.cfg.ocr_slice_width,
                    overlap_height_ratio=self.cfg.ocr_overlap_height_ratio,
                    overlap_width_ratio=self.cfg.ocr_overlap_width_ratio,
                ),
            )
        elif self.cfg.ocr_route == "ocrmac":
            ocr_result = run_ocrmac_sahi(
                stage1_input,
                image_id=Path(self.image_path).name,
                cfg=OcrMacSahiConfig(
                    framework=self.cfg.ocrmac_framework,
                    recognition_level=self.cfg.ocrmac_recognition_level,
                    slice_height=self.cfg.ocr_slice_height,
                    slice_width=self.cfg.ocr_slice_width,
                    overlap_height_ratio=self.cfg.ocr_overlap_height_ratio,
                    overlap_width_ratio=self.cfg.ocr_overlap_width_ratio,
                    enable_rotated_ocr=self.cfg.ocr_enable_rotated,
                ),
            )
        else:
            raise ValueError(f"Unsupported ocr_route: {self.cfg.ocr_route}")
        ocr_result["summary"]["route"] = self.cfg.ocr_route
        if self.cfg.ocr_route == "gemini":
            ocr_result["summary"]["configured_postprocess_match_threshold"] = self.cfg.gemini_postprocess_match_threshold
        self._save_json("stage2_ocr_regions", ocr_result["regions_payload"])
        self._save_json("stage2_ocr_summary", ocr_result["summary"])
        self._save_json("stage2_ocr_exception_candidates", ocr_result["exception_candidates"])
        self._save_img("stage2_ocr_overlay", ocr_result["overlay_image"])
        if self.cfg.ocr_route == "gemini":
            self._save_json("stage2_gemini_patch_requests", ocr_result.get("patch_requests", []))
            self._save_json("stage2_gemini_patch_raw", ocr_result.get("patch_raw", []))
            self._save_json("stage2_gemini_crop_raw", ocr_result.get("crop_raw", []))

    def stage2b_ocr_tag_refinement(self) -> None:
        """Reclassify OCR unknown→instrument_tag regions near S4 instrument tag bboxes.

        Runs after stage4_instrument_tag_fusion so that S4 instrument tag bboxes
        are available for proximity-based reclassification.
        """
        s4_payload = self._load_json_artifact_or_default(
            "stage4_instrument_tags", {"instrument_tags": []}
        )
        instrument_tag_bboxes = s4_payload.get("instrument_tags", [])
        if not instrument_tag_bboxes:
            print("[INFO] No S4 instrument tags found, skipping OCR refinement")
            return

        ocr_payload = self._load_json_artifact("stage2_ocr_regions")
        text_regions = ocr_payload.get("text_regions", [])

        from garnet.ocrmac_sahi import _reclassify_nearby_tags
        refined = _reclassify_nearby_tags(text_regions, instrument_tag_bboxes, proximity_px=120.0)

        # Compute reclassification stats
        reclassified = sum(
            1 for r, orig in zip(refined, text_regions)
            if r.get("class") == "instrument_tag" and orig.get("class") == "unknown"
        )

        ocr_payload["text_regions"] = refined
        self._save_json("stage2_ocr_regions", ocr_payload)
        self._save_json("stage2_ocr_refinement_summary", {
            "reclassified_to_instrument_tag": reclassified,
            "total_regions": len(refined),
            "proximity_px": 120.0,
        })
        print(f"[INFO] OCR refinement: {reclassified} unknown → instrument_tag")

    # ---------- Stage 4 ----------
    def stage4_object_detection(self) -> None:
        """Run YOLO+SAHI object detection and derive topology markers from arrow/node classes."""
        detection_result = run_object_detection_sahi(
            self.image_path,
            image_id=Path(self.image_path).name,
            cfg=DetectionSahiConfig(
                weight_path=self.cfg.detection_weight_path,
                image_size=self.cfg.detection_image_size,
                overlap_ratio=self.cfg.detection_overlap_ratio,
                postprocess_type=self.cfg.detection_postprocess_type,
                postprocess_match_metric=self.cfg.detection_postprocess_match_metric,
                postprocess_match_threshold=self.cfg.detection_postprocess_match_threshold,
            ),
            connection_ports={},  # empty: skip midpoint heuristic ports; stage5 owns all port rendering
        )
        self._save_json("stage4_objects", detection_result["objects_payload"])
        self._save_json("stage4_objects_summary", detection_result["summary"])
        self._save_img("stage4_objects_overlay", detection_result["overlay_image"])
        topology_marker_result = run_topology_marker_router(
            image_id=Path(self.image_path).name,
            objects=detection_result["objects_payload"].get("objects", []),
        )
        self._save_json("stage4_topology_markers", topology_marker_result["topology_markers_payload"])
        self._save_json("stage4_topology_marker_summary", topology_marker_result["summary"])

    def stage4_line_number_fusion(self) -> None:
        """Fuse OCR text regions with detected objects to identify pipe line numbers."""
        object_payload = self._load_json_artifact("stage4_objects")
        ocr_payload = self._load_json_artifact("stage2_ocr_regions")
        fusion_result = run_line_number_fusion_stage(
            image_id=Path(self.image_path).name,
            image_bgr=self._ensure_image_loaded(),
            object_regions=object_payload.get("objects", []),
            text_regions=ocr_payload.get("text_regions", []),
            max_distance_px=self.cfg.line_number_fusion_max_distance_px,
        )
        self._save_json("stage4_line_numbers", fusion_result["line_numbers_payload"])
        self._save_json("stage4_line_number_summary", fusion_result["summary"])
        self._save_img("stage4_line_number_overlay", fusion_result["overlay_image"])

    def stage4_instrument_tag_fusion(self) -> None:
        """Fuse OCR text regions with detected objects to identify instrument tags."""
        object_payload = self._load_json_artifact("stage4_objects")
        ocr_payload = self._load_json_artifact("stage2_ocr_regions")
        fusion_result = run_instrument_tag_fusion_stage(
            image_id=Path(self.image_path).name,
            image_bgr=self._ensure_image_loaded(),
            object_regions=object_payload.get("objects", []),
            text_regions=ocr_payload.get("text_regions", []),
            max_distance_px=self.cfg.instrument_tag_fusion_max_distance_px,
        )
        self._save_json("stage4_instrument_tags", fusion_result["instrument_tags_payload"])
        self._save_json("stage4_instrument_tag_summary", fusion_result["summary"])
        self._save_img("stage4_instrument_tag_overlay", fusion_result["overlay_image"])

    def stage4_equipment_tag_fusion(self) -> None:
        ocr_payload = self._load_json_artifact_or_default("stage2_ocr_regions", {"text_regions": []})
        text_regions = ocr_payload.get("text_regions", [])
        fusion_result = run_equipment_tag_fusion_stage(
            image_id=Path(self.image_path).name,
            image_bgr=self._ensure_image_loaded() if text_regions else self.image_bgr,
            object_regions=text_regions,
            max_distance_px=self.cfg.equipment_tag_fusion_max_distance_px,
        )
        self._save_json("stage4_equipment_tags", fusion_result["equipment_tags_payload"])
        self._save_json("stage4_equipment_tag_summary", fusion_result["summary"])
        self._save_img("stage4_equipment_tag_overlay", fusion_result["overlay_image"])

    # ---------- Stage 5 ----------
    def stage5_pipe_mask(self) -> None:
        """Generate analysis and continuity pipe masks from OCR/object-suppressed candidates."""
        gray_path = self.out_dir / "stage1_gray.png"
        adaptive_path = self.out_dir / "stage1_binary_adaptive.png"
        otsu_path = self.out_dir / "stage1_binary_otsu.png"
        if not gray_path.exists() or not adaptive_path.exists() or not otsu_path.exists():
            raise FileNotFoundError("Stage 5 requires Stage 1 grayscale and binary artifacts")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 5 pipe-mask generation")

        gray_image = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        adaptive_mask = cv2.imread(str(adaptive_path), cv2.IMREAD_GRAYSCALE)
        otsu_mask = cv2.imread(str(otsu_path), cv2.IMREAD_GRAYSCALE)
        if gray_image is None or adaptive_mask is None or otsu_mask is None:
            raise RuntimeError("Failed to load Stage 1 artifacts for Stage 5")

        ocr_regions = self._load_json_artifact("stage2_ocr_regions").get("text_regions", [])
        object_regions = self._load_json_artifact("stage4_objects").get("objects", [])
        pipe_mask_result = run_pipe_mask_stage(
            image_bgr=self._ensure_image_loaded(),
            gray_image=gray_image,
            adaptive_mask=adaptive_mask,
            otsu_mask=otsu_mask,
            ocr_regions=ocr_regions,
            object_regions=object_regions,
            image_id=Path(self.image_path).name,
            ocr_padding=self.cfg.pipe_mask_ocr_padding,
            object_inset=self.cfg.pipe_mask_object_inset,
            min_component_area=self.cfg.pipe_mask_min_component_area,
            preserve_ocr_classes=self.cfg.pipe_mask_preserve_ocr_classes,
            preserve_object_classes=self.cfg.pipe_mask_preserve_object_classes,
        )
        self._save_img("stage5_pipe_mask", pipe_mask_result["mask_image"])
        self._save_img("stage5_pipe_mask_overlay", pipe_mask_result["overlay_image"])
        self._save_json("stage5_pipe_mask_summary", pipe_mask_result["summary"])

    # ------------------------------------------------------------------
    # VLM port detection (replaces heuristic get_connection_ports)
    # ------------------------------------------------------------------

    def _detect_port_cv(
        self, image: np.ndarray, bbox: dict[str, int], track_len: int = 60
    ) -> Optional[tuple[int, int, str]]:
        """Detect pipe port using edge scanning + outward tracking (no VLM).

        Scans all 4 edges of the bbox for non-background pixel clusters,
        then tracks outward perpendicular to each edge. The edge+cluster
        with the longest contiguous non-background run wins.

        Returns (x, y, direction) or None if no pipe found.
        """
        import cv2 as _cv2

        x1, y1 = bbox["x_min"], bbox["y_min"]
        x2, y2 = bbox["x_max"], bbox["y_max"]
        h_img, w_img = image.shape[:2]

        # Convert to grayscale for pixel checks
        if len(image.shape) == 3:
            gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Background threshold (median of a strip outside the bbox)
        bg_strip_y = max(0, y1 - 10)
        bg_strip = gray[bg_strip_y:y1, x1:x2] if y1 > 10 else gray[0:10, x1:x2]
        bg_val = float(np.median(bg_strip)) if bg_strip.size > 0 else 200

        # Define edges: (name, start_xy, end_xy, step_direction, track_dx, track_dy)
        edges = [
            ("TOP",    (x1, y1), (x2, y1), (1, 0),   (0, -1)),
            ("BOTTOM", (x1, y2), (x2, y2), (1, 0),   (0,  1)),
            ("LEFT",   (x1, y1), (x1, y2), (0, 1),   (-1, 0)),
            ("RIGHT",  (x2, y1), (x2, y2), (0, 1),   (1,  0)),
        ]

        best_score = 0
        best_result = None

        for edge_name, start, end, step_dir, track_dir in edges:
            sx, sy = start
            ex, ey = end
            length = max(abs(ex - sx), abs(ey - sy))

            for i in range(length):
                px = sx + i * step_dir[0]
                py = sy + i * step_dir[1]
                if px < 0 or px >= w_img or py < 0 or py >= h_img:
                    continue

                # Check if this pixel is pipe (dark)
                pixel_val = gray[py, px]
                if pixel_val > bg_val * 0.7:  # near background → skip
                    continue

                # Found a dark pixel — track outward
                track_count = 0
                tx, ty = px, py
                for _ in range(track_len):
                    tx += track_dir[0]
                    ty += track_dir[1]
                    if tx < 0 or tx >= w_img or ty < 0 or ty >= h_img:
                        break
                    if gray[ty, tx] < bg_val * 0.6:
                        track_count += 1
                    else:
                        break  # hit background

                if track_count > best_score:
                    best_score = track_count
                    direction = {"TOP": "UP", "BOTTOM": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}[edge_name]
                    best_result = (px, py, direction)

                # Skip the rest of this cluster
                while i + 1 < length:
                    nx = sx + (i + 1) * step_dir[0]
                    ny = sy + (i + 1) * step_dir[1]
                    if nx < 0 or nx >= w_img or ny < 0 or ny >= h_img:
                        i += 1
                        continue
                    if gray[ny, nx] > bg_val * 0.7:
                        break
                    i += 1

        if best_result and best_score >= 3:
            px, py, direction = best_result
            # Snap to bbox edge
            if direction == "UP":
                py = y1
            elif direction == "DOWN":
                py = y2
            elif direction == "LEFT":
                px = x1
            elif direction == "RIGHT":
                px = x2
            return (int(px), int(py), direction)

        return None

    def _compute_connection_ports_vlm(
        self, objects: list[dict[str, Any]]
    ) -> dict[str, list[tuple[int, int, str]]]:
        """Use VLM to determine pipe ports on all page connection symbols.

        Returns {object_id: [(port_x, port_y, direction), ...]} — same format
        as get_connection_ports so downstream consumers are unchanged.
        """
        import time as _time
        ports: dict[str, list[tuple[int, int, str]]] = {}
        image = self._ensure_image_loaded()

        conn_objects = [
            o for o in objects
            if o.get("class_name") in ("page_connection", "page connection",
                                        "connection", "utility connection",
                                        "page connection symbol")
        ]
        if not conn_objects:
            # Log what classes are present for debugging
            all_classes = {o.get("class_name", "?") for o in objects}
            logger.info("No connection objects found. Stage4 classes: %s", sorted(all_classes))
            return ports

        logger.info("Port detection (%s): %d connection objects", self.cfg.port_detection_mode, len(conn_objects))
        for i, obj in enumerate(conn_objects):
            obj_id = obj["id"]
            bbox = obj["bbox"]

            if self.cfg.port_detection_mode == "cv":
                # CV edge-scan only
                result = self._detect_port_cv(image, bbox)
                if result:
                    px, py, direction = result
                    ports[obj_id] = [(px, py, direction)]
                    logger.info("  %s -> %s (%d,%d) [CV]", obj_id, direction, px, py)
                else:
                    logger.warning("  %s -> CV failed, skipping", obj_id)
                continue

            # VLM mode (with CV fallback if vlm+cv)
            other_bboxes = [
                o["bbox"] for j, o in enumerate(conn_objects)
                if j != i
            ]

            result = compute_port_vlm(
                image, bbox,
                model=self.cfg.port_detection_model,
                mask_bboxes=other_bboxes if other_bboxes else None,
            )
            if result:
                px, py, direction = result
                ports[obj_id] = [(px, py, direction)]
                logger.info("  %s -> %s (%d,%d) [VLM]", obj_id, direction, px, py)
            elif self.cfg.port_detection_mode == "vlm+cv":
                result_cv = self._detect_port_cv(image, bbox)
                if result_cv:
                    px, py, direction = result_cv
                    ports[obj_id] = [(px, py, direction)]
                    logger.info("  %s -> %s (%d,%d) [CV fallback]", obj_id, direction, px, py)
                else:
                    logger.warning("  %s -> VLM+CV both failed, skipping", obj_id)
            else:
                logger.warning("  %s -> VLM failed, skipping", obj_id)
            _time.sleep(0.5)  # rate limit

        logger.info("VLM port detection done: %d/%d ports found",
                 len(ports), len(conn_objects))
        return ports

    # ---------- Stage 5b: CV Pipe Tracing ----------
    def stage5b_pipe_trace(self) -> None:
        """Trace pipes from each connection port to their terminals using CV.

        Uses the pipe mask (stage5) to walk from each port pixel-by-pixel.
        Detects turns, inline objects, and terminals (page connections,
        instrument tags, equipment, tee junctions, sheet edges, dead ends).

        Saves stage5b_trace_results.json and stage5b_trace_overlay.png.
        """
        from garnet.visual_primitives.cv_pipe_tracer import CVPipeTracer, TraceToken, TerminalType

        import cv2 as _cv2
        import time as _time

        ports = self._load_json_artifact("stage5_connection_ports")
        if not ports:
            logger.warning("No connection ports found — skipping pipe trace")
            return

        objects = self._load_json_artifact("stage4_objects").get("objects", [])
        pipe_mask = _cv2.imread(
            str(self.out_dir / "stage5_pipe_mask.png"), _cv2.IMREAD_GRAYSCALE
        )
        if pipe_mask is None:
            logger.error("Cannot load stage5_pipe_mask.png")
            return

        # Morphological close: bridge 1-2px gaps at corners and text edges
        # without merging distinct parallel pipes (3x3 kernel is safe).
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
        pipe_mask = _cv2.morphologyEx(pipe_mask, _cv2.MORPH_CLOSE, kernel)

        image = self._ensure_image_loaded()

        # Separate stage4 objects by type
        page_connections = [
            o for o in objects
            if o.get("class_name") in ("page_connection", "page connection",
                                        "connection", "utility connection",
                                        "page connection symbol")
        ]
        instrument_tags = [
            o for o in objects
            if o.get("class_name") in (
                "instrument tag", "instrument dcs", "instrument logic",
            )
        ]
        equipment = [
            o for o in objects
            if o.get("class_name") in (
                "vessel", "column", "pump", "compressor", "blower",
                "heat_exchanger", "tank", "reactor", "knockout_drum",
                "filter", "cooler", "heater",
            )
        ]

        # Inline symbols (valves, reducers, etc.)
        inline_classes = {
            "gate_valve", "globe_valve", "check_valve", "ball_valve",
            "butterfly_valve", "control_valve", "pressure_relief_valve",
            "reducer", "spectacle_blind", "strainer",
        }
        inline_symbols = [
            o for o in objects
            if o.get("class_name") in inline_classes
        ]

        # Extend pipe mask into equipment and instrument bboxes
        # so the tracer can walk into terminal objects instead of
        # stopping at the mask edge.
        _all_terminals = equipment + instrument_tags
        if _all_terminals:
            pipe_mask = self._extend_mask_to_terminals(
                pipe_mask, _all_terminals, max_gap=80,
            )
            logger.info(
                "Extended pipe mask toward %d terminals",
                len(_all_terminals),
            )

        # Trace from each port
        visited = np.zeros_like(pipe_mask)
        all_results: dict[str, dict] = {}

        logger.info("CV pipe trace: %d ports", len(ports))
        t0 = _time.monotonic()

        for obj_id, port_list in ports.items():
            for px, py, direction in port_list:
                tracer = CVPipeTracer(
                    pipe_mask=pipe_mask,
                    image=image,
                    page_connections=page_connections,
                    instrument_tags=instrument_tags,
                    equipment_objects=equipment,
                    visited_mask=visited,
                )
                tracer.set_inline_symbols(inline_symbols)

                result = tracer.trace(px, py, direction, source_obj_id=obj_id)

                all_results[obj_id] = {
                    "port": {"x": px, "y": py, "direction": direction},
                    "terminal_type": result.terminal_type,
                    "terminal_x": result.terminal_x,
                    "terminal_y": result.terminal_y,
                    "terminal_obj_id": result.terminal_obj_id,
                    "segments": [
                        {
                            "x1": s.x1, "y1": s.y1,
                            "x2": s.x2, "y2": s.y2,
                            "direction": s.direction,
                            "length_px": s.length_px,
                        }
                        for s in result.segments
                    ],
                    "turns": [
                        {"x": tx, "y": ty, "new_dir": td}
                        for tx, ty, td in result.turns
                    ],
                    "hits": [
                        {"class": h.class_name, "x": h.x, "y": h.y}
                        for h in result.hits
                    ],
                    "trace_length_px": result.trace_length_px,
                    "status": result.status,
                }
                logger.info(
                    "  %s -> %s (%d px, %d segs)",
                    obj_id, result.terminal_type,
                    result.trace_length_px, len(result.segments),
                )

        elapsed = _time.monotonic() - t0
        logger.info("CV pipe trace done: %d traces in %.1fs", len(all_results), elapsed)

        self._save_json("stage5b_trace_results", all_results)

        # Draw overlay
        overlay = image.copy()
        colors = {
            "page_connection": (0, 180, 0),
            "instrument_tag": (0, 120, 0),
            "equipment": (0, 100, 180),
            "tee_junction": (180, 0, 180),
            "sheet_edge": (100, 100, 100),
            "dead_end": (0, 0, 200),
            "max_steps": (200, 100, 0),
        }
        for obj_id, data in all_results.items():
            # Draw trace path
            for seg in data["segments"]:
                _cv2.line(
                    overlay,
                    (seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]),
                    (0, 200, 0), 2,
                )
                # Arrow at midpoint
                mx = (seg["x1"] + seg["x2"]) // 2
                my = (seg["y1"] + seg["y2"]) // 2
                _cv2.circle(overlay, (mx, my), 3, (0, 160, 0), -1)

            # Terminal marker
            tx, ty = data["terminal_x"], data["terminal_y"]
            ttype = data.get("terminal_type", "unknown")
            color = colors.get(ttype, (128, 128, 128))
            _cv2.circle(overlay, (tx, ty), 8, color, -1)
            _cv2.circle(overlay, (tx, ty), 8, (255, 255, 255), 1)

            # Label
            label = f"{obj_id}:{ttype}"
            _cv2.putText(
                overlay, label,
                (tx + 12, ty - 12),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        self._save_img("stage5b_trace_overlay", overlay)

    # ---------- Stage 5: Geometric line-detection alternative ----------
    def stage5_geometric_line_detection(self) -> None:
        """Geometric pipeline: adaptive threshold → corner detection → Telea inpaint
        → contour extraction → collinear + endpoint merge → segment mask."""
        gray_path = self.out_dir / "stage1_gray.png"
        if not gray_path.exists():
            raise FileNotFoundError("Stage 5 (geometric) requires Stage 1 grayscale artifact")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 5 geometric line detection")

        gray_image = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            raise RuntimeError(f"Failed to load Stage 1 grayscale: {gray_path}")

        ocr_regions = self._load_json_artifact("stage2_ocr_regions").get("text_regions", [])
        object_regions = self._load_json_artifact("stage4_objects").get("objects", [])

        result = run_line_detection_inpaint(
            stage1_gray=gray_image,
            text_regions=ocr_regions,
            object_regions=object_regions,
            image_id=Path(self.image_path).name,
        )

        # Render segments to binary mask so Stage 6+ consume it unchanged
        h, w = gray_image.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for seg in result["segments"]:
            cv2.line(mask, (seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]), 255, thickness=2)
        self._save_img("stage5_pipe_mask", mask)

        # Save overlay for visual inspection
        image_bgr = self._ensure_image_loaded()
        overlay = render_line_overlay(image_bgr, result["segments"])
        self._save_img("stage5_geometric_line_overlay", overlay)

        # Save segments and summary as JSON (convert numpy ints to Python ints)
        json_segments = [
            {k: int(v) if isinstance(v, (np.integer,)) else v for k, v in seg.items()}
            for seg in result["segments"]
        ]
        self._save_json("stage5_geometric_segments", json_segments)
        self._save_json("stage5_geometric_summary", result["summary"])

        # Load pipe mask for mask-validation of ports
        import cv2 as _cv2
        pipe_mask = _cv2.imread(str(self.out_dir / "stage5_pipe_mask.png"), _cv2.IMREAD_GRAYSCALE)

        # Compute connection ports using VLM (firmed process)
        objects = self._load_json_artifact("stage4_objects").get("objects", [])
        ports = self._compute_connection_ports_vlm(objects)
        self._save_json("stage5_connection_ports", ports)

        # Overlay true port markers onto the stage4_objects_overlay image
        self._add_port_markers_to_overlay(ports)

    # ---------- Stage 5 dispatcher ----------
    def stage5_dispatcher(self) -> None:
        if self.cfg.use_geometric_line_detection:
            self.stage5_geometric_line_detection()
        else:
            self.stage5_pipe_mask()

    # ---------- Stage 6 ----------
    def stage6_morphological_sealing(self) -> None:
        """Apply morphological closing to the continuity mask for unbroken topology extraction."""
        pipe_mask_path = self.out_dir / "stage5_pipe_continuity_mask.png"
        if not pipe_mask_path.exists():
            raise FileNotFoundError("Stage 6 requires Stage 5 continuity mask artifact")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 6 morphological sealing")

        pipe_mask = cv2.imread(str(pipe_mask_path), cv2.IMREAD_GRAYSCALE)
        if pipe_mask is None:
            raise RuntimeError(f"Failed to load Stage 5 pipe mask: {pipe_mask_path}")

        seal_result = run_pipe_seal_stage(
            image_bgr=self._ensure_image_loaded(),
            pipe_mask=pipe_mask,
            image_id=Path(self.image_path).name,
            horizontal_close_kernel=self.cfg.pipe_seal_horizontal_close_kernel,
            vertical_close_kernel=self.cfg.pipe_seal_vertical_close_kernel,
            min_component_area=self.cfg.pipe_seal_min_component_area,
        )
        self._save_img("stage6_pipe_mask_sealed", seal_result["sealed_mask_image"])
        self._save_img("stage6_pipe_mask_sealed_overlay", seal_result["overlay_image"])
        self._save_json("stage6_pipe_mask_sealed_summary", seal_result["summary"])

    # ---------- Stage 7 ----------
    def stage7_skeleton_generation(self) -> None:
        """Compute medial-axis skeleton from the continuity-sealed pipe mask."""
        sealed_mask_path = self.out_dir / "stage6_pipe_mask_sealed.png"
        if not sealed_mask_path.exists():
            raise FileNotFoundError(f"Stage 7 requires Stage 6 artifact: {sealed_mask_path}")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 7 skeleton generation")

        sealed_mask = cv2.imread(str(sealed_mask_path), cv2.IMREAD_GRAYSCALE)
        if sealed_mask is None:
            raise RuntimeError(f"Failed to load Stage 6 sealed mask: {sealed_mask_path}")

        skeleton_result = run_pipe_skeleton_stage(
            image_bgr=self._ensure_image_loaded(),
            sealed_mask=sealed_mask,
            image_id=Path(self.image_path).name,
        )
        self._save_img("stage7_pipe_skeleton", skeleton_result["skeleton_image"])
        self._save_img("stage7_pipe_skeleton_overlay", skeleton_result["overlay_image"])
        self._save_json("stage7_pipe_skeleton_summary", skeleton_result["summary"])

    # ---------- Stage 8 ----------
    def stage8_skeleton_node_detection(self) -> None:
        """Detect skeleton endpoints and junctions from the skeleton image."""
        skeleton_path = self.out_dir / "stage7_pipe_skeleton.png"
        if not skeleton_path.exists():
            raise FileNotFoundError(f"Stage 8 requires Stage 7 artifact: {skeleton_path}")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 8 skeleton node detection")

        skeleton_mask = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
        if skeleton_mask is None:
            raise RuntimeError(f"Failed to load Stage 7 skeleton: {skeleton_path}")

        node_result = run_pipe_node_stage(
            image_bgr=self._ensure_image_loaded(),
            skeleton_mask=skeleton_mask,
            image_id=Path(self.image_path).name,
        )
        self._save_img("stage8_endpoints", node_result["endpoint_image"])
        self._save_img("stage8_junctions", node_result["junction_image"])
        self._save_img("stage8_nodes_overlay", node_result["overlay_image"])
        self._save_json("stage8_node_summary", node_result["summary"])

    # ---------- Stage 9 ----------
    def stage9_node_clustering(self) -> None:
        """Cluster nearby skeleton nodes using DBSCAN into consolidated graph nodes."""
        endpoints_path = self.out_dir / "stage8_endpoints.png"
        junctions_path = self.out_dir / "stage8_junctions.png"
        if not endpoints_path.exists() or not junctions_path.exists():
            raise FileNotFoundError("Stage 9 requires Stage 8 endpoint and junction artifacts")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 9 node clustering")

        endpoint_mask = cv2.imread(str(endpoints_path), cv2.IMREAD_GRAYSCALE)
        junction_mask = cv2.imread(str(junctions_path), cv2.IMREAD_GRAYSCALE)
        if endpoint_mask is None or junction_mask is None:
            raise RuntimeError("Failed to load Stage 8 node masks")

        cluster_result = run_pipe_node_cluster_stage(
            image_bgr=self._ensure_image_loaded(),
            endpoint_mask=endpoint_mask,
            junction_mask=junction_mask,
            image_id=Path(self.image_path).name,
            cluster_eps=self.cfg.node_cluster_eps,
            cluster_min_samples=self.cfg.node_cluster_min_samples,
        )
        self._save_img("stage9_endpoint_clusters", cluster_result["endpoint_cluster_image"])
        self._save_img("stage9_junction_clusters", cluster_result["junction_cluster_image"])
        self._save_img("stage9_node_clusters_overlay", cluster_result["overlay_image"])
        self._save_json("stage9_node_clusters", cluster_result["clusters_payload"])
        self._save_json("stage9_node_cluster_summary", cluster_result["summary"])

    # ---------- Stage 10 ----------
    def stage10_edge_tracing(self) -> None:
        """Resolve crossings, then trace pipe edges between clustered nodes."""
        sealed_mask_path = self.out_dir / "stage6_pipe_mask_sealed.png"
        skeleton_path = self.out_dir / "stage7_pipe_skeleton.png"
        node_clusters_path = self.out_dir / "stage9_node_clusters.json"
        if not sealed_mask_path.exists() or not skeleton_path.exists() or not node_clusters_path.exists():
            raise FileNotFoundError("Stage 10 requires Stage 6 sealed mask, Stage 7 skeleton, and Stage 9 clustered nodes")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 10 edge tracing")

        sealed_mask = cv2.imread(str(sealed_mask_path), cv2.IMREAD_GRAYSCALE)
        skeleton_mask = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
        if sealed_mask is None or skeleton_mask is None:
            raise RuntimeError("Failed to load Stage 6 sealed mask or Stage 7 skeleton")

        clusters_payload = self._load_json_artifact("stage9_node_clusters")
        topology_markers_path = self.out_dir / "stage4_topology_markers.json"
        topology_markers_payload = {"topology_markers": []}
        if topology_markers_path.exists():
            topology_markers_payload = self._load_json_artifact("stage4_topology_markers")
        crossing_result = run_pipe_crossing_stage(
            image_bgr=self._ensure_image_loaded(),
            sealed_mask=sealed_mask,
            skeleton_mask=skeleton_mask,
            node_clusters=clusters_payload.get("clusters", []),
            topology_markers=topology_markers_payload.get("topology_markers", []),
            image_id=Path(self.image_path).name,
            branch_stub_length_px=self.cfg.crossing_branch_stub_length_px,
            branch_merge_angle_tolerance_deg=self.cfg.crossing_branch_merge_angle_tolerance_deg,
            opposite_angle_tolerance_deg=self.cfg.crossing_opposite_angle_tolerance_deg,
            center_blob_radius_px=self.cfg.crossing_center_blob_radius_px,
            center_blob_threshold=self.cfg.crossing_center_blob_threshold,
            stage4_marker_match_distance_px=self.cfg.crossing_stage4_marker_match_distance_px,
        )
        edge_result = run_pipe_edge_stage(
            image_bgr=self._ensure_image_loaded(),
            skeleton_mask=skeleton_mask,
            node_clusters=clusters_payload.get("clusters", []),
            image_id=Path(self.image_path).name,
            min_edge_length_px=self.cfg.min_edge_length_px,
            crossing_resolution=crossing_result["crossings_payload"].get("candidates", []),
        )
        self._save_img("stage10_crossing_resolution_overlay", crossing_result["overlay_image"])
        self._save_json("stage10_crossing_resolution", crossing_result["crossings_payload"])
        self._save_json("stage10_crossing_resolution_summary", crossing_result["summary"])
        self._save_img("stage10_pipe_edges_overlay", edge_result["overlay_image"])
        self._save_json("stage10_pipe_edges", edge_result["edges_payload"])
        self._save_json("stage10_pipe_edge_summary", edge_result["summary"])
        # Phase 2: continuity-aware outputs for Stage 11/12
        if "continuity_result" in edge_result:
            self._save_json("stage10_continuity_result", edge_result["continuity_result"])
        if "gap_summary" in edge_result:
            self._save_json("stage10_gap_summary", {"gaps": edge_result["gap_summary"]})

    def stage10b_polyline_simplification(self) -> None:
        edges_payload = self._load_json_artifact("stage10_pipe_edges")
        simplification_result = run_polyline_simplification_stage(
            edges=edges_payload.get("edges", []),
            image_id=Path(self.image_path).name,
            epsilon=self.cfg.polyline_simplify_epsilon,
        )
        self._save_json("stage10b_pipe_edges_simplified", simplification_result["edges_payload"])
        self._save_json("stage10b_polyline_simplification_summary", simplification_result["summary"])

    def stage10c_edge_direction(self) -> None:
        edges_payload = self._load_json_artifact("stage10b_pipe_edges_simplified")
        object_payload = self._load_json_artifact("stage4_objects")
        direction_result = run_edge_direction_stage(
            edges=edges_payload.get("edges", []),
            objects=object_payload.get("objects", []),
            image_id=Path(self.image_path).name,
            arrow_proximity_px=self.cfg.arrow_proximity_px,
        )
        self._save_json("stage10c_edge_direction", direction_result["edges_payload"])
        self._save_json("stage10c_arrow_assignments", {"arrow_assignments": direction_result["arrow_assignments"]})
        self._save_json("stage10c_edge_direction_summary", direction_result["summary"])

    def stage10d_edge_split(self) -> None:
        edges_payload = self._load_json_artifact("stage10c_edge_direction")
        object_payload = self._load_json_artifact("stage4_objects")
        connections_path = self.out_dir / "stage12_edge_connections.json"
        if connections_path.exists():
            connections_payload = self._load_json_artifact("stage12_edge_connections")
            edge_connections = connections_payload.get("edge_connections", [])
        else:
            node_clusters_payload = self._load_json_artifact("stage9_node_clusters")
            connectivity_result = build_pipe_edge_connectivity(
                edges=edges_payload.get("edges", []),
                node_clusters=node_clusters_payload.get("clusters", []),
                object_regions=object_payload.get("objects", []),
                inline_connector_classes=self.cfg.graph_inline_connector_classes,
                inline_match_distance_px=self.cfg.graph_inline_connector_match_distance_px,
            )
            edge_connections = connectivity_result["connections"]
        inline_connections = [
            {
                **connection,
                "inline_match_distance_px": self.cfg.graph_inline_connector_match_distance_px,
            }
            for connection in edge_connections
            if str(connection.get("kind", "")) == "inline_element"
        ]
        split_result = split_edges_at_inline_elements(
            edges=edges_payload.get("edges", []),
            inline_connections=inline_connections,
            objects=object_payload.get("objects", []),
            confidence_threshold=self.cfg.inline_split_confidence_threshold,
        )
        split_result["edges_payload"]["image_id"] = Path(self.image_path).name
        split_result["summary"]["image_id"] = Path(self.image_path).name
        self._save_json("stage10d_split_edges", split_result["edges_payload"])
        self._save_json("stage10d_split_nodes", {"nodes": split_result["split_nodes"]})
        self._save_json("stage10d_split_report", split_result["split_report"])
        self._save_json("stage10d_split_summary", split_result["summary"])

    # ---------- Stage 11 ----------
    def stage11_junction_review(self) -> None:
        """Review crossing candidates and classify as confirmed junctions or unresolved."""
        crossing_payload_path = self.out_dir / "stage10_crossing_resolution.json"
        if not crossing_payload_path.exists():
            raise FileNotFoundError("Stage 11 requires Stage 10 crossing resolution artifacts")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 11 junction review")

        junction_result = run_pipe_junction_stage(
            image_bgr=self._ensure_image_loaded(),
            crossing_candidates=self._load_json_artifact("stage10_crossing_resolution").get("candidates", []),
            image_id=Path(self.image_path).name,
        )
        self._save_img("stage11_confirmed_junctions", junction_result["confirmed_junction_image"])
        self._save_img("stage11_unresolved_junctions", junction_result["unresolved_junction_image"])
        self._save_img("stage11_junction_review_overlay", junction_result["overlay_image"])
        self._save_json("stage11_junctions", junction_result["junctions_payload"])
        self._save_json("stage11_junction_review_summary", junction_result["summary"])

    def stage12_geometric_graph_assembly(self) -> None:
        """Phase 3 geometric bypass: build Stage 12 graph directly from Stage 5 segments."""
        object_payload = self._load_json_artifact("stage4_objects")
        text_payload = self._load_json_artifact("stage4_line_numbers")
        instrument_tag_payload = self._load_json_artifact("stage4_instrument_tags")
        equipment_tag_payload = self._load_json_artifact_or_default("stage4_equipment_tags", {"equipment_tags": []})
        segments_payload = self._load_json_artifact("stage5_geometric_segments")
        if isinstance(segments_payload, dict):
            segments = segments_payload.get("segments", [])
        else:
            segments = segments_payload
        if not isinstance(segments, list):
            raise ValueError("stage5_geometric_segments must be a list or contain a 'segments' list")

        image_id = Path(self.image_path).name
        runs = chain_geometric_segments(segments)
        junctions = detect_junctions_from_runs(runs)
        geo_graph_result = build_graph_from_runs_and_junctions(runs, junctions, image_id=image_id)
        node_clusters = geo_graph_result["node_clusters"]
        raw_edges = geo_graph_result["edges_payload"].get("edges", [])

        self._save_json("phase3_runs", {"image_id": image_id, "runs": runs})
        self._save_json("phase3_junctions", {"image_id": image_id, "junctions": junctions})
        self._save_json("phase3_graph", geo_graph_result["graph_payload"])
        self._save_json("phase3_graph_summary", geo_graph_result["summary"])
        self._save_json("phase3_node_clusters", {"image_id": image_id, "clusters": node_clusters})
        self._save_json("phase3_pipe_edges", {"image_id": image_id, "pass_type": "sheet", "edges": raw_edges})

        direction_result = run_edge_direction_stage(
            edges=raw_edges,
            objects=object_payload.get("objects", []),
            image_id=image_id,
            arrow_proximity_px=self.cfg.arrow_proximity_px,
        )
        directed_edges = direction_result["edges_payload"].get("edges", [])
        self._save_json("phase3_edge_direction", direction_result["edges_payload"])
        self._save_json("phase3_arrow_assignments", {"arrow_assignments": direction_result["arrow_assignments"]})
        self._save_json("phase3_edge_direction_summary", direction_result["summary"])

        overlay_edge_filter_result = _filter_border_like_edges(
            directed_edges,
            self._ensure_image_loaded().shape,
        )

        edge_terminal_result = classify_pipe_edge_terminals(
            edges=directed_edges,
            node_clusters=node_clusters,
            object_regions=object_payload.get("objects", []),
            equipment_terminal_classes=self.cfg.terminal_equipment_classes,
            connection_terminal_classes=self.cfg.terminal_connection_classes,
            inline_passthrough_classes=self.cfg.terminal_inline_passthrough_classes,
            match_distance_px=self.cfg.terminal_match_distance_px,
        )
        edge_terminal_map = {
            str(item.get("edge_id", "")): item
            for item in edge_terminal_result["edge_terminals"]
            if item.get("edge_id") is not None
        }

        attachment_result = run_pipe_equipment_attachment_stage(
            image_id=image_id,
            objects=object_payload.get("objects", []),
            edges=directed_edges,
            attachment_classes=self.cfg.equipment_attachment_classes,
            max_distance_px=self.cfg.equipment_attachment_max_distance_px,
            k_candidate_edges=self.cfg.equipment_attachment_k_candidate_edges,
        )
        connection_attachment_result = run_pipe_equipment_attachment_stage(
            image_id=image_id,
            objects=object_payload.get("objects", []),
            edges=directed_edges,
            attachment_classes=self.cfg.connection_attachment_classes,
            max_distance_px=self.cfg.connection_attachment_max_distance_px,
            k_candidate_edges=self.cfg.connection_attachment_k_candidate_edges,
        )

        # S5: Run edge connectivity first, then detect only the gaps that
        # weren't already handled (dedup against existing connections).
        edge_connectivity_result = build_pipe_edge_connectivity(
            edges=directed_edges,
            node_clusters=node_clusters,
            object_regions=object_payload.get("objects", []),
            inline_connector_classes=self.cfg.graph_inline_connector_classes,
            inline_match_distance_px=self.cfg.graph_inline_connector_match_distance_px,
            connection_seed_edge_ids={
                str(item.get("edge_id", ""))
                for item in connection_attachment_result["attachments_payload"].get("accepted", [])
                if item.get("edge_id") is not None
            },
        )

        # S5-01: Detect remaining near-edge gaps (exclude pairs already connected above)
        from garnet.geometric_graph_builder import detect_phase3_gaps

        phase3_gaps = detect_phase3_gaps(
            edges=directed_edges,
            gap_threshold_px=GAP_THRESHOLD_PX,
            existing_connections=edge_connectivity_result["connections"],
        )

        # S5: Wire all detected gaps as gap_seed connections (quality filter
        # moved to pipe_edge_connectivity — it handles strict/good/weak tiers there)
        if phase3_gaps:
            edge_connectivity_result_2 = build_pipe_edge_connectivity(
                edges=directed_edges,
                node_clusters=node_clusters,
                object_regions=object_payload.get("objects", []),
                inline_connector_classes=self.cfg.graph_inline_connector_classes,
                inline_match_distance_px=self.cfg.graph_inline_connector_match_distance_px,
                connection_seed_edge_ids={
                    str(item.get("edge_id", ""))
                    for item in connection_attachment_result["attachments_payload"].get("accepted", [])
                    if item.get("edge_id") is not None
                },
                gap_seed_connections=phase3_gaps,
            )
            # Merge: take all connections from the second call that are gap_seed kind
            existing_ids = {frozenset((c["source_edge_id"], c["target_edge_id"])) for c in edge_connectivity_result["connections"]}
            for conn in edge_connectivity_result_2["connections"]:
                pair = frozenset((conn["source_edge_id"], conn["target_edge_id"]))
                if conn.get("kind") == "gap_seed" and pair not in existing_ids:
                    edge_connectivity_result["connections"].append(conn)
                    edge_connectivity_result["summary"]["edge_connection_count"] += 1

        from garnet.continuity_aware_connections import validate_connections_against_gaps

        connection_validation = validate_connections_against_gaps(
            edges=directed_edges,
            connections=edge_connectivity_result["connections"],
            gap_summary=phase3_gaps,
        )

        self._save_json("phase3_gaps", {
            "image_id": image_id,
            "gaps": phase3_gaps,
            "gap_coverage_pct": round(
                len(connection_validation.get("connected_gaps", [])) / len(phase3_gaps) * 100, 1
            ) if phase3_gaps else 100.0,
        })

        overlay_edges = [
            {
                **edge,
                "edge_terminals": edge_terminal_map.get(str(edge.get("id", ""))),
            }
            for edge in directed_edges
        ]
        text_attachment_result = run_pipe_text_attachment_stage(
            image_id=image_id,
            image_bgr=self._ensure_image_loaded(),
            text_regions=text_payload.get("line_numbers", []),
            edges=overlay_edges,
            max_distance_px=self.cfg.line_text_attachment_max_distance_px,
            text_class="line_number",
        )
        instrument_tag_attachment_result = run_pipe_text_attachment_stage(
            image_id=image_id,
            image_bgr=self._ensure_image_loaded(),
            text_regions=instrument_tag_payload.get("instrument_tags", []),
            edges=overlay_edges,
            max_distance_px=self.cfg.line_text_attachment_max_distance_px,
            text_class="instrument_semantic",
        )

        equipment_nodes = []
        for attachment in attachment_result["attachments_payload"].get("accepted", []):
            bbox = attachment.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            equipment_nodes.append(
                {
                    "id": f"equipment::{attachment['det_id']}",
                    "type": attachment.get("class_name", "equipment"),
                    "position": {
                        "x": float((bbox[0] + bbox[2]) / 2),
                        "y": float((bbox[1] + bbox[3]) / 2),
                    },
                    "bbox": bbox,
                }
            )
        equipment_tag_attachment_result = run_node_text_attachment_stage(
            image_id=image_id,
            image_bgr=self._ensure_image_loaded(),
            text_regions=equipment_tag_payload.get("equipment_tags", []),
            nodes=equipment_nodes,
            max_distance_px=self.cfg.equipment_tag_attachment_max_distance_px,
            text_class="equipment_tag",
        )

        combined_text_overlay = render_text_attachment_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=overlay_edges,
            attachments=
                text_attachment_result["attachments_payload"].get("accepted", [])
                + instrument_tag_attachment_result["attachments_payload"].get("accepted", []),
        )
        connection_overlay = render_connection_attachment_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=overlay_edges,
            attachments=connection_attachment_result["attachments_payload"].get("accepted", []),
            edge_connections=edge_connectivity_result["connections"],
        )
        junction_decision_overlay = render_junction_decision_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=directed_edges,
            edge_connections=edge_connectivity_result["connections"],
            rejected_junction_connections=edge_connectivity_result.get("rejected_junction_connections", []),
        )
        candidate_link_overlay = render_candidate_link_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=directed_edges,
            candidate_links=edge_connectivity_result.get("candidate_link_graph", {}).get("links", []),
        )

        # ── Build graph_result directly from the collapsed geo_graph ─────────────
        # Using geo_graph_result["graph_payload"] directly preserves the pass_through
        # collapse (merged edges, no degenerate PT nodes). Calling run_pipe_graph_stage
        # would rebuild from node_clusters/edges and lose that collapse.
        graph_result = {
            "graph_payload": geo_graph_result["graph_payload"],
            "summary": {
                "image_id": image_id,
                "pass_type": "sheet",
                "node_count": len(geo_graph_result["graph_payload"]["nodes"]),
                "edge_count": len(geo_graph_result["graph_payload"]["edges"]),
                "merged_pass_through_count": len([e for e in geo_graph_result["graph_payload"].get("edges", []) if e.get("merged_from_pass_through")]),
                "geometric_bypass": {
                    "segment_count": len(segments),
                    "run_count": len(runs),
                    "junction_count": len([n for n in geo_graph_result["graph_payload"]["nodes"] if n.get("type") == "junction"]),
                    "terminal_count": len([n for n in geo_graph_result["graph_payload"]["nodes"] if n.get("type") == "terminal"]),
                },
                "edge_direction": direction_result["summary"],
                "source_artifacts": [
                    "stage5_geometric_segments.json",
                    "phase3_runs.json",
                    "phase3_junctions.json",
                    "phase3_pipe_edges.json",
                    "phase3_edge_direction.json",
                ],
            },
        }

        self._save_json("stage12_equipment_attachments", attachment_result["attachments_payload"])
        self._save_json("stage12_equipment_attachment_summary", attachment_result["summary"])
        self._save_json("stage12_connection_attachments", connection_attachment_result["attachments_payload"])
        self._save_json("stage12_connection_attachment_summary", connection_attachment_result["summary"])
        self._save_img("stage12_connection_attachment_overlay", connection_overlay)
        self._save_img("stage12_junction_decision_overlay", junction_decision_overlay)
        self._save_json("stage12_edge_terminals", {"edge_terminals": edge_terminal_result["edge_terminals"]})
        self._save_json("stage12_edge_terminal_summary", edge_terminal_result["summary"])
        self._save_json("stage12_arrow_assignments", {"arrow_assignments": direction_result["arrow_assignments"]})
        self._save_json("stage12_edge_connections", {"edge_connections": edge_connectivity_result["connections"]})
        self._save_json("stage12_edge_connection_summary", edge_connectivity_result["summary"])
        self._save_json("stage12_candidate_links", edge_connectivity_result.get("candidate_link_graph", {}))
        self._save_json(
            "stage12_candidate_link_summary",
            edge_connectivity_result.get("candidate_link_graph", {}).get("summary", {}),
        )
        self._save_json("stage12_selected_candidate_links", edge_connectivity_result.get("selected_candidate_links", {}))
        self._save_json("stage12_candidate_link_diff", edge_connectivity_result.get("candidate_link_diff", {}))
        self._save_json(
            "stage12_candidate_link_selection_summary",
            edge_connectivity_result.get("candidate_link_selection_summary", {}),
        )
        self._save_img("stage12_candidate_link_overlay", candidate_link_overlay)
        self._save_json(
            "stage12_rejected_junction_connections",
            {"rejected_junction_connections": edge_connectivity_result.get("rejected_junction_connections", [])},
        )
        self._save_json(
            "stage12_rejected_junction_connection_summary",
            {
                "image_id": image_id,
                "pass_type": "sheet",
                "rejected_junction_alignment_connection_count": edge_connectivity_result["summary"].get(
                    "rejected_junction_alignment_connection_count", 0
                ),
                "rejected_junction_alignment_reason_counts": edge_connectivity_result["summary"].get(
                    "rejected_junction_alignment_reason_counts", {}
                ),
                "invalid_shared_junction_fallback_candidate_count": edge_connectivity_result["summary"].get(
                    "invalid_shared_junction_fallback_candidate_count", 0
                ),
                "accepted_junction_straight_through_count": edge_connectivity_result["summary"].get(
                    "accepted_junction_straight_through_count", 0
                ),
            },
        )
        self._save_json("stage12_text_attachments", text_attachment_result["attachments_payload"])
        self._save_json("stage12_text_attachment_summary", text_attachment_result["summary"])
        self._save_img("stage12_text_attachment_overlay", combined_text_overlay)
        self._save_json("stage12_overlay_edges_filtered", overlay_edge_filter_result["filtered_edges_payload"])
        self._save_json("stage12_overlay_edges_filtered_summary", overlay_edge_filter_result["summary"])
        self._save_json("stage12_instrument_tag_attachments", instrument_tag_attachment_result["attachments_payload"])
        self._save_json("stage12_instrument_tag_attachment_summary", instrument_tag_attachment_result["summary"])
        self._save_json("stage12_equipment_tag_attachments", equipment_tag_attachment_result["attachments_payload"])
        self._save_json("stage12_equipment_tag_attachment_summary", equipment_tag_attachment_result["summary"])
        self._save_img("stage12_equipment_tag_attachment_overlay", equipment_tag_attachment_result["overlay_image"])
        self._save_json("stage12_graph", graph_result["graph_payload"])
        self._save_json("stage12_graph_summary", graph_result["summary"])
        self._save_json("stage12_connection_validation", connection_validation)
        self._save_json("stage12_connection_validation_summary", connection_validation.get("gap_connection_summary", {}))

        # S5-02: Detect near-boundary terminals (after graph is built)
        from garnet.geometric_graph_builder import detect_boundary_terminals

        boundary_terminals = detect_boundary_terminals(
            edges=directed_edges,
            nodes=graph_result["graph_payload"].get("nodes", []),
            image_shape=self._ensure_image_loaded().shape,
            boundary_margin_px=50.0,
        )
        self._save_json("phase3_boundary_terminals", {
            "image_id": image_id,
            "boundary_terminals": boundary_terminals,
        })

    # ---------- Stage 12 ----------
    def stage12_edge_topology(self) -> None:
        """Classify terminals, attach objects, and bridge connectivity across connection objects."""
        object_payload = self._load_json_artifact("stage4_objects")
        text_payload = self._load_json_artifact("stage4_line_numbers")
        instrument_tag_payload = self._load_json_artifact("stage4_instrument_tags")
        equipment_tag_payload = self._load_json_artifact_or_default("stage4_equipment_tags", {"equipment_tags": []})
        node_clusters_payload = self._load_json_artifact("stage9_node_clusters")
        edges_payload = self._load_json_artifact("stage10d_split_edges")
        split_nodes_payload = self._load_json_artifact("stage10d_split_nodes")
        polyline_simplification_summary = self._load_json_artifact("stage10b_polyline_simplification_summary")
        edge_direction_summary = self._load_json_artifact("stage10c_edge_direction_summary")
        edge_split_summary = self._load_json_artifact("stage10d_split_summary")
        crossing_payload = self._load_json_artifact("stage10_crossing_resolution")
        junctions_payload = self._load_json_artifact("stage11_junctions")
        overlay_edge_filter_result = _filter_border_like_edges(
            edges_payload.get("edges", []),
            self._ensure_image_loaded().shape,
        )
        overlay_edges = overlay_edge_filter_result["kept_edges"]
        edge_terminal_result = classify_pipe_edge_terminals(
            edges=edges_payload.get("edges", []),
            node_clusters=node_clusters_payload.get("clusters", []),
            object_regions=object_payload.get("objects", []),
            equipment_terminal_classes=self.cfg.terminal_equipment_classes,
            connection_terminal_classes=self.cfg.terminal_connection_classes,
            inline_passthrough_classes=self.cfg.terminal_inline_passthrough_classes,
            match_distance_px=self.cfg.terminal_match_distance_px,
        )
        edge_terminal_map = {
            str(item.get("edge_id", "")): item
            for item in edge_terminal_result["edge_terminals"]
            if item.get("edge_id") is not None
        }
        attachment_result = run_pipe_equipment_attachment_stage(
            image_id=Path(self.image_path).name,
            objects=object_payload.get("objects", []),
            edges=edges_payload.get("edges", []),
            attachment_classes=self.cfg.equipment_attachment_classes,
            max_distance_px=self.cfg.equipment_attachment_max_distance_px,
            k_candidate_edges=self.cfg.equipment_attachment_k_candidate_edges,
        )
        connection_attachment_result = run_pipe_equipment_attachment_stage(
            image_id=Path(self.image_path).name,
            objects=object_payload.get("objects", []),
            edges=edges_payload.get("edges", []),
            attachment_classes=self.cfg.connection_attachment_classes,
            max_distance_px=self.cfg.connection_attachment_max_distance_px,
            k_candidate_edges=self.cfg.connection_attachment_k_candidate_edges,
        )
        # Phase 2: Load Stage 10 continuity data
        continuity_payload = self._load_json_artifact_or_default(
            "stage10_continuity_result",
            {"orphan_edges": 0, "gap_candidate_edges": 0, "validated_edges": 0, "provisional_edges": 0},
        )
        gap_summary_payload = self._load_json_artifact_or_default(
            "stage10_gap_summary",
            {"gaps": []},
        )

        # Phase 2: Merge Stage 10 continuity metadata into edges before graph assembly
        from garnet.continuity_aware_connections import (
            merge_continuity_into_graph,
            validate_connections_against_gaps,
        )
        edges_raw = edges_payload.get("edges", [])
        enriched_edges, enriched_nodes = merge_continuity_into_graph(
            edges=edges_raw,
            nodes=[],  # nodes built later from node_clusters
            continuity_result=continuity_payload,
            gap_summary=gap_summary_payload.get("gaps", []),
        )

        # Phase 2: Pass gap_summary to edge connectivity
        edge_connectivity_result = build_pipe_edge_connectivity(
            edges=enriched_edges,
            node_clusters=node_clusters_payload.get("clusters", []),
            object_regions=object_payload.get("objects", []),
            inline_connector_classes=self.cfg.graph_inline_connector_classes,
            inline_match_distance_px=self.cfg.graph_inline_connector_match_distance_px,
            connection_seed_edge_ids={
                str(item.get("edge_id", ""))
                for item in connection_attachment_result["attachments_payload"].get("accepted", [])
                if item.get("edge_id") is not None
            },
        )

        connection_bridges = []
        connection_bridge_distance_px = 120.0
        connection_classes = {"connection", "page connection", "utility connection"}
        for obj in object_payload.get("objects", []):
            class_name = str(obj.get("class_name", "")).strip().lower()
            if class_name not in connection_classes:
                continue
            bbox = obj.get("bbox", {})
            if not bbox:
                continue
            conn_center = (
                (float(bbox["x_min"]) + float(bbox["x_max"])) / 2.0,
                (float(bbox["y_min"]) + float(bbox["y_max"])) / 2.0,
            )
            nearby_endpoints = []
            for edge in edges_payload.get("edges", []):
                edge_id = str(edge.get("id", ""))
                polyline = edge.get("polyline", [])
                if len(polyline) < 2:
                    continue
                for endpoint_name, point in (("start", polyline[0]), ("end", polyline[-1])):
                    dist = (
                        (float(point["col"]) - conn_center[0]) ** 2
                        + (float(point["row"]) - conn_center[1]) ** 2
                    ) ** 0.5
                    if dist <= connection_bridge_distance_px:
                        nearby_endpoints.append((edge_id, endpoint_name, dist))

            if len(nearby_endpoints) < 2:
                continue
            nearby_endpoints.sort(key=lambda item: item[2])
            endpoint_a, endpoint_b = nearby_endpoints[0], nearby_endpoints[1]
            if endpoint_a[0] == endpoint_b[0]:
                continue
            connection_bridges.append(
                {
                    "kind": "connection_object_bridge",
                    "connection_class": class_name,
                    "connection_id": str(obj.get("id", "")),
                    "source_edge_id": endpoint_a[0],
                    "source_endpoint": endpoint_a[1],
                    "target_edge_id": endpoint_b[0],
                    "target_endpoint": endpoint_b[1],
                    "gap_px": round((endpoint_a[2] + endpoint_b[2]) / 2, 2),
                }
            )

        edge_connectivity_result["connections"].extend(connection_bridges)
        edge_connectivity_result["summary"]["connection_object_bridge_count"] = len(connection_bridges)
        edge_connectivity_result["summary"]["edge_connection_count"] = len(edge_connectivity_result["connections"])

        overlay_edges = [
            {
                **edge,
                "edge_terminals": edge_terminal_map.get(str(edge.get("id", ""))),
            }
            for edge in enriched_edges  # Phase 2: use continuity-enriched edges
        ]
        connection_overlay = render_connection_attachment_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=overlay_edges,
            attachments=connection_attachment_result["attachments_payload"].get("accepted", []),
            edge_connections=edge_connectivity_result["connections"],
        )
        filtered_edges_payload = {
            **overlay_edge_filter_result["filtered_edges_payload"],
            "edges": overlay_edges,
        }

        self._save_json("stage12_filtered_edges", filtered_edges_payload)
        self._save_json("stage12_filtered_edges_summary", overlay_edge_filter_result["summary"])
        self._save_json("stage12_edge_terminals", {"edge_terminals": edge_terminal_result["edge_terminals"]})
        self._save_json("stage12_edge_terminal_summary", edge_terminal_result["summary"])
        self._save_json("stage12_equipment_attachments", attachment_result["attachments_payload"])
        self._save_json("stage12_equipment_attachment_summary", attachment_result["summary"])
        self._save_json("stage12_connection_attachments", connection_attachment_result["attachments_payload"])
        self._save_json("stage12_connection_attachment_summary", connection_attachment_result["summary"])
        self._save_img("stage12_connection_attachment_overlay", connection_overlay)
        self._save_json("stage12_connection_bridges", {"bridges": connection_bridges})
        self._save_json("stage12_edge_connections", {"edge_connections": edge_connectivity_result["connections"]})
        self._save_json("stage12_edge_connection_summary", edge_connectivity_result["summary"])

    # ---------- Stage 13 ----------
    def stage13_text_attachment(self) -> None:
        """Attach line numbers and instrument tags to pipe edges."""
        text_payload = self._load_json_artifact("stage4_line_numbers")
        instrument_tag_payload = self._load_json_artifact("stage4_instrument_tags")
        filtered_edges_payload = self._load_json_artifact("stage12_filtered_edges")
        overlay_edges = filtered_edges_payload.get("edges", [])

        text_attachment_result = run_pipe_text_attachment_stage(
            image_id=Path(self.image_path).name,
            image_bgr=self._ensure_image_loaded(),
            text_regions=text_payload.get("line_numbers", []),
            edges=overlay_edges,
            max_distance_px=self.cfg.line_text_attachment_max_distance_px,
            text_class="line_number",
        )
        instrument_tag_attachment_result = run_pipe_text_attachment_stage(
            image_id=Path(self.image_path).name,
            image_bgr=self._ensure_image_loaded(),
            text_regions=instrument_tag_payload.get("instrument_tags", []),
            edges=overlay_edges,
            max_distance_px=self.cfg.line_text_attachment_max_distance_px,
            text_class="instrument_semantic",
        )
        equipment_nodes = []
        for attachment in attachment_result["attachments_payload"].get("accepted", []):
            bbox = attachment.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            equipment_nodes.append(
                {
                    "id": f"equipment::{attachment['det_id']}",
                    "type": attachment.get("class_name", "equipment"),
                    "position": {
                        "x": float((bbox[0] + bbox[2]) / 2),
                        "y": float((bbox[1] + bbox[3]) / 2),
                    },
                    "bbox": bbox,
                }
            )
        equipment_tag_attachment_result = run_node_text_attachment_stage(
            image_id=Path(self.image_path).name,
            image_bgr=self._ensure_image_loaded(),
            text_regions=equipment_tag_payload.get("equipment_tags", []),
            nodes=equipment_nodes,
            max_distance_px=self.cfg.equipment_tag_attachment_max_distance_px,
            text_class="equipment_tag",
        )
        combined_text_overlay = render_text_attachment_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=overlay_edges,
            attachments=
                text_attachment_result["attachments_payload"].get("accepted", [])
                + instrument_tag_attachment_result["attachments_payload"].get("accepted", []),
        )
        connection_overlay = render_connection_attachment_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=overlay_edges,
            attachments=connection_attachment_result["attachments_payload"].get("accepted", []),
            edge_connections=edge_connectivity_result["connections"],
        )
        junction_decision_overlay = render_junction_decision_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=enriched_edges,
            edge_connections=edge_connectivity_result["connections"],
            rejected_junction_connections=edge_connectivity_result.get("rejected_junction_connections", []),
        )
        candidate_link_overlay = render_candidate_link_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=enriched_edges,
            candidate_links=edge_connectivity_result.get("candidate_link_graph", {}).get("links", []),
        )

        graph_result = run_pipe_graph_stage(
            image_id=Path(self.image_path).name,
            node_clusters=node_clusters_payload.get("clusters", []),
            edges=enriched_edges,  # Phase 2: continuity-enriched edges
            confirmed_junctions=junctions_payload.get("confirmed_junctions", []),
            unresolved_junctions=junctions_payload.get("unresolved_junctions", []),
            split_nodes=split_nodes_payload.get("nodes", []),
            crossing_candidates=crossing_payload.get("candidates", []),
            equipment_attachments=attachment_result["attachments_payload"].get("accepted", []),
            connection_attachments=connection_attachment_result["attachments_payload"].get("accepted", []),
            text_attachments=text_attachment_result["attachments_payload"].get("accepted", []),
            instrument_tag_attachments=instrument_tag_attachment_result["attachments_payload"].get("accepted", []),
            equipment_tag_attachments=equipment_tag_attachment_result["attachments_payload"].get("accepted", []),
            edge_terminals=edge_terminal_result["edge_terminals"],
            edge_connections=edge_connectivity_result["connections"],
        )
        graph_result["summary"]["polyline_simplification"] = polyline_simplification_summary
        graph_result["summary"]["edge_direction"] = edge_direction_summary
        graph_result["summary"]["edge_split"] = edge_split_summary
        graph_result["summary"]["source_artifacts"] = [
            artifact
            for artifact in graph_result["summary"].get("source_artifacts", [])
            if artifact != "stage10_pipe_edges.json"
        ] + [
            "stage10b_pipe_edges_simplified.json",
            "stage10b_polyline_simplification_summary.json",
            "stage10c_edge_direction.json",
            "stage10c_edge_direction_summary.json",
            "stage10d_split_edges.json",
            "stage10d_split_nodes.json",
            "stage10d_split_summary.json",
        ]
        self._save_json("stage12_equipment_attachments", attachment_result["attachments_payload"])
        self._save_json("stage12_equipment_attachment_summary", attachment_result["summary"])
        self._save_json("stage12_connection_attachments", connection_attachment_result["attachments_payload"])
        self._save_json("stage12_connection_attachment_summary", connection_attachment_result["summary"])
        self._save_img("stage12_connection_attachment_overlay", connection_overlay)
        self._save_img("stage12_junction_decision_overlay", junction_decision_overlay)
        self._save_json("stage12_edge_terminals", {"edge_terminals": edge_terminal_result["edge_terminals"]})
        self._save_json("stage12_edge_terminal_summary", edge_terminal_result["summary"])
        self._save_json("stage12_arrow_assignments", self._load_json_artifact("stage10c_arrow_assignments"))
        self._save_json("stage12_edge_connections", {"edge_connections": edge_connectivity_result["connections"]})
        self._save_json("stage12_edge_connection_summary", edge_connectivity_result["summary"])
        self._save_json("stage12_candidate_links", edge_connectivity_result.get("candidate_link_graph", {}))
        self._save_json(
            "stage12_candidate_link_summary",
            edge_connectivity_result.get("candidate_link_graph", {}).get("summary", {}),
        )
        self._save_json(
            "stage12_selected_candidate_links",
            edge_connectivity_result.get("selected_candidate_links", {}),
        )
        self._save_json("stage12_candidate_link_diff", edge_connectivity_result.get("candidate_link_diff", {}))
        self._save_json(
            "stage12_candidate_link_selection_summary",
            edge_connectivity_result.get("candidate_link_selection_summary", {}),
        )
        self._save_img("stage12_candidate_link_overlay", candidate_link_overlay)
        self._save_json(
            "stage12_rejected_junction_connections",
            {"rejected_junction_connections": edge_connectivity_result.get("rejected_junction_connections", [])},
        )
        self._save_json(
            "stage12_rejected_junction_connection_summary",
            {
                "image_id": Path(self.image_path).name,
                "pass_type": "sheet",
                "rejected_junction_alignment_connection_count": edge_connectivity_result["summary"].get(
                    "rejected_junction_alignment_connection_count", 0
                ),
                "rejected_junction_alignment_reason_counts": edge_connectivity_result["summary"].get(
                    "rejected_junction_alignment_reason_counts", {}
                ),
                "invalid_shared_junction_fallback_candidate_count": edge_connectivity_result["summary"].get(
                    "invalid_shared_junction_fallback_candidate_count", 0
                ),
                "accepted_junction_straight_through_count": edge_connectivity_result["summary"].get(
                    "accepted_junction_straight_through_count", 0
                ),
            },
        )
        self._save_json("stage12_text_attachments", text_attachment_result["attachments_payload"])
        self._save_json("stage12_text_attachment_summary", text_attachment_result["summary"])
        self._save_img("stage12_text_attachment_overlay", combined_text_overlay)
        self._save_json("stage12_overlay_edges_filtered", overlay_edge_filter_result["filtered_edges_payload"])
        self._save_json("stage12_overlay_edges_filtered_summary", overlay_edge_filter_result["summary"])
        self._save_json("stage12_instrument_tag_attachments", instrument_tag_attachment_result["attachments_payload"])
        self._save_json("stage12_instrument_tag_attachment_summary", instrument_tag_attachment_result["summary"])
        self._save_json("stage12_equipment_tag_attachments", equipment_tag_attachment_result["attachments_payload"])
        self._save_json("stage12_equipment_tag_attachment_summary", equipment_tag_attachment_result["summary"])
        self._save_img("stage12_equipment_tag_attachment_overlay", equipment_tag_attachment_result["overlay_image"])
        self._save_json("stage12_graph", graph_result["graph_payload"])
        self._save_json("stage12_graph_summary", graph_result["summary"])
        # Phase 2: save continuity connection validation results
        self._save_json("stage12_connection_validation", connection_validation)
        self._save_json(
            "stage12_connection_validation_summary",
            connection_validation.get("gap_connection_summary", {}),
        )

    def stage12c_page_connector_labeling(self) -> None:
        from garnet.page_connector import find_nearby_text

        connection_payload = self._load_json_artifact_or_default("stage12_connection_attachments", {"accepted": []})
        equipment_payload = self._load_json_artifact_or_default("stage12_equipment_attachments", {"accepted": []})
        accepted = [
            a
            for a in connection_payload.get("accepted", []) + equipment_payload.get("accepted", [])
            if a.get("class_name") == "page connection"
        ]
        ocr_payload = self._load_json_artifact("stage2_ocr_regions")
        text_regions = ocr_payload.get("text_regions", [])
        all_labels = []
        for att in accepted:
            bbox = att.get("bbox", {})
            labels = find_nearby_text(bbox, text_regions, max_distance_px=80.0)
            all_labels.append({"object_id": att.get("object_id") or att.get("det_id"), "labels": labels})
        self._save_json("stage12_page_connector_labels", {"connectors": all_labels})
        self._save_json(
            "stage12_page_connector_labels_summary",
            {
                "total_connectors": len(accepted),
                "total_labels": sum(len(l["labels"]) for l in all_labels),
            },
        )

    def stage12b_graph_export(self) -> None:
        graph_payload = self._load_json_artifact("stage12_graph")
        object_payload = self._load_json_artifact("stage4_objects")
        line_number_payload = self._load_json_artifact("stage4_line_numbers")
        instrument_tag_payload = self._load_json_artifact("stage4_instrument_tags")
        page_connector_labels_payload = self._load_json_artifact_or_default(
            "stage12_page_connector_labels",
            {"connectors": []},
        )
        connection_attachments_payload = self._load_json_artifact_or_default(
            "stage12_connection_attachments",
            {"accepted": []},
        )
        normalization_summary = self._load_json_artifact("stage1_normalization_summary")
        graph_v1_payload = build_graph_v1_payload(
            stage12_graph=graph_payload,
            objects_payload=object_payload,
            line_numbers_payload=line_number_payload,
            instrument_tags_payload=instrument_tag_payload,
            page_connector_labels_payload=page_connector_labels_payload,
            connection_attachments_payload=connection_attachments_payload,
            image_dimensions=normalization_summary.get("dimensions", {}),
        )
        self._save_json("stage12b_graph_v1", graph_v1_payload)

    # ---------- Stage 13 + 14 ----------
    def stage13_graph_qa(self) -> None:
        graph_payload = self._load_json_artifact("stage12_graph")
        qa_result = run_pipe_graph_qa_stage(
            image_id=Path(self.image_path).name,
            graph_payload=graph_payload,
            image_bgr=self._ensure_image_loaded(),
        )
        self._save_json("stage15_graph_anomalies", qa_result["anomaly_report"])
        self._save_img("stage15_graph_components_overlay", qa_result["component_overlay_image"])
        self._save_json("stage15_review_queue", qa_result["review_queue"])
        self._save_json("stage15_graph_qa_summary", qa_result["summary"])

    def stage14_continuity_check(self) -> None:
        """Run pipe continuity rules (Rules 1-10) against the assembled graph."""
        graph_payload = self._load_json_artifact("stage12_graph")
        equip_payload = self._load_json_artifact("stage12_equipment_attachments")
        conn_payload = self._load_json_artifact("stage12_connection_attachments")
        continuity_result = run_continuity_checker_stage(
            image_id=Path(self.image_path).name,
            graph_payload=graph_payload,
            equipment_attachments_payload=equip_payload,
            connection_attachments_payload=conn_payload,
            image_bgr=self._ensure_image_loaded(),
        )
        self._save_json("stage14_continuity_result", continuity_result["continuity_result"])
        self._save_json("stage14_violations", continuity_result["violations"])
        self._save_img("stage14_continuity_violations_overlay", continuity_result["overlay_image"])
        self._save_json("stage14_continuity_summary", continuity_result["summary"])

    # ---------- Stage 15 ----------
    def stage15_recovery_loop(self) -> None:
        from garnet.recovery_loop import run_recovery_stage

        decisions = run_recovery_stage(str(self.out_dir), max_iterations=3)
        self._save_json("stage5_recovery_decisions", decisions)

    # ---------- Stage 16 ----------
    def stage16_connection_overlay(self) -> None:
        """
        Stage 16: render connection + pipe-segment overlay.

        Uses render_overlay() from render_connection_pipeline_overlay.py to draw:
        - Red pipe segments connected to accepted page-connection anchors
        - Orange inline element connectors
        - Blue page-connection marker boxes + anchor dots + labels

        Runs after Stage 12 (needs connection_attachments + edge_connections)
        and uses stage4_objects as the background reference.
        """
        out = self.out_dir
        overlay_path = out / "stage16_connection_pipeline_overlay.png"

        render_overlay(
            connection_attachments_path=str(out / "stage12_connection_attachments.json"),
            edge_connections_path=str(out / "stage12_edge_connections.json"),
            edge_terminals_path=str(out / "stage12_edge_terminals.json"),
            graph_path=str(out / "stage12_graph.json"),
            objects_path=str(out / "stage4_objects.json"),
            output_path=str(overlay_path),
            image_base_path=str(self.image_path),
        )


def main() -> None:
    parser = argparse.ArgumentParser("P&ID pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--ocr-route", choices=["easyocr", "gemini", "paddleocr", "ocrmac"], default="ocrmac")
    parser.add_argument("--stop-after", type=int, default=2, help="Run up to this stage (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, or 15)")
    args = parser.parse_args()
    pipe = PIDPipeline(
        args.image,
        output_dir=args.out,
        cfg=PipelineConfig(
            ocr_route=args.ocr_route,
            use_geometric_line_detection=args.geometric,
        ),
    )
    pipe.run(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
