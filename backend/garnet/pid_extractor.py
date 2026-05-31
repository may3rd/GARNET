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
import math
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
from garnet.trace_graph_builder import build_trace_graph_from_stage11, render_stage12_graph_overlay
from garnet.trace_graph_qa import run_stage12_trace_graph_qa
from garnet.stage13_review_package import build_stage13_review_package, render_stage13_review_overlay
from garnet.stage14_review_decisions import apply_stage14_review_decisions
from garnet.stage15_process_exports import build_stage15_process_exports, render_stage15_inline_mto_overlay

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
LINE_NUMBER_REVIEW_ASSUMPTION = "accepted_line_numbers_are_human_reviewed"


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


def _mark_line_number_review_state(association: dict[str, Any], *, accepted: bool) -> dict[str, Any]:
    result = dict(association)
    if accepted:
        result.update(
            {
                "review_state": "accepted",
                "review_source": "human_assumed",
                "review_required": False,
            }
        )
    else:
        result.update(
            {
                "review_state": "needs_review",
                "review_source": "system",
                "review_required": True,
            }
        )
    return result


def build_stage11_line_number_review_payload(
    *,
    image_id: str,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    traces_without_line_number: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "image_id": image_id,
        "review_assumption": LINE_NUMBER_REVIEW_ASSUMPTION,
        "accepted": accepted,
        "needs_review": rejected,
        "traces_without_line_number": traces_without_line_number,
    }
    summary = {
        "image_id": image_id,
        "accepted_count": len(accepted),
        "needs_review_count": len(rejected),
        "trace_without_line_number_count": len(traces_without_line_number),
        "simulated_assignment_count": len([item for item in accepted if item.get("source") == "simulated_hitl"]),
        "review_assumption": LINE_NUMBER_REVIEW_ASSUMPTION,
    }
    return payload, summary


def _stable_choice_index(key: str, count: int) -> int:
    if count <= 0:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(key)) % count


def simulate_line_number_hitl_for_missing_traces(
    edges: list[dict[str, Any]],
    reviewed_line_numbers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Temporary deterministic stand-in for human line-number correction."""
    reviewed_pool = [
        item
        for item in reviewed_line_numbers
        if str(item.get("review_state") or "") == "accepted" and str(item.get("id") or item.get("source_object_id") or "")
    ]
    if not reviewed_pool:
        return []

    assignments: list[dict[str, Any]] = []
    for edge in edges:
        attachments = edge.setdefault("attachments", {})
        line_numbers = attachments.setdefault("line_numbers", [])
        if line_numbers:
            continue
        trace_id = str(edge.get("trace_id") or "")
        template = reviewed_pool[_stable_choice_index(trace_id, len(reviewed_pool))]
        line_id = str(template.get("id") or template.get("source_object_id") or "")
        assignment = {
            "id": line_id,
            "source_object_id": template.get("source_object_id", line_id),
            "class_name": template.get("class_name", ""),
            "bbox": template.get("bbox"),
            "text": template.get("text", ""),
            "normalized_text": template.get("normalized_text", template.get("text", "")),
            "confidence": template.get("confidence"),
            "trace_id": trace_id,
            "trace_kind": edge.get("trace_kind"),
            "source": "simulated_hitl",
            "review_state": "accepted",
            "review_source": "human_simulated",
            "review_required": False,
            "simulated_from_line_number_id": line_id,
        }
        line_numbers.append(assignment)
        assignments.append(assignment)
    return assignments


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
    pipe_mask_inline_object_inset: int = 12
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
    trace_association_equipment_port_max_distance_px: float = 16.0
    trace_association_inline_object_max_distance_px: float = 24.0
    trace_association_text_max_distance_px: float = 100.0
    trace_association_instrument_max_distance_px: float = 90.0
    trace_association_arrow_max_distance_px: float = 45.0
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
        self.out_dir = Path(output_dir)
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
        stages = [
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
        ]
        if self.cfg.use_geometric_line_detection:
            # Stage 5b is the trace source for the geometric route, so skip
            # skeleton stages 6-10 and attach semantics directly in Stage 11.
            return [
                (1, "stage1_input_normalization", self.stage1_input_normalization),
                (2, "stage2_ocr_discovery", self.stage2_ocr_discovery),
                (4, "stage4_object_detection", self.stage4_object_detection),
                (4, "stage4_line_number_fusion", self.stage4_line_number_fusion),
                (4, "stage4_instrument_tag_fusion", self.stage4_instrument_tag_fusion),
                (5, "stage5_pipe_mask", self.stage5_pipe_mask),
                (5, "stage5b_pipe_trace", self.stage5b_pipe_trace),
                (11, "stage11_trace_associations", self.stage11_trace_associations),
                (12, "stage12_geometric_graph_assembly", self.stage12_geometric_graph_assembly),
                (12, "stage12c_page_connector_labeling", self.stage12c_page_connector_labeling),
                (12, "stage12b_graph_export", self.stage12b_graph_export),
                (13, "stage13_graph_qa", self.stage13_graph_qa),
                (14, "stage14_apply_review_decisions", self.stage14_apply_review_decisions),
                (15, "stage15_process_exports", self.stage15_process_exports),
                (16, "stage16_connection_overlay", self.stage16_connection_overlay),
            ]
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
        # Skip if OCR regions already exist
        ocr_regions_path = self.out_dir / "stage2_ocr_regions.json"
        if ocr_regions_path.exists():
            logger.info(f"Skipping Stage 2 OCR discovery: {ocr_regions_path} already exists")
            # Still need to load and verify the Stage 1 input exists for consistency
            stage1_input = self.out_dir / "stage1_gray.png"
            if not stage1_input.exists():
                raise FileNotFoundError(f"Stage 2 requires Stage 1 artifact: {stage1_input}")
            return

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
            inline_object_inset=self.cfg.pipe_mask_inline_object_inset,
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

    def _snap_port_to_pipe_centerline(
        self,
        pipe_mask: np.ndarray,
        port: tuple[int, int, str],
        search_radius: int = 12,
    ) -> tuple[int, int, str]:
        """Move a bbox-edge port onto the center of the local pipe thickness."""
        x, y, direction = port
        direction = direction.upper()
        h, w = pipe_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return (int(x), int(y), direction)

        def _closest_run(values: list[int], current: int) -> Optional[tuple[int, int]]:
            if not values:
                return None
            values = sorted(set(values))
            runs: list[tuple[int, int]] = []
            start = prev = values[0]
            for value in values[1:]:
                if value == prev + 1:
                    prev = value
                    continue
                runs.append((start, prev))
                start = prev = value
            runs.append((start, prev))
            return min(
                runs,
                key=lambda r: (
                    0
                    if r[0] <= current <= r[1]
                    else min(abs(current - r[0]), abs(current - r[1]))
                ),
            )

        if direction in ("UP", "DOWN"):
            cols = [
                cx for cx in range(max(0, x - search_radius), min(w, x + search_radius + 1))
                if pipe_mask[y, cx] > 0
            ]
            run = _closest_run(cols, x)
            if run:
                x = (run[0] + run[1]) // 2
        else:
            rows = [
                cy for cy in range(max(0, y - search_radius), min(h, y + search_radius + 1))
                if pipe_mask[cy, x] > 0
            ]
            run = _closest_run(rows, y)
            if run:
                y = (run[0] + run[1]) // 2
        return (int(x), int(y), direction)

    def _snap_ports_to_pipe_centerlines(
        self,
        ports: dict[str, list],
        pipe_mask: np.ndarray,
    ) -> dict[str, list[tuple[int, int, str]]]:
        snapped: dict[str, list[tuple[int, int, str]]] = {}
        for obj_id, port_list in ports.items():
            snapped[obj_id] = []
            for port in port_list:
                if len(port) != 3:
                    continue
                px, py, direction = port
                snapped[obj_id].append(
                    self._snap_port_to_pipe_centerline(
                        pipe_mask,
                        (int(px), int(py), str(direction)),
                    )
                )
        return snapped

    def _point_near_segment(
        self,
        x: int,
        y: int,
        seg: dict[str, Any],
        tolerance: int = 5,
    ) -> bool:
        x1 = int(seg["x1"])
        y1 = int(seg["y1"])
        x2 = int(seg["x2"])
        y2 = int(seg["y2"])
        if seg["direction"] in ("LEFT", "RIGHT"):
            return (
                min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
                and abs(y - y1) <= tolerance
            )
        return (
            min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
            and abs(x - x1) <= tolerance
        )

    def _point_inside_any_bbox(
        self,
        x: int,
        y: int,
        objects: list[dict[str, Any]],
        margin: int = 4,
    ) -> bool:
        for obj in objects:
            bbox = obj.get("bbox")
            if not bbox:
                continue
            if (
                bbox["x_min"] - margin <= x <= bbox["x_max"] + margin
                and bbox["y_min"] - margin <= y <= bbox["y_max"] + margin
            ):
                return True
        return False

    def _branch_already_traced(
        self,
        x: int,
        y: int,
        branch_direction: str,
        all_results: dict[str, dict],
        source_obj_id: str,
        source_segment_index: int,
        tolerance: int = 5,
    ) -> bool:
        dx, dy = {
            "UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0),
        }[branch_direction]
        probe_x = x + dx * 12
        probe_y = y + dy * 12
        for obj_id, result in all_results.items():
            for seg_index, seg in enumerate(result.get("segments", [])):
                if obj_id == source_obj_id and seg_index == source_segment_index:
                    continue
                opposite = {
                    "UP": "DOWN",
                    "DOWN": "UP",
                    "LEFT": "RIGHT",
                    "RIGHT": "LEFT",
                }
                if seg.get("direction") not in (branch_direction, opposite[branch_direction]):
                    continue
                if self._point_near_segment(probe_x, probe_y, seg, tolerance=tolerance):
                    return True
        return False

    def _has_orthogonal_branch_run(
        self,
        pipe_mask: np.ndarray,
        x: int,
        y: int,
        direction: str,
        min_run: int = 25,
        max_exact_gap: int = 3,
    ) -> bool:
        deltas = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }
        dx, dy = deltas[direction]
        h, w = pipe_mask.shape
        exact_hits = 0
        current_gap = 0
        max_gap = 0
        for step in range(1, min_run + 1):
            px = x + dx * step
            py = y + dy * step
            ok = 0 <= px < w and 0 <= py < h and pipe_mask[py, px] > 0
            if ok:
                exact_hits += 1
                current_gap = 0
            else:
                current_gap += 1
                max_gap = max(max_gap, current_gap)

        # A real branch may have small raster gaps, but it should not drift like
        # a diagonal leader line or symbol stroke.
        return exact_hits >= int(round(min_run * 0.72)) and max_gap <= max_exact_gap

    def _inline_object_on_axis(
        self,
        x: int,
        y: int,
        direction: str,
        inline_symbols: list[dict[str, Any]],
        max_distance: int = 35,
        axis_margin: int = 3,
    ) -> Optional[dict[str, Any]]:
        best: Optional[tuple[int, dict[str, Any]]] = None
        for obj in inline_symbols:
            bbox = obj.get("bbox")
            if not bbox:
                continue
            if direction == "UP":
                if not (bbox["x_min"] - axis_margin <= x <= bbox["x_max"] + axis_margin):
                    continue
                distance = y - int(bbox["y_max"])
            elif direction == "DOWN":
                if not (bbox["x_min"] - axis_margin <= x <= bbox["x_max"] + axis_margin):
                    continue
                distance = int(bbox["y_min"]) - y
            elif direction == "LEFT":
                if not (bbox["y_min"] - axis_margin <= y <= bbox["y_max"] + axis_margin):
                    continue
                distance = x - int(bbox["x_max"])
            else:
                if not (bbox["y_min"] - axis_margin <= y <= bbox["y_max"] + axis_margin):
                    continue
                distance = int(bbox["x_min"]) - x

            if distance < -axis_margin or distance > max_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, obj)
        return best[1] if best else None

    def _has_inline_bridge_branch_run(
        self,
        pipe_mask: np.ndarray,
        x: int,
        y: int,
        direction: str,
        inline_symbols: list[dict[str, Any]],
        min_run: int = 25,
    ) -> bool:
        deltas = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }
        dx, dy = deltas[direction]
        h, w = pipe_mask.shape

        def has_pipe_near(px: int, py: int, band: int = 1) -> bool:
            if direction in ("LEFT", "RIGHT"):
                return any(
                    0 <= py + off < h and 0 <= px < w and pipe_mask[py + off, px] > 0
                    for off in range(-band, band + 1)
                )
            return any(
                0 <= px + off < w and 0 <= py < h and pipe_mask[py, px + off] > 0
                for off in range(-band, band + 1)
            )

        lead_run = 0
        for step in range(1, 10):
            if has_pipe_near(x + dx * step, y + dy * step):
                lead_run += 1
            else:
                break
        if lead_run < 3:
            return False

        inline_obj = self._inline_object_on_axis(x, y, direction, inline_symbols)
        if not inline_obj:
            return False

        bbox = inline_obj["bbox"]
        if direction == "UP":
            exit_x, exit_y = x, int(bbox["y_min"]) - 1
        elif direction == "DOWN":
            exit_x, exit_y = x, int(bbox["y_max"]) + 1
        elif direction == "LEFT":
            exit_x, exit_y = int(bbox["x_min"]) - 1, y
        else:
            exit_x, exit_y = int(bbox["x_max"]) + 1, y

        resumed = False
        resume_x = exit_x
        resume_y = exit_y
        for offset in range(0, 16):
            px = exit_x + dx * offset
            py = exit_y + dy * offset
            if has_pipe_near(px, py):
                resume_x, resume_y = px, py
                resumed = True
                break
        if not resumed:
            return False

        after_hits = 0
        for step in range(0, 16):
            if has_pipe_near(resume_x + dx * step, resume_y + dy * step):
                after_hits += 1
        span = max(abs(resume_x - x), abs(resume_y - y))
        return span >= min_run and after_hits >= 5

    def _has_branch_candidate_run(
        self,
        pipe_mask: np.ndarray,
        x: int,
        y: int,
        direction: str,
        inline_symbols: list[dict[str, Any]],
        min_run: int = 25,
    ) -> bool:
        from garnet.visual_primitives.cv_pipe_tracer import _has_connected_side_pipe

        if _has_connected_side_pipe(pipe_mask, x, y, direction, min_run=min_run):
            return self._has_orthogonal_branch_run(pipe_mask, x, y, direction, min_run=min_run)
        return self._has_inline_bridge_branch_run(
            pipe_mask,
            x,
            y,
            direction,
            inline_symbols,
            min_run=min_run,
        )

    def _cluster_branch_candidates(
        self,
        raw_candidates: list[dict[str, Any]],
        radius: int = 8,
    ) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            match = None
            for cluster in clusters:
                if cluster["branch_direction"] != candidate["branch_direction"]:
                    continue
                if (
                    abs(cluster["x"] - candidate["x"]) <= radius
                    and abs(cluster["y"] - candidate["y"]) <= radius
                ):
                    match = cluster
                    break
            if match is None:
                clusters.append({**candidate, "members": [candidate]})
                continue
            match["members"].append(candidate)
            members = match["members"]
            match["x"] = int(round(sum(m["x"] for m in members) / len(members)))
            match["y"] = int(round(sum(m["y"] for m in members) / len(members)))
            if candidate["status"] == "queued":
                match["status"] = "queued"
                match["reason"] = candidate["reason"]
            if candidate.get("node_obj_id") and not match.get("node_obj_id"):
                match["node_obj_id"] = candidate["node_obj_id"]
                match["reason"] = candidate.get("reason", match["reason"])
        return clusters

    def _detect_stage5b_branch_candidates(
        self,
        pipe_mask: np.ndarray,
        all_results: dict[str, dict],
        inline_symbols: list[dict[str, Any]],
        node_symbols: Optional[list[dict[str, Any]]] = None,
        equipment_objects: Optional[list[dict[str, Any]]] = None,
        sample_step: int = 5,
        min_branch_run: int = 25,
    ) -> list[dict[str, Any]]:
        from garnet.visual_primitives.cv_pipe_tracer import (
            TURN_LEFT,
            TURN_RIGHT,
        )

        tee_points = [
            (int(result["terminal_x"]), int(result["terminal_y"]))
            for result in all_results.values()
            if result.get("terminal_type") == "tee_junction"
        ]
        turn_points = [
            (int(turn["x"]), int(turn["y"]))
            for result in all_results.values()
            for turn in result.get("turns", [])
        ]
        existing_points = tee_points + turn_points

        raw_candidates: list[dict[str, Any]] = []
        for obj_id, result in all_results.items():
            for seg_index, seg in enumerate(result.get("segments", [])):
                direction = seg["direction"]
                branch_dirs = (TURN_LEFT[direction], TURN_RIGHT[direction])
                x1 = int(seg["x1"])
                y1 = int(seg["y1"])
                x2 = int(seg["x2"])
                y2 = int(seg["y2"])
                if direction in ("LEFT", "RIGHT"):
                    length = abs(x2 - x1)
                else:
                    length = abs(y2 - y1)
                if length < min_branch_run:
                    continue
                dx, dy = {
                    "UP": (0, -1),
                    "DOWN": (0, 1),
                    "LEFT": (-1, 0),
                    "RIGHT": (1, 0),
                }[direction]

                for dist in range(min_branch_run, max(min_branch_run, length - min_branch_run) + 1, sample_step):
                    if direction in ("LEFT", "RIGHT"):
                        x = x1 + dx * dist
                        y = int(round(y1 + ((y2 - y1) * dist / length)))
                    else:
                        x = int(round(x1 + ((x2 - x1) * dist / length)))
                        y = y1 + dy * dist
                    if self._point_inside_any_bbox(x, y, inline_symbols, margin=0):
                        continue

                    for branch_direction in branch_dirs:
                        if not self._has_branch_candidate_run(
                            pipe_mask,
                            x,
                            y,
                            branch_direction,
                            inline_symbols,
                            min_run=min_branch_run,
                        ):
                            continue

                        status = "queued"
                        reason = "untraced_branch"
                        if self._point_inside_any_bbox(x, y, equipment_objects or [], margin=2):
                            status = "rejected_inside_equipment"
                            reason = "candidate_inside_equipment_bbox"
                        elif any(abs(x - tx) <= 10 and abs(y - ty) <= 10 for tx, ty in tee_points):
                            status = "done_existing_tee"
                            reason = "near_existing_tee_terminal"
                        elif any(abs(x - tx) <= 8 and abs(y - ty) <= 8 for tx, ty in turn_points):
                            status = "done_existing_turn"
                            reason = "near_existing_turn"
                        elif self._branch_already_traced(
                            x, y, branch_direction, all_results, obj_id, seg_index
                        ):
                            status = "done_already_traced"
                            reason = "branch_direction_covered_by_existing_segment"

                        raw_candidates.append({
                            "x": int(x),
                            "y": int(y),
                            "branch_direction": branch_direction,
                            "source_trace_id": obj_id,
                            "source_segment_index": seg_index,
                            "source_direction": direction,
                            "status": status,
                            "reason": reason,
                        })

        for node in node_symbols or []:
            bbox = node.get("bbox")
            if not bbox:
                continue
            x = int(round((bbox["x_min"] + bbox["x_max"]) / 2))
            y = int(round((bbox["y_min"] + bbox["y_max"]) / 2))
            if any(abs(x - px) <= 10 and abs(y - py) <= 10 for px, py in existing_points):
                continue
            if self._point_inside_any_bbox(x, y, equipment_objects or [], margin=2):
                continue

            matched_segment: Optional[tuple[str, int, dict[str, Any]]] = None
            for obj_id, result in all_results.items():
                for seg_index, seg in enumerate(result.get("segments", [])):
                    if self._point_near_segment(x, y, seg, tolerance=6):
                        matched_segment = (obj_id, seg_index, seg)
                        break
                if matched_segment:
                    break
            if not matched_segment:
                continue

            source_obj_id, source_segment_index, source_seg = matched_segment
            source_direction = str(source_seg["direction"])
            for branch_direction in (TURN_LEFT[source_direction], TURN_RIGHT[source_direction]):
                if not self._has_branch_candidate_run(
                    pipe_mask,
                    x,
                    y,
                    branch_direction,
                    inline_symbols,
                    min_run=min_branch_run,
                ):
                    continue
                raw_candidates.append({
                    "x": x,
                    "y": y,
                    "branch_direction": branch_direction,
                    "source_trace_id": source_obj_id,
                    "source_segment_index": source_segment_index,
                    "source_direction": source_direction,
                    "status": "queued",
                    "reason": "node_object_branch",
                    "node_obj_id": node.get("id", ""),
                })

        candidates = self._cluster_branch_candidates(raw_candidates)
        for candidate in candidates:
            result = all_results.get(str(candidate.get("source_trace_id", "")), {})
            segments = result.get("segments", [])
            seg_index = candidate.get("source_segment_index")
            if not isinstance(seg_index, int) or seg_index < 0 or seg_index >= len(segments):
                candidate["status"] = "rejected_not_on_source_trace"
                candidate["reason"] = "missing_source_segment"
                continue
            if not self._point_near_segment(
                int(candidate["x"]),
                int(candidate["y"]),
                segments[seg_index],
                tolerance=6,
            ):
                candidate["status"] = "rejected_not_on_source_trace"
                candidate["reason"] = "candidate_off_source_centerline"
        for index, candidate in enumerate(candidates, start=1):
            candidate["id"] = f"branch_{index:06d}"
            candidate["member_count"] = len(candidate.pop("members", []))
        return candidates

    def _draw_stage5b_branch_candidate_overlay(
        self,
        image: np.ndarray,
        candidates: list[dict[str, Any]],
        base_results: Optional[dict[str, dict]] = None,
        branch_results: Optional[dict[str, dict]] = None,
    ) -> np.ndarray:
        import cv2 as _cv2

        overlay = image.copy()
        self._draw_stage5b_result_paths(
            overlay,
            base_results or {},
            line_color=(0, 170, 0),
            terminal_color=(0, 120, 0),
            thickness=2,
            label_terminals=False,
        )
        self._draw_stage5b_result_paths(
            overlay,
            branch_results or {},
            line_color=(0, 0, 255),
            terminal_color=(0, 0, 255),
            thickness=3,
            label_terminals=True,
        )
        colors = {
            "queued": (0, 0, 255),
            "done_existing_tee": (0, 180, 0),
            "done_existing_turn": (0, 180, 0),
            "done_already_traced": (0, 180, 0),
            "done_branch_connection": (0, 180, 0),
            "done_used_by_branch": (0, 180, 0),
            "pending_branch_connection": (0, 180, 180),
        }
        deltas = {
            "UP": (0, -16), "DOWN": (0, 16),
            "LEFT": (-16, 0), "RIGHT": (16, 0),
        }
        for candidate in candidates:
            x = int(candidate["x"])
            y = int(candidate["y"])
            status = candidate.get("status", "queued")
            color = colors.get(status, (128, 128, 128))
            _cv2.circle(overlay, (x, y), 7, color, -1)
            _cv2.circle(overlay, (x, y), 7, (255, 255, 255), 1)
            dx, dy = deltas.get(candidate.get("branch_direction", ""), (0, 0))
            if dx or dy:
                _cv2.arrowedLine(
                    overlay,
                    (x, y),
                    (x + dx, y + dy),
                    color,
                    2,
                    tipLength=0.35,
                )
            label = candidate.get("id", "").replace("branch_", "b")
            _cv2.putText(
                overlay,
                label,
                (x + 9, y - 9),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        return overlay

    def _draw_stage5b_result_paths(
        self,
        overlay: np.ndarray,
        results: dict[str, dict],
        line_color: tuple[int, int, int],
        terminal_color: tuple[int, int, int],
        thickness: int,
        label_terminals: bool,
    ) -> None:
        import cv2 as _cv2

        for trace_id, data in (results or {}).items():
            if not data.get("segments"):
                continue
            if data.get("status") not in (None, "ok", "traced"):
                continue
            for seg in data.get("segments", []):
                _cv2.line(
                    overlay,
                    (int(seg["x1"]), int(seg["y1"])),
                    (int(seg["x2"]), int(seg["y2"])),
                    line_color,
                    thickness,
                )
            tx = int(data.get("terminal_x", 0))
            ty = int(data.get("terminal_y", 0))
            if tx or ty:
                _cv2.circle(overlay, (tx, ty), 4 + thickness, terminal_color, -1)
                if label_terminals:
                    _cv2.putText(
                        overlay,
                        str(trace_id).replace("branch_", "b"),
                        (tx + 8, ty + 14),
                        _cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        terminal_color,
                        1,
                    )

    def _draw_stage5b_branch_trace_overlay(
        self,
        image: np.ndarray,
        branch_results: dict[str, dict],
    ) -> np.ndarray:
        import cv2 as _cv2

        overlay = image.copy()
        for branch_id, data in (branch_results or {}).items():
            if data.get("status") != "traced":
                continue
            for seg in data.get("segments", []):
                _cv2.line(
                    overlay,
                    (int(seg["x1"]), int(seg["y1"])),
                    (int(seg["x2"]), int(seg["y2"])),
                    (0, 0, 255),
                    3,
                )
            tx = int(data.get("terminal_x", 0))
            ty = int(data.get("terminal_y", 0))
            _cv2.circle(overlay, (tx, ty), 6, (0, 0, 255), -1)
            _cv2.putText(
                overlay,
                f"{branch_id}:branch",
                (tx + 8, ty + 14),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
        return overlay

    def _find_matching_stage5b_branch_candidate(
        self,
        candidate: dict[str, Any],
        candidates: list[dict[str, Any]],
        radius: int = 10,
    ) -> Optional[dict[str, Any]]:
        for existing in candidates:
            if existing.get("branch_direction") != candidate.get("branch_direction"):
                continue
            if (
                abs(int(existing["x"]) - int(candidate["x"])) <= radius
                and abs(int(existing["y"]) - int(candidate["y"])) <= radius
            ):
                return existing
        return None

    def _stage5b_branch_result_as_trace_source(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "port": result.get("port", {}),
            "terminal_type": result.get("terminal_type"),
            "terminal_x": result.get("terminal_x"),
            "terminal_y": result.get("terminal_y"),
            "terminal_obj_id": result.get("terminal_obj_id"),
            "segments": result.get("segments", []),
            "turns": result.get("turns", []),
            "hits": result.get("hits", []),
            "trace_length_px": result.get("trace_length_px", 0),
            "status": result.get("status"),
        }

    def _stage5b_branch_start_reached_by_prior_result(
        self,
        candidate: dict[str, Any],
        branch_results: dict[str, dict],
        tolerance: int = 12,
    ) -> Optional[tuple[str, dict]]:
        node_obj_id = str(candidate.get("node_obj_id") or "")
        branch_direction = str(candidate.get("branch_direction") or "").upper()
        opposite = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }
        if branch_direction not in opposite:
            return None

        cx = int(candidate.get("x", 0))
        cy = int(candidate.get("y", 0))
        for prior_id, prior in branch_results.items():
            if prior.get("status") != "traced":
                continue
            terminal_matches = False
            if node_obj_id and str(prior.get("terminal_obj_id") or "") == node_obj_id:
                terminal_matches = True
            elif (
                prior.get("terminal_x") is not None
                and prior.get("terminal_y") is not None
                and abs(int(prior["terminal_x"]) - cx) <= tolerance
                and abs(int(prior["terminal_y"]) - cy) <= tolerance
            ):
                terminal_matches = True
            if not terminal_matches:
                continue
            segments = prior.get("segments") or []
            if not segments:
                continue
            last_dir = str(segments[-1].get("direction") or "").upper()
            if last_dir == opposite[branch_direction]:
                return prior_id, prior
        return None

    def _find_stage5b_existing_path_intersection(
        self,
        candidate: dict[str, Any],
        segments: list[dict[str, Any]],
        existing_results: dict[str, dict],
        min_distance_from_start: int = 30,
        tolerance: int = 5,
        min_straight_continuation: int = 25,
    ) -> Optional[tuple[int, int, int, int, str]]:
        """Find the first point where a branch reaches an already traced path."""
        deltas = {
            "UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0),
        }
        source_trace_id = str(candidate.get("source_trace_id", ""))
        source_segment_index = candidate.get("source_segment_index")
        distance_from_start = 0
        for seg_index, seg in enumerate(segments):
            direction = str(seg.get("direction", "")).upper()
            dx, dy = deltas.get(direction, (0, 0))
            if dx == 0 and dy == 0:
                continue
            x1 = int(seg["x1"])
            y1 = int(seg["y1"])
            seg_len = int(seg.get("length_px") or max(abs(int(seg["x2"]) - x1), abs(int(seg["y2"]) - y1)))
            for step in range(0, seg_len + 1):
                path_distance = distance_from_start + step
                if path_distance <= min_distance_from_start:
                    continue
                px = x1 + dx * step
                py = y1 + dy * step
                for existing_id, existing in existing_results.items():
                    for existing_seg_index, existing_seg in enumerate(existing.get("segments", [])):
                        if (
                            str(existing_id) == source_trace_id
                            and existing_seg_index == source_segment_index
                            and path_distance <= min_distance_from_start * 2
                        ):
                            continue
                        if self._point_near_segment(px, py, existing_seg, tolerance=tolerance):
                            remaining_on_seg = max(0, seg_len - step)
                            if remaining_on_seg >= min_straight_continuation:
                                continue
                            return seg_index, px, py, path_distance, str(existing_id)
            distance_from_start += seg_len
        return None

    def _truncate_stage5b_branch_at_intersection(
        self,
        segments: list[dict[str, Any]],
        trace_length: int,
        intersection: tuple[int, int, int, int, str],
    ) -> tuple[list[dict[str, Any]], int, int, int, str]:
        seg_index, ix, iy, path_distance, existing_id = intersection
        truncated = [dict(seg) for seg in segments[:seg_index + 1]]
        if truncated:
            last = truncated[-1]
            direction = str(last.get("direction", "")).upper()
            if direction in ("UP", "DOWN"):
                ix = int(last["x1"])
            elif direction in ("LEFT", "RIGHT"):
                iy = int(last["y1"])
            last["x2"] = ix
            last["y2"] = iy
            last["length_px"] = max(abs(ix - int(last["x1"])), abs(iy - int(last["y1"])))
        return truncated, path_distance, ix, iy, existing_id

    def _trace_stage5b_branch_candidates(
        self,
        pipe_mask: np.ndarray,
        image: np.ndarray,
        candidates: list[dict[str, Any]],
        page_connections: list[dict[str, Any]],
        instrument_tags: list[dict[str, Any]],
        equipment: list[dict[str, Any]],
        inline_symbols: list[dict[str, Any]],
        node_symbols: list[dict[str, Any]],
        visited: np.ndarray,
        terminal_candidates: Optional[list[dict[str, Any]]] = None,
        existing_trace_sources: Optional[dict[str, dict]] = None,
    ) -> dict[str, dict]:
        from garnet.visual_primitives.cv_pipe_tracer import (
            CVPipeTracer,
            TURN_LEFT,
            TURN_RIGHT,
            _has_connected_side_pipe,
        )

        deltas = {
            "UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0),
        }
        h_mask, w_mask = pipe_mask.shape
        branch_results: dict[str, dict] = {}
        all_terminal_candidates = terminal_candidates or candidates
        candidate_by_id = {
            str(candidate["id"]): candidate
            for candidate in all_terminal_candidates
        }
        used_branch_pairs: dict[str, str] = {}
        pending_branch_pairs: dict[str, str] = {}
        for candidate in candidates:
            branch_id = candidate["id"]
            if branch_id in used_branch_pairs and branch_id not in pending_branch_pairs:
                candidate["status"] = "done_used_by_branch"
                candidate["reason"] = "paired_branch_already_traced"
                candidate["paired_branch_id"] = used_branch_pairs[branch_id]
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": "done_used_by_branch",
                    "paired_branch_id": used_branch_pairs[branch_id],
                    "candidate": candidate,
                }
                continue
            if candidate.get("status") != "queued":
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": candidate.get("status"),
                    "candidate": candidate,
                }
                continue

            direction = str(candidate["branch_direction"]).upper()
            dx, dy = deltas[direction]
            start_x = int(candidate["x"]) + dx * 3
            start_y = int(candidate["y"]) + dy * 3
            found_start = False
            for offset in range(3, 31):
                tx = int(candidate["x"]) + dx * offset
                ty = int(candidate["y"]) + dy * offset
                if 0 <= tx < w_mask and 0 <= ty < h_mask and pipe_mask[ty, tx] > 0:
                    start_x, start_y = tx, ty
                    found_start = True
                    break
            if not found_start:
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": "no_pipe_at_branch_start",
                    "candidate": candidate,
                }
                continue
            prior_reach = self._stage5b_branch_start_reached_by_prior_result(
                candidate,
                branch_results,
            )
            if prior_reach is not None:
                prior_id, _prior = prior_reach
                candidate["status"] = "done_used_by_branch"
                candidate["reason"] = "source_node_reached_by_prior_branch"
                candidate["paired_branch_id"] = prior_id
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": "source_node_reached_by_prior_branch",
                    "paired_branch_id": prior_id,
                    "candidate": candidate,
                }
                continue

            branch_terminals = []
            for other in all_terminal_candidates:
                other_id = other.get("id")
                if other_id == branch_id:
                    continue
                ox = int(other["x"])
                oy = int(other["y"])
                branch_terminals.append({
                    "id": other_id,
                    "class_name": "branch_candidate",
                    "bbox": {
                        "x_min": ox - 8,
                        "y_min": oy - 8,
                        "x_max": ox + 8,
                        "y_max": oy + 8,
                    },
                })
            source_node_id = str(candidate.get("node_obj_id", ""))
            junction_markers = [
                node for node in node_symbols
                if str(node.get("id", "")) != source_node_id
            ]
            tracer = CVPipeTracer(
                pipe_mask=pipe_mask,
                image=image,
                page_connections=page_connections + branch_terminals,
                instrument_tags=instrument_tags,
                equipment_objects=equipment,
                junction_markers=junction_markers,
                visited_mask=visited,
            )
            tracer.set_inline_symbols(inline_symbols)
            result = tracer.trace(
                start_x,
                start_y,
                direction,
                source_obj_id=branch_id,
            )
            terminal_inside_exact_bbox = False
            terminal_margin_gap: Optional[int] = None
            if result.terminal_type == "instrument_tag" and result.terminal_obj_id:
                terminal_obj = next(
                    (tag for tag in instrument_tags if tag.get("id") == result.terminal_obj_id),
                    None,
                )
                terminal_bbox = terminal_obj.get("bbox") if terminal_obj else None
                if terminal_bbox:
                    terminal_inside_exact_bbox = (
                        terminal_bbox["x_min"] <= result.terminal_x <= terminal_bbox["x_max"]
                        and terminal_bbox["y_min"] <= result.terminal_y <= terminal_bbox["y_max"]
                    )
                    if result.segments:
                        last_dir = result.segments[-1].direction
                        if last_dir == "UP" and result.terminal_y > terminal_bbox["y_max"]:
                            terminal_margin_gap = result.terminal_y - int(terminal_bbox["y_max"])
                        elif last_dir == "DOWN" and result.terminal_y < terminal_bbox["y_min"]:
                            terminal_margin_gap = int(terminal_bbox["y_min"]) - result.terminal_y
                        elif last_dir == "LEFT" and result.terminal_x > terminal_bbox["x_max"]:
                            terminal_margin_gap = result.terminal_x - int(terminal_bbox["x_max"])
                        elif last_dir == "RIGHT" and result.terminal_x < terminal_bbox["x_min"]:
                            terminal_margin_gap = int(terminal_bbox["x_min"]) - result.terminal_x

            recover_side_turn = (
                result.terminal_type == "dead_end"
                or (
                    result.terminal_type == "instrument_tag"
                    and not terminal_inside_exact_bbox
                    and terminal_margin_gap is not None
                    and terminal_margin_gap <= 10
                )
            )
            extra_turns = []
            extra_result = None
            recovery_point = None
            if recover_side_turn and result.segments:
                last_seg = result.segments[-1]
                last_dir = last_seg.direction
                sdx, sdy = deltas[last_dir]
                for backtrack in range(0, 21):
                    bx = result.terminal_x - sdx * backtrack
                    by = result.terminal_y - sdy * backtrack
                    for turn_dir in (TURN_LEFT[last_dir], TURN_RIGHT[last_dir]):
                        if not _has_connected_side_pipe(pipe_mask, bx, by, turn_dir, min_run=25):
                            continue
                        tdx, tdy = deltas[turn_dir]
                        cont_x = bx + tdx * 3
                        cont_y = by + tdy * 3
                        found_cont = False
                        for offset in range(3, 31):
                            tx = bx + tdx * offset
                            ty = by + tdy * offset
                            if 0 <= tx < w_mask and 0 <= ty < h_mask and pipe_mask[ty, tx] > 0:
                                cont_x, cont_y = tx, ty
                                found_cont = True
                                break
                        if not found_cont:
                            continue
                        recovery_tracer = CVPipeTracer(
                            pipe_mask=pipe_mask,
                            image=image,
                            page_connections=page_connections + branch_terminals,
                            instrument_tags=instrument_tags,
                            equipment_objects=equipment,
                            junction_markers=junction_markers,
                            visited_mask=visited,
                        )
                        recovery_tracer.set_inline_symbols(inline_symbols)
                        extra_result = recovery_tracer.trace(
                            cont_x,
                            cont_y,
                            turn_dir,
                            source_obj_id=branch_id,
                        )
                        recovery_point = (bx, by, turn_dir)
                        extra_turns.append({"x": bx, "y": by, "new_dir": turn_dir})
                        break
                    if extra_result is not None:
                        break

            segments = [
                {
                    "x1": s.x1, "y1": s.y1,
                    "x2": s.x2, "y2": s.y2,
                    "direction": s.direction,
                    "length_px": s.length_px,
                }
                for s in result.segments
            ]
            trace_length = result.trace_length_px
            terminal_type = result.terminal_type
            terminal_x = result.terminal_x
            terminal_y = result.terminal_y
            terminal_obj_id = result.terminal_obj_id
            turns = [
                {"x": tx, "y": ty, "new_dir": td}
                for tx, ty, td in result.turns
            ]
            hits = [
                {"class": h.class_name, "x": h.x, "y": h.y}
                for h in result.hits
            ]
            if extra_result is not None and recovery_point is not None and segments:
                bx, by, turn_dir = recovery_point
                last = segments[-1]
                old_len = int(last["length_px"])
                new_len = max(abs(bx - int(last["x1"])), abs(by - int(last["y1"])))
                last["x2"] = bx
                last["y2"] = by
                last["length_px"] = new_len
                trace_length += new_len - old_len
                turns.extend(extra_turns)
                turns.extend(
                    {"x": tx, "y": ty, "new_dir": td}
                    for tx, ty, td in extra_result.turns
                )
                hits.extend(
                    {"class": h.class_name, "x": h.x, "y": h.y}
                    for h in extra_result.hits
                )
                extra_segments = [
                    {
                        "x1": s.x1, "y1": s.y1,
                        "x2": s.x2, "y2": s.y2,
                        "direction": s.direction,
                        "length_px": s.length_px,
                    }
                    for s in extra_result.segments
                ]
                segments.extend(extra_segments)
                trace_length += extra_result.trace_length_px
                terminal_type = extra_result.terminal_type
                terminal_x = extra_result.terminal_x
                terminal_y = extra_result.terminal_y
                terminal_obj_id = extra_result.terminal_obj_id

            if not str(terminal_obj_id or "").startswith("branch_"):
                existing_paths = dict(existing_trace_sources or {})
                existing_paths.update(branch_results)
                intersection = self._find_stage5b_existing_path_intersection(
                    candidate,
                    segments,
                    existing_paths,
                )
                if intersection is not None:
                    segments, trace_length, terminal_x, terminal_y, hit_trace_id = (
                        self._truncate_stage5b_branch_at_intersection(
                            segments,
                            trace_length,
                            intersection,
                        )
                    )
                    terminal_type = "tee_junction"
                    terminal_obj_id = hit_trace_id

            if (terminal_obj_id or "").startswith("branch_"):
                terminal_type = "branch_connection"
                paired_branch_id = str(terminal_obj_id)
                used_branch_pairs[branch_id] = paired_branch_id
                used_branch_pairs[paired_branch_id] = branch_id
                candidate["status"] = "done_branch_connection"
                candidate["reason"] = "traced_to_branch_candidate"
                candidate["paired_branch_id"] = paired_branch_id
                paired_candidate = candidate_by_id.get(paired_branch_id)
                if paired_candidate is not None:
                    paired_candidate["paired_branch_id"] = branch_id
                    if paired_branch_id in branch_results:
                        paired_candidate["status"] = "done_used_by_branch"
                        paired_candidate["reason"] = "reverse_branch_trace_preferred"
                    else:
                        paired_candidate["status"] = "pending_branch_connection"
                        paired_candidate["reason"] = "paired_branch_candidate_pending_reverse_check"
                        pending_branch_pairs[paired_branch_id] = branch_id
                    if paired_candidate.get("node_obj_id"):
                        terminal_type = "tee_junction"
                        terminal_obj_id = str(paired_candidate["node_obj_id"])
                        terminal_x = int(paired_candidate["x"])
                        terminal_y = int(paired_candidate["y"])
            branch_results[branch_id] = {
                "status": "traced",
                "candidate": candidate,
                "port": {"x": start_x, "y": start_y, "direction": direction},
                "terminal_type": terminal_type,
                "terminal_x": terminal_x,
                "terminal_y": terminal_y,
                "terminal_obj_id": terminal_obj_id,
                "segments": segments,
                "turns": turns,
                "hits": hits,
                "trace_length_px": trace_length,
            }
            self._extend_stage5b_result_to_terminal(branch_results[branch_id])
            if branch_id in pending_branch_pairs and terminal_type == "branch_connection":
                previous_branch_id = pending_branch_pairs.pop(branch_id)
                previous_result = branch_results.get(previous_branch_id)
                if previous_result is not None:
                    previous_candidate = candidate_by_id.get(previous_branch_id)
                    if previous_candidate is not None:
                        previous_candidate["status"] = "done_used_by_branch"
                        previous_candidate["reason"] = "reverse_branch_trace_preferred"
                        previous_candidate["paired_branch_id"] = branch_id
                    previous_result["status"] = "skipped"
                    previous_result["skip_reason"] = "done_used_by_branch"
                    previous_result["paired_branch_id"] = branch_id
                    previous_result["preferred_branch_id"] = branch_id
                candidate["status"] = "done_branch_connection"
                candidate["reason"] = "reverse_branch_trace_preferred"
                candidate["paired_branch_id"] = previous_branch_id
            if terminal_type == "branch_connection" and terminal_obj_id in branch_results:
                previous_result = branch_results.get(str(terminal_obj_id))
                if previous_result is not None:
                    previous_result["status"] = "skipped"
                    previous_result["skip_reason"] = "done_used_by_branch"
                    previous_result["paired_branch_id"] = branch_id
                    previous_result["preferred_branch_id"] = branch_id
                paired_candidate = candidate_by_id.get(str(terminal_obj_id))
                if paired_candidate is not None:
                    paired_candidate["status"] = "done_used_by_branch"
                    paired_candidate["reason"] = "reverse_branch_trace_preferred"
                    paired_candidate["paired_branch_id"] = branch_id

        for pending_branch_id, traced_branch_id in pending_branch_pairs.items():
            pending_candidate = candidate_by_id.get(pending_branch_id)
            if pending_candidate is not None and pending_candidate.get("status") == "pending_branch_connection":
                pending_candidate["status"] = "done_branch_connection"
                pending_candidate["reason"] = "paired_branch_candidate_traced"
                pending_candidate["paired_branch_id"] = traced_branch_id
            pending_result = branch_results.get(pending_branch_id)
            if pending_result is not None and pending_result.get("status") == "skipped":
                pending_result["skip_reason"] = "done_used_by_branch"
                pending_result["paired_branch_id"] = traced_branch_id
                pending_result["preferred_branch_id"] = traced_branch_id
        return branch_results

    def _trace_stage5b_branches_iterative(
        self,
        pipe_mask: np.ndarray,
        image: np.ndarray,
        base_results: dict[str, dict],
        page_connections: list[dict[str, Any]],
        instrument_tags: list[dict[str, Any]],
        equipment: list[dict[str, Any]],
        inline_symbols: list[dict[str, Any]],
        node_symbols: list[dict[str, Any]],
        visited: np.ndarray,
        max_iterations: int = 5,
        candidate_overlay_base: Optional[np.ndarray] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict], list[dict[str, Any]]]:
        trace_sources = dict(base_results)
        all_candidates: list[dict[str, Any]] = []
        all_branch_results: dict[str, dict] = {}
        iteration_summaries: list[dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            detected_candidates = self._detect_stage5b_branch_candidates(
                pipe_mask=pipe_mask,
                all_results=trace_sources,
                inline_symbols=inline_symbols,
                node_symbols=node_symbols,
                equipment_objects=equipment,
            )
            new_candidates: list[dict[str, Any]] = []
            for detected in detected_candidates:
                existing = self._find_matching_stage5b_branch_candidate(
                    detected,
                    all_candidates,
                )
                if existing is not None:
                    if (
                        str(existing.get("status", "")).startswith("rejected_")
                        and detected.get("status") == "queued"
                    ):
                        existing.update({
                            key: value
                            for key, value in detected.items()
                            if key not in {"id", "member_count"}
                        })
                        existing["iteration"] = iteration
                    continue

                candidate = {
                    key: value
                    for key, value in detected.items()
                    if key != "id"
                }
                candidate["id"] = f"branch_{len(all_candidates) + 1:06d}"
                candidate["iteration"] = iteration
                all_candidates.append(candidate)
                new_candidates.append(candidate)

            queued_candidates = [
                candidate
                for candidate in new_candidates
                if candidate.get("status") == "queued"
            ]
            if not new_candidates:
                iteration_summaries.append({
                    "iteration": iteration,
                    "new_candidates": 0,
                    "queued": 0,
                    "traced": 0,
                    "new_trace_sources": 0,
                    "stop_reason": "no_new_candidates",
                })
                if candidate_overlay_base is not None:
                    candidate_overlay = self._draw_stage5b_branch_candidate_overlay(
                        candidate_overlay_base,
                        new_candidates,
                        base_results=base_results,
                        branch_results=all_branch_results,
                    )
                    self._save_img("stage5b_branch_candidates_overlay", candidate_overlay)
                    self._save_img(f"stage5b_branch_candidates_iter_{iteration:02d}_overlay", candidate_overlay)
                break
            if not queued_candidates:
                iteration_summaries.append({
                    "iteration": iteration,
                    "new_candidates": len(new_candidates),
                    "queued": 0,
                    "traced": 0,
                    "new_trace_sources": 0,
                    "stop_reason": "no_new_queued_candidates",
                })
                if candidate_overlay_base is not None:
                    candidate_overlay = self._draw_stage5b_branch_candidate_overlay(
                        candidate_overlay_base,
                        new_candidates,
                        base_results=base_results,
                        branch_results=all_branch_results,
                    )
                    self._save_img("stage5b_branch_candidates_overlay", candidate_overlay)
                    self._save_img(f"stage5b_branch_candidates_iter_{iteration:02d}_overlay", candidate_overlay)
                break

            prior_branch_results = dict(all_branch_results)
            branch_results = self._trace_stage5b_branch_candidates(
                pipe_mask=pipe_mask,
                image=image,
                candidates=queued_candidates,
                page_connections=page_connections,
                instrument_tags=instrument_tags,
                equipment=equipment,
                inline_symbols=inline_symbols,
                node_symbols=node_symbols,
                visited=visited,
                terminal_candidates=all_candidates,
                existing_trace_sources=trace_sources,
            )
            all_branch_results.update(branch_results)

            new_trace_sources = 0
            traced_count = 0
            for branch_id, result in branch_results.items():
                if result.get("status") != "traced" or not result.get("segments"):
                    continue
                traced_count += 1
                trace_sources[branch_id] = self._stage5b_branch_result_as_trace_source(result)
                new_trace_sources += 1

            iteration_summaries.append({
                "iteration": iteration,
                "new_candidates": len(new_candidates),
                "queued": len(queued_candidates),
                "traced": traced_count,
                "new_trace_sources": new_trace_sources,
                "stop_reason": "max_iterations" if iteration == max_iterations else None,
            })
            if candidate_overlay_base is not None:
                candidate_overlay = self._draw_stage5b_branch_candidate_overlay(
                    candidate_overlay_base,
                    new_candidates,
                    base_results=base_results,
                    branch_results=prior_branch_results,
                )
                self._save_img("stage5b_branch_candidates_overlay", candidate_overlay)
                self._save_img(f"stage5b_branch_candidates_iter_{iteration:02d}_overlay", candidate_overlay)
            if new_trace_sources == 0:
                iteration_summaries[-1]["stop_reason"] = "no_new_trace_sources"
                break

        return all_candidates, all_branch_results, iteration_summaries

    def _align_stage5b_branch_node_terminals(
        self,
        branch_results: dict[str, dict],
        node_symbols: list[dict[str, Any]],
    ) -> None:
        """Move saved tee endpoints to the node bbox center after discovery.

        This keeps the iterative branch search using the raw traced geometry,
        while the final JSON/overlay lands node terminals on the visible tee dot.
        """
        node_by_id = {
            str(node.get("id", "")): node
            for node in node_symbols
            if node.get("bbox")
        }
        for result in branch_results.values():
            if result.get("status") != "traced":
                continue
            if result.get("terminal_type") != "tee_junction":
                continue
            terminal_obj_id = str(result.get("terminal_obj_id") or "")
            node = node_by_id.get(terminal_obj_id)
            if not node:
                continue
            bbox = node["bbox"]
            terminal_x = (int(bbox["x_min"]) + int(bbox["x_max"])) // 2
            terminal_y = (int(bbox["y_min"]) + int(bbox["y_max"])) // 2
            result["terminal_x"] = terminal_x
            result["terminal_y"] = terminal_y
            segments = result.get("segments") or []
            if not segments:
                continue
            last = segments[-1]
            same_axis = (
                last["direction"] in ("LEFT", "RIGHT")
                and abs(int(last["y2"]) - terminal_y) <= 3
            ) or (
                last["direction"] in ("UP", "DOWN")
                and abs(int(last["x2"]) - terminal_x) <= 3
            )
            if not same_axis:
                continue
            old_len = int(last.get("length_px", 0))
            last["x2"] = terminal_x
            last["y2"] = terminal_y
            new_len = max(
                abs(int(last["x2"]) - int(last["x1"])),
                abs(int(last["y2"]) - int(last["y1"])),
            )
            last["length_px"] = new_len
            result["trace_length_px"] = int(result.get("trace_length_px", 0)) + new_len - old_len

    def _extend_stage5b_result_to_terminal(self, result: dict[str, Any]) -> None:
        segments = result.get("segments") or []
        if not segments:
            return
        terminal_x = result.get("terminal_x")
        terminal_y = result.get("terminal_y")
        if terminal_x is None or terminal_y is None:
            return
        last = segments[-1]
        same_axis = (
            last["direction"] in ("LEFT", "RIGHT")
            and abs(int(last["y2"]) - int(terminal_y)) <= 3
        ) or (
            last["direction"] in ("UP", "DOWN")
            and abs(int(last["x2"]) - int(terminal_x)) <= 3
        )
        if not same_axis:
            return
        if int(last["x2"]) == int(terminal_x) and int(last["y2"]) == int(terminal_y):
            return
        old_len = int(last.get("length_px", 0))
        if last["direction"] in ("LEFT", "RIGHT"):
            last["x2"] = int(terminal_x)
            last["y2"] = int(last["y1"])
        else:
            last["x2"] = int(last["x1"])
            last["y2"] = int(terminal_y)
        new_len = max(
            abs(int(last["x2"]) - int(last["x1"])),
            abs(int(last["y2"]) - int(last["y1"])),
        )
        last["length_px"] = new_len
        result["trace_length_px"] = int(result.get("trace_length_px", 0)) + new_len - old_len

    def _find_existing_stage5b_terminal_trace(
        self,
        all_results: dict[str, dict],
        obj_id: str,
        point: Optional[tuple[int, int]] = None,
        point_tolerance: int = 12,
    ) -> Optional[tuple[str, dict]]:
        is_equipment_source = str(obj_id).startswith("equip_")
        for existing_trace_id, existing in all_results.items():
            terminal_matches_obj = (
                not is_equipment_source
                and str(existing.get("terminal_obj_id", "")) == str(obj_id)
            )
            terminal_matches_point = False
            if point is not None:
                tx = existing.get("terminal_x")
                ty = existing.get("terminal_y")
                terminal_matches_point = (
                    tx is not None
                    and ty is not None
                    and abs(int(tx) - int(point[0])) <= point_tolerance
                    and abs(int(ty) - int(point[1])) <= point_tolerance
                )
            if not (terminal_matches_obj or terminal_matches_point):
                continue
            if existing.get("status") not in {None, "ok"}:
                continue
            if not existing.get("segments"):
                continue
            return existing_trace_id, existing
        return None

    def _align_stage5b_result_to_near_node(
        self,
        result: dict[str, Any],
        node_symbols: list[dict[str, Any]],
        max_distance: int = 20,
    ) -> None:
        if result.get("terminal_type") != "tee_junction":
            return
        if result.get("terminal_obj_id"):
            return
        tx = result.get("terminal_x")
        ty = result.get("terminal_y")
        if tx is None or ty is None:
            return
        nearest = None
        nearest_dist = max_distance + 1
        for node in node_symbols:
            bbox = node.get("bbox")
            if not bbox:
                continue
            cx = (int(bbox["x_min"]) + int(bbox["x_max"])) // 2
            cy = (int(bbox["y_min"]) + int(bbox["y_max"])) // 2
            dist = max(abs(int(tx) - cx), abs(int(ty) - cy))
            if dist < nearest_dist:
                nearest = (str(node.get("id", "")), cx, cy)
                nearest_dist = dist
        if nearest is None:
            return
        node_id, cx, cy = nearest
        result["terminal_obj_id"] = node_id
        result["terminal_x"] = cx
        result["terminal_y"] = cy
        segments = result.get("segments") or []
        if not segments:
            return
        last = segments[-1]
        old_len = int(last.get("length_px", 0))
        last["x2"] = cx
        last["y2"] = cy
        new_len = max(
            abs(int(last["x2"]) - int(last["x1"])),
            abs(int(last["y2"]) - int(last["y1"])),
        )
        last["length_px"] = new_len
        result["trace_length_px"] = int(result.get("trace_length_px", 0)) + new_len - old_len

    def _reverse_stage5b_trace_result(
        self,
        *,
        obj_id: str,
        port_index: int,
        port: tuple[int, int, str],
        existing_trace_id: str,
        existing: dict[str, Any],
    ) -> dict[str, Any]:
        opposite = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }
        segments = []
        for seg in reversed(existing.get("segments") or []):
            old_dir = str(seg.get("direction", "")).upper()
            segments.append({
                "x1": int(seg.get("x2", 0)),
                "y1": int(seg.get("y2", 0)),
                "x2": int(seg.get("x1", 0)),
                "y2": int(seg.get("y1", 0)),
                "direction": opposite.get(old_dir, old_dir),
                "length_px": int(seg.get("length_px", 0)),
            })

        turns = [
            {
                "x": int(prev_seg["x2"]),
                "y": int(prev_seg["y2"]),
                "new_dir": str(next_seg.get("direction", "")),
            }
            for prev_seg, next_seg in zip(segments, segments[1:])
        ]

        source_port = existing.get("port") or {}
        terminal_x = int(source_port.get("x", segments[-1]["x2"] if segments else port[0]))
        terminal_y = int(source_port.get("y", segments[-1]["y2"] if segments else port[1]))
        terminal_obj_id = str(existing.get("source_obj_id") or "").split(":")[0]
        terminal_type = "equipment" if terminal_obj_id.startswith("equip_") else "page_connection"

        return {
            "source_obj_id": obj_id,
            "port_index": port_index,
            "port": {"x": int(port[0]), "y": int(port[1]), "direction": str(port[2])},
            "terminal_type": terminal_type,
            "terminal_x": terminal_x,
            "terminal_y": terminal_y,
            "terminal_obj_id": terminal_obj_id,
            "segments": segments,
            "turns": turns,
            "hits": list(existing.get("hits") or []),
            "trace_length_px": int(existing.get("trace_length_px") or 0),
            "status": "reused_existing_trace",
            "reused_trace_id": existing_trace_id,
        }

    def _skip_stage5b_existing_trace_result(
        self,
        *,
        obj_id: str,
        port_index: int,
        port: tuple[int, int, str],
        existing_trace_id: str,
        existing: dict[str, Any],
        reason: str = "source_reached_by_existing_trace",
    ) -> dict[str, Any]:
        return {
            "source_obj_id": obj_id,
            "port_index": port_index,
            "port": {"x": int(port[0]), "y": int(port[1]), "direction": str(port[2])},
            "terminal_type": existing.get("terminal_type"),
            "terminal_x": existing.get("terminal_x"),
            "terminal_y": existing.get("terminal_y"),
            "terminal_obj_id": existing.get("terminal_obj_id"),
            "segments": [],
            "turns": [],
            "hits": [],
            "trace_length_px": 0,
            "status": "skipped_existing_trace",
            "skip_reason": reason,
            "reused_trace_id": existing_trace_id,
        }

    def _find_existing_stage5b_source_port_trace(
        self,
        all_results: dict[str, dict],
        obj_id: str,
        point: tuple[int, int],
        direction: str,
        point_tolerance: int = 35,
    ) -> Optional[tuple[str, dict]]:
        for existing_trace_id, existing in all_results.items():
            if str(existing.get("source_obj_id", "")) != str(obj_id):
                continue
            if str((existing.get("port") or {}).get("direction", "")).upper() != direction.upper():
                continue
            if existing.get("status") != "skipped_existing_trace":
                continue
            port = existing.get("port") or {}
            px = port.get("x")
            py = port.get("y")
            if px is None or py is None:
                continue
            if abs(int(px) - int(point[0])) <= point_tolerance and abs(int(py) - int(point[1])) <= point_tolerance:
                return existing_trace_id, existing
        return None

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

    def _detect_equipment_ports_cv(
        self, image: np.ndarray, bbox: dict[str, int], track_len: int = 60
    ) -> list[tuple[int, int, str]]:
        """Detect ALL pipe attachment ports on an equipment bbox.

        Unlike page connections (single pipe), equipment can have multiple
        nozzles — inlet, outlet, drain, vent, etc.  Returns all valid ports
        found on any edge of the bbox.

        Returns list of (x, y, direction) — empty if no pipes found.
        """
        import cv2 as _cv2

        x1, y1 = bbox["x_min"], bbox["y_min"]
        x2, y2 = bbox["x_max"], bbox["y_max"]
        h_img, w_img = image.shape[:2]

        if len(image.shape) == 3:
            gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        bg_strip_y = max(0, y1 - 10)
        bg_strip = gray[bg_strip_y:y1, x1:x2] if y1 > 10 else gray[0:10, x1:x2]
        bg_val = float(np.median(bg_strip)) if bg_strip.size > 0 else 200

        edges = [
            ("TOP",    (x1, y1), (x2, y1), (1, 0),   (0, -1)),
            ("BOTTOM", (x1, y2), (x2, y2), (1, 0),   (0,  1)),
            ("LEFT",   (x1, y1), (x1, y2), (0, 1),   (-1, 0)),
            ("RIGHT",  (x2, y1), (x2, y2), (0, 1),   (1,  0)),
        ]
        direction_map = {"TOP": "UP", "BOTTOM": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}

        all_ports: list[tuple[int, int, str]] = []

        for edge_name, start, end, step_dir, track_dir in edges:
            sx, sy = start
            ex, ey = end
            length = max(abs(ex - sx), abs(ey - sy))

            i = 0
            while i < length:
                px = sx + i * step_dir[0]
                py = sy + i * step_dir[1]
                if px < 0 or px >= w_img or py < 0 or py >= h_img:
                    i += 1
                    continue

                pixel_val = gray[py, px]
                if pixel_val > bg_val * 0.7:
                    i += 1
                    continue

                # Track outward
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
                        break

                if track_count >= 3:
                    direction = direction_map[edge_name]
                    out_x, out_y = px, py
                    if direction == "UP":
                        out_y = y1
                    elif direction == "DOWN":
                        out_y = y2
                    elif direction == "LEFT":
                        out_x = x1
                    elif direction == "RIGHT":
                        out_x = x2
                    all_ports.append((int(out_x), int(out_y), direction))

                # Skip 15 px ahead — equipment nozzles are well-separated
                i += 15
                continue

        return all_ports

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

        logger.info("Connection port detection: %d/%d ports found",
                 len(ports), len(conn_objects))

        # --- Equipment port detection from LabelMe ground truth ---
        # YOLO PPCL model currently lacks vessel/column/HEX classes, so we use
        # the LabelMe JSON as the source of truth for equipment bboxes.
        equip_labels = {
            "vessel", "column", "pump", "compressor", "blower",
            "heat exchanger", "tank", "reactor", "mixer", "pot",
            "knockout drum", "filter", "cooler", "heater",
            "injection pump",
        }
        equip_shapes = self._load_equipment_labelme()
        if equip_shapes:
            logger.info("Equipment port detection (LabelMe): %d equipment shapes", len(equip_shapes))
            for i, shape in enumerate(equip_shapes):
                label = shape.get("label", "").strip()
                if label.lower() not in equip_labels:
                    continue
                pts = shape.get("points", [])
                if len(pts) != 2:
                    continue
                x1 = int(round(pts[0][0]))
                y1 = int(round(pts[0][1]))
                x2 = int(round(pts[1][0]))
                y2 = int(round(pts[1][1]))
                bbox = {"x_min": min(x1, x2), "y_min": min(y1, y2),
                        "x_max": max(x1, x2), "y_max": max(y1, y2)}
                eq_id = f"equip_{i}_{label.replace(' ', '_')}"
                eq_ports = self._detect_equipment_ports_cv(image, bbox)
                if eq_ports:
                    ports[eq_id] = eq_ports
                    port_str = ", ".join(f"{d}({x},{y})" for x, y, d in eq_ports)
                    logger.info("  %s (%s) -> %d ports: %s",
                              eq_id, label, len(eq_ports), port_str)
                else:
                    logger.info("  %s (%s) -> no ports detected", eq_id, label)

        equip_port_ids = {k for k in ports if k.startswith("equip_")}
        conn_port_ids = {k for k in ports if not k.startswith("equip_")}
        logger.info("Total ports: %d objects (%d conn + %d equip)",
                 len(ports), len(conn_port_ids), len(equip_port_ids))
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

        # Compute connection ports if not already cached
        ports_path = self.out_dir / "stage5_connection_ports.json"
        if not ports_path.exists():
            logger.info("Computing connection ports (first run)")
            objects_all = self._load_json_artifact("stage4_objects").get("objects", [])
            ports = self._compute_connection_ports_vlm(objects_all)
        else:
            ports = self._load_json_artifact("stage5_connection_ports")

        if not ports:
            logger.warning("No connection ports found — skipping pipe trace")
            return

        ports = self._snap_ports_to_pipe_centerlines(ports, pipe_mask)
        self._save_json("stage5_connection_ports", ports)

        objects = self._load_json_artifact("stage4_objects").get("objects", [])
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

        # Load LabelMe equipment bboxes (YOLO PPCL lacks vessel/column/HEX classes)
        equip_labels = {
            "vessel", "column", "pump", "compressor", "blower",
            "heat exchanger", "tank", "reactor", "mixer", "pot",
            "knockout drum", "filter", "cooler", "heater",
            "injection pump",
        }
        labelme_shapes = self._load_equipment_labelme()
        for i, shape in enumerate(labelme_shapes):
            label = shape.get("label", "").strip()
            if label.lower() not in equip_labels:
                continue
            pts = shape.get("points", [])
            if len(pts) != 2:
                continue
            x1 = int(round(pts[0][0]))
            y1 = int(round(pts[0][1]))
            x2 = int(round(pts[1][0]))
            y2 = int(round(pts[1][1]))
            equipment.append({
                "id": f"equip_{i}_{label.replace(' ', '_')}",
                "class_name": label,
                "bbox": {
                    "x_min": min(x1, x2), "y_min": min(y1, y2),
                    "x_max": max(x1, x2), "y_max": max(y1, y2),
                },
            })
        if labelme_shapes:
            logger.info("Added %d LabelMe equipment bboxes for tracer terminals",
                         len([s for s in labelme_shapes
                              if s.get("label", "").strip().lower() in equip_labels
                              and len(s.get("points", [])) == 2]))

        # Inline symbols (valves, reducers, etc.)
        inline_classes = {
            "gate_valve", "globe_valve", "check_valve", "ball_valve",
            "butterfly_valve", "control_valve", "pressure_relief_valve",
            "reducer", "spectacle_blind", "strainer",
            "gate valve", "globe valve", "check valve", "ball valve",
            "butterfly valve", "control valve", "pressure relief valve",
            "spectacle blind",
        }
        inline_symbols = [
            o for o in objects
            if o.get("class_name") in inline_classes
        ]
        node_symbols = [
            o for o in objects
            if o.get("class_name") == "node"
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

        total_ports = sum(len(port_list) for port_list in ports.values())
        logger.info("CV pipe trace: %d objects, %d ports", len(ports), total_ports)
        t0 = _time.monotonic()

        port_items = sorted(
            ports.items(),
            key=lambda item: (1 if str(item[0]).startswith("equip_") else 0, str(item[0])),
        )

        for obj_id, port_list in port_items:
            for port_index, (px, py, direction) in enumerate(port_list, start=1):
                trace_id = obj_id if len(port_list) == 1 else f"{obj_id}:port_{port_index:02d}"
                is_page_connection_source = any(pc.get("id") == obj_id for pc in page_connections)
                existing_terminal = (
                    None
                    if is_page_connection_source
                    else self._find_existing_stage5b_terminal_trace(
                        all_results,
                        obj_id,
                        point=(int(px), int(py)),
                    )
                )
                if existing_terminal is None and str(obj_id).startswith("equip_"):
                    existing_terminal = self._find_existing_stage5b_source_port_trace(
                        all_results,
                        obj_id,
                        point=(int(px), int(py)),
                        direction=str(direction),
                    )
                if existing_terminal is not None:
                    existing_trace_id, existing = existing_terminal
                    all_results[trace_id] = self._skip_stage5b_existing_trace_result(
                        obj_id=obj_id,
                        port_index=port_index,
                        port=(px, py, direction),
                        existing_trace_id=existing_trace_id,
                        existing=existing,
                    )
                    logger.info(
                        "  %s -> reused %s (%d px, %d segs)",
                        trace_id,
                        existing_trace_id,
                        all_results[trace_id]["trace_length_px"],
                        len(all_results[trace_id]["segments"]),
                    )
                    continue
                tracer = CVPipeTracer(
                    pipe_mask=pipe_mask,
                    image=image,
                    page_connections=page_connections,
                    instrument_tags=instrument_tags,
                    equipment_objects=equipment,
                    junction_markers=node_symbols,
                    visited_mask=visited,
                )
                tracer.set_inline_symbols(inline_symbols)

                result = tracer.trace(px, py, direction, source_obj_id=obj_id)
                terminal_type = result.terminal_type

                all_results[trace_id] = {
                    "source_obj_id": obj_id,
                    "port_index": port_index,
                    "port": {"x": px, "y": py, "direction": direction},
                    "terminal_type": terminal_type,
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
                self._extend_stage5b_result_to_terminal(all_results[trace_id])
                self._align_stage5b_result_to_near_node(all_results[trace_id], node_symbols)
                logger.info(
                    "  %s -> %s (%d px, %d segs)",
                    trace_id, result.terminal_type,
                    result.trace_length_px, len(result.segments),
                )

        elapsed = _time.monotonic() - t0
        logger.info("CV pipe trace done: %d traces in %.1fs", len(all_results), elapsed)

        self._save_json("stage5b_trace_results", all_results)
        for stale_overlay in self.out_dir.glob("stage5b_branch_candidates_iter_*_overlay.png"):
            stale_overlay.unlink()
        branch_candidates, branch_results, branch_iterations = self._trace_stage5b_branches_iterative(
            pipe_mask=pipe_mask,
            image=image,
            base_results=all_results,
            page_connections=page_connections,
            instrument_tags=instrument_tags,
            equipment=equipment,
            inline_symbols=inline_symbols,
            node_symbols=node_symbols,
            visited=visited,
            candidate_overlay_base=image,
        )
        self._align_stage5b_branch_node_terminals(branch_results, node_symbols)
        self._save_json("stage5b_branch_candidates", {
            "candidates": branch_candidates,
            "iterations": branch_iterations,
            "summary": {
                "total": len(branch_candidates),
                "queued": len([c for c in branch_candidates if c.get("status") == "queued"]),
                "done": len([c for c in branch_candidates if str(c.get("status", "")).startswith("done_")]),
                "rejected": len([c for c in branch_candidates if str(c.get("status", "")).startswith("rejected_")]),
                "iteration_count": len(branch_iterations),
            },
        })
        self._save_json("stage5b_branch_trace_results", {
            "branches": branch_results,
            "iterations": branch_iterations,
            "summary": {
                "total": len(branch_results),
                "traced": len([r for r in branch_results.values() if r.get("status") == "traced"]),
                "skipped": len([r for r in branch_results.values() if r.get("status") == "skipped"]),
                "iteration_count": len(branch_iterations),
            },
        })

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
        # Build object lookup for source bbox drawing
        id_to_obj = {o["id"]: o for o in objects}

        # Human-readable P&ID labels
        _CLASS_LABEL_MAP = {
            "gate_valve": "Gate Valve",
            "globe_valve": "Globe Valve",
            "check_valve": "Check Valve",
            "ball_valve": "Ball Valve",
            "butterfly_valve": "Butterfly Valve",
            "control_valve": "Control Valve",
            "pressure_relief_valve": "Pressure Relief Valve",
            "reducer": "Reducer",
            "spectacle_blind": "Spectacle Blind",
            "strainer": "Strainer",
            "pump": "Pump",
            "vessel": "Vessel",
            "column": "Column",
            "heat_exchanger": "Heat Exchanger",
            "tank": "Tank",
            "compressor": "Compressor",
            "blower": "Blower",
            "filter": "Filter",
            "cooler": "Cooler",
            "heater": "Heater",
            "reactor": "Reactor",
            "knockout_drum": "Knockout Drum",
            "page connection": "Page Conn.",
            "connection": "Connection",
            "utility connection": "Utility Conn.",
            "instrument tag": "Instr. Tag",
            "instrument dcs": "Instr. (DCS)",
            "instrument logic": "Instr. (Logic)",
            "line number": "Line No.",
            "arrow": "Flow Arrow",
            "node": "Node",
            "sampling point": "Samp. Point",
        }

        for obj_id, data in all_results.items():
            # --- Draw source object bbox ---
            source_obj_id = str(data.get("source_obj_id") or obj_id)
            src_obj = id_to_obj.get(source_obj_id)
            if src_obj:
                src_bbox = src_obj["bbox"]
                _cv2.rectangle(
                    overlay,
                    (src_bbox["x_min"], src_bbox["y_min"]),
                    (src_bbox["x_max"], src_bbox["y_max"]),
                    (0, 200, 0), 2,
                )
                # Human-readable class label above bbox
                cls_raw = src_obj.get("class_name", "?")
                cls_label = _CLASS_LABEL_MAP.get(cls_raw, cls_raw.replace("_", " ").title())
                label_text = f"{cls_label} ({source_obj_id})"
                _cv2.putText(
                    overlay, label_text,
                    (src_bbox["x_min"], src_bbox["y_min"] - 6),
                    _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2,
                )

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

            # Port marker (start point)
            px, py = data["port"]["x"], data["port"]["y"]
            _cv2.circle(overlay, (px, py), 5, (0, 255, 0), -1)
            _cv2.circle(overlay, (px, py), 5, (255, 255, 255), 1)

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

        # Draw equipment ports (cyan markers from stage5_connection_ports)
        self._draw_equipment_port_markers(overlay, ports)

        # Draw equipment bboxes from LabelMe ground truth
        self._draw_equipment_ground_truth(overlay)

        self._save_img("stage5b_trace_overlay", overlay)
        branch_overlay = self._draw_stage5b_branch_candidate_overlay(
            overlay,
            branch_candidates,
            branch_results=branch_results,
        )
        self._save_img("stage5b_branch_candidates_overlay", branch_overlay)
        branch_trace_overlay = self._draw_stage5b_branch_trace_overlay(
            overlay,
            branch_results,
        )
        self._save_img("stage5b_branch_trace_overlay", branch_trace_overlay)

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
        if pipe_mask is not None:
            ports = self._snap_ports_to_pipe_centerlines(ports, pipe_mask)
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
    def _trace_assoc_point_to_segment(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> tuple[float, float, float, float]:
        abx = bx - ax
        aby = by - ay
        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= 0:
            return ax, ay, 0.0, math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / ab_len_sq))
        qx = ax + t * abx
        qy = ay + t * aby
        return qx, qy, t, math.hypot(px - qx, py - qy)

    def _trace_assoc_bbox_points(self, bbox: dict[str, Any]) -> list[tuple[float, float]]:
        x_min = float(bbox["x_min"])
        y_min = float(bbox["y_min"])
        x_max = float(bbox["x_max"])
        y_max = float(bbox["y_max"])
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return [
            (cx, cy),
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (cx, y_min),
            (cx, y_max),
            (x_min, cy),
            (x_max, cy),
        ]

    def _trace_assoc_polyline_from_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[list[int]]:
        polyline: list[list[int]] = []
        for segment in segments:
            p1 = [int(segment["x1"]), int(segment["y1"])]
            p2 = [int(segment["x2"]), int(segment["y2"])]
            if not polyline or polyline[-1] != p1:
                polyline.append(p1)
            if polyline[-1] != p2:
                polyline.append(p2)
        return polyline

    def _stage11_source_metadata(
        self,
        source_obj_id: str,
        objects_by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        obj = objects_by_id.get(source_obj_id)
        if obj is not None:
            return {
                "source_obj_id": source_obj_id,
                "source_obj_type": obj.get("class_name"),
                "source_bbox": obj.get("bbox"),
            }
        if source_obj_id.startswith("equip_"):
            return {
                "source_obj_id": source_obj_id,
                "source_obj_type": "equipment",
                "source_bbox": None,
            }
        if source_obj_id.startswith("branch_"):
            return {
                "source_obj_id": source_obj_id,
                "source_obj_type": "branch_candidate",
                "source_bbox": None,
            }
        return {
            "source_obj_id": source_obj_id,
            "source_obj_type": None,
            "source_bbox": None,
        }

    def _load_stage5b_trace_edges(
        self,
        objects_by_id: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        objects_by_id = objects_by_id or {}
        trace_payload = self._load_json_artifact("stage5b_trace_results")
        branch_payload = self._load_json_artifact_or_default("stage5b_branch_trace_results", {"branches": {}})

        edges: list[dict[str, Any]] = []
        for trace_id, trace in trace_payload.items():
            segments = trace.get("segments", [])
            if not segments:
                continue
            source_obj_id = str(trace.get("source_obj_id", trace_id))
            edges.append({
                "trace_id": str(trace_id),
                "trace_kind": "port",
                **self._stage11_source_metadata(source_obj_id, objects_by_id),
                "port_index": trace.get("port_index"),
                "port": trace.get("port"),
                "terminal_type": trace.get("terminal_type"),
                "terminal_obj_id": trace.get("terminal_obj_id"),
                "terminal_xy": [trace.get("terminal_x"), trace.get("terminal_y")],
                "segments": segments,
                "polyline": self._trace_assoc_polyline_from_segments(segments),
                "turns": trace.get("turns", []),
                "hits": trace.get("hits", []),
                "trace_length_px": trace.get("trace_length_px", 0),
                "status": trace.get("status", "ok"),
                "attachments": {},
                "warnings": [],
            })

        for branch_id, branch in branch_payload.get("branches", {}).items():
            if branch.get("status") != "traced" or not branch.get("segments"):
                continue
            segments = branch.get("segments", [])
            source_obj_id = str(branch_id)
            edges.append({
                "trace_id": str(branch_id),
                "trace_kind": "branch",
                **self._stage11_source_metadata(source_obj_id, objects_by_id),
                "candidate": branch.get("candidate", {}),
                "port": branch.get("port"),
                "terminal_type": branch.get("terminal_type"),
                "terminal_obj_id": branch.get("terminal_obj_id"),
                "terminal_xy": [branch.get("terminal_x"), branch.get("terminal_y")],
                "segments": segments,
                "polyline": self._trace_assoc_polyline_from_segments(segments),
                "turns": branch.get("turns", []),
                "hits": branch.get("hits", []),
                "trace_length_px": branch.get("trace_length_px", 0),
                "status": branch.get("status", "traced"),
                "paired_branch_id": branch.get("paired_branch_id"),
                "attachments": {},
                "warnings": [],
            })
        return edges

    def _trace_assoc_nearest_segment(
        self,
        point: tuple[float, float],
        edges: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        best: Optional[dict[str, Any]] = None
        px, py = point
        for edge in edges:
            cumulative = 0.0
            for index, segment in enumerate(edge.get("segments", [])):
                ax = float(segment["x1"])
                ay = float(segment["y1"])
                bx = float(segment["x2"])
                by = float(segment["y2"])
                qx, qy, t, distance = self._trace_assoc_point_to_segment(px, py, ax, ay, bx, by)
                seg_len = max(abs(bx - ax), abs(by - ay))
                along = cumulative + t * seg_len
                if best is None or distance < best["distance_px"]:
                    best = {
                        "trace_id": edge["trace_id"],
                        "trace_kind": edge["trace_kind"],
                        "segment_index": index,
                        "projected_xy": [round(qx, 2), round(qy, 2)],
                        "distance_px": round(distance, 2),
                        "t": round(t, 4),
                        "trace_distance_px": round(along, 2),
                    }
                cumulative += seg_len
        return best

    def _trace_assoc_nearest_bbox(
        self,
        bbox: dict[str, Any],
        edges: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        best: Optional[dict[str, Any]] = None
        for point in self._trace_assoc_bbox_points(bbox):
            candidate = self._trace_assoc_nearest_segment(point, edges)
            if candidate is None:
                continue
            if best is None or candidate["distance_px"] < best["distance_px"]:
                best = candidate
        return best

    def _trace_assoc_add(
        self,
        edges_by_id: dict[str, dict[str, Any]],
        trace_id: str,
        group: str,
        association: dict[str, Any],
    ) -> None:
        edge = edges_by_id.get(trace_id)
        if edge is None:
            return
        edge.setdefault("attachments", {}).setdefault(group, []).append(association)

    def _trace_assoc_attach_bbox_items(
        self,
        *,
        edges: list[dict[str, Any]],
        edges_by_id: dict[str, dict[str, Any]],
        group: str,
        items: list[dict[str, Any]],
        max_distance_px: float,
        id_key: str = "id",
        class_key: str = "class_name",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for item in items:
            bbox = item.get("bbox")
            item_id = str(item.get(id_key, ""))
            if not bbox:
                rejected.append({"id": item_id, "reason": "missing_bbox", "source": item})
                continue
            nearest = self._trace_assoc_nearest_bbox(bbox, edges)
            if nearest is None:
                rejected.append({"id": item_id, "reason": "no_trace_edges", "source": item})
                continue
            association = {
                "id": item_id,
                "source_object_id": item.get("source_object_id", item_id),
                "class_name": item.get(class_key, item.get("semantic_class", "")),
                "bbox": bbox,
                "text": item.get("text", ""),
                "normalized_text": item.get("normalized_text", ""),
                "confidence": item.get("confidence", item.get("fused_confidence", item.get("detection_confidence"))),
                **nearest,
            }
            if nearest["distance_px"] <= max_distance_px:
                if group == "line_numbers":
                    association = _mark_line_number_review_state(association, accepted=True)
                accepted.append(association)
                self._trace_assoc_add(edges_by_id, nearest["trace_id"], group, association)
            else:
                rejected_association = {
                    **association,
                    "reason": "distance_over_threshold",
                    "max_distance_px": max_distance_px,
                }
                if group == "line_numbers":
                    rejected_association = _mark_line_number_review_state(rejected_association, accepted=False)
                rejected.append(rejected_association)
        return accepted, rejected

    def _draw_stage11_trace_association_overlay(
        self,
        edges: list[dict[str, Any]],
        associations: dict[str, Any],
    ) -> np.ndarray:
        import cv2 as _cv2

        image = self._ensure_image_loaded()
        overlay = image.copy()
        for edge in edges:
            color = (0, 180, 0) if edge.get("trace_kind") == "port" else (0, 0, 220)
            for segment in edge.get("segments", []):
                _cv2.line(
                    overlay,
                    (int(segment["x1"]), int(segment["y1"])),
                    (int(segment["x2"]), int(segment["y2"])),
                    color,
                    2,
                )

        draw_specs = [
            ("equipment_ports", (255, 255, 0), 4),
            ("inline_objects", (0, 165, 255), 4),
            ("line_numbers", (255, 0, 255), 3),
            ("instrument_tags", (0, 255, 255), 3),
            ("flow_arrows", (255, 0, 0), 4),
            ("terminals", (180, 0, 180), 5),
        ]
        for group, color, radius in draw_specs:
            for item in associations.get(group, {}).get("accepted", []):
                projected = item.get("projected_xy")
                if not projected:
                    continue
                x = int(round(float(projected[0])))
                y = int(round(float(projected[1])))
                _cv2.circle(overlay, (x, y), radius, color, -1)
                if group == "line_numbers":
                    label = str(item.get("normalized_text") or item.get("text") or item.get("id") or "")
                    if len(label) > 48:
                        label = label[:45] + "..."
                    if label:
                        _cv2.putText(
                            overlay,
                            label,
                            (x + 8, y - 8),
                            _cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            color,
                            1,
                            _cv2.LINE_AA,
                        )
        return overlay

    def stage11_trace_associations(self) -> None:
        """Attach semantic detections to Stage 5b traced pipe paths."""
        image_id = Path(self.image_path).name
        object_payload = self._load_json_artifact("stage4_objects")
        objects = object_payload.get("objects", [])
        objects_by_id = {str(obj.get("id", "")): obj for obj in objects}
        edges = self._load_stage5b_trace_edges(objects_by_id)
        edges_by_id = {edge["trace_id"]: edge for edge in edges}
        line_payload = self._load_json_artifact_or_default("stage4_line_numbers", {"line_numbers": []})
        instrument_payload = self._load_json_artifact_or_default("stage4_instrument_tags", {"instrument_tags": []})
        branch_payload = self._load_json_artifact_or_default("stage5b_branch_trace_results", {"branches": {}})
        ports_payload = self._load_json_artifact_or_default("stage5_connection_ports", {})

        associations: dict[str, Any] = {
            "equipment_ports": {"accepted": [], "rejected": []},
            "inline_objects": {"accepted": [], "rejected": []},
            "line_numbers": {"accepted": [], "rejected": []},
            "instrument_tags": {"accepted": [], "rejected": []},
            "flow_arrows": {"accepted": [], "rejected": []},
            "terminals": {"accepted": [], "rejected": []},
        }

        # Equipment/page ports are deterministic because Stage 5b trace ids are
        # derived from the source object id and port index.
        for obj_id, port_list in ports_payload.items():
            for port_index, port in enumerate(port_list, start=1):
                if len(port) < 3:
                    continue
                trace_id = obj_id if len(port_list) == 1 else f"{obj_id}:port_{port_index:02d}"
                point = (float(port[0]), float(port[1]))
                nearest = self._trace_assoc_nearest_segment(point, edges)
                if trace_id in edges_by_id:
                    own_nearest = self._trace_assoc_nearest_segment(point, [edges_by_id[trace_id]])
                    if own_nearest is not None:
                        nearest = own_nearest
                if nearest is None:
                    associations["equipment_ports"]["rejected"].append({
                        "id": f"{obj_id}:port_{port_index:02d}",
                        "reason": "no_trace_edges",
                        "source_obj_id": obj_id,
                        "port_index": port_index,
                        "port": port,
                    })
                    continue
                association = {
                    "id": f"{obj_id}:port_{port_index:02d}",
                    "source_obj_id": obj_id,
                    "port_index": port_index,
                    "port_xy": [int(port[0]), int(port[1])],
                    "direction": str(port[2]),
                    **nearest,
                }
                if nearest["distance_px"] <= self.cfg.trace_association_equipment_port_max_distance_px:
                    associations["equipment_ports"]["accepted"].append(association)
                    self._trace_assoc_add(edges_by_id, nearest["trace_id"], "equipment_ports", association)
                else:
                    associations["equipment_ports"]["rejected"].append({
                        **association,
                        "reason": "distance_over_threshold",
                        "max_distance_px": self.cfg.trace_association_equipment_port_max_distance_px,
                    })

        inline_classes = {
            "gate_valve", "globe_valve", "check_valve", "ball_valve",
            "butterfly_valve", "control_valve", "pressure_relief_valve",
            "reducer", "spectacle_blind", "strainer",
            "gate valve", "globe valve", "check valve", "ball valve",
            "butterfly valve", "control valve", "pressure relief valve",
            "spectacle blind",
        }
        inline_objects = [obj for obj in objects if obj.get("class_name") in inline_classes]
        accepted, rejected = self._trace_assoc_attach_bbox_items(
            edges=edges,
            edges_by_id=edges_by_id,
            group="inline_objects",
            items=inline_objects,
            max_distance_px=self.cfg.trace_association_inline_object_max_distance_px,
        )
        associations["inline_objects"]["accepted"].extend(accepted)
        associations["inline_objects"]["rejected"].extend(rejected)

        # Preserve Stage 5b inline hits even when the detector bbox was not close
        # enough to pass the independent bbox matcher.
        seen_inline = {
            (item.get("trace_id"), item.get("class_name"), int(round(float(item.get("projected_xy", [0, 0])[0]))), int(round(float(item.get("projected_xy", [0, 0])[1]))))
            for item in associations["inline_objects"]["accepted"]
        }
        for edge in edges:
            for hit_index, hit in enumerate(edge.get("hits", []), start=1):
                point = (float(hit.get("x", 0)), float(hit.get("y", 0)))
                nearest = self._trace_assoc_nearest_segment(point, [edge])
                if nearest is None:
                    continue
                key = (
                    edge["trace_id"],
                    hit.get("class", hit.get("class_name", "")),
                    int(round(float(nearest["projected_xy"][0]))),
                    int(round(float(nearest["projected_xy"][1]))),
                )
                if key in seen_inline:
                    continue
                association = {
                    "id": f"{edge['trace_id']}:hit_{hit_index:03d}",
                    "class_name": hit.get("class", hit.get("class_name", "")),
                    "hit_xy": [int(point[0]), int(point[1])],
                    "source": "stage5b_hit",
                    **nearest,
                }
                associations["inline_objects"]["accepted"].append(association)
                self._trace_assoc_add(edges_by_id, edge["trace_id"], "inline_objects", association)
                seen_inline.add(key)

        accepted, rejected = self._trace_assoc_attach_bbox_items(
            edges=edges,
            edges_by_id=edges_by_id,
            group="line_numbers",
            items=line_payload.get("line_numbers", []),
            max_distance_px=self.cfg.trace_association_text_max_distance_px,
        )
        associations["line_numbers"]["accepted"] = accepted
        associations["line_numbers"]["rejected"] = rejected

        accepted, rejected = self._trace_assoc_attach_bbox_items(
            edges=edges,
            edges_by_id=edges_by_id,
            group="instrument_tags",
            items=instrument_payload.get("instrument_tags", []),
            max_distance_px=self.cfg.trace_association_instrument_max_distance_px,
        )
        associations["instrument_tags"]["accepted"] = accepted
        associations["instrument_tags"]["rejected"] = rejected

        arrows = [obj for obj in objects if obj.get("class_name") == "arrow"]
        accepted, rejected = self._trace_assoc_attach_bbox_items(
            edges=edges,
            edges_by_id=edges_by_id,
            group="flow_arrows",
            items=arrows,
            max_distance_px=self.cfg.trace_association_arrow_max_distance_px,
        )
        associations["flow_arrows"]["accepted"] = accepted
        associations["flow_arrows"]["rejected"] = rejected

        for edge in edges:
            terminal_xy = edge.get("terminal_xy") or []
            if len(terminal_xy) != 2 or terminal_xy[0] is None or terminal_xy[1] is None:
                continue
            nearest = self._trace_assoc_nearest_segment((float(terminal_xy[0]), float(terminal_xy[1])), [edge])
            if nearest is None:
                continue
            association = {
                "id": f"{edge['trace_id']}:terminal",
                "terminal_type": edge.get("terminal_type"),
                "terminal_obj_id": edge.get("terminal_obj_id"),
                "terminal_xy": terminal_xy,
                **nearest,
            }
            associations["terminals"]["accepted"].append(association)
            self._trace_assoc_add(edges_by_id, edge["trace_id"], "terminals", association)

        skipped_branches = [
            {"id": branch_id, **branch}
            for branch_id, branch in branch_payload.get("branches", {}).items()
            if branch.get("status") != "traced"
        ]
        simulated_line_number_assignments = simulate_line_number_hitl_for_missing_traces(
            edges,
            associations["line_numbers"]["accepted"],
        )
        associations["line_numbers"]["accepted"].extend(simulated_line_number_assignments)
        traces_without_line_number = [
            edge["trace_id"]
            for edge in edges
            if not edge.get("attachments", {}).get("line_numbers")
        ]
        dead_end_traces = [
            edge["trace_id"]
            for edge in edges
            if edge.get("terminal_type") == "dead_end"
        ]
        line_number_review_payload, line_number_review_summary = build_stage11_line_number_review_payload(
            image_id=image_id,
            accepted=associations["line_numbers"]["accepted"],
            rejected=associations["line_numbers"]["rejected"],
            traces_without_line_number=traces_without_line_number,
        )

        payload = {
            "image_id": image_id,
            "trace_source": "stage5b",
            "trace_edges": edges,
            "associations": associations,
            "unresolved": {
                "skipped_branches": skipped_branches,
                "traces_without_line_number": traces_without_line_number,
                "dead_end_traces": dead_end_traces,
                "unattached_line_numbers": associations["line_numbers"]["rejected"],
                "unattached_instrument_tags": associations["instrument_tags"]["rejected"],
            },
        }
        summary = {
            "image_id": image_id,
            "trace_edge_count": len(edges),
            "port_trace_count": len([edge for edge in edges if edge.get("trace_kind") == "port"]),
            "branch_trace_count": len([edge for edge in edges if edge.get("trace_kind") == "branch"]),
            "skipped_branch_count": len(skipped_branches),
            "dead_end_trace_count": len(dead_end_traces),
            "trace_without_line_number_count": len(traces_without_line_number),
            "accepted_counts": {
                key: len(value.get("accepted", []))
                for key, value in associations.items()
            },
            "simulated_line_number_assignment_count": len(simulated_line_number_assignments),
            "rejected_counts": {
                key: len(value.get("rejected", []))
                for key, value in associations.items()
            },
        }

        self._save_json("stage11_trace_associations", payload)
        self._save_json("stage11_trace_association_summary", summary)
        self._save_json("stage11_line_number_review", line_number_review_payload)
        self._save_json("stage11_line_number_review_summary", line_number_review_summary)
        self._save_img(
            "stage11_trace_association_overlay",
            self._draw_stage11_trace_association_overlay(edges, associations),
        )

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
        stage11_path = self.out_dir / "stage11_trace_associations.json"
        if stage11_path.exists():
            image_id = Path(self.image_path).name
            stage11_payload = self._load_json_artifact("stage11_trace_associations")
            trace_graph_result = build_trace_graph_from_stage11(stage11_payload, image_id=image_id)
            self._save_json("stage12_graph", trace_graph_result["graph_payload"])
            self._save_json("stage12_graph_summary", trace_graph_result["summary"])
            self._save_json("stage12_trace_edge_nodes", trace_graph_result["trace_edge_nodes_payload"])
            self._save_json("stage12_review_queue", trace_graph_result["review_queue_payload"])
            self._save_json("stage12_review_queue_summary", trace_graph_result["review_queue_summary"])
            self._save_json("stage12_graph_normalization", trace_graph_result["normalization_payload"])
            self._save_json("stage12_graph_normalization_summary", trace_graph_result["normalization_summary"])
            self._save_img(
                "stage12_graph_overlay",
                render_stage12_graph_overlay(self._ensure_image_loaded(), trace_graph_result["graph_payload"]),
            )
            trace_graph_qa_result = run_stage12_trace_graph_qa(
                image_id=image_id,
                graph_payload=trace_graph_result["graph_payload"],
                image_bgr=self._ensure_image_loaded(),
            )
            self._save_json("stage12_graph_qa", trace_graph_qa_result["qa_payload"])
            self._save_json("stage12_graph_qa_summary", trace_graph_qa_result["summary"])
            self._save_img("stage12_graph_qa_overlay", trace_graph_qa_result["overlay_image"])
            return

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
        stage12_qa_path = self.out_dir / "stage12_graph_qa.json"
        if stage12_qa_path.exists():
            image_id = Path(self.image_path).name
            result = build_stage13_review_package(
                image_id=image_id,
                graph_payload=graph_payload,
                stage12_qa_payload=self._load_json_artifact("stage12_graph_qa"),
                stage12_review_queue_payload=self._load_json_artifact_or_default("stage12_review_queue", {"review_queue": []}),
                stage11_line_number_review_payload=self._load_json_artifact_or_default(
                    "stage11_line_number_review",
                    {"line_number_review": []},
                ),
            )
            self._save_json("stage13_review_items", result["review_items_payload"])
            self._save_json("stage13_review_summary", result["summary"])
            self._save_img(
                "stage13_review_overlay",
                render_stage13_review_overlay(self._ensure_image_loaded(), result["review_items_payload"]),
            )
            return

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

    def stage14_apply_review_decisions(self) -> None:
        """Apply Stage 13 review decisions to produce the corrected trace graph."""
        image_id = Path(self.image_path).name
        result = apply_stage14_review_decisions(
            image_id=image_id,
            graph_payload=self._load_json_artifact("stage12_graph"),
            review_items_payload=self._load_json_artifact("stage13_review_items"),
            decisions_payload=self._load_json_artifact_or_default("stage13_review_decisions", {"decisions": []}),
        )
        self._save_json("stage14_corrected_graph", result["corrected_graph_payload"])
        self._save_json("stage14_review_resolutions", result["review_resolution_payload"])
        self._save_json("stage14_correction_audit", result["correction_audit_payload"])
        self._save_json("stage14_correction_summary", result["summary"])

    # ---------- Stage 15 ----------
    def stage15_recovery_loop(self) -> None:
        from garnet.recovery_loop import run_recovery_stage

        decisions = run_recovery_stage(str(self.out_dir), max_iterations=3)
        self._save_json("stage5_recovery_decisions", decisions)

    def stage15_process_exports(self) -> None:
        """Export process-facing tables from the corrected trace graph."""
        image_id = Path(self.image_path).name
        result = build_stage15_process_exports(
            image_id=image_id,
            corrected_graph_payload=self._load_json_artifact("stage14_corrected_graph"),
        )
        self._save_json("stage15_line_list", result["line_list_payload"])
        self._save_json("stage15_equipment_connectivity", result["equipment_connectivity_payload"])
        self._save_json("stage15_inline_mto", result["inline_mto_payload"])
        self._save_json("stage15_inline_observations", result["inline_observations_payload"])
        self._save_json("stage15_instrument_index", result["instrument_index_payload"])
        self._save_json("stage15_process_export_summary", result["summary"])
        self._save_img(
            "stage15_inline_mto_overlay",
            render_stage15_inline_mto_overlay(self._ensure_image_loaded(), result["inline_mto_payload"]),
        )

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

    def stage12_graph_assembly(self) -> None:
        """Placeholder — non-geometric graph assembly not yet implemented."""
        raise NotImplementedError("stage12_graph_assembly: use --geometric path")

    def _find_equipment_json(self) -> Optional[str]:
        """Find the LabelMe equipment JSON matching the current image."""
        import glob as _glob
        import os as _os

        stem = _os.path.splitext(_os.path.basename(self.image_path))[0]
        json_path = _os.path.join(_os.path.dirname(self.image_path), f"{stem}.json")
        if _os.path.isfile(json_path):
            return json_path
        return None

    def _load_equipment_labelme(self) -> list[dict]:
        """Load LabelMe equipment shapes for the current image.

        Returns all shapes (not just equipment — filtering happens at call site).
        """
        import json as _json
        json_path = self._find_equipment_json()
        if json_path is None:
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        return data.get("shapes", [])

    def _draw_equipment_port_markers(
        self, overlay: np.ndarray, ports: dict[str, list[tuple[int, int, str]]]
    ) -> None:
        """Draw equipment port markers (cyan circles) on the trace overlay.

        Equipment ports come from stage5_connection_ports.json and are
        identified by the 'equip_' prefix in their object ID.
        """
        import cv2 as _cv2

        for obj_id, port_list in ports.items():
            if not obj_id.startswith("equip_"):
                continue
            for port_index, (px, py, direction) in enumerate(port_list, start=1):
                # Cyan filled circle with white outline
                _cv2.circle(overlay, (px, py), 6, (255, 200, 0), -1)
                _cv2.circle(overlay, (px, py), 6, (255, 255, 255), 1)
                _cv2.putText(
                    overlay,
                    f"p{port_index:02d}",
                    (px + 8, py - 8),
                    _cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 120, 120),
                    2,
                )
                # Direction arrow
                dd = {"UP": (0, -18), "DOWN": (0, 18), "LEFT": (-18, 0), "RIGHT": (18, 0)}
                if direction in dd:
                    ax = px + dd[direction][0]
                    ay = py + dd[direction][1]
                    _cv2.arrowedLine(overlay, (px, py), (ax, ay), (255, 200, 0), 2, tipLength=0.3)

    def _draw_equipment_ground_truth(self, overlay: np.ndarray) -> None:
        """Draw LabelMe equipment bboxes on the trace overlay."""
        import json as _json
        import os as _os

        json_path = self._find_equipment_json()
        if json_path is None:
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        # Equipment that gets distinct overlay styling
        equip_labels = {
            "vessel", "pump", "compressor", "blower", "column",
            "heat exchanger", "tank", "reactor", "mixer", "pot",
            "knockout drum", "filter", "cooler", "heater",
            "injection pump",
        }
        equip_color = (220, 120, 0)  # orange
        equip_thickness = 2

        import cv2 as _cv2

        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label.lower() not in equip_labels:
                continue
            pts = shape.get("points", [])
            if len(pts) != 2:
                continue
            x1, y1 = int(round(pts[0][0])), int(round(pts[0][1]))
            x2, y2 = int(round(pts[1][0])), int(round(pts[1][1]))
            _cv2.rectangle(overlay, (x1, y1), (x2, y2), equip_color, equip_thickness)
            _cv2.putText(
                overlay, label.title(),
                (x1, y1 - 8),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.55, equip_color, 2,
            )


def main() -> None:
    parser = argparse.ArgumentParser("P&ID pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--ocr-route", choices=["easyocr", "gemini", "paddleocr", "ocrmac"], default="ocrmac")
    parser.add_argument("--stop-after", type=int, default=2, help="Run up to this stage (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, or 15)")
    parser.add_argument("--geometric", action="store_true", default=False,
                        help="Use geometric line detection in stage 5")
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
