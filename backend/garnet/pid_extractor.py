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
from garnet.gemini_ocr_sahi import GeminiOcrSahiConfig, run_gemini_ocr_sahi
from garnet.instrument_tag_fusion import run_instrument_tag_fusion_stage
from garnet.line_number_fusion import run_line_number_fusion_stage
from garnet.model_defaults import pick_default_weight_file
from garnet.object_detection_sahi import DetectionSahiConfig, run_object_detection_sahi
from garnet.ocrmac_sahi import OcrMacSahiConfig, run_ocrmac_sahi
from garnet.pipe_edges import run_pipe_edge_stage
from garnet.pipe_equipment_attachment import run_pipe_equipment_attachment_stage
from garnet.pipe_graph import run_pipe_graph_stage
from garnet.pipe_graph_qa import run_pipe_graph_qa_stage
from garnet.pipe_edge_connectivity import build_pipe_edge_connectivity
from garnet.pipe_crossings import run_pipe_crossing_stage
from garnet.pipe_junctions import run_pipe_junction_stage
from garnet.pipe_text_attachment import (
    _filter_border_like_edges,
    render_connection_attachment_overlay,
    render_text_attachment_overlay,
    run_pipe_text_attachment_stage,
)
from garnet.paddle_ocr_sahi import PaddleOcrSahiConfig, run_paddle_ocr_sahi
from garnet.pipe_mask import run_pipe_mask_stage
from garnet.pipe_node_clusters import run_pipe_node_cluster_stage
from garnet.pipe_nodes import run_pipe_node_stage
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
    pipe_mask_ocr_padding: int = 1
    pipe_mask_object_inset: int = 1
    pipe_mask_min_component_area: int = 16
    pipe_mask_preserve_ocr_classes: tuple[str, ...] = ()
    pipe_mask_preserve_object_classes: tuple[str, ...] = (
        "arrow",
        "node",
    )
    pipe_seal_horizontal_close_kernel: int = 5
    pipe_seal_vertical_close_kernel: int = 5
    pipe_seal_min_component_area: int = 16
    node_cluster_eps: float = 6.0
    node_cluster_min_samples: int = 1
    min_edge_length_px: int = 2
    crossing_branch_stub_length_px: int = 8
    crossing_branch_merge_angle_tolerance_deg: float = 18.0
    crossing_opposite_angle_tolerance_deg: float = 35.0
    crossing_center_blob_radius_px: int = 4
    crossing_center_blob_threshold: float = 0.5
    crossing_stage4_marker_match_distance_px: float = 24.0
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
    terminal_match_distance_px: float = 48.0
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
        out_dir: str | Path = DEFAULT_OUT,
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

    def _manifest_path(self) -> Path:
        return self.out_dir / "stage_manifest.json"

    def _write_stage_manifest(self) -> None:
        with open(self._manifest_path(), "w") as f:
            json.dump(self.stage_manifest, f, indent=2)
        logger.info(f"saved {self._manifest_path()}")

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

    # ---------- Stage 5 ----------
    def stage5_pipe_mask(self) -> None:
        """Generate provisional pipe-only binary mask by suppressing OCR text and detected objects."""
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

    # ---------- Stage 6 ----------
    def stage6_morphological_sealing(self) -> None:
        """Apply morphological closing to seal gaps in the pipe mask."""
        pipe_mask_path = self.out_dir / "stage5_pipe_mask.png"
        if not pipe_mask_path.exists():
            raise FileNotFoundError(f"Stage 6 requires Stage 5 artifact: {pipe_mask_path}")
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
        """Compute medial-axis skeleton from the sealed pipe mask."""
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

    # ---------- Stage 12 ----------
    def stage12_edge_topology(self) -> None:
        """Classify edge terminals, attach equipment/connections, and build edge connectivity."""
        object_payload = self._load_json_artifact("stage4_objects")
        node_clusters_payload = self._load_json_artifact("stage9_node_clusters")
        edges_payload = self._load_json_artifact("stage10_pipe_edges")
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
        edge_connectivity_result = build_pipe_edge_connectivity(
            edges=edges_payload.get("edges", []),
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
        overlay_edges = [
            {
                **edge,
                "edge_terminals": edge_terminal_map.get(str(edge.get("id", ""))),
            }
            for edge in overlay_edges
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
        combined_text_overlay = render_text_attachment_overlay(
            image_bgr=self._ensure_image_loaded(),
            edges=overlay_edges,
            attachments=
                text_attachment_result["attachments_payload"].get("accepted", [])
                + instrument_tag_attachment_result["attachments_payload"].get("accepted", []),
        )
        self._save_json("stage13_text_attachments", text_attachment_result["attachments_payload"])
        self._save_json("stage13_text_attachment_summary", text_attachment_result["summary"])
        self._save_json("stage13_instrument_tag_attachments", instrument_tag_attachment_result["attachments_payload"])
        self._save_json("stage13_instrument_tag_attachment_summary", instrument_tag_attachment_result["summary"])
        self._save_img("stage13_text_attachment_overlay", combined_text_overlay)

    # ---------- Stage 14 ----------
    def stage14_graph_assembly(self) -> None:
        """Build the final graph from all topology, attachment, and text evidence."""
        node_clusters_payload = self._load_json_artifact("stage9_node_clusters")
        edges_payload = self._load_json_artifact("stage10_pipe_edges")
        crossing_payload = self._load_json_artifact("stage10_crossing_resolution")
        junctions_payload = self._load_json_artifact("stage11_junctions")
        attachment_payload = self._load_json_artifact("stage12_equipment_attachments")
        connection_attachment_payload = self._load_json_artifact("stage12_connection_attachments")
        edge_terminal_payload = self._load_json_artifact("stage12_edge_terminals")
        edge_connections_payload = self._load_json_artifact("stage12_edge_connections")
        text_attachment_payload = self._load_json_artifact("stage13_text_attachments")
        instrument_tag_attachment_payload = self._load_json_artifact("stage13_instrument_tag_attachments")

        graph_result = run_pipe_graph_stage(
            image_id=Path(self.image_path).name,
            node_clusters=node_clusters_payload.get("clusters", []),
            edges=edges_payload.get("edges", []),
            confirmed_junctions=junctions_payload.get("confirmed_junctions", []),
            unresolved_junctions=junctions_payload.get("unresolved_junctions", []),
            crossing_candidates=crossing_payload.get("candidates", []),
            equipment_attachments=attachment_payload.get("accepted", []),
            connection_attachments=connection_attachment_payload.get("accepted", []),
            text_attachments=text_attachment_payload.get("accepted", []),
            instrument_tag_attachments=instrument_tag_attachment_payload.get("accepted", []),
            edge_terminals=edge_terminal_payload.get("edge_terminals", []),
            edge_connections=edge_connections_payload.get("edge_connections", []),
        )
        self._save_json("stage14_graph", graph_result["graph_payload"])
        self._save_json("stage14_graph_summary", graph_result["summary"])

    # ---------- Stage 15 ----------
    def stage15_graph_qa(self) -> None:
        """Run graph QA checks and produce anomaly report and review queue."""
        graph_payload = self._load_json_artifact("stage14_graph")
        qa_result = run_pipe_graph_qa_stage(
            image_id=Path(self.image_path).name,
            graph_payload=graph_payload,
            image_bgr=self._ensure_image_loaded(),
        )
        self._save_json("stage15_graph_anomalies", qa_result["anomaly_report"])
        self._save_img("stage15_graph_components_overlay", qa_result["component_overlay_image"])
        self._save_json("stage15_review_queue", qa_result["review_queue"])
        self._save_json("stage15_graph_qa_summary", qa_result["summary"])


def main() -> None:
    parser = argparse.ArgumentParser("P&ID pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--ocr-route", choices=["easyocr", "gemini", "paddleocr", "ocrmac"], default="ocrmac")
    parser.add_argument("--stop-after", type=int, default=2, help="Run up to this stage (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, or 15)")
    args = parser.parse_args()
    pipe = PIDPipeline(args.image, out_dir=args.out, cfg=PipelineConfig(ocr_route=args.ocr_route))
    pipe.run(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
