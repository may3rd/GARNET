"""
Stage-based P&ID digitizing pipeline.

Current executable pipeline:
- Stage 1: normalize the source image and save binary review artifacts.
- Stage 2: run the selected OCR route and emit text-region artifacts.
- Stage 3: reserved HITL review/input outside this runner, currently used for
  major equipment bounding boxes supplied through LabelMe/frontend review.
- Stage 4: run fixed-baseline object detection, topology-marker routing, line
  number fusion, and instrument-tag fusion.
- Stage 5: build a provisional pipe mask, compute/snap connection ports, trace
  port paths, and iteratively trace tee-branch paths with CV geometry.
  Heavy per-trace and per-branch diagnostic images are saved only when
  debug_artifacts is enabled.
- Stage 6: associate traced paths with ports, inline objects, line numbers,
  instruments, flow arrows, and terminals. Missing line numbers are currently
  filled by a deterministic simulated-HITL placeholder.
- Stage 7: assemble and normalize the geometric trace graph, label page
  connectors, run graph QA, and export the v1 graph payload.
- Stage 8: build the graph/line-number HITL review package.
- Stage 9: apply review decisions, or pass through when no decisions exist.
- Stage 10: export process-facing line list, equipment connectivity, inline
  MTO, inline observations, and instrument index.
- Stage 11: render a connection-pipeline overlay for review.

The stage numbering intentionally skips Stage 3 in this runner because HITL
input is managed outside the automatic CLI flow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from garnet.path_tracer.stage5b_pipeline import Stage5bPipelineMixin
from garnet.review_state import build_stage4_line_numbers_from_review_state
from garnet.trace_associations import (
    apply_stage6_line_number_review,
    build_trace_associations,
    render_trace_association_overlay,
)

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
BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = BACKEND_DIR.parent
STAGE_NUMBERING_NOTE = (
    "Stage numbering intentionally skips Stage 3 in this runner; Stage 3 is "
    "external HITL equipment/geometry review input, while automated execution "
    "continues with Stage 4."
)
EQUIPMENT_LABELS = {
    "vessel",
    "column",
    "pump",
    "compressor",
    "blower",
    "heat exchanger",
    "heat_exchanger",
    "tank",
    "reactor",
    "mixer",
    "pot",
    "knockout drum",
    "knockout_drum",
    "filter",
    "cooler",
    "heater",
    "injection pump",
    "injection_pump",
}


def load_pipeline_env() -> None:
    load_dotenv(ROOT_DIR / ".env", override=False)
    load_dotenv(BACKEND_DIR / ".env", override=False)


def normalize_for_save(img: np.ndarray) -> np.ndarray:
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _signature_mismatches(saved: Any, current: Any, prefix: str = "") -> list[str]:
    if isinstance(saved, dict) and isinstance(current, dict):
        mismatches: list[str] = []
        for key in sorted(set(saved) | set(current)):
            field = f"{prefix}.{key}" if prefix else str(key)
            if key not in saved or key not in current:
                mismatches.append(field)
            else:
                mismatches.extend(_signature_mismatches(saved[key], current[key], field))
        return mismatches
    return [] if saved == current else [prefix]


def _default_detection_weight_path() -> str:
    from garnet.model_defaults import pick_default_weight_file

    return pick_default_weight_file("ultralytics") or "yolo_weights/yolo26n_PPCL_640_20260227.pt"


def run_easyocr_sahi(*args: Any, **kwargs: Any) -> Any:
    from garnet.easyocr_sahi import run_easyocr_sahi as _run

    return _run(*args, **kwargs)


def run_gemini_ocr_sahi(*args: Any, **kwargs: Any) -> Any:
    from garnet.gemini_ocr_sahi import run_gemini_ocr_sahi as _run

    return _run(*args, **kwargs)


def run_paddle_ocr_sahi(*args: Any, **kwargs: Any) -> Any:
    from garnet.paddle_ocr_sahi import run_paddle_ocr_sahi as _run

    return _run(*args, **kwargs)


def run_ocrmac_sahi(*args: Any, **kwargs: Any) -> Any:
    from garnet.ocrmac_sahi import run_ocrmac_sahi as _run

    return _run(*args, **kwargs)


def run_object_detection_sahi(*args: Any, **kwargs: Any) -> Any:
    from garnet.object_detection_sahi import run_object_detection_sahi as _run

    return _run(*args, **kwargs)


def run_line_number_fusion_stage(*args: Any, **kwargs: Any) -> Any:
    from garnet.line_number_fusion import run_line_number_fusion_stage as _run

    return _run(*args, **kwargs)


def run_instrument_tag_fusion_stage(*args: Any, **kwargs: Any) -> Any:
    from garnet.instrument_tag_fusion import run_instrument_tag_fusion_stage as _run

    return _run(*args, **kwargs)


def run_pipe_mask_stage(*args: Any, **kwargs: Any) -> Any:
    from garnet.pipe_mask import run_pipe_mask_stage as _run

    return _run(*args, **kwargs)


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
    ocrmac_recognition_level: str = "fast"
    detection_weight_path: str = field(default_factory=_default_detection_weight_path)
    detection_image_size: int = 640
    detection_overlap_ratio: float = 0.2
    detection_postprocess_type: str = "GREEDYNMM"
    detection_postprocess_match_metric: str = "IOS"
    detection_postprocess_match_threshold: float = 0.1
    line_number_fusion_max_distance_px: float = 80.0
    instrument_tag_fusion_max_distance_px: float = 60.0
    debug_artifacts: bool = False
    pipe_mask_ocr_padding: int = 1
    pipe_mask_object_inset: int = 1
    pipe_mask_inline_object_inset: int = 12
    pipe_mask_min_component_area: int = 16
    pipe_mask_preserve_ocr_classes: tuple[str, ...] = ()
    pipe_mask_preserve_object_classes: tuple[str, ...] = (
        "arrow",
        "node",
    )
    # --- Stage 5b CV pipe-tracing tuning ---
    trace_max_steps: int = 5000
    trace_min_step: int = 5
    trace_straight_min_step: int = 10
    trace_turn_min_step: int = 3
    trace_lookahead_px: int = 30
    trace_raycast_max_snap_shift_px: int = 4
    trace_branch_min_run_px: int = 25
    trace_branch_candidate_sample_step_px: int = 5
    trace_branch_cluster_radius_px: int = 8
    trace_branch_max_iterations: int = 5
    # --- Additional Stage 5b tracer tuning knobs (defaults = prior hardcoded) ---
    trace_centerline_radius_px: int = 8
    trace_side_path_inline_probe_px: int = 60
    trace_raycast_start_px: int = 20
    trace_raycast_max_px: int = 50
    trace_raycast_step_px: int = 2
    trace_anchor_turn_max_gap_px: int = 60
    trace_turn_terminal_scan_px: int = 70
    trace_turn_terminal_scan_far_px: int = 160
    trace_axis_rewind_px: int = 12
    trace_sheet_edge_margin_px: int = 10
    trace_warmup_steps: int = 20
    trace_turn_gap_max_px: int = 60
    trace_branch_side_turn_probe_px: int = 8
    trace_branch_side_turn_terminal_px: int = 90
    trace_turn_probe_px: int = 8
    trace_tee_search_px: int = 8
    trace_terminal_current_margin_px: int = 2
    trace_terminal_current_tag_margin_px: int = 4
    trace_terminal_current_dcs_margin_px: int = 6
    trace_terminal_current_equipment_margin_px: int = 4
    trace_terminal_ahead_margin_px: int = 2
    trace_terminal_ahead_equipment_margin_px: int = 4
    trace_inline_hit_margin_px: int = 2
    trace_inline_exit_margin_px: int = 6
    trace_branch_point_tolerance_px: int = 10
    trace_branch_turn_tolerance_px: int = 8
    trace_association_equipment_port_max_distance_px: float = 16.0
    trace_association_inline_object_max_distance_px: float = 24.0
    trace_association_text_max_distance_px: float = 100.0
    trace_association_instrument_max_distance_px: float = 90.0
    trace_association_arrow_max_distance_px: float = 45.0


class PIDPipeline(Stage5bPipelineMixin):
    def __init__(
        self,
        image_path: str,
        output_dir: str | Path = DEFAULT_OUT,
        cfg: PipelineConfig | None = None,
        stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        document_id: str | None = None,
    ) -> None:
        self.image_path = str(image_path)
        self.out_dir = Path(output_dir)
        self.cfg = cfg or PipelineConfig()
        self.stage_callback = stage_callback
        self.document_id = str(document_id).strip() if document_id and str(document_id).strip() else None

        self.image_bgr: Optional[np.ndarray] = None
        self.stage_manifest: Dict[str, Any] = {}
        self._current_stage_artifacts: list[str] = []

    def _image_id(self) -> str:
        return self.document_id or Path(self.image_path).name

    # ---------- Stage runner ----------
    def _stage_definitions(self) -> List[Tuple[int, str, Callable[[], None]]]:
        """Return the ordered stage list executed by the pipeline."""
        return [
            (1, "stage1_input_normalization", self.stage1_input_normalization),
            (2, "stage2_ocr_discovery", self.stage2_ocr_discovery),
            (4, "stage4_object_detection", self.stage4_object_detection),
            (4, "stage4_line_number_fusion", self.stage4_line_number_fusion),
            (4, "stage4_instrument_tag_fusion", self.stage4_instrument_tag_fusion),
            (5, "stage5_pipe_mask", self.stage5_pipe_mask),
            (5, "stage5b_pipe_trace", self.stage5b_pipe_trace),
            (6, "stage6_trace_associations", self.stage6_trace_associations),
            (7, "stage7_geometric_graph_assembly", self.stage7_geometric_graph_assembly),
            (7, "stage7c_page_connector_labeling", self.stage7c_page_connector_labeling),
            (7, "stage7b_graph_export", self.stage7b_graph_export),
            (8, "stage8_graph_qa", self.stage8_graph_qa),
            (9, "stage9_apply_review_decisions", self.stage9_apply_review_decisions),
            (10, "stage10_process_exports", self.stage10_process_exports),
            (11, "stage11_connection_overlay", self.stage11_connection_overlay),
        ]

    def _manifest_path(self) -> Path:
        return self.out_dir / "stage_manifest.json"

    def _write_stage_manifest(self) -> None:
        path = self._manifest_path()
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.stage_manifest, f, indent=2)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        logger.info(f"saved {path}")

    def _notify_stage_callback(self, payload: Dict[str, Any]) -> None:
        """Forward stage lifecycle events to the optional callback."""
        if self.stage_callback is not None:
            self.stage_callback(payload)

    def _build_run_signature(self) -> Dict[str, Any]:
        input_path = Path(self.image_path)
        weight_path = Path(self.cfg.detection_weight_path).expanduser()
        if not weight_path.is_absolute():
            weight_path = BACKEND_DIR / weight_path
        return {
            "input": {
                "sha256": _sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
            },
            "config": json.loads(json.dumps(asdict(self.cfg), sort_keys=True)),
            "detection_weight": {
                "path": self.cfg.detection_weight_path,
                "sha256": _sha256_file(weight_path) if weight_path.is_file() else None,
            },
            "document_id": self._image_id(),
            "git_revision": _git_revision(),
        }

    def _validate_resume_manifest(
        self,
        manifest: Dict[str, Any],
        current_signature: Dict[str, Any],
        stages: List[Tuple[int, str, Callable[[], None]]],
    ) -> set[str]:
        if manifest.get("manifest_version") != 2 or not isinstance(manifest.get("run_signature"), dict):
            raise ValueError("Cannot resume: manifest version 2 with run_signature is required")
        compared_fields = ("input", "config", "detection_weight", "document_id")
        mismatches = _signature_mismatches(
            {field: manifest["run_signature"].get(field) for field in compared_fields},
            {field: current_signature.get(field) for field in compared_fields},
        )
        if mismatches:
            raise ValueError(f"Cannot resume: run signature mismatch: {', '.join(mismatches)}")

        latest_by_name: dict[str, Dict[str, Any]] = {}
        for entry in manifest.get("stages", []):
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                latest_by_name[entry["name"]] = entry

        completed: set[str] = set()
        found_gap = False
        for _stage_num, stage_name, _stage_fn in stages:
            entry = latest_by_name.get(stage_name)
            if not entry or entry.get("status") != "completed":
                found_gap = True
                continue
            if found_gap:
                raise ValueError(f"Cannot resume: completed stages are not contiguous at {stage_name}")
            artifacts = entry.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError(f"Cannot resume: completed stage {stage_name} has no registered artifacts")
            for artifact in artifacts:
                if (
                    not isinstance(artifact, str)
                    or not artifact
                    or artifact in {".", ".."}
                    or Path(artifact).name != artifact
                ):
                    raise ValueError(f"Cannot resume: invalid artifact name for {stage_name}: {artifact!r}")
                artifact_path = self.out_dir / artifact
                if not artifact_path.exists():
                    raise ValueError(f"Cannot resume: completed artifact is missing: {artifact}")
                if artifact_path.is_symlink() or not artifact_path.is_file():
                    raise ValueError(f"Cannot resume: completed artifact is not a regular file: {artifact}")
            completed.add(stage_name)
        return completed

    def _reset_stage_manifest(self, stop_after: int, run_signature: Dict[str, Any]) -> None:
        """Initialize a fresh stage manifest for the current run."""
        self.stage_manifest = {
            "manifest_version": 2,
            "run_signature": run_signature,
            "image_path": self.image_path,
            "out_dir": str(self.out_dir),
            "stop_after": stop_after,
            "ocr_route": self.cfg.ocr_route,
            "detection_weight_path": self.cfg.detection_weight_path,
            "debug_artifacts": self.cfg.debug_artifacts,
            "stage_numbering_note": STAGE_NUMBERING_NOTE,
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
        # Re-running a stage (rework/resume) must update the existing manifest
        # entry rather than append a duplicate, so the stage list stays clean.
        existing = next((item for item in self.stage_manifest["stages"] if item.get("name") == stage_name), None)
        if existing is not None:
            existing.clear()
            existing.update(entry)
            entry = existing
        else:
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

    def run(self, stop_after: int = 11, resume: bool = False) -> None:
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
        current_signature = self._build_run_signature()
        if resume and manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest.setdefault("stages", [])
            manifest_image_path = manifest.get("image_path")
            if manifest_image_path not in (None, self.image_path):
                raise ValueError(
                    f"Cannot resume from manifest for different image: {manifest_image_path} != {self.image_path}"
                )
            manifest_out_dir = manifest.get("out_dir")
            if manifest_out_dir not in (None, str(self.out_dir)):
                raise ValueError(
                    f"Cannot resume from manifest for different output directory: {manifest_out_dir} != {self.out_dir}"
                )
            completed_stage_names = self._validate_resume_manifest(manifest, current_signature, stages)
            self.stage_manifest = manifest
            self.stage_manifest["image_path"] = self.image_path
            self.stage_manifest["out_dir"] = str(self.out_dir)
            self.stage_manifest["stop_after"] = stop_after
            self.stage_manifest["ocr_route"] = self.cfg.ocr_route
            self.stage_manifest["detection_weight_path"] = self.cfg.detection_weight_path
            self.stage_manifest["debug_artifacts"] = self.cfg.debug_artifacts
            self.stage_manifest["stage_numbering_note"] = STAGE_NUMBERING_NOTE
            if completed_stage_names:
                last_completed_stage = max(
                    completed_stage_names,
                    key=lambda name: next(index for index, (_num, stage_name, _fn) in enumerate(stages) if stage_name == name),
                )
                logger.info("Resuming pipeline from %s after %s", manifest_path, last_completed_stage)
            self._write_stage_manifest()
        else:
            self._reset_stage_manifest(stop_after, current_signature)

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
            if not cv2.imwrite(str(path), out):
                raise OSError(f"Failed to save image: {path}")
        elif Image is not None:
            Image.fromarray(out).save(str(path))
        else:  # pragma: no cover
            raise RuntimeError("No image backend available")
        self._register_artifact(path.name)
        logger.info(f"saved {path}")

    def _save_json(self, name: str, data: Any) -> None:
        """Persist a JSON artifact to the output directory and register it."""
        path = self.out_dir / f"{name}.json"
        tmp_path = path.with_name(f".{path.name}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
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

    def _load_json_artifact_compat(self, name: str, legacy_name: str) -> Any:
        """Load a renamed artifact, falling back to the pre-renumbered name."""
        path = self.out_dir / f"{name}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return self._load_json_artifact(legacy_name)

    def _load_json_artifact_or_default_compat(self, name: str, legacy_name: str, default: Any) -> Any:
        path = self.out_dir / f"{name}.json"
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return self._load_json_artifact_or_default(legacy_name, default)

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
            from garnet.easyocr_sahi import EasyOcrSahiConfig

            ocr_result = run_easyocr_sahi(
                stage1_input,
                image_id=self._image_id(),
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
            from garnet.gemini_ocr_sahi import GeminiOcrSahiConfig

            ocr_result = run_gemini_ocr_sahi(
                stage1_input,
                image_id=self._image_id(),
                cfg=GeminiOcrSahiConfig(
                    postprocess_match_threshold=self.cfg.gemini_postprocess_match_threshold,
                ),
            )
        elif self.cfg.ocr_route == "paddleocr":
            from garnet.paddle_ocr_sahi import PaddleOcrSahiConfig

            ocr_result = run_paddle_ocr_sahi(
                stage1_input,
                image_id=self._image_id(),
                cfg=PaddleOcrSahiConfig(
                    slice_height=self.cfg.ocr_slice_height,
                    slice_width=self.cfg.ocr_slice_width,
                    overlap_height_ratio=self.cfg.ocr_overlap_height_ratio,
                    overlap_width_ratio=self.cfg.ocr_overlap_width_ratio,
                ),
            )
        elif self.cfg.ocr_route == "ocrmac":
            from garnet.ocrmac_sahi import OcrMacSahiConfig

            ocr_result = run_ocrmac_sahi(
                stage1_input,
                image_id=self._image_id(),
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
        from garnet.object_detection_sahi import DetectionSahiConfig
        from garnet.topology_markers import run_topology_marker_router

        detection_result = run_object_detection_sahi(
            self.image_path,
            image_id=self._image_id(),
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
            image_id=self._image_id(),
            objects=detection_result["objects_payload"].get("objects", []),
        )
        self._save_json("stage4_topology_markers", topology_marker_result["topology_markers_payload"])
        self._save_json("stage4_topology_marker_summary", topology_marker_result["summary"])

    def stage4_line_number_fusion(self) -> None:
        """Fuse OCR text regions with detected objects to identify pipe line numbers."""
        object_payload = self._load_json_artifact("stage4_objects")
        ocr_payload = self._load_json_artifact("stage2_ocr_regions")
        fusion_result = run_line_number_fusion_stage(
            image_id=self._image_id(),
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
            image_id=self._image_id(),
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
            image_id=self._image_id(),
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

    # ---------- Stage 6+: trace association, graph assembly, QA, review, exports ----------
    def stage6_trace_associations(self) -> None:
        """Attach semantic evidence to Stage 5b traced pipe paths."""
        image_id = self._image_id()
        reviewed_line_payload = build_stage4_line_numbers_from_review_state(
            self.out_dir,
            {"image_path": self.image_path},
        )
        if reviewed_line_payload is not None:
            self._save_json("stage4_line_numbers", reviewed_line_payload)
        object_payload = self._load_json_artifact("stage4_objects")
        line_payload = self._load_json_artifact_or_default("stage4_line_numbers", {"line_numbers": []})
        instrument_payload = self._load_json_artifact_or_default("stage4_instrument_tags", {"instrument_tags": []})
        trace_payload = self._load_json_artifact("stage5b_trace_results")
        branch_payload = self._load_json_artifact_or_default("stage5b_branch_trace_results", {"branches": {}})
        ports_payload = self._load_json_artifact_or_default("stage5_connection_ports", {})

        result = build_trace_associations(
            image_id=image_id,
            objects=object_payload.get("objects", []),
            trace_payload=trace_payload,
            branch_payload=branch_payload,
            ports_payload=ports_payload,
            line_numbers=line_payload.get("line_numbers", []),
            instrument_tags=instrument_payload.get("instrument_tags", []),
            equipment_port_max_distance_px=self.cfg.trace_association_equipment_port_max_distance_px,
            inline_object_max_distance_px=self.cfg.trace_association_inline_object_max_distance_px,
            text_max_distance_px=self.cfg.trace_association_text_max_distance_px,
            instrument_max_distance_px=self.cfg.trace_association_instrument_max_distance_px,
            arrow_max_distance_px=self.cfg.trace_association_arrow_max_distance_px,
        )

        self._save_json("stage6_trace_associations", result["trace_associations_payload"])
        self._save_json("stage6_trace_association_summary", result["trace_association_summary"])
        self._save_json("stage6_line_number_review", result["line_number_review_payload"])
        self._save_json("stage6_line_number_review_summary", result["line_number_review_summary"])
        self._save_img(
            "stage6_trace_association_overlay",
            render_trace_association_overlay(
                self._ensure_image_loaded(),
                result["trace_edges"],
                result["associations"],
            ),
        )


    def stage7_geometric_graph_assembly(self) -> None:
        """Build and QA the geometric graph directly from Stage 6 traced paths."""
        from garnet.trace_graph_builder import (
            build_trace_graph_from_stage11 as build_trace_graph_from_stage6,
            render_stage12_graph_overlay as render_stage7_graph_overlay,
        )
        from garnet.trace_graph_qa import run_stage12_trace_graph_qa as run_stage7_trace_graph_qa

        stage6_path = self.out_dir / "stage6_trace_associations.json"
        stage11_path = self.out_dir / "stage11_trace_associations.json"
        if not stage6_path.exists() and not stage11_path.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {stage6_path} "
                f"(legacy fallback also missing: {stage11_path})"
            )

        image_id = self._image_id()
        stage6_payload = self._load_json_artifact_compat(
            "stage6_trace_associations",
            "stage11_trace_associations",
        )
        stage6_payload = apply_stage6_line_number_review(
            stage6_payload,
            self._load_json_artifact_or_default("stage6_line_number_review", {}),
        )
        trace_graph_result = build_trace_graph_from_stage6(stage6_payload, image_id=image_id)
        self._save_json("stage7_graph", trace_graph_result["graph_payload"])
        self._save_json("stage7_graph_summary", trace_graph_result["summary"])
        self._save_json("stage7_trace_edge_nodes", trace_graph_result["trace_edge_nodes_payload"])
        self._save_json("stage7_review_queue", trace_graph_result["review_queue_payload"])
        self._save_json("stage7_review_queue_summary", trace_graph_result["review_queue_summary"])
        self._save_json("stage7_graph_normalization", trace_graph_result["normalization_payload"])
        self._save_json("stage7_graph_normalization_summary", trace_graph_result["normalization_summary"])
        self._save_img(
            "stage7_graph_overlay",
            render_stage7_graph_overlay(self._ensure_image_loaded(), trace_graph_result["graph_payload"]),
        )
        trace_graph_qa_result = run_stage7_trace_graph_qa(
            image_id=image_id,
            graph_payload=trace_graph_result["graph_payload"],
            image_bgr=self._ensure_image_loaded(),
        )
        self._save_json("stage7_graph_qa", trace_graph_qa_result["qa_payload"])
        self._save_json("stage7_graph_qa_summary", trace_graph_qa_result["summary"])
        self._save_img("stage7_graph_qa_overlay", trace_graph_qa_result["overlay_image"])

    def stage7c_page_connector_labeling(self) -> None:
        """Attach nearby OCR labels to accepted page-connection objects."""
        from garnet.page_connector import find_nearby_text, line_number_by_trace_id, select_connector_metadata

        accepted = [
            obj
            for obj in self._load_json_artifact("stage4_objects").get("objects", [])
            if obj.get("class_name") == "page connection" and obj.get("review_state") != "rejected"
        ]
        ocr_payload = self._load_json_artifact("stage2_ocr_regions")
        text_regions = ocr_payload.get("text_regions", [])
        # Fall back to the line number attached to the traced pipe that terminates
        # at each connector when no line-number OCR label sits near the symbol.
        trace_assoc = self._load_json_artifact_or_default("stage6_trace_associations", {})
        line_number_by_trace = line_number_by_trace_id(trace_assoc)
        all_labels = []
        for obj in accepted:
            bbox = obj.get("bbox", {})
            labels = find_nearby_text(bbox, text_regions, max_distance_px=80.0)
            obj_id = obj.get("id") or obj.get("det_id")
            all_labels.append(
                {
                    "object_id": obj_id,
                    "labels": labels,
                    **select_connector_metadata(
                        labels,
                        fallback_line_number=line_number_by_trace.get(str(obj_id), ""),
                    ),
                }
            )
        self._save_json("stage7_page_connector_labels", {"connectors": all_labels})
        self._save_json(
            "stage7_page_connector_labels_summary",
            {
                "total_connectors": len(accepted),
                "total_labels": sum(len(l["labels"]) for l in all_labels),
            },
        )

    def stage7b_graph_export(self) -> None:
        """Export the Stage 7 graph into the API/frontend graph-v1 payload."""
        from garnet.graph_export_adapter import build_graph_v1_payload

        graph_payload = self._load_json_artifact_compat("stage7_graph", "stage12_graph")
        object_payload = self._load_json_artifact("stage4_objects")
        line_number_payload = self._load_json_artifact("stage4_line_numbers")
        instrument_tag_payload = self._load_json_artifact("stage4_instrument_tags")
        page_connector_labels_payload = self._load_json_artifact_or_default_compat(
            "stage7_page_connector_labels",
            "stage12_page_connector_labels",
            {"connectors": []},
        )
        normalization_summary = self._load_json_artifact("stage1_normalization_summary")
        graph_v1_payload = build_graph_v1_payload(
            stage12_graph=graph_payload,
            objects_payload=object_payload,
            line_numbers_payload=line_number_payload,
            instrument_tags_payload=instrument_tag_payload,
            page_connector_labels_payload=page_connector_labels_payload,
            image_dimensions=normalization_summary.get("dimensions", {}),
        )
        self._save_json("stage7b_graph_v1", graph_v1_payload)

    # ---------- Stage 8 + 9 ----------
    def stage8_graph_qa(self) -> None:
        """Build the HITL review package from graph QA and line-number review state."""
        from garnet.stage8_review_package import build_stage8_review_package, render_stage8_review_overlay

        graph_payload = self._load_json_artifact_compat("stage7_graph", "stage12_graph")
        stage7_qa_path = self.out_dir / "stage7_graph_qa.json"
        stage12_qa_path = self.out_dir / "stage12_graph_qa.json"
        if not stage7_qa_path.exists() and not stage12_qa_path.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {stage7_qa_path} "
                f"(legacy fallback also missing: {stage12_qa_path})"
            )

        image_id = self._image_id()
        result = build_stage8_review_package(
            image_id=image_id,
            graph_payload=graph_payload,
            stage7_qa_payload=self._load_json_artifact_compat("stage7_graph_qa", "stage12_graph_qa"),
            stage7_review_queue_payload=self._load_json_artifact_or_default_compat("stage7_review_queue", "stage12_review_queue", {"review_queue": []}),
            stage6_line_number_review_payload=self._load_json_artifact_or_default_compat(
                "stage6_line_number_review",
                "stage11_line_number_review",
                {"line_number_review": []},
            ),
        )
        self._save_json("stage8_review_items", result["review_items_payload"])
        self._save_json("stage8_review_summary", result["summary"])
        self._save_img(
            "stage8_review_overlay",
            render_stage8_review_overlay(self._ensure_image_loaded(), result["review_items_payload"]),
        )


    def stage9_apply_review_decisions(self) -> None:
        """Apply Stage 8 review decisions to produce the corrected trace graph.

        When no decisions are present, this stage keeps the graph unchanged and
        records a pass-through correction audit.
        """
        from garnet.stage9_review_decisions import apply_stage9_review_decisions

        image_id = self._image_id()
        result = apply_stage9_review_decisions(
            image_id=image_id,
            graph_payload=self._load_json_artifact_compat("stage7_graph", "stage12_graph"),
            review_items_payload=self._load_json_artifact_compat("stage8_review_items", "stage13_review_items"),
            decisions_payload=self._load_json_artifact_or_default_compat("stage8_review_decisions", "stage13_review_decisions", {"decisions": []}),
        )
        self._save_json("stage9_corrected_graph", result["corrected_graph_payload"])
        self._save_json("stage9_review_resolutions", result["review_resolution_payload"])
        self._save_json("stage9_correction_audit", result["correction_audit_payload"])
        self._save_json("stage9_correction_summary", result["summary"])

    # ---------- Stage 10 ----------

    def stage10_process_exports(self) -> None:
        """Export process-facing tables from the corrected trace graph.

        Outputs include line list, equipment connectivity, unique physical
        inline-object MTO, inline observations, instrument index, and review
        overlays for inline objects and associated line numbers.
        """
        from garnet.stage10_process_exports import (
            build_stage10_process_exports,
            render_stage10_inline_mto_overlay,
            render_stage10_line_number_overlay,
        )

        image_id = self._image_id()
        result = build_stage10_process_exports(
            image_id=image_id,
            corrected_graph_payload=self._load_json_artifact("stage9_corrected_graph"),
        )
        self._save_json("stage10_line_list", result["line_list_payload"])
        self._save_json("stage10_equipment_connectivity", result["equipment_connectivity_payload"])
        self._save_json("stage10_inline_mto", result["inline_mto_payload"])
        self._save_json("stage10_inline_observations", result["inline_observations_payload"])
        self._save_json("stage10_instrument_index", result["instrument_index_payload"])
        self._save_json("stage10_process_export_summary", result["summary"])
        self._save_img(
            "stage10_inline_mto_overlay",
            render_stage10_inline_mto_overlay(self._ensure_image_loaded(), result["inline_mto_payload"]),
        )
        self._save_img(
            "stage10_line_number_overlay",
            render_stage10_line_number_overlay(
                self._ensure_image_loaded(),
                result["line_list_payload"],
                self._load_json_artifact("stage9_corrected_graph"),
            ),
        )

    # ---------- Stage 11 ----------
    def stage11_connection_overlay(self) -> None:
        """Render the final current-graph overlay."""
        from garnet.trace_graph_builder import render_stage12_graph_overlay

        self._save_img(
            "stage11_connection_pipeline_overlay",
            render_stage12_graph_overlay(
                self._ensure_image_loaded(),
                self._load_json_artifact("stage7_graph"),
            ),
        )


    def _find_equipment_json(self) -> Optional[str]:
        """Find the LabelMe equipment JSON matching the current image."""
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

    def _normalize_equipment_object(
        self,
        item: dict[str, Any],
        *,
        fallback_id: str,
        source: str,
    ) -> dict[str, Any] | None:
        label = str(item.get("class_name") or item.get("label") or "").strip()
        if label.lower() not in EQUIPMENT_LABELS:
            return None
        bbox = item.get("bbox")
        if not isinstance(bbox, dict):
            pts = item.get("points", [])
            if len(pts) != 2:
                return None
            x1 = int(round(float(pts[0][0])))
            y1 = int(round(float(pts[0][1])))
            x2 = int(round(float(pts[1][0])))
            y2 = int(round(float(pts[1][1])))
            bbox = {
                "x_min": min(x1, x2),
                "y_min": min(y1, y2),
                "x_max": max(x1, x2),
                "y_max": max(y1, y2),
            }
        try:
            norm_bbox = {
                "x_min": int(round(float(bbox["x_min"]))),
                "y_min": int(round(float(bbox["y_min"]))),
                "x_max": int(round(float(bbox["x_max"]))),
                "y_max": int(round(float(bbox["y_max"]))),
            }
        except (KeyError, TypeError, ValueError):
            return None
        if norm_bbox["x_max"] <= norm_bbox["x_min"] or norm_bbox["y_max"] <= norm_bbox["y_min"]:
            return None
        result = dict(item)
        result.update(
            {
                "id": str(item.get("id") or fallback_id),
                "class_name": label,
                "bbox": norm_bbox,
                "source": str(item.get("source") or source),
            }
        )
        result.setdefault("review_state", "accepted" if source == "hitl" else "fallback")
        return result

    def _load_stage3_equipment_bboxes(self) -> list[dict[str, Any]]:
        payload = self._load_json_artifact_or_default("stage3_equipment_bboxes", {})
        raw_items = payload.get("equipment", []) if isinstance(payload, dict) else []
        equipment: list[dict[str, Any]] = []
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_equipment_object(
                item,
                fallback_id=f"equip_{index:03d}",
                source="hitl",
            )
            if normalized is not None:
                equipment.append(normalized)
        return equipment

    def _load_labelme_equipment_bboxes(self) -> list[dict[str, Any]]:
        equipment: list[dict[str, Any]] = []
        for index, shape in enumerate(self._load_equipment_labelme()):
            normalized = self._normalize_equipment_object(
                shape,
                fallback_id=f"equip_{index}_{str(shape.get('label', 'equipment')).replace(' ', '_')}",
                source="labelme_fallback",
            )
            if normalized is not None:
                equipment.append(normalized)
        return equipment

    def _load_equipment_bboxes_for_stage5b(self) -> list[dict[str, Any]]:
        stage3_equipment = self._load_stage3_equipment_bboxes()
        if stage3_equipment:
            logger.info("Loaded %d Stage 3 equipment bboxes", len(stage3_equipment))
            return stage3_equipment
        labelme_equipment = self._load_labelme_equipment_bboxes()
        if labelme_equipment:
            logger.info("Loaded %d LabelMe equipment bboxes as Stage 3 fallback", len(labelme_equipment))
        return labelme_equipment

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
        """Draw Stage 3 equipment bboxes, falling back to LabelMe bboxes."""
        equipment = self._load_equipment_bboxes_for_stage5b()
        if not equipment:
            return
        equip_color = (220, 120, 0)  # orange
        equip_thickness = 2

        import cv2 as _cv2

        for item in equipment:
            label = str(item.get("class_name", "equipment")).strip()
            bbox = item["bbox"]
            x1, y1 = bbox["x_min"], bbox["y_min"]
            x2, y2 = bbox["x_max"], bbox["y_max"]
            _cv2.rectangle(overlay, (x1, y1), (x2, y2), equip_color, equip_thickness)
            _cv2.putText(
                overlay, label.title(),
                (x1, y1 - 8),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.55, equip_color, 2,
            )


def _resolve_cli_weight_file(weight_file: str) -> str:
    """Resolve and validate the optional Stage 4 YOLO weight path."""
    if not weight_file:
        return PipelineConfig().detection_weight_path
    raw_path = Path(weight_file).expanduser()
    resolved_path = raw_path if raw_path.is_absolute() else BACKEND_DIR / raw_path
    if not resolved_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_file}")
    try:
        return str(resolved_path.relative_to(BACKEND_DIR))
    except ValueError:
        return str(resolved_path)


def main() -> None:
    load_pipeline_env()
    parser = argparse.ArgumentParser("P&ID pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--ocr-route", choices=["easyocr", "gemini", "paddleocr", "ocrmac"], default="ocrmac")
    parser.add_argument(
        "--weight-file",
        default="",
        help=(
            "Stage 4 YOLO weight file. Relative paths are resolved from the backend directory, "
            "for example yolo_weights/my_model.pt. Defaults to the configured yolo_weights model."
        ),
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=11,
        help=(
            "Run up to this automated stage. Valid values are "
            "1, 2, 4, 5, 6, 7, 8, 9, 10, or 11. Stage 3 is external HITL input. Default: 11 (full pipeline)."
        ),
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        default=False,
        help="Save heavy diagnostic artifacts such as Stage 5b per-trace images and branch-candidate iteration overlays.",
    )
    args = parser.parse_args()
    detection_weight_path = _resolve_cli_weight_file(args.weight_file)
    pipe = PIDPipeline(
        args.image,
        output_dir=args.out,
        cfg=PipelineConfig(
            ocr_route=args.ocr_route,
            detection_weight_path=detection_weight_path,
            debug_artifacts=args.debug_artifacts,
        ),
    )
    pipe.run(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
