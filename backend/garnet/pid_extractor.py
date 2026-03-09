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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from garnet.easyocr_sahi import EasyOcrSahiConfig, run_easyocr_sahi
from garnet.gemini_ocr_sahi import GeminiOcrSahiConfig, run_gemini_ocr_sahi
from garnet.model_defaults import pick_default_weight_file
from garnet.object_detection_sahi import DetectionSahiConfig, run_object_detection_sahi
from garnet.pipe_edges import run_pipe_edge_stage
from garnet.pipe_graph import run_pipe_graph_stage
from garnet.pipe_junctions import run_pipe_junction_stage
from garnet.paddle_ocr_sahi import PaddleOcrSahiConfig, run_paddle_ocr_sahi
from garnet.pipe_mask import run_pipe_mask_stage
from garnet.pipe_node_clusters import run_pipe_node_cluster_stage
from garnet.pipe_nodes import run_pipe_node_stage
from garnet.pipe_seal import run_pipe_seal_stage
from garnet.pipe_skeleton import run_pipe_skeleton_stage

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


@dataclass
class PipelineConfig:
    adaptive_block_size: int = 21
    adaptive_c: int = 5
    blur_kernel: int = 5
    ocr_route: str = "easyocr"
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
    detection_weight_path: str = pick_default_weight_file("ultralytics") or "yolo_weights/yolo26n_PPCL_640_20260227.pt"
    detection_image_size: int = 640
    detection_overlap_ratio: float = 0.2
    detection_postprocess_type: str = "GREEDYNMM"
    detection_postprocess_match_metric: str = "IOS"
    detection_postprocess_match_threshold: float = 0.1
    pipe_mask_ocr_padding: int = 1
    pipe_mask_object_inset: int = 1
    pipe_mask_min_component_area: int = 16
    pipe_seal_horizontal_close_kernel: int = 5
    pipe_seal_vertical_close_kernel: int = 5
    pipe_seal_min_component_area: int = 16
    node_cluster_eps: float = 6.0
    node_cluster_min_samples: int = 1
    min_edge_length_px: int = 2


class PIDPipeline:
    def __init__(
        self,
        image_path: str,
        out_dir: str | Path = DEFAULT_OUT,
        cfg: PipelineConfig | None = None,
        stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **_: Any,
    ) -> None:
        self.image_path = str(image_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or PipelineConfig()
        self.stage_callback = stage_callback

        self.image_bgr: Optional[np.ndarray] = None
        self.stage_manifest: Dict[str, Any] = {}

    # ---------- Stage runner ----------
    def _stage_definitions(self) -> List[Tuple[int, str, Callable[[], None]]]:
        return [
            (1, "stage1_input_normalization", self.stage1_input_normalization),
            (2, "stage2_ocr_discovery", self.stage2_ocr_discovery),
            (4, "stage4_object_detection", self.stage4_object_detection),
            (5, "stage5_pipe_mask", self.stage5_pipe_mask),
            (6, "stage6_morphological_sealing", self.stage6_morphological_sealing),
            (7, "stage7_skeleton_generation", self.stage7_skeleton_generation),
            (8, "stage8_skeleton_node_detection", self.stage8_skeleton_node_detection),
            (9, "stage9_node_clustering", self.stage9_node_clustering),
            (10, "stage10_edge_tracing", self.stage10_edge_tracing),
            (11, "stage11_junction_review", self.stage11_junction_review),
            (12, "stage12_graph_assembly", self.stage12_graph_assembly),
        ]

    def _manifest_path(self) -> Path:
        return self.out_dir / "stage_manifest.json"

    def _write_stage_manifest(self) -> None:
        with open(self._manifest_path(), "w") as f:
            json.dump(self.stage_manifest, f, indent=2)
        logger.info(f"saved {self._manifest_path()}")

    def _notify_stage_callback(self, payload: Dict[str, Any]) -> None:
        if self.stage_callback is not None:
            self.stage_callback(payload)

    def _reset_stage_manifest(self, stop_after: int) -> None:
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

    def _stage_artifacts_since(self, started_at: float) -> List[str]:
        artifacts: List[str] = []
        for path in self.out_dir.iterdir():
            if not path.is_file() or path.name == "stage_manifest.json":
                continue
            if path.stat().st_mtime >= started_at:
                artifacts.append(path.name)
        return sorted(artifacts)

    def _run_stage(self, stage_num: int, stage_name: str, stage_fn: Callable[[], None]) -> None:
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
        try:
            stage_fn()
        except Exception as exc:
            entry["status"] = "failed"
            entry["ended_at"] = time.time()
            entry["duration_sec"] = round(entry["ended_at"] - started_at, 6)
            entry["artifacts"] = self._stage_artifacts_since(started_at)
            entry["error"] = str(exc)
            self._write_stage_manifest()
            self._notify_stage_callback({"event": "stage_failed", "stage": entry.copy(), "manifest": self.stage_manifest})
            raise
        entry["status"] = "completed"
        entry["ended_at"] = time.time()
        entry["duration_sec"] = round(entry["ended_at"] - started_at, 6)
        entry["artifacts"] = self._stage_artifacts_since(started_at)
        self._write_stage_manifest()
        self._notify_stage_callback({"event": "stage_completed", "stage": entry.copy(), "manifest": self.stage_manifest})

    def run(self, stop_after: int = 1) -> None:
        stages = self._stage_definitions()
        valid_stop_after = {num for num, _, _ in stages}
        if stop_after not in valid_stop_after:
            raise ValueError(f"stop_after must be one of {sorted(valid_stop_after)}, got {stop_after}")
        self._reset_stage_manifest(stop_after)
        for stage_num, stage_name, stage_fn in stages:
            if stage_num > stop_after:
                break
            self._run_stage(stage_num, stage_name, stage_fn)

    # ---------- Persistence ----------
    def _save_img(self, name: str, img: np.ndarray) -> None:
        path = self.out_dir / f"{name}.png"
        out = img
        if out.dtype == bool:
            out = out.astype(np.uint8) * 255
        elif out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        if cv2 is not None:
            cv2.imwrite(str(path), out)
        elif Image is not None:
            Image.fromarray(out).save(str(path))
        else:  # pragma: no cover
            raise RuntimeError("No image backend available")
        logger.info(f"saved {path}")

    def _save_json(self, name: str, data: Any) -> None:
        path = self.out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"saved {path}")

    def _load_json_artifact(self, name: str) -> Any:
        path = self.out_dir / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")
        with open(path, "r") as f:
            return json.load(f)

    # ---------- Stage 1 ----------
    def _ensure_image_loaded(self) -> np.ndarray:
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

    # ---------- Stage 5 ----------
    def stage5_pipe_mask(self) -> None:
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
        )
        self._save_img("stage5_pipe_mask", pipe_mask_result["mask_image"])
        self._save_img("stage5_pipe_mask_overlay", pipe_mask_result["overlay_image"])
        self._save_json("stage5_pipe_mask_summary", pipe_mask_result["summary"])

    # ---------- Stage 6 ----------
    def stage6_morphological_sealing(self) -> None:
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
        skeleton_path = self.out_dir / "stage7_pipe_skeleton.png"
        node_clusters_path = self.out_dir / "stage9_node_clusters.json"
        if not skeleton_path.exists() or not node_clusters_path.exists():
            raise FileNotFoundError("Stage 10 requires Stage 7 skeleton and Stage 9 clustered nodes")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 10 edge tracing")

        skeleton_mask = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
        if skeleton_mask is None:
            raise RuntimeError(f"Failed to load Stage 7 skeleton: {skeleton_path}")

        clusters_payload = self._load_json_artifact("stage9_node_clusters")
        edge_result = run_pipe_edge_stage(
            image_bgr=self._ensure_image_loaded(),
            skeleton_mask=skeleton_mask,
            node_clusters=clusters_payload.get("clusters", []),
            image_id=Path(self.image_path).name,
            min_edge_length_px=self.cfg.min_edge_length_px,
        )
        self._save_img("stage10_pipe_edges_overlay", edge_result["overlay_image"])
        self._save_json("stage10_pipe_edges", edge_result["edges_payload"])
        self._save_json("stage10_pipe_edge_summary", edge_result["summary"])

    # ---------- Stage 11 ----------
    def stage11_junction_review(self) -> None:
        skeleton_path = self.out_dir / "stage7_pipe_skeleton.png"
        node_clusters_path = self.out_dir / "stage9_node_clusters.json"
        if not skeleton_path.exists() or not node_clusters_path.exists():
            raise FileNotFoundError("Stage 11 requires Stage 7 skeleton and Stage 9 clustered nodes")
        if cv2 is None:
            raise RuntimeError("cv2 is required for Stage 11 junction review")

        skeleton_mask = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
        if skeleton_mask is None:
            raise RuntimeError(f"Failed to load Stage 7 skeleton: {skeleton_path}")

        clusters_payload = self._load_json_artifact("stage9_node_clusters")
        junction_result = run_pipe_junction_stage(
            image_bgr=self._ensure_image_loaded(),
            skeleton_mask=skeleton_mask,
            node_clusters=clusters_payload.get("clusters", []),
            image_id=Path(self.image_path).name,
        )
        self._save_img("stage11_confirmed_junctions", junction_result["confirmed_junction_image"])
        self._save_img("stage11_unresolved_junctions", junction_result["unresolved_junction_image"])
        self._save_img("stage11_junction_review_overlay", junction_result["overlay_image"])
        self._save_json("stage11_junctions", junction_result["junctions_payload"])
        self._save_json("stage11_junction_review_summary", junction_result["summary"])

    # ---------- Stage 12 ----------
    def stage12_graph_assembly(self) -> None:
        node_clusters_payload = self._load_json_artifact("stage9_node_clusters")
        edges_payload = self._load_json_artifact("stage10_pipe_edges")
        junctions_payload = self._load_json_artifact("stage11_junctions")

        graph_result = run_pipe_graph_stage(
            image_id=Path(self.image_path).name,
            node_clusters=node_clusters_payload.get("clusters", []),
            edges=edges_payload.get("edges", []),
            confirmed_junctions=junctions_payload.get("confirmed_junctions", []),
            unresolved_junctions=junctions_payload.get("unresolved_junctions", []),
        )
        self._save_json("stage12_graph", graph_result["graph_payload"])
        self._save_json("stage12_graph_summary", graph_result["summary"])


def main() -> None:
    parser = argparse.ArgumentParser("P&ID pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--ocr-route", choices=["easyocr", "gemini", "paddleocr"], default="easyocr")
    parser.add_argument("--stop-after", type=int, default=2, help="Run up to this stage (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, or 12)")
    args = parser.parse_args()
    pipe = PIDPipeline(args.image, out_dir=args.out, cfg=PipelineConfig(ocr_route=args.ocr_route))
    pipe.run(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
