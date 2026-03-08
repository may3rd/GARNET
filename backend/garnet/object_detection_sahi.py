from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DetectionSahiConfig:
    model_type: str = "ultralytics"
    weight_path: str = "yolo_weights/yolo11n_PPCL_640_20250204.pt"
    config_path: str = "datasets/yaml/data.yaml"
    conf_th: float = 0.8
    image_size: int = 640
    overlap_ratio: float = 0.2
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.1


def _build_detection_model(cfg: DetectionSahiConfig) -> Any:
    return AutoDetectionModel.from_pretrained(
        model_type=cfg.model_type,
        model_path=str(BACKEND_DIR / cfg.weight_path),
        config_path=str(BACKEND_DIR / cfg.config_path),
        confidence_threshold=cfg.conf_th,
        image_size=cfg.image_size,
    )


def _draw_overlay(image_bgr: np.ndarray, objects: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_bgr.copy()
    for obj in objects:
        bbox = obj["bbox"]
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            (255, 0, 0),
            2,
        )
    return overlay


def run_object_detection_sahi(
    image_path: str | Path,
    image_id: str = "",
    cfg: DetectionSahiConfig | None = None,
) -> dict[str, Any]:
    config = cfg or DetectionSahiConfig()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image for detection: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection_model = _build_detection_model(config)
    result = get_sliced_prediction(
        image=image_rgb,
        detection_model=detection_model,
        slice_height=config.image_size,
        slice_width=config.image_size,
        overlap_height_ratio=config.overlap_ratio,
        overlap_width_ratio=config.overlap_ratio,
        postprocess_type=config.postprocess_type,
        postprocess_match_metric=config.postprocess_match_metric,
        postprocess_match_threshold=config.postprocess_match_threshold,
        verbose=0,
    )

    objects: list[dict[str, Any]] = []
    for idx, detection in enumerate(result.object_prediction_list, start=1):
        bbox_xyxy = detection.bbox.to_xyxy()
        objects.append(
            {
                "id": f"obj_{idx:06d}",
                "class_name": detection.category.name,
                "confidence": round(float(detection.score.value), 4),
                "bbox": {
                    "x_min": int(bbox_xyxy[0]),
                    "y_min": int(bbox_xyxy[1]),
                    "x_max": int(bbox_xyxy[2]),
                    "y_max": int(bbox_xyxy[3]),
                },
                "source_model": config.model_type,
                "source_weight": config.weight_path,
            }
        )

    class_counts = dict(sorted(Counter(obj["class_name"] for obj in objects).items()))
    summary = {
        "image_id": image_id,
        "pass_type": "sheet",
        "route": config.model_type,
        "object_count": len(objects),
        "class_counts": class_counts,
        "image_size": config.image_size,
        "overlap_ratio": config.overlap_ratio,
        "postprocess_type": config.postprocess_type,
        "postprocess_match_metric": config.postprocess_match_metric,
        "postprocess_match_threshold": config.postprocess_match_threshold,
        "source_model": config.model_type,
        "source_weight": config.weight_path,
    }
    return {
        "objects_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "objects": objects,
        },
        "summary": summary,
        "overlay_image": _draw_overlay(image_bgr, objects),
    }
