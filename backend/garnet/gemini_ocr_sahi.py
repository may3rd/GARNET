from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


PromptBundle = dict[str, str]
InferFn = Callable[[np.ndarray], dict[str, Any]]

PROMPT_DIR = Path(__file__).resolve().parent / "OCR_prompts"
OCR_CLASSES = [
    "equipment_tag",
    "line_number",
    "instrument_tag",
    "valve_tag",
    "utility_label",
    "process_label",
    "note",
    "dimension",
    "title_block",
    "table_text",
    "legend_text",
    "unknown",
]
CATEGORY_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(OCR_CLASSES)}
CATEGORY_ID_TO_NAME = {idx + 1: name for idx, name in enumerate(OCR_CLASSES)}


@dataclass(frozen=True)
class GeminiOcrSahiConfig:
    patch_size: int = 1024
    patch_overlap: int = 128
    low_confidence_threshold: float = 0.3
    prompt_dir: Path = PROMPT_DIR
    model_name: str = "google/gemini-3-flash-preview"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.1
    openrouter_api_key: str | None = None
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.1


def _load_prompt_bundle(prompt_dir: Path | None = None) -> PromptBundle:
    base = Path(prompt_dir or PROMPT_DIR)
    files = {
        "full_page_system": base / "full_page_pass_system_prompt.md",
        "full_page_user": base / "full_page_pass_user_prompt.md",
        "crop_system": base / "crop_pass_system_prompt.md",
        "crop_user": base / "crop_pass_user_prompt.md",
    }
    bundle: PromptBundle = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing OCR prompt file: {path}")
        bundle[key] = path.read_text(encoding="utf-8")
    return bundle


def _build_patch_grid(width: int, height: int, patch_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    stride = max(patch_size - overlap, 1)

    def starts(limit: int) -> list[int]:
        if limit <= patch_size:
            return [0]
        items = [0]
        pos = 0
        while pos + patch_size < limit:
            pos = min(pos + stride, limit - patch_size)
            if pos == items[-1]:
                break
            items.append(pos)
        return items

    boxes: list[tuple[int, int, int, int]] = []
    for y in starts(height):
        for x in starts(width):
            boxes.append((x, y, min(x + patch_size, width), min(y + patch_size, height)))
    return boxes


def _extract_patch(
    image: np.ndarray,
    source_box: tuple[int, int, int, int],
    patch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    x1, y1, x2, y2 = source_box
    crop = image[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]
    if crop_h == 0 or crop_w == 0:
        raise ValueError("Cannot extract an empty patch")

    scale = min(patch_size / crop_w, patch_size / crop_h)
    resized_w = max(1, int(round(crop_w * scale)))
    resized_h = max(1, int(round(crop_h * scale)))

    if cv2 is None:  # pragma: no cover
        raise RuntimeError("cv2 is required for Gemini OCR patch extraction")
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        patch = np.zeros((patch_size, patch_size), dtype=image.dtype)
    else:
        patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
    pad_x = (patch_size - resized_w) // 2
    pad_y = (patch_size - resized_h) // 2
    patch[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

    transform = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "source_box": {
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2,
        },
        "resized_width": resized_w,
        "resized_height": resized_h,
        "patch_size": patch_size,
    }
    return patch, transform


def _map_bbox_to_sheet(
    bbox: dict[str, Any],
    transform: dict[str, Any],
    sheet_width: int,
    sheet_height: int,
) -> dict[str, int]:
    scale = float(transform["scale"])
    pad_x = int(transform["pad_x"])
    pad_y = int(transform["pad_y"])
    source = transform["source_box"]

    def convert(value: float, offset: int, source_offset: int, max_value: int) -> int:
        pixel = int(round((value - offset) / scale)) + source_offset
        return max(0, min(pixel, max_value))

    x_min = convert(float(bbox["x_min"]), pad_x, int(source["x_min"]), sheet_width)
    y_min = convert(float(bbox["y_min"]), pad_y, int(source["y_min"]), sheet_height)
    x_max = convert(float(bbox["x_max"]), pad_x, int(source["x_min"]), sheet_width)
    y_max = convert(float(bbox["y_max"]), pad_y, int(source["y_min"]), sheet_height)
    return {
        "x_min": min(x_min, x_max),
        "y_min": min(y_min, y_max),
        "x_max": max(x_min, x_max),
        "y_max": max(y_min, y_max),
    }


def _bbox_ios(a: dict[str, Any], b: dict[str, Any]) -> float:
    x_left = max(float(a["x_min"]), float(b["x_min"]))
    y_top = max(float(a["y_min"]), float(b["y_min"]))
    x_right = min(float(a["x_max"]), float(b["x_max"]))
    y_bottom = min(float(a["y_max"]), float(b["y_max"]))
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (float(a["x_max"]) - float(a["x_min"])) * (float(a["y_max"]) - float(a["y_min"])))
    area_b = max(0.0, (float(b["x_max"]) - float(b["x_min"])) * (float(b["y_max"]) - float(b["y_min"])))
    smaller = min(area_a, area_b)
    if smaller <= 0:
        return 0.0
    return intersection / smaller


def _choose_best_candidate(
    merged_bbox: dict[str, Any],
    candidates: list[dict[str, Any]],
    match_threshold: float,
) -> dict[str, Any] | None:
    matches = [candidate for candidate in candidates if _bbox_ios(merged_bbox, candidate["bbox"]) >= match_threshold]
    if not matches:
        return None
    return max(matches, key=lambda item: (float(item.get("confidence", 0.0)), _bbox_ios(merged_bbox, item["bbox"])))


def _sheet_region(region: dict[str, Any], bbox: dict[str, int], region_id: str) -> dict[str, Any]:
    return {
        "id": region_id,
        "text": region.get("text", ""),
        "normalized_text": region.get("normalized_text", ""),
        "class": region.get("class", "unknown"),
        "confidence": float(region.get("confidence", 0.0)),
        "bbox": bbox,
        "rotation": float(region.get("rotation", 0)),
        "reading_direction": region.get("reading_direction", "unknown"),
        "legibility": region.get("legibility", "illegible"),
    }


def _draw_overlay(image: np.ndarray, regions: list[dict[str, Any]]) -> np.ndarray:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("cv2 is required for Gemini OCR overlay rendering")
    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    for region in regions:
        bbox = region["bbox"]
        cv2.rectangle(
            overlay,
            (int(bbox["x_min"]), int(bbox["y_min"])),
            (int(bbox["x_max"]), int(bbox["y_max"])),
            (255, 0, 0),
            2,
        )
    return overlay


def _infer_with_openrouter(
    patch_image: np.ndarray,
    *,
    pass_kind: str,
    prompt_bundle: PromptBundle,
    image_id: str,
    cfg: GeminiOcrSahiConfig,
) -> dict[str, Any]:
    from openai import OpenAI  # type: ignore

    api_key = (cfg.openrouter_api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for Gemini OCR route")
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("cv2 is required for Gemini OCR route")

    ok, buffer = cv2.imencode(".jpg", patch_image)
    if not ok:
        raise RuntimeError("Failed to encode Gemini OCR input patch")
    image_data = base64.b64encode(buffer).decode("utf-8")

    if pass_kind == "full_page":
        system_prompt = prompt_bundle["full_page_system"]
        user_prompt = prompt_bundle["full_page_user"]
    else:
        system_prompt = prompt_bundle["crop_system"]
        user_prompt = prompt_bundle["crop_user"]

    client = OpenAI(base_url=cfg.base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=cfg.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            },
        ],
        temperature=cfg.temperature,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not isinstance(content, str):
        raise RuntimeError("Gemini OCR response was not a JSON string")
    return json.loads(content)


class GeminiOcrSahiDetectionModel(DetectionModel):
    required_packages = ["openai", "numpy", "opencv-python", "sahi"]

    def __init__(
        self,
        *,
        cfg: GeminiOcrSahiConfig,
        prompt_bundle: PromptBundle,
        infer_fn: Callable[..., dict[str, Any]],
        image_id: str,
    ) -> None:
        self.cfg = cfg
        self.prompt_bundle = prompt_bundle
        self.infer_fn = infer_fn
        self.image_id = image_id
        self.slice_requests: list[dict[str, Any]] = []
        self.patch_raw: list[dict[str, Any]] = []
        self.crop_raw: list[dict[str, Any]] = []
        self.all_candidates: list[dict[str, Any]] = []
        self.crop_fallback_count = 0
        self._last_slice_payload: dict[str, Any] | None = None
        self._current_slice_shape: tuple[int, int] = (0, 0)
        super().__init__(
            load_at_init=False,
            confidence_threshold=cfg.low_confidence_threshold,
            category_mapping={str(idx): name for idx, name in CATEGORY_ID_TO_NAME.items()},
        )

    def load_model(self) -> None:
        self.model = object()

    def perform_inference(self, image: np.ndarray) -> None:
        self._current_slice_shape = image.shape[:2]
        full_page = self.infer_fn(
            image,
            pass_kind="full_page",
            prompt_bundle=self.prompt_bundle,
            image_id=self.image_id,
        )
        regions = list(full_page.get("text_regions", []))
        crop_response = None
        use_crop_fallback = not regions or max(float(region.get("confidence", 0.0)) for region in regions) < self.cfg.low_confidence_threshold
        if use_crop_fallback:
            crop_response = self.infer_fn(
                image,
                pass_kind="crop",
                prompt_bundle=self.prompt_bundle,
                image_id=self.image_id,
            )
            crop_regions = list(crop_response.get("text_regions", []))
            if crop_regions:
                regions = crop_regions
            self.crop_fallback_count += 1

        self._last_slice_payload = {
            "full_page_response": full_page,
            "crop_response": crop_response,
            "used_crop_fallback": use_crop_fallback,
        }
        self._original_predictions = regions

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ) -> None:
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift_amount = shift_amount_list[0]
        full_shape = full_shape_list[0] if full_shape_list else None
        slice_h, slice_w = self._current_slice_shape
        source_box = {
            "x_min": int(shift_amount[0]),
            "y_min": int(shift_amount[1]),
            "x_max": int(shift_amount[0] + slice_w),
            "y_max": int(shift_amount[1] + slice_h),
        }

        slice_index = len(self.slice_requests)
        self.slice_requests.append({"slice_index": slice_index, "source_box": source_box})
        if self._last_slice_payload is not None:
            self.patch_raw.append({"slice_index": slice_index, "source_box": source_box, "response": self._last_slice_payload["full_page_response"]})
            if self._last_slice_payload["crop_response"] is not None:
                self.crop_raw.append({"slice_index": slice_index, "source_box": source_box, "response": self._last_slice_payload["crop_response"]})

        object_predictions: list[ObjectPrediction] = []
        for region in self._original_predictions:
            bbox = region.get("bbox", {})
            local_bbox = [
                int(round(float(bbox["x_min"]))),
                int(round(float(bbox["y_min"]))),
                int(round(float(bbox["x_max"]))),
                int(round(float(bbox["y_max"]))),
            ]
            class_name = region.get("class", "unknown")
            category_id = CATEGORY_NAME_TO_ID.get(class_name, CATEGORY_NAME_TO_ID["unknown"])
            object_predictions.append(
                ObjectPrediction(
                    bbox=local_bbox,
                    category_id=category_id,
                    category_name=CATEGORY_ID_TO_NAME[category_id],
                    score=float(region.get("confidence", 0.0)),
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
            )
            self.all_candidates.append(
                {
                    "text": region.get("text", ""),
                    "normalized_text": region.get("normalized_text", ""),
                    "class": class_name if class_name in CATEGORY_NAME_TO_ID else "unknown",
                    "confidence": float(region.get("confidence", 0.0)),
                    "bbox": {
                        "x_min": local_bbox[0] + int(shift_amount[0]),
                        "y_min": local_bbox[1] + int(shift_amount[1]),
                        "x_max": local_bbox[2] + int(shift_amount[0]),
                        "y_max": local_bbox[3] + int(shift_amount[1]),
                    },
                    "rotation": float(region.get("rotation", 0)),
                    "reading_direction": region.get("reading_direction", "unknown"),
                    "legibility": region.get("legibility", "illegible"),
                }
            )

        self._object_prediction_list_per_image = [object_predictions]
        self._object_prediction_list = object_predictions


def run_gemini_ocr_sahi(
    image_path: str | Path,
    image_id: str = "",
    cfg: GeminiOcrSahiConfig | None = None,
    infer_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = cfg or GeminiOcrSahiConfig()
    prompt_bundle = _load_prompt_bundle(config.prompt_dir)
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("cv2 is required for Gemini OCR route")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    infer = infer_fn or (
        lambda patch_image, *, pass_kind, prompt_bundle, image_id: _infer_with_openrouter(
            patch_image,
            pass_kind=pass_kind,
            prompt_bundle=prompt_bundle,
            image_id=image_id,
            cfg=config,
        )
    )
    detector = GeminiOcrSahiDetectionModel(
        cfg=config,
        prompt_bundle=prompt_bundle,
        infer_fn=infer,
        image_id=image_id,
    )
    overlap_ratio = config.patch_overlap / config.patch_size if config.patch_size else 0.0
    prediction_result = get_sliced_prediction(
        image=image,
        detection_model=detector,
        slice_height=config.patch_size,
        slice_width=config.patch_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=False,
        postprocess_type=config.postprocess_type,
        postprocess_match_metric=config.postprocess_match_metric,
        postprocess_match_threshold=config.postprocess_match_threshold,
        verbose=0,
        auto_slice_resolution=False,
    )

    sheet_regions: list[dict[str, Any]] = []
    exception_candidates: list[dict[str, Any]] = []
    for object_prediction in prediction_result.object_prediction_list:
        bbox_xyxy = object_prediction.bbox.to_xyxy()
        merged_bbox = {
            "x_min": int(bbox_xyxy[0]),
            "y_min": int(bbox_xyxy[1]),
            "x_max": int(bbox_xyxy[2]),
            "y_max": int(bbox_xyxy[3]),
        }
        chosen = _choose_best_candidate(
            merged_bbox,
            detector.all_candidates,
            match_threshold=config.postprocess_match_threshold,
        )
        if chosen is None:
            chosen = {
                "text": "",
                "normalized_text": "",
                "class": object_prediction.category.name,
                "confidence": float(object_prediction.score.value),
                "rotation": 0,
                "reading_direction": "unknown",
                "legibility": "illegible",
            }
        sheet_region = _sheet_region(chosen, merged_bbox, f"ocr_{len(sheet_regions) + 1:06d}")
        sheet_regions.append(sheet_region)
        if float(sheet_region["confidence"]) < config.low_confidence_threshold:
            exception_candidates.append(
                {
                    "id": sheet_region["id"],
                    "reason_codes": ["low_confidence"],
                    "source_route": "gemini",
                    "bbox": sheet_region["bbox"],
                    "text": sheet_region["text"],
                    "confidence": sheet_region["confidence"],
                }
            )

    regions_payload = {
        "image_id": image_id,
        "pass_type": "sheet",
        "text_regions": sheet_regions,
    }
    summary = {
        "image_id": image_id,
        "pass_type": "sheet",
        "route": "gemini",
        "patch_count": len(detector.slice_requests),
        "raw_detection_count": len(detector.all_candidates),
        "merged_region_count": len(sheet_regions),
        "exception_candidate_count": len(exception_candidates),
        "patch_size": config.patch_size,
        "patch_overlap": config.patch_overlap,
        "crop_fallback_count": detector.crop_fallback_count,
        "low_confidence_threshold": config.low_confidence_threshold,
        "postprocess_type": config.postprocess_type,
        "postprocess_match_metric": config.postprocess_match_metric,
        "postprocess_match_threshold": config.postprocess_match_threshold,
    }
    overlay_image = _draw_overlay(image, sheet_regions)
    return {
        "regions_payload": regions_payload,
        "summary": summary,
        "exception_candidates": exception_candidates,
        "overlay_image": overlay_image,
        "patch_requests": detector.slice_requests,
        "patch_raw": detector.patch_raw,
        "crop_raw": detector.crop_raw,
    }
