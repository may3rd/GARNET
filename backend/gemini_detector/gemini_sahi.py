import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from openai import OpenAI
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list


@dataclass(frozen=True)
class GeminiSahiConfig:
    openrouter_api_key: str
    model_name: str = "google/gemini-3-flash-preview"
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.7


class GeminiSahiInferenceError(RuntimeError):
    """Raised when model inference cannot produce a valid prediction payload."""


class GeminiSahiDetector(DetectionModel):
    required_packages = ["openai", "numpy", "opencv-python", "sahi"]
    """
    SAHI DetectionModel adapter that uses an OpenRouter-hosted Gemini model to produce
    normalized 0..1000 bounding boxes for:
      - instrument_tag (category_id=1)
      - line_number (category_id=2)

    Notes:
    - `model_path` is preserved for SAHI compatibility, but prefer `set_config(...)`.
    - This model is stateful per inference call; after each inference, the latest
      parsed JSON is available via `last_model_output`.
    """

    def __init__(self, *args, **kwargs):
        cfg = kwargs.pop("cfg", None)
        debug_enabled = bool(kwargs.pop("bebug", kwargs.pop("debug", False)))
        self.bebug = debug_enabled
        self.debug = debug_enabled
        self.debug_root = Path(kwargs.pop("debug_root", "results/debug"))
        self.raise_on_error = bool(kwargs.pop("raise_on_error", True))
        self._debug_lock = Lock()
        self._cfg: GeminiSahiConfig | None = cfg
        self._last_model_output: dict[str, Any] | None = None
        self._last_error: str | None = None
        if "load_at_init" not in kwargs and cfg is None and not kwargs.get("model_path"):
            kwargs["load_at_init"] = False
        super().__init__(*args, **kwargs)

    def set_config(self, cfg: GeminiSahiConfig) -> None:
        self._cfg = cfg

    def _cfg_value(self, name: str, default: Any) -> Any:
        cfg = self._cfg
        if cfg is None:
            return default
        return getattr(cfg, name, default)

    @property
    def last_model_output(self) -> dict[str, Any] | None:
        return self._last_model_output

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def load_model(self):
        """
        Initialize the OpenRouter client.

        Back-compat:
        - If config was not provided, treat `model_path` as the API key.
        """
        if self._cfg is None:
            api_key = self.model_path or os.environ.get("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise ValueError("Missing OpenRouter API key (set GeminiSahiConfig or model_path).")
            self._cfg = GeminiSahiConfig(
                openrouter_api_key=api_key,
                model_name=os.environ.get("OPENROUTER_MODEL", "google/gemini-3-flash-preview").strip()
                or "google/gemini-3-flash-preview",
            )

        self.model = OpenAI(
            base_url=self._cfg_value("base_url", "https://openrouter.ai/api/v1"),
            api_key=self._cfg_value("openrouter_api_key", ""),
        )
        self.client = self.model
        self.model_name = self._cfg_value("model_name", "google/gemini-3-flash-preview")
        if not self.category_mapping:
            self.category_mapping = {"1": "instrument_tag", "2": "line_number"}

    def set_model(self, model: Any, **kwargs):
        """
        Set a preloaded OpenAI-compatible client object.
        """
        self.model = model
        self.client = model
        self.model_name = kwargs.get("model_name", getattr(self, "model_name", None) or "google/gemini-3-flash-preview")
        if not self.category_mapping:
            self.category_mapping = {"1": "instrument_tag", "2": "line_number"}

    @staticmethod
    def _get_hw(image: np.ndarray) -> tuple[int, int]:
        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])
        return int(image.shape[0]), int(image.shape[1])

    @staticmethod
    def _bbox_1000_to_pixel(bbox: list[Any], width: int, height: int) -> list[float] | None:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        try:
            xmin, ymin, xmax, ymax = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        return [
            (xmin * width) / 1000.0,
            (ymin * height) / 1000.0,
            (xmax * width) / 1000.0,
            (ymax * height) / 1000.0,
        ]

    def _next_debug_dir(self) -> Path:
        self.debug_root.mkdir(parents=True, exist_ok=True)
        max_idx = -1
        for p in self.debug_root.iterdir():
            if p.is_dir() and p.name.isdigit():
                max_idx = max(max_idx, int(p.name))
        out = self.debug_root / f"{max_idx + 1:02d}"
        out.mkdir(parents=True, exist_ok=False)
        return out

    @staticmethod
    def _image_for_cv_write(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _save_debug_inference(
        self,
        image: np.ndarray,
        model_output: dict[str, Any] | None,
        parsed_predictions: list[dict[str, Any]],
        error: str | None = None,
    ) -> None:
        if not self.bebug:
            return
        try:
            with self._debug_lock:
                out_dir = self._next_debug_dir()
            cv2.imwrite(str(out_dir / "input.jpg"), self._image_for_cv_write(image))
            payload = {
                "error": error,
                "model_output": model_output,
                "parsed_predictions": parsed_predictions,
            }
            (out_dir / "annotations.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as save_err:
            print(f"Failed to save debug inference artifacts: {save_err}")

    def perform_inference(self, image: np.ndarray):
        """
        Takes a slice (numpy array), sends to Gemini, and stores results in
        `self._original_predictions` as expected by SAHI.
        """
        self._last_model_output = None
        self._last_error = None
        self._original_predictions = []
        if self.model is None:
            self.load_model()

        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            msg = "Failed to encode input image to JPEG."
            self._last_error = msg
            self._save_debug_inference(
                image=image,
                model_output=None,
                parsed_predictions=self._original_predictions,
                error=msg,
            )
            if self.raise_on_error:
                raise GeminiSahiInferenceError(msg)
            return
        base64_image = base64.b64encode(buffer).decode("utf-8")

        system_prompt = """
**Role:** You are a Senior Piping Designer and Computer Vision Expert specialized in digitizing P&IDs (Piping and Instrumentation Diagrams).

**Objective:** Extract precise bounding boxes and text content for **Instrument Tags** and **Line Numbers**.

**Visual Definitions (Strict Adherence Required):**

1.  **Instrument Tags:**
    *   **Visual Anchor:** Look for text enclosed within **Circles** (discrete instruments) or **Circles in Squares** (computer functions).
    *   **Text Pattern:** Typically 2-4 letters (e.g., PT, FIT, TIC) followed by a number (e.g., 101, 202A).
    *   **Orientation:** Text is usually horizontal, but the bubble may be attached to lines.
    *   **Exclusion:** Ignore circles without text (e.g., valves, pumps).
    *   **Precise Bbx:** Bounding box must be precisely around the circle.

2.  **Line Numbers:**
    *   **Visual Anchor:** Look for long text strings floating parallel to piping lines. They often have arrows pointing to the pipe.
    *   **Text Pattern:** Must follow a piping spec format. Look for patterns containing: `Size` (e.g., 4", 100mm) + `Service` (e.g., P, WP, HC) + `Sequence` (Numbers) + `Spec` (e.g., A1, CS).
    *   **Orientation:** **CRITICAL:** Line numbers are often rotated 90 degrees (vertical). You must detect vertical text.
    *   **Exclusion:** Ignore text with multiple lines, isolated numbers, or text not matching the piping spec format.

**Coordinate Precision Rules:**
*   Output coordinates in [x_min, y_min, x_max, y_max] that normalized to 1000x1000 pixels.
*   **Tight Fit:** The bounding box must encompass the *entire* text string and the enclosing shape (for tags). Do not include the leader line (the line connecting the bubble to the pipe).
*   **Overlap:** If a Line Number crosses a pipe, capture the text, not the pipe line.

**JSON Syntax & Escaping Railguards (CRITICAL):**
P&ID Line Numbers frequently contain the double-quote symbol (`"`) to denote inches (e.g., 4"). This character breaks JSON structure if not escaped properly.

1.  **Mandatory Escaping:** You MUST escape all double quotes appearing *inside* the text string using a backslash (`\\`).
2.  **Validation Check:** Before outputting, verify that every `"` used as a unit of measurement has a preceding `\\`.

**Examples:**
*   ❌ **WRONG (Invalid JSON):**
    `"text_content": "4"-CUL-25-002018"`
    *(Reason: The quote after 4 closes the string prematurely.)*

*   ✅ **RIGHT (Valid JSON):**
    `"text_content": "4\\"-CUL-25-002018"`

*   ✅ **RIGHT (Alternative):**
    `"text_content": "1 1/2\\"-VG-25-002021"`
    
**Output Format:**
Return ONLY a valid JSON object. Do not include markdown formatting.
{
  "annotations": [
    {
      "category_id": 1,
      "bbox": [x_min, y_min, x_max, y_max],
      "text_content": "PV-1001",
      "confidence": 0.0-1.0
    }
  ],
  "categories": [
    {"id": 1, "name": "instrument_tag"},
    {"id": 2, "name": "line_number"}
  ]
}
        """.strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
Analyze the attached P&ID section.

**Step 1: Scan for Geometry.**
First, visually scan the image for all circular and rectangular bubbles. These are your Instrument Tags.
Second, scan along all thick piping lines to find text strings describing the pipe. These are your Line Numbers.

**Step 2: OCR & Validation.**
Read the text.
*   If text is vertical, read it from bottom-to-top or top-to-bottom as per standard drafting rules.
*   If a Tag is "TI-101", ensure the box covers the circle around it.
*   If a Line Number is "4-P-1001-A1", ensure you capture the full string, not just "1001".

**Step 3: Output.**
Generate the JSON response. **Return the JSON output exactly as specified.**
""".strip(),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    },
                ],
                temperature=float(self._cfg_value("temperature", 0.7)),
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if not isinstance(content, str):
                msg = "Model response content is not a JSON string."
                self._last_error = msg
                self._save_debug_inference(
                    image=image,
                    model_output=None,
                    parsed_predictions=self._original_predictions,
                    error=msg,
                )
                if self.raise_on_error:
                    raise GeminiSahiInferenceError(msg)
                return
            output = json.loads(content)
            self._last_model_output = output

            raw_annotations = output.get("annotations", [])
            raw_predictions = output.get("predictions", [])

            h, w = self._get_hw(image)
            category_map: dict[int, str] = {1: "instrument_tag", 2: "line_number"}
            for cat in output.get("categories", []):
                if isinstance(cat, dict) and "id" in cat and "name" in cat:
                    try:
                        category_map[int(cat["id"])] = str(cat["name"])
                    except (TypeError, ValueError):
                        continue

            if raw_annotations:
                for ann in raw_annotations:
                    if not isinstance(ann, dict):
                        continue
                    pixel_bbox = self._bbox_1000_to_pixel(ann.get("bbox"), width=w, height=h)
                    if pixel_bbox is None:
                        continue
                    try:
                        category_id = int(ann.get("category_id"))
                    except (TypeError, ValueError):
                        continue
                    self._original_predictions.append(
                        {
                            "bbox": pixel_bbox,
                            "label": category_map.get(category_id, "unknown"),
                            "category_id": category_id,
                            "score": float(ann.get("confidence", 0.95)),
                            "text_content": ann.get("text_content"),
                        }
                    )
            else:
                for pred in raw_predictions:
                    if not isinstance(pred, (list, tuple)) or len(pred) < 5:
                        continue
                    ymin, xmin, ymax, xmax, label = pred[:5]
                    pixel_bbox = self._bbox_1000_to_pixel([xmin, ymin, xmax, ymax], width=w, height=h)
                    if pixel_bbox is None:
                        continue
                    label_name = str(label)
                    category_id = 1 if label_name == "instrument_tag" else 2 if label_name == "line_number" else None
                    if category_id is None:
                        continue
                    self._original_predictions.append(
                        {
                            "bbox": pixel_bbox,
                            "label": label_name,
                            "category_id": category_id,
                            "score": 0.95,
                        }
                    )
            self._save_debug_inference(
                image=image,
                model_output=output,
                parsed_predictions=self._original_predictions,
            )

        except Exception as e:
            self._last_error = str(e)
            self._save_debug_inference(
                image=image,
                model_output=self._last_model_output,
                parsed_predictions=self._original_predictions,
                error=self._last_error,
            )
            self._original_predictions = []
            if self.raise_on_error:
                if isinstance(e, GeminiSahiInferenceError):
                    raise
                raise GeminiSahiInferenceError(self._last_error) from e
            print(f"Error during API call: {e}")

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ):
        shift_amount_list = fix_shift_amount_list(shift_amount_list or [[0, 0]])
        full_shape_list = fix_full_shape_list(full_shape_list)
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        object_prediction_list: list[ObjectPrediction] = []
        for res in self._original_predictions or []:
            bbox = res.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
                score = float(res.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if score < self.confidence_threshold:
                continue
            x1, y1 = max(0.0, x1), max(0.0, y1)
            if full_shape is not None:
                x2 = min(float(full_shape[1]), x2)
                y2 = min(float(full_shape[0]), y2)
            if not (x1 < x2 and y1 < y2):
                continue
            category_id = res.get("category_id")
            try:
                category_id = int(category_id)
            except (TypeError, ValueError):
                continue
            category_name = res.get("label") or (self.category_mapping or {}).get(str(category_id)) or "unknown"
            prediction = ObjectPrediction(
                bbox=[x1, y1, x2, y2],
                category_id=category_id,
                category_name=category_name,
                score=score,
                segmentation=None,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            object_prediction_list.append(prediction)

        self._object_prediction_list_per_image = [object_prediction_list]

    @property
    def category_names(self):
        if self.category_mapping:
            try:
                return [v for _, v in sorted(self.category_mapping.items(), key=lambda kv: int(kv[0]))]
            except (TypeError, ValueError):
                return list(self.category_mapping.values())
        return ["instrument_tag", "line_number"]

    @property
    def num_categories(self):
        return len(self.category_names)
