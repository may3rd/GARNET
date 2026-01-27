"""
GARNET API Service - Pure API backend for React frontend

This is the API-only backend service. Run with:
    uvicorn api:app --reload --port 8001

The React frontend should run separately on port 5173 (dev) or be built for production.
"""

from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from sahi import AutoDetectionModel, DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import crop_object_predictions

import torch
import easyocr

import cv2
import numpy as np
import json
import math
import glob
import os
import datetime
import pandas as pd
import logging
import uuid
import inspect

from garnet import utils
import garnet.Settings as Settings
import yaml

# Configure logging.
# When running under `uvicorn`, its logging config can override/disable other loggers.
# To ensure logs show in the terminal, write through `uvicorn.error` and attach our
# file handler there. When not under uvicorn, fall back to a normal StreamHandler.
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "garnet.log")


def _ensure_logger_has_file_handler(target: logging.Logger) -> None:
    if any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_file
        for h in target.handlers
    ):
        return
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    target.addHandler(file_handler)


def _ensure_logger_has_stream_handler(target: logging.Logger) -> None:
    if any(isinstance(h, logging.StreamHandler) for h in target.handlers):
        return
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    target.addHandler(stream_handler)


logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
_ensure_logger_has_file_handler(logger)
_ensure_logger_has_stream_handler(logger)

# Filter a noisy Ultralytics warning by message substring.
# We still attempt to pass `task='detect'` explicitly, but this keeps logs clean
# across varying SAHI/Ultralytics versions.
class _UltralyticsTaskGuessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "Unable to automatically guess model task" not in record.getMessage()
        except Exception:
            return True


logging.getLogger("ultralytics").addFilter(_UltralyticsTaskGuessFilter())

# In-memory store for detection results
RESULTS_STORE: dict[str, dict] = {}

# Create Settings object
settings = Settings.Settings()

# Cache detection models across requests (useful for batch processing).
MODEL_CACHE: dict[tuple, DetectionModel] = {}


def get_cached_detection_model(
    selected_model: str,
    weight_file: str,
    conf_th: float,
    image_size: int,
) -> DetectionModel:
    cache_key = (selected_model, weight_file, conf_th, image_size)
    cached = MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from_pretrained_kwargs: dict = {
        "model_type": selected_model,
        "model_path": weight_file,
        "confidence_threshold": conf_th,
    }

    sig = inspect.signature(AutoDetectionModel.from_pretrained)
    if "image_size" in sig.parameters:
        from_pretrained_kwargs["image_size"] = image_size

    if selected_model == "ultralytics":
        if "task" in sig.parameters:
            from_pretrained_kwargs["task"] = "detect"
        elif "model_kwargs" in sig.parameters:
            from_pretrained_kwargs["model_kwargs"] = {"task": "detect"}

    detection_model = AutoDetectionModel.from_pretrained(**from_pretrained_kwargs)
    MODEL_CACHE[cache_key] = detection_model
    return detection_model


def load_class_names_from_yaml(yaml_path: str) -> list[str]:
    """Load class names from YOLO-style YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                names = data['names']
                if isinstance(names, dict):
                    sorted_names = [names[k] for k in sorted(names.keys())]
                    return sorted_names
                elif isinstance(names, list):
                    return names
    except Exception as e:
        logger.error(f"Error loading class names from {yaml_path}: {e}")
    return []


def is_mps_available():
    return torch.backends.mps.is_available()


def list_weight_files(weight_paths: list = None) -> list:
    """Return the list of model in the `weight_paths` paths."""
    if weight_paths is None:
        weight_paths = [os.path.join(settings.MODEL_PATH, "*.onnx"),
                        os.path.join(settings.MODEL_PATH, "*.pt")]
    logger.log(logging.INFO, f"Load weight files from {weight_paths}")
    weight_files = []

    for path in weight_paths:
        file_list = glob.glob(path)
        file_list.sort()
        for item in file_list:
            weight_files.append({"item": item})
            logger.log(logging.INFO, f"Found weight file: {item}")

    logger.log(logging.INFO, f"Found {len(weight_files)} weight files.")
    return weight_files


def list_config_files() -> list:
    """Return the list of the yaml data for corresponding weight model."""
    logger.log(logging.INFO, f"Load config files")
    config_files = []
    file_list = glob.glob(r"datasets/yaml/*.yaml")
    file_list.sort()

    for item in file_list:
        config_files.append({"item": item})
        logger.log(logging.INFO, f"Found config file: {item}")

    logger.log(logging.INFO, f"Found {len(config_files)} config files.")
    return config_files


def extract_item_list(items: list, key: str = "item") -> list[str]:
    result: list[str] = []
    for entry in items:
        if isinstance(entry, dict) and key in entry:
            value = entry.get(key)
            if isinstance(value, str):
                result.append(value)
        elif isinstance(entry, str):
            result.append(entry)
    return result


def get_result_or_404(result_id: str) -> dict:
    result = RESULTS_STORE.get(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result


def pick_default_weight_file(model_type: str) -> str | None:
    weight_files = extract_item_list(MODEL_LIST)
    if not weight_files:
        return None
    if model_type == "ultralytics":
        for item in weight_files:
            if item.endswith(".pt"):
                return item
    return weight_files[0]


logger.log(logging.INFO, f"* *********************************** *")
logger.log(logging.INFO, f"* Preloading weight files and configs *")
logger.log(logging.INFO, f"* *********************************** *")

MODEL_LIST = list_weight_files()
CONFIG_FILE_LIST = list_config_files()


def extract_text_from_image(image: np.ndarray, objects: list[dict]) -> list[list[str]]:
    """Extract text from image using easyOCR."""
    logger.log(logging.INFO, f"Extract text from image with {len(objects)} objects.")
    if len(objects) < 1:
        return []
    try:
        logger.log(logging.INFO, f"Create easyOCR reader.")
        reader = easyocr.Reader(['en'])
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"\u00bd'
        return_list = []

        logger.log(logging.INFO, f"Start extracting text from image.")
        for index, obj in enumerate(objects):
            logger.log(logging.INFO, f"Cropping object {index+1} from image.")
            x_start, y_start, x_end, y_end = (
                obj["Left"],
                obj["Top"],
                obj["Left"] + obj["Width"],
                obj["Top"] + obj["Height"]
            )

            cropped_img_name = os.path.join(
                settings.TEXT_PATH, f"cropped_image_with_text_{index}.png")
            cropped_img = image[y_start:y_end, x_start:x_end]

            if obj["Object"] in settings.VERTICAL_TEXT:
                (height, wide) = cropped_img.shape[:2]
                if height > wide:
                    cropped_img = utils.rotate_image(cropped_img, 270)
                    logger.log(logging.INFO, f"Rotated cropped image for vertical text.")

            logger.log(logging.INFO, f"Making line thickness 2.")
            inverted_img = cv2.bitwise_not(cropped_img)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated_img = cv2.dilate(inverted_img, kernel, iterations=1)
            cropped_img = cv2.bitwise_not(dilated_img)

            logger.log(logging.INFO, f"Saving cropped image with text to {cropped_img_name}.")
            cv2.imwrite(cropped_img_name, cropped_img)

            logger.log(logging.INFO, f"Reading text from cropped image with text.")
            result = reader.readtext(
                cropped_img,
                decoder="wordbeamsearch",
                batch_size=4,
                paragraph=True,
                detail=0,
                mag_ratio=1.0,
                text_threshold=0.7,
                low_text=0.2,
                allowlist=allowlist,
            )

            logger.log(logging.INFO, f"Saving read result to list.")
            return_list.append(result)

        logger.log(logging.INFO, f"Returning the list of read text.")
        return return_list
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return []


# Initialize FastAPI app
app = FastAPI(
    title="GARNET API",
    description="P&ID Object Detection API Service",
    version="1.0.0"
)

# Mount static files for serving images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "GARNET API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "garnet-api"}


@app.get("/api/model-types")
async def api_model_types():
    """Get available model types."""
    return settings.MODEL_TYPES


@app.get("/api/models")
async def api_models():
    """Compatibility endpoint returning model type values."""
    try:
        return [item["value"] for item in settings.MODEL_TYPES if "value" in item]
    except Exception as exc:
        logger.error(f"Error loading model types: {exc}")
        return ["ultralytics"]


@app.get("/api/weight-files")
async def api_weight_files():
    """Get available weight files."""
    try:
        return extract_item_list(MODEL_LIST)
    except Exception as exc:
        logger.error(f"Error loading weight files: {exc}")
        return []


@app.get("/api/config-files")
async def api_config_files():
    """Get available config files."""
    try:
        return extract_item_list(CONFIG_FILE_LIST)
    except Exception as exc:
        logger.error(f"Error loading config files: {exc}")
        return ["datasets/yaml/data.yaml"]


@app.post("/api/detect")
async def api_detect(
    file_input: UploadFile = File(...),
    selected_model: str = Form("ultralytics"),
    weight_file: str = Form(""),
    config_file: str = Form("datasets/yaml/data.yaml"),
    conf_th: float = Form(0.8),
    image_size: int = Form(640),
    overlap_ratio: float = Form(0.2),
    text_OCR: bool = Form(False),
):
    """
    JSON API endpoint for object detection.
    Returns detection results as JSON.
    """
    logger.log(logging.INFO, f"API detect: model={selected_model}, weight={weight_file}")

    if not weight_file:
        weight_file = pick_default_weight_file(selected_model) or ""
        logger.log(logging.INFO, f"API detect: using default weight file: {weight_file}")
    if not weight_file:
        raise HTTPException(
            status_code=400,
            detail="No weight file available. Add a model under yolo_weights or select a weight file.",
        )

    # Read input image file
    input_filename = file_input.filename
    input_image_str = file_input.file.read()
    file_input.file.close()

    input_image_array = np.frombuffer(input_image_str, np.uint8)
    image = cv2.imdecode(input_image_array, cv2.IMREAD_COLOR)
    original_image = np.copy(image)
    processed_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    logger.log(logging.INFO, f"API detect: image shape: {image.shape}")

    # Calculate overlap and normalize image_size to a multiple of 32 (YOLO stride).
    image_size = (int(math.ceil((image_size + 1) / 32)) - 1) * 32
    logger.log(logging.INFO, f"API detect: adjusted image_size: {image_size}")

    # Set up the model (Ultralytics + SAHI)
    detection_model = get_cached_detection_model(
        selected_model=selected_model,
        weight_file=weight_file,
        conf_th=conf_th,
        image_size=image_size,
    )

    result = get_sliced_prediction(
        processed_image,
        detection_model,
        slice_height=image_size,
        slice_width=image_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type="NMM",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.2,
        verbose=0,
    )

    result_id = uuid.uuid4().hex
    image_filename = f"prediction_results_{result_id}.png"
    image_path = os.path.join("static", "images", image_filename)
    cv2.imwrite(image_path, original_image)

    # Process results
    table_data = []
    symbol_with_text = []
    category_object_count = [0 for _ in range(len(list(detection_model.category_mapping.values())))]

    for index, prediction in enumerate(result.object_prediction_list):
        bbox = prediction.bbox
        x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        width = x_max - x_min
        height = y_max - y_min
        object_category = prediction.category.name
        object_category_id = prediction.category.id

        table_data.append({
            "Index": index + 1,
            "Object": object_category,
            "CategoryID": object_category_id,
            "ObjectID": category_object_count[object_category_id] + 1,
            "Left": math.floor(x_min),
            "Top": math.floor(y_min),
            "Width": math.ceil(width),
            "Height": math.ceil(height),
            "Score": round(float(prediction.score.value), 3),
            "Text": f"{object_category} - no. {category_object_count[object_category_id] + 1}",
        })

        category_object_count[object_category_id] += 1

        if object_category in settings.SYMBOL_WITH_TEXT:
            symbol_with_text.append(table_data[-1])

    # OCR if enabled
    if text_OCR and len(symbol_with_text) > 0:
        text_list = extract_text_from_image(image, symbol_with_text)
        if len(text_list) > 0:
            for i in range(len(text_list)):
                idx = symbol_with_text[i]["Index"] - 1
                txt_to_display = " ".join(text_list[i])
                table_data[idx]["Text"] = txt_to_display

    # Sort by category then object ID
    sorted_data = sorted(table_data, key=lambda x: (x['CategoryID'], x['ObjectID']))
    for i in range(len(sorted_data)):
        sorted_data[i]["Index"] = i + 1

    result_payload = {
        "id": result_id,
        "objects": sorted_data,
        "image_url": f"/static/images/{image_filename}",
        "image_width": int(original_image.shape[1]),
        "image_height": int(original_image.shape[0]),
        "count": len(sorted_data),
    }
    RESULTS_STORE[result_id] = result_payload
    return JSONResponse(result_payload)


@app.get("/api/results/{result_id}")
async def api_get_result(result_id: str):
    """Fetch a previously detected result by ID."""
    return get_result_or_404(result_id)


@app.patch("/api/results/{result_id}/objects/{obj_id}")
async def api_patch_object(result_id: str, obj_id: int, payload: dict = Body(...)):
    """Update a detected object by Index within a stored result."""
    result = get_result_or_404(result_id)
    objects = result.get("objects", [])
    allowed_fields = {
        "Object", "CategoryID", "ObjectID", "Left", "Top", "Width", "Height",
        "Score", "Text", "ReviewStatus",
    }

    target = None
    for obj in objects:
        if obj.get("Index") == obj_id:
            target = obj
            break

    if not target:
        raise HTTPException(status_code=404, detail="Object not found")

    updates = {k: v for k, v in payload.items() if k in allowed_fields}
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    target.update(updates)
    return target


@app.post("/api/results/{result_id}/objects")
async def api_create_object(result_id: str, payload: dict = Body(...)):
    """Create a new detected object inside a stored result."""
    result = get_result_or_404(result_id)
    objects = result.get("objects", [])
    required_fields = {"Object", "Left", "Top", "Width", "Height"}
    if not required_fields.issubset(payload.keys()):
        raise HTTPException(status_code=400, detail="Missing required fields for new object")

    object_name = str(payload.get("Object", "")).strip()
    if not object_name:
        raise HTTPException(status_code=400, detail="Object name is required")

    category_id = payload.get("CategoryID")
    if category_id is None:
        matched = next((obj for obj in objects if obj.get("Object") == object_name), None)
        category_id = matched.get("CategoryID") if matched else 0

    try:
        left = float(payload.get("Left"))
        top = float(payload.get("Top"))
        width = float(payload.get("Width"))
        height = float(payload.get("Height"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid geometry for new object")

    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Width and height must be positive")

    object_id = payload.get("ObjectID")
    if object_id is None:
        object_id = sum(1 for obj in objects if obj.get("CategoryID") == category_id) + 1

    next_index = max((obj.get("Index", 0) for obj in objects), default=0) + 1
    new_obj = {
        "Index": next_index,
        "Object": object_name,
        "CategoryID": int(category_id),
        "ObjectID": int(object_id),
        "Left": int(math.floor(left)),
        "Top": int(math.floor(top)),
        "Width": int(math.ceil(width)),
        "Height": int(math.ceil(height)),
        "Score": float(payload.get("Score", 1.0)),
        "Text": str(payload.get("Text") or object_name),
        "ReviewStatus": payload.get("ReviewStatus"),
    }
    objects.append(new_obj)
    result["count"] = len(objects)
    return new_obj


@app.delete("/api/results/{result_id}/objects/{obj_id}")
async def api_delete_object(result_id: str, obj_id: int):
    """Delete a detected object by Index within a stored result."""
    result = get_result_or_404(result_id)
    objects = result.get("objects", [])
    updated_objects = [obj for obj in objects if obj.get("Index") != obj_id]
    if len(updated_objects) == len(objects):
        raise HTTPException(status_code=404, detail="Object not found")
    result["objects"] = updated_objects
    result["count"] = len(updated_objects)
    return {"status": "deleted"}


if __name__ == "__main__":
    logger.log(logging.INFO, f"* *********************************** *")
    logger.log(logging.INFO, f"*     Starting GARNET API Service     *")
    logger.log(logging.INFO, f"* *********************************** *")
    uvicorn.run("api:app", reload=True, port=8001)
