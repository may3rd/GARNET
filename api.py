"""
GARNET API Service - Pure API backend for React frontend

This is the API-only backend service. Run with:
    uvicorn api:app --reload --port 8001

The React frontend should run separately on port 5173 (dev) or be built for production.
"""

from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Body, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

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
import time
import threading
import pandas as pd
import logging
import uuid
import inspect

from garnet import utils
import garnet.Settings as Settings
import yaml

# =============================================================================
# Environment-based Configuration
# =============================================================================

class AppConfig:
    """Application configuration from environment variables."""
    # Environment
    ENV = os.getenv("ENV", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # Server
    HOST = os.getenv("HOST", "localhost")
    PORT = int(os.getenv("PORT", "8001"))
    
    # CORS
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:4173")
    
    # File Upload Limits
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS = os.getenv("ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.webp,.bmp,.tiff").split(",")
    
    # Cache Configuration
    RESULTS_CACHE_MAX_SIZE = int(os.getenv("RESULTS_CACHE_MAX_SIZE", "100"))
    RESULTS_CACHE_TTL = int(os.getenv("RESULTS_CACHE_TTL", "3600"))
    
    # Cleanup Configuration
    PREDICTION_IMAGE_TTL_HOURS = int(os.getenv("PREDICTION_IMAGE_TTL_HOURS", "24"))
    CLEANUP_INTERVAL_MINUTES = int(os.getenv("CLEANUP_INTERVAL_MINUTES", "60"))
    
    # Model Defaults
    DEFAULT_CONF_THRESHOLD = float(os.getenv("DEFAULT_CONF_THRESHOLD", "0.8"))
    DEFAULT_IMAGE_SIZE = int(os.getenv("DEFAULT_IMAGE_SIZE", "640"))
    DEFAULT_OVERLAP_RATIO = float(os.getenv("DEFAULT_OVERLAP_RATIO", "0.2"))
    
    # Paths
    PREDICTIONS_DIR = os.getenv("PREDICTIONS_DIR", "static/images/predictions")


config = AppConfig()

# =============================================================================
# Logging Configuration
# =============================================================================

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
class _UltralyticsTaskGuessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "Unable to automatically guess model task" not in record.getMessage()
        except Exception:
            return True


logging.getLogger("ultralytics").addFilter(_UltralyticsTaskGuessFilter())

# =============================================================================
# Pydantic Models for Input Validation
# =============================================================================

class DetectRequest(BaseModel):
    """Validated detection request parameters."""
    selected_model: str = Field(default="ultralytics", description="Model type to use")
    weight_file: str = Field(default="", description="Path to model weights")
    config_file: str = Field(default="datasets/yaml/data.yaml", description="Path to config YAML")
    conf_th: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence threshold")
    image_size: int = Field(default=640, ge=320, le=2048, description="Image size for inference")
    overlap_ratio: float = Field(default=0.2, ge=0.0, le=0.95, description="Slice overlap ratio")
    text_ocr: bool = Field(default=False, description="Enable OCR for text extraction")

    @validator('weight_file')
    def validate_weight_file(cls, v):
        if v and not os.path.exists(v):
            # Allow empty string (will use default)
            if v.strip():
                raise ValueError(f"Weight file not found: {v}")
        return v

    @validator('config_file')
    def validate_config_file(cls, v):
        if v and not os.path.exists(v):
            raise ValueError(f"Config file not found: {v}")
        return v


class CreateObjectRequest(BaseModel):
    """Validated request for creating a new object."""
    Object: str = Field(..., min_length=1, max_length=100, description="Object category name")
    Left: float = Field(..., ge=0, description="Left coordinate")
    Top: float = Field(..., ge=0, description="Top coordinate")
    Width: float = Field(..., gt=0, le=10000, description="Width (must be positive)")
    Height: float = Field(..., gt=0, le=10000, description="Height (must be positive)")
    Text: str = Field(default="", max_length=500, description="Associated text")
    CategoryID: int | None = Field(default=None, ge=0, description="Category ID (optional)")
    ObjectID: int | None = Field(default=None, ge=1, description="Object ID (optional)")
    Score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class UpdateObjectRequest(BaseModel):
    """Validated request for updating an object."""
    Object: str | None = Field(default=None, min_length=1, max_length=100)
    Left: float | None = Field(default=None, ge=0)
    Top: float | None = Field(default=None, ge=0)
    Width: float | None = Field(default=None, gt=0, le=10000)
    Height: float | None = Field(default=None, gt=0, le=10000)
    Text: str | None = Field(default=None, max_length=500)
    Score: float | None = Field(default=None, ge=0.0, le=1.0)
    ReviewStatus: str | None = Field(default=None, max_length=50)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    timestamp: str
    version: str
    environment: str
    models_loaded: bool
    models_available: int
    memory_usage_mb: float


# =============================================================================
# Global State
# =============================================================================

# In-memory store for detection results
RESULTS_STORE: dict[str, dict] = {}

# Create Settings object
settings = Settings.Settings()

# Cache detection models across requests
MODEL_CACHE: dict[tuple, DetectionModel] = {}

# Model loading status for health check
MODELS_LOADED = False
MODEL_LOAD_ERROR = None

# Ensure predictions directory exists
os.makedirs(config.PREDICTIONS_DIR, exist_ok=True)

# =============================================================================
# Cleanup Job for Old Prediction Images
# =============================================================================

def cleanup_old_predictions():
    """Remove prediction images older than configured TTL."""
    try:
        cutoff_time = time.time() - (config.PREDICTION_IMAGE_TTL_HOURS * 3600)
        cleaned_count = 0
        
        for filename in os.listdir(config.PREDICTIONS_DIR):
            filepath = os.path.join(config.PREDICTIONS_DIR, filename)
            if os.path.isfile(filepath):
                file_mtime = os.path.getmtime(filepath)
                if file_mtime < cutoff_time:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                    except OSError as e:
                        logger.warning(f"Failed to remove old prediction {filepath}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old prediction images")
    except Exception as e:
        logger.error(f"Error during prediction cleanup: {e}")


def start_cleanup_scheduler():
    """Start background thread for periodic cleanup."""
    def cleanup_loop():
        while True:
            cleanup_old_predictions()
            time.sleep(config.CLEANUP_INTERVAL_MINUTES * 60)
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info(f"Started cleanup scheduler (interval: {config.CLEANUP_INTERVAL_MINUTES} minutes)")


# =============================================================================
# Model and Utility Functions
# =============================================================================

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


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def list_weight_files(weight_paths: list | None = None) -> list:
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


def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file."""
    # Check file extension
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in config.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file extension. Allowed: {', '.join(config.ALLOWED_IMAGE_EXTENSIONS)}"
        )


# =============================================================================
# Application Initialization
# =============================================================================

logger.log(logging.INFO, f"* *********************************** *")
logger.log(logging.INFO, f"* Preloading weight files and configs *")
logger.log(logging.INFO, f"* *********************************** *")

MODEL_LIST = list_weight_files()
CONFIG_FILE_LIST = list_config_files()

# Preload at least one model to verify it works
try:
    default_weight = pick_default_weight_file("ultralytics")
    if default_weight:
        logger.info(f"Preloading default model: {default_weight}")
        _ = get_cached_detection_model("ultralytics", default_weight, 0.8, 640)
        MODELS_LOADED = True
        logger.info("Model preloaded successfully")
    else:
        logger.warning("No weight files found for preloading")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    logger.error(f"Failed to preload model: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="GARNET API",
    description="P&ID Object Detection API Service",
    version="1.0.0"
)

# Mount static files for serving images
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS configuration from environment
origins = [o.strip() for o in config.ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)

# Start cleanup scheduler
start_cleanup_scheduler()

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "GARNET API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "environment": config.ENV
    }


@app.get("/api/health", response_model=HealthResponse)
async def api_health():
    """Health check endpoint with model loading verification."""
    return HealthResponse(
        status="healthy" if MODELS_LOADED else "degraded",
        service="garnet-api",
        timestamp=datetime.datetime.utcnow().isoformat(),
        version="1.0.0",
        environment=config.ENV,
        models_loaded=MODELS_LOADED,
        models_available=len(MODEL_CACHE),
        memory_usage_mb=round(get_memory_usage_mb(), 2)
    )


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

    # Validate image file
    validate_image_file(file_input)

    # Read and validate file size
    input_image_str = await file_input.read()
    file_size = len(input_image_str)
    
    if file_size > config.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
        )
    
    await file_input.close()

    # Validate form parameters using Pydantic
    try:
        params = DetectRequest(
            selected_model=selected_model,
            weight_file=weight_file,
            config_file=config_file,
            conf_th=conf_th,
            image_size=image_size,
            overlap_ratio=overlap_ratio,
            text_ocr=text_OCR
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    if not params.weight_file:
        params.weight_file = pick_default_weight_file(params.selected_model) or ""
        logger.log(logging.INFO, f"API detect: using default weight file: {params.weight_file}")
    if not params.weight_file:
        raise HTTPException(
            status_code=400,
            detail="No weight file available. Add a model under yolo_weights or select a weight file.",
        )

    # Decode image
    input_image_array = np.frombuffer(input_image_str, np.uint8)
    image = cv2.imdecode(input_image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image file. Could not decode image."
        )
    
    original_image = np.copy(image)
    processed_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    logger.log(logging.INFO, f"API detect: image shape: {image.shape}")

    # Calculate overlap and normalize image_size to a multiple of 32 (YOLO stride).
    adjusted_image_size = (int(math.ceil((params.image_size + 1) / 32)) - 1) * 32
    logger.log(logging.INFO, f"API detect: adjusted image_size: {adjusted_image_size}")

    # Set up the model (Ultralytics + SAHI)
    detection_model = get_cached_detection_model(
        selected_model=params.selected_model,
        weight_file=params.weight_file,
        conf_th=params.conf_th,
        image_size=adjusted_image_size,
    )

    result = get_sliced_prediction(
        processed_image,
        detection_model,
        slice_height=adjusted_image_size,
        slice_width=adjusted_image_size,
        overlap_height_ratio=params.overlap_ratio,
        overlap_width_ratio=params.overlap_ratio,
        postprocess_type="NMM",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.2,
        verbose=0,
    )

    result_id = uuid.uuid4().hex
    image_filename = f"prediction_results_{result_id}.png"
    image_path = os.path.join(config.PREDICTIONS_DIR, image_filename)
    cv2.imwrite(image_path, original_image)

    # Process results
    table_data = []
    symbol_with_text = []
    category_mapping = getattr(detection_model, 'category_mapping', {}) or {}
    category_object_count = [0 for _ in range(len(list(category_mapping.values())))]

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
    if params.text_ocr and len(symbol_with_text) > 0:
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
        "image_url": f"/static/images/predictions/{image_filename}",
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
async def api_patch_object(result_id: str, obj_id: int, payload: UpdateObjectRequest):
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

    # Convert Pydantic model to dict, excluding None values
    updates = payload.dict(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")

    target.update(updates)
    return target


@app.post("/api/results/{result_id}/objects")
async def api_create_object(result_id: str, payload: CreateObjectRequest):
    """Create a new detected object inside a stored result."""
    result = get_result_or_404(result_id)
    objects = result.get("objects", [])
    
    # Get existing object to determine CategoryID if not provided
    category_id = payload.CategoryID
    if category_id is None:
        matched = next((obj for obj in objects if obj.get("Object") == payload.Object), None)
        category_id = matched.get("CategoryID") if matched else 0

    # Get ObjectID if not provided
    object_id = payload.ObjectID
    if object_id is None:
        object_id = sum(1 for obj in objects if obj.get("CategoryID") == category_id) + 1

    next_index = max((obj.get("Index", 0) for obj in objects), default=0) + 1
    new_obj = {
        "Index": next_index,
        "Object": payload.Object,
        "CategoryID": int(category_id),
        "ObjectID": int(object_id),
        "Left": int(math.floor(payload.Left)),
        "Top": int(math.floor(payload.Top)),
        "Width": int(math.ceil(payload.Width)),
        "Height": int(math.ceil(payload.Height)),
        "Score": float(payload.Score),
        "Text": payload.Text if payload.Text else payload.Object,
        "ReviewStatus": None,
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
    logger.log(logging.INFO, f"*     Environment: {config.ENV:21} *")
    logger.log(logging.INFO, f"* *********************************** *")
    uvicorn.run("api:app", reload=config.DEBUG, port=config.PORT, host=config.HOST)
