# Not used, but kept for reference

from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, JSONResponse

from sahi import AutoDetectionModel, DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import crop_object_predictions

# import onnxruntime as ort
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

# Configure logging
log_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'garnet.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

# In-memory store for detection results.
RESULTS_STORE: dict[str, dict] = {}

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BACKEND_DIR, "runs")
DETECT_DIR = os.path.join(RUNS_DIR, "detect")
ULTRALYTICS_RUNS_DIR = os.path.join(BACKEND_DIR, ".ultralytics_runs")
os.makedirs(DETECT_DIR, exist_ok=True)
os.makedirs(ULTRALYTICS_RUNS_DIR, exist_ok=True)

MODEL_LIST: list[dict] | None = None
CONFIG_FILE_LIST: list[dict] | None = None


def load_class_names_from_yaml(yaml_path: str) -> list[str]:
    """Load class names from YOLO-style YAML file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            if 'names' in data:
                names = data['names']
                if isinstance(names, dict):
                    # Sort by id assuming integer keys
                    sorted_names = [names[k] for k in sorted(names.keys())]
                    return sorted_names
                elif isinstance(names, list):
                    return names
    except Exception as e:
        logger.error(f"Error loading class names from {yaml_path}: {e}")
    return []

# Test logging
# logger.info("Application started")
# logger.debug("Debug message")
# logger.warning("Warning message")
# logger.error("Error message")
# logger.critical("Critical message")


# Create Settings object
settings = Settings.Settings()


def is_mps_available():
    return torch.backends.mps.is_available()


def list_weight_files(weight_paths: list = [os.path.join(settings.MODEL_PATH, "*.onnx"),
                                            os.path.join(settings.MODEL_PATH, "*.pt")]) -> list:
    """
    Return the list of model in the `weight_paths` paths.
    """
    # logger.log(logging.INFO, f"Load weight files from {weight_paths}")
    weight_files = []

    for path in weight_paths:
        file_list = glob.glob(path)
        file_list.sort()
        for item in file_list:
            weight_files.append({"item": item})
            # logger.log(logging.INFO, f"Found weight file: {item}")

    logger.log(logging.INFO, f"Found {len(weight_files)} weight files.")
    return weight_files


def list_config_files() -> list:
    """
    Return the list of the yaml data for corresponding weight model.
    """
    logger.log(logging.INFO, f"Load config files")
    config_files = []
    file_list = glob.glob(r"datasets/yaml/*.yaml")
    file_list.sort()

    for item in file_list:
        config_files.append({"item": item})
        # logger.log(logging.INFO, f"Found config file: {item}")

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
    model_list, _ = ensure_model_and_config_loaded()
    weight_files = extract_item_list(model_list)
    if not weight_files:
        return None
    if model_type == "ultralytics":
        for item in weight_files:
            if item.endswith(".pt"):
                return item
    return weight_files[0]


def ensure_model_and_config_loaded() -> tuple[list[dict], list[dict]]:
    global MODEL_LIST, CONFIG_FILE_LIST
    if MODEL_LIST is None or CONFIG_FILE_LIST is None:
        logger.log(logging.INFO, f"Preloading weight files and configs")
        MODEL_LIST = list_weight_files()
        CONFIG_FILE_LIST = list_config_files()
    return MODEL_LIST, CONFIG_FILE_LIST  # type: ignore[return-value]


def configure_ultralytics_runs_dir() -> None:
    try:
        from ultralytics import settings as ultralytics_settings

        if ultralytics_settings.get("runs_dir") != ULTRALYTICS_RUNS_DIR:
            ultralytics_settings.update({"runs_dir": ULTRALYTICS_RUNS_DIR})
            logger.log(
                logging.INFO,
                f"Set Ultralytics runs_dir to {ULTRALYTICS_RUNS_DIR}",
            )
    except Exception as exc:
        logger.warning(f"Failed to configure Ultralytics runs_dir: {exc}")


configure_ultralytics_runs_dir()



def extract_text_from_image(
    image: np.ndarray,
    objects: list[dict]
) -> list[list[str]]:
    '''
    extract text from image using easyOCR

    param: image (cv2 image)
    param: objects

    return: list of read text
    '''

    logger.log(
        logging.INFO, f"Extract text from image with {len(objects)} objects.")
    # if objects is None or zero member then return nothing
    if len(objects) < 1:
        return []
    try:
        # Create Reader for text OCR
        logger.log(logging.INFO, f"Create easyOCR reader.")
        reader = easyocr.Reader(['en'])

        # Create allowlist for object detection
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"½'

        return_list = []

        logger.log(logging.INFO, f"Start extracting text from image.")
        # Loop through objects list
        for index, object in enumerate(objects):
            # Crop the object from the image
            logger.log(logging.INFO, f"Cropping object {index+1} from image.")
            x_start, y_start, x_end, y_end = (
                object["Left"],
                object["Top"],
                object["Left"] + object["Width"],
                object["Top"] + object["Height"]
            )

            cropped_img_name = os.path.join(
                settings.TEXT_PATH, f"cropped_image_with_text_{index}.png")
            cropped_img = image[y_start:y_end, x_start:x_end]

            # rotate if object is page connection and dimension wide is less than height
            if object["Object"] in settings.VERTICAL_TEXT:
                (height, wide) = cropped_img.shape[:2]
                if height > wide:
                    cropped_img = utils.rotate_image(cropped_img, 270)
                    logger.log(
                        logging.INFO, f"Rotated cropped image for vertical text.")

            # # remove circle from instrument tag
            # if "instrument" in object["Object"]:
            #     logger.log(logging.INFO, f"Removing circle from instrument tag.")
            #     cropped_img = utils.remove_circular_lines(
            #         cropped_img,
            #         param1=50,
            #         param2=80,
            #         minRadius=30,
            #         maxRadius=100,
            #         thickness=3,
            #         outside=False,
            #         )

            # make line thickness 2
            logger.log(logging.INFO, f"Making line thickness 2.")
            inverted_img = cv2.bitwise_not(cropped_img)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            dilated_img = cv2.dilate(inverted_img, kernel, iterations=1)
            cropped_img = cv2.bitwise_not(dilated_img)

            # save a processed cropped image
            logger.log(
                logging.INFO, f"Saving cropped image with text to {cropped_img_name}.")
            cv2.imwrite(cropped_img_name, cropped_img)

            # read text in processed cropped image
            logger.log(
                logging.INFO, f"Reading text from cropped image with text.")
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

            # save read result to list
            logger.log(logging.INFO, f"Saving read result to list.")
            return_list.append(result)

        # return the list of read text
        logger.log(logging.INFO, f"Returning the list of read text.")
        return return_list
    except Exception as e:
        logger.error(f"Error extracting text: {str(e)}")
        return []


app = FastAPI()
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# JSON API Endpoints for React Frontend
# ============================================


@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/api/pdf-extract")
async def api_pdf_extract(
    file_input: UploadFile = File(...),
):
    """
    Extract pages from a PDF as base64-encoded PNG images.
    Returns JSON with page count and list of base64 image strings.
    """
    import base64
    from io import BytesIO

    try:
        from pdf2image import convert_from_bytes
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pdf2image is not installed. Run: pip install pdf2image",
        )

    MAX_PAGES = 50
    DPI = 300

    logger.info(f"PDF extract: received file {file_input.filename}")

    if not file_input.filename or not file_input.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        pdf_bytes = await file_input.read()
        images = convert_from_bytes(pdf_bytes, dpi=DPI)
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)}")

    if len(images) > MAX_PAGES:
        raise HTTPException(
            status_code=400,
            detail=f"PDF has {len(images)} pages, maximum allowed is {MAX_PAGES}",
        )

    pages: list[str] = []
    for i, img in enumerate(images):
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        pages.append(b64)
        logger.info(f"PDF extract: converted page {i + 1}/{len(images)}")

    return {"count": len(pages), "pages": pages}

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
        model_list, _ = ensure_model_and_config_loaded()
        return extract_item_list(model_list)
    except Exception as exc:
        logger.error(f"Error loading weight files: {exc}")
        return []


@app.get("/api/config-files")
async def api_config_files():
    """Get available config files."""
    try:
        _, config_file_list = ensure_model_and_config_loaded()
        return extract_item_list(config_file_list)
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
    text_OCR: bool = Form(False),
):
    """
    JSON API endpoint for object detection.
    Returns detection results as JSON instead of HTML.
    """
    from fastapi.responses import JSONResponse

    logger.log(
        logging.INFO, f"API detect: model={selected_model}, weight={weight_file}")

    if not weight_file:
        weight_file = pick_default_weight_file(selected_model) or ""
        logger.log(
            logging.INFO, f"API detect: using default weight file: {weight_file}")
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
    logger.log(
        logging.INFO, f"API detect: original image shape: {image.shape}, processed image shape: {processed_image.shape}")

    # Set up the model
    from_pretrained_kwargs: dict = {
        "model_type": selected_model,
        "model_path": weight_file,
        "confidence_threshold": conf_th,
    }
    sig = inspect.signature(AutoDetectionModel.from_pretrained)
    if selected_model == "ultralytics":
        predict_kwargs = {
            "save": False,
            "save_txt": False,
            "save_conf": False,
            "save_crop": False,
            "show": False,
            "verbose": False,
        }
        if "task" in sig.parameters:
            from_pretrained_kwargs["task"] = "detect"
        if "model_kwargs" in sig.parameters:
            from_pretrained_kwargs["model_kwargs"] = {
                "task": "detect",
                **predict_kwargs,
            }
    detection_model: DetectionModel = AutoDetectionModel.from_pretrained(
        **from_pretrained_kwargs
    )

    # Calculate overlap and run detection
    overlap_ratio = 0.2
    image_size = (int(math.ceil((image_size + 1) / 32)) - 1) * 32
    logger.log(logging.INFO, f"API detect: adjusted image_size: {image_size}")

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
        exclude_classes_by_name=["line number", "instrument tag", "instrument dcs", "instrument logic", "trip function"],
    )

    result_id = uuid.uuid4().hex
    image_filename = f"prediction_results_{result_id}.png"
    image_path = os.path.join(DETECT_DIR, image_filename)
    cv2.imwrite(image_path, original_image)

    # Process results
    table_data = []
    symbol_with_text = []
    category_object_count = [0 for _ in range(
        len(list(detection_model.category_mapping.values())))]  # type: ignore

    logger.log(logging.INFO, f"API detect: number of predictions: {len(result.object_prediction_list)}")
    
    for index, prediction in enumerate(result.object_prediction_list):
        bbox = prediction.bbox
        x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        width = x_max - x_min
        height = y_max - y_min
        object_category = prediction.category.name
        object_category_id = prediction.category.id
        # logger.log(
        #     logging.INFO, f"API detect: prediction {index}: raw bbox minx={int(x_min)}, miny={int(y_min)}, maxx={int(x_max)}, maxy={int(y_max)}, width={int(width)}, height={int(height)}")

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
    sorted_data = sorted(table_data, key=lambda x: (
        x['CategoryID'], x['ObjectID']))
    for i in range(len(sorted_data)):
        sorted_data[i]["Index"] = i + 1

    result_payload = {
        "id": result_id,
        "objects": sorted_data,
        "image_url": f"/runs/detect/{image_filename}",
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
        "Object",
        "CategoryID",
        "ObjectID",
        "Left",
        "Top",
        "Width",
        "Height",
        "Score",
        "Text",
        "ReviewStatus",
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
        raise HTTPException(
            status_code=400, detail="No valid fields to update")

    target.update(updates)
    return target


@app.post("/api/results/{result_id}/objects")
async def api_create_object(result_id: str, payload: dict = Body(...)):
    """Create a new detected object inside a stored result."""
    result = get_result_or_404(result_id)
    objects = result.get("objects", [])
    required_fields = {"Object", "Left", "Top", "Width", "Height"}
    if not required_fields.issubset(payload.keys()):
        raise HTTPException(
            status_code=400, detail="Missing required fields for new object")

    object_name = str(payload.get("Object", "")).strip()
    if not object_name:
        raise HTTPException(status_code=400, detail="Object name is required")

    category_id = payload.get("CategoryID")
    if category_id is None:
        matched = next((obj for obj in objects if obj.get(
            "Object") == object_name), None)
        category_id = matched.get("CategoryID") if matched else 0

    try:
        left = float(payload.get("Left", 0))
        top = float(payload.get("Top", 0))
        width = float(payload.get("Width", 0))
        height = float(payload.get("Height", 0))
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400, detail="Invalid geometry for new object")

    if width <= 0 or height <= 0:
        raise HTTPException(
            status_code=400, detail="Width and height must be positive")

    object_id = payload.get("ObjectID")
    if object_id is None:
        object_id = sum(1 for obj in objects if obj.get(
            "CategoryID") == category_id) + 1

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
    logger.log(logging.INFO, f"*           Starting GARNET           *")
    logger.log(logging.INFO, f"* *********************************** *")
    uvicorn.run("main:app", reload=True, port=8001)
