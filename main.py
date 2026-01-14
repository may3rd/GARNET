from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import Response

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

from garnet import utils
import garnet.Settings as Settings
import yaml
from garnet.azure_inference import CustomVisionSAHIDetector, CustomVisionConfig

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
    """
    Return the list of the yaml data for corresponding weight model.
    """
    logger.log(logging.INFO, f"Load config files")
    config_files = []
    file_list = glob.glob(r"datasets/yaml/*.yaml")
    file_list.sort()

    for item in file_list:
        config_files.append({"item": item})
        logger.log(logging.INFO, f"Found config file: {item}")

    logger.log(logging.INFO, f"Found {len(config_files)} config files.")
    return config_files


logger.log(logging.INFO, f"* *********************************** *")
logger.log(logging.INFO, f"* Preloading weight files and configs *")
logger.log(logging.INFO, f"* *********************************** *")

MODEL_LIST = list_weight_files()
CONFIG_FILE_LIST = list_config_files()


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
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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


@app.get("/api/model-types")
async def api_model_types():
    """Get available model types."""
    return settings.MODEL_TYPES


@app.get("/api/weight-files")
async def api_weight_files():
    """Get available weight files."""
    return MODEL_LIST


@app.get("/api/config-files")
async def api_config_files():
    """Get available config files."""
    return CONFIG_FILE_LIST


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
    if selected_model == "azure_custom_vision":
        class_names = load_class_names_from_yaml(config_file)
        config = CustomVisionConfig(
            model_path=weight_file,
            class_names=class_names,
            input_size=(image_size, image_size),
            confidence_threshold=conf_th
        )
        detection_model = CustomVisionSAHIDetector(config)
    else:
        detection_model: DetectionModel = AutoDetectionModel.from_pretrained(
            model_type=selected_model,
            model_path=weight_file,
            confidence_threshold=conf_th,
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
    )

    # Save the original image for canvas display
    cv2.imwrite('static/images/prediction_results.png', original_image)

    # Process results
    table_data = []
    symbol_with_text = []
    category_object_count = [0 for _ in range(
        len(list(detection_model.category_mapping.values())))]  # type: ignore

    for index, prediction in enumerate(result.object_prediction_list):
        bbox = prediction.bbox
        x_min, y_min, x_max, y_max = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        width = x_max - x_min
        height = y_max - y_min
        object_category = prediction.category.name
        object_category_id = prediction.category.id
        logger.log(
            logging.INFO, f"API detect: prediction {index}: raw bbox minx={x_min}, miny={y_min}, maxx={x_max}, maxy={y_max}, width={width}, height={height}")

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

    return JSONResponse({
        "objects": sorted_data,
        "image_url": "/static/images/prediction_results.png",
        "count": len(sorted_data),
    })

# ============================================
# Original HTML Endpoints
# ============================================


@app.get("/")
async def main(request: Request):
    # Create dummy table
    logger.log(logging.INFO, "First time run, creating dummy table.")
    table_data = []
    count = 15

    for idx in range(count):
        table_data.append({
            "Index": idx+1,
            "CategoryID": 0,
            "Object": "Object to be detected",
            "Score": 0.0,
            "Id": 0,
            "Text": "Object:" + str(idx),
        })

    # Create JSON data to return to template
    logger.log(logging.INFO, "Creating JSON data to return to template.")
    json_data = json.dumps(table_data)
    checkboxes = []
    checkboxes.append({
        "id": 0,
        "desc": "Object to be detcted",
        "count": count,
    })

    logger.log(logging.INFO, "Returning template response.")
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "runFlag": False,
            "table_data": table_data,
            "json_data": json_data,
            "weight_files": MODEL_LIST,
            "config_files": CONFIG_FILE_LIST,
            "model_types": settings.MODEL_TYPES,
            "input_filename": "",
            "output_text": "Not run yet!",
            "category_id": checkboxes,
        }
    )


"""
    Inference the input file with selected method.
"""


@app.post("/submit")
async def inferencing_image_and_text(
        request: Request,
        file_input: UploadFile = File(...),
        selected_model: str = Form("ultralytics"),
        weight_file: str = Form(os.path.join(
            settings.MODEL_PATH, "yolov8_640_20231022.pt")),
        config_file: str = Form("datasets/yaml/data.yaml"),
        conf_th: float = Form(0.8),
        image_size: int = Form(640),
        text_OCR: bool = Form(False),
):
    logger.log(
        logging.INFO, f"Start inferencing image and text with selected model {selected_model}.")
    logger.log(logging.INFO, f"Weight file is {weight_file}.")
    # logger.log(logging.INFO, f"Config file is {config_file}.")
    logger.log(logging.INFO, f"Confidence threshold is {conf_th}.")
    logger.log(logging.INFO, f"Image size is {image_size}.")
    logger.log(logging.INFO, f"Text OCR is {text_OCR}.")
    logger.log(logging.INFO, f"Input file name is {file_input.filename}.")

    # Read input image file
    logger.log(
        logging.INFO, f"Reading input image file {file_input.filename}.")
    print("Input file name:", file_input.filename)
    input_filename = file_input.filename
    input_image_str = file_input.file.read()
    file_input.file.close()

    logger.log(logging.INFO, f"Converting input image string to numpy array.")
    input_image_array = np.frombuffer(input_image_str, np.uint8)

    # Convert input image array to CV2
    logger.log(logging.INFO, f"Converting input image array to CV2 image.")
    image = cv2.imdecode(input_image_array, cv2.IMREAD_COLOR)
    original_image = np.copy(image)
    processed_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Create inferencing model
    logger.log(
        logging.INFO, f"Creating inferencing model {selected_model} and weight file {weight_file}.")
    print("start detecting by using", selected_model,
          "model with conf =", conf_th)
    print("weight_path is", weight_file)

    # Not required in SAHI v.0.11.32
    # # Set category_mapping for ONNX model, required by updated version of SAHI
    # logger.log(logging.INFO, f"Setting category mapping if the model is ONNX.")
    # if "yolov8onnx" == selected_model:
    #     import onnx
    #     import ast
    #     model = onnx.load(weight_file)
    #     props = { p.key: p.value for p in model.metadata_props }
    #     names = ast.literal_eval(props['names'])
    #     category_mapping = { str(key): value for key, value in names.items() }
    # else:
    #     category_mapping = None

    # # Create session options
    # sess_options = ort.SessionOptions()
    # providers = ['CPUExecutionProvider']  # Default to CPU

    # # Check if MPS is available and set the provider accordingly
    # logger.log(logging.INFO, f"Checking if MPS is available.")
    # if is_mps_available():
    #     providers = ['CoreMLExecutionProvider']
    #     device = 'mps'
    # else:
    #     device = 'cpu'

    # logger.log(logging.INFO, f"The model will be run on {device}.")

    # Set up the model to be used for inferencing.
    logger.log(logging.INFO, f"Setting up the model to be used for inferencing.")

    if selected_model == "azure_custom_vision":
        logger.info("Initializing Custom Vision Detector")
        class_names = load_class_names_from_yaml(config_file)
        if not class_names:
            logger.warning(
                f"No class names found in {config_file}, or file not found. Inference might fail if model does not provide them.")

        config = CustomVisionConfig(
            model_path=weight_file,
            class_names=class_names,
            input_size=(image_size, image_size),
            confidence_threshold=conf_th
        )
        detection_model = CustomVisionSAHIDetector(config)
    else:
        detection_model: DetectionModel = AutoDetectionModel.from_pretrained(
            model_type=selected_model,
            model_path=weight_file,
            confidence_threshold=conf_th,
            # device=device,
        )
    # Calculate the overlap ratio
    logger.log(logging.INFO, f"Calculating the overlap ratio.")
    overlap_ratio = 0.2  # float(32 / image_size)

    # Correct the image size to use with the model
    logger.log(logging.INFO, f"Correcting the image size to use with the model.")
    image_size = (int(math.ceil((image_size + 1) / 32)) - 1) * 32

    # Run the inferencing model
    # use verbose = 2 to see predection time
    print(f"Run the sliced prediction of {image_size}x{image_size} slices.")

    logger.log(
        logging.INFO, f"Running the sliced prediction of {image_size}x{image_size} slices.")
    result = get_sliced_prediction(
        processed_image,
        detection_model,
        slice_height=image_size,
        slice_width=image_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        # Prostprocessing algorithm to use (None, "GREEDYNMM", "NMS")
        postprocess_type="NMM",
        # Match metric for NMS postprocessing (IOS, IOU)
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.2,  # Match threshold for NMS postprocessing
        verbose=2,
    )

    # Extract the result from inferencing model
    # result.export_visuals(
    #    export_dir="static/images/",  # save the output picture for display
    #    text_size=0.5,
    #    rect_th=2,
    #    hide_labels=True,
    #    hide_conf=True,
    #    file_name="prediction_results",  # output file name
    # )

    # Write the original image and the bounding boxes will be created by fabric.js
    logger.log(
        logging.INFO, f"Writing the original image and the bounding boxes will be created by fabric.js.")
    cv2.imwrite('static/images/prediction_results.png', original_image)

    # Obtain the prediction list from model results.
    object_prediction_list = result.object_prediction_list

    # Create COCO annotation file
    coco = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "COCO format annotation file for object detection",
            "contributor": "GARNET",
            "url": "https://github.com/may3r/GARNET",
            "date_created": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")
        },
        "licenses": [{
            "id": 0,
            "name": "MIT License",
            "url": "https://github.com/may3r/GARNET/blob/main/LICENSE"
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Get the image size
    height, width, _ = original_image.shape

    image_info = {
        "id": 0,
        "file_name": file_input.filename,
        "width": width,
        "height": height,
        "date_captured": "",
        "license": 0,
        "coco_url": "",
        "flickr_url": ""
    }

    coco["images"].append(image_info)

    coco["annotations"] = result.to_coco_annotations()

    # Change image_id to 0
    for i in range(len(coco["annotations"])):
        coco["annotations"][i]["image_id"] = 0

    # Create category mapping for COCO annotation
    for category_id, category_name in detection_model.category_mapping.items():  # type: ignore
        category_info = {
            "id": category_id,
            "name": category_name,
            "supercategory": ""
        }
        coco["categories"].append(category_info)

    # Save COCO annotation file
    logger.log(
        logging.INFO, f"Saving COCO annotation file to {os.path.join(settings.OUTPUT_PATH, 'coco_annotation.json')}.")
    with open(os.path.join(settings.OUTPUT_PATH, "coco_annotation.json"), "w") as f:
        json.dump(coco, f)

    # Crops bounding boxes over the source image and exports to directory
    logger.log(
        logging.INFO, f"Crops bounding boxes over the source image and exports to directory.")

    def delete_all_files_in_folder(folder_path):
        # Get a list of all files in the folder
        files = glob.glob(os.path.join(folder_path, "*"))

        # Iterate over the list of files and remove each one
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")

    logger.log(
        logging.INFO, f"Deleting all files in {settings.CROPPED_OBJECT_PATH}.")
    delete_all_files_in_folder(settings.CROPPED_OBJECT_PATH)

    logger.log(
        logging.INFO, f"Cropping object predictions and saving to {settings.CROPPED_OBJECT_PATH}.")
    crop_object_predictions(
        processed_image, object_prediction_list, settings.CROPPED_OBJECT_PATH)

    # Initialize data list and index
    logger.log(logging.INFO, f"Initializing data list and index.")
    table_data = []
    symbol_with_text = []
    category_ids = set()
    index = 0

    # Get the prediction list from inferenced result
    prediction_list = result.object_prediction_list

    # Create output text
    output_text = str(input_filename) + \
        f": found {str(len(prediction_list))} objects."
    print("Found", len(prediction_list), "objects.")
    logger.log(logging.INFO, f"Found {len(prediction_list)} objects.")

    # Count the number of objects for each category
    category_object_count = [0 for i in range(
        len(list(detection_model.category_mapping.values())))]  # type: ignore
    category_names = {}
    # Loop through prediction list and extract data for HTML table
    # Extarct bboxes from prediction result
    for prediction in prediction_list:
        bbox = prediction.bbox
        x_min = bbox.minx
        y_min = bbox.miny
        x_max = bbox.maxx
        y_max = bbox.maxy
        width = x_max - x_min
        height = y_max - y_min
        object_category = prediction.category.name
        object_category_id = prediction.category.id
        index += 1

        category_ids.add(object_category_id)

        # save data to use in HTML canvas
        table_data.append({
            "Index": index,
            "Object": object_category,
            "CategoryID": object_category_id,
            "ObjectID": category_object_count[object_category_id] + 1,
            "Left": math.floor(x_min),
            "Top": math.floor(y_min),
            "Width": math.ceil(width),
            "Height": math.ceil(height),
            "Score": round(float(prediction.score.value), 3),
            "Text": f"{object_category} - no. {str(category_object_count[object_category_id] + 1)}",
        })

        category_object_count[object_category_id] = category_object_count[object_category_id] + 1

        # Add current object to symbol with text list
        if object_category in settings.SYMBOL_WITH_TEXT:
            symbol_with_text.append(table_data[-1])

    print(category_names)

    # Log the number of objects found for each category
    logger.log(logging.INFO, f"Number of objects found for each category.")
    for i in range(len(category_object_count)):
        if category_object_count[i] > 0:
            # type: ignore
            logger.log(
                logging.INFO, f"Category {i}: {detection_model.category_mapping[str(i)]} - {category_object_count[i]}")

    if text_OCR:
        # Extract the text from prediciton
        output_text = output_text + \
            f": {len(symbol_with_text)} objects with text."
        print("Found", len(symbol_with_text), "object to be text.")
        logger.log(
            logging.INFO, f"Found {len(symbol_with_text)} object to be text.")
        # Delete all files in text directory
        logger.log(
            logging.INFO, f"Deleting all files in {settings.TEXT_PATH}.")
        delete_all_files_in_folder(settings.TEXT_PATH)

        # Extract text from image and save to text directory
        logger.log(
            logging.INFO, f"Extracting text from image and saving to {settings.TEXT_PATH}.")
        text_list = extract_text_from_image(image, symbol_with_text)

        # Update table_data with text
        logger.log(logging.INFO, f"Updating table_data with text.")
        if len(text_list) > 0:
            for i in range(len(text_list)):
                index = symbol_with_text[i]["Index"] - 1
                txt_to_display = " ".join(text_list[i])
                print(txt_to_display)
                table_data[index]["Text"] = txt_to_display

    # sort table_data by 'CategoryID' then 'ObjectID'
    logger.log(
        logging.INFO, f"Sorting table_data by 'CategoryID' then 'ObjectID'.")
    sorted_data = sorted(table_data, key=lambda x: (
        x['CategoryID'], x['ObjectID']))

    for i in range(len(sorted_data)):
        sorted_data[i]["Index"] = i + 1

    # Convert data table to JSON data
    json_data = json.dumps(sorted_data)

    # save category id and name for create checkbox table
    category_ids_list = list(category_ids)
    category_ids_list.sort()
    category_mapping = list(
        detection_model.category_mapping.values())  # type: ignore
    category_id_found = [item["CategoryID"] for item in table_data]

    checkboxes = []
    for i in range(len(category_ids)):
        checkboxes.append({
            "id": category_ids_list[i],
            "desc": category_mapping[category_ids_list[i]],
            "count": category_id_found.count(category_ids_list[i]),
        })

    # Save json data to file in output directory
    logger.log(
        logging.INFO, f"Saving json data to file in {settings.OUTPUT_PATH}.")
    with open(os.path.join(settings.OUTPUT_PATH, "data.json"), "w") as f:
        json.dump(sorted_data, f)

    # Export sorted_data to excel
    logger.log(
        logging.INFO, f"Exporting sorted_data to excel in {settings.OUTPUT_PATH}.")
    df = pd.DataFrame(sorted_data)
    df.to_excel(os.path.join(settings.OUTPUT_PATH, "data.xlsx"), index=False)

    logger.log(
        logging.INFO, f"End inferencing image and text with selected model {selected_model}.")
    # Generate timestamp for cache busting to ensure browser loads new prediction image
    cache_buster = datetime.datetime.now().timestamp()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "run_flag": True,
            "cache_buster": cache_buster,
            "table_data": sorted_data,
            "json_data": json_data,
            "weight_files": MODEL_LIST,
            "config_files": CONFIG_FILE_LIST,
            "model_types": settings.MODEL_TYPES,
            "input_filename": input_filename,
            "output_text": output_text,
            "category_id": checkboxes,
        }
    )

if __name__ == "__main__":
    logger.log(logging.INFO, f"* *********************************** *")
    logger.log(logging.INFO, f"*           Starting GARNET           *")
    logger.log(logging.INFO, f"* *********************************** *")
    uvicorn.run("main:app", reload=True)
