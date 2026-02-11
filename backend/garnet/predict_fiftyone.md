# SAHI-FiftyOne Inference and Export Tool

This Python script performs the following tasks:

1. **Inference & Visualization:**

    - Processes all images within a given folder using SAHI (Slicing Aided Hyper Inference) for object detection.
    - Visualizes the detection results using the FiftyOne app.

2. **Exporting Results:**
    - Saves the inference results as a COCO-format JSON file.
    - Exports per-image detection details to an Excel workbook, with each image’s detections in a separate worksheet.

## Table of Contents

-   [Usage](#usage)
-   [Function Documentation](#function-documentation)
    -   [visualize_folder_in_fiftyone](#visualize_folder_in_fiftyone)
    -   [save_predictions_to_coco](#save_predictions_to_coco)
    -   [save_predictions_to_excel](#save_predictions_to_excel)
-   [Example](#example)

## Usage

Run the script from the command line. You can adjust the parameters in the `if __name__ == "__main__":` block, or import the functions into your own code.

For example, to process images in the folder `test/fiftyone_test` using a YOLOv8 model, execute:

```bash
python predict_fiftyone.py
```

The script will:

-   Process all images with extensions such as PNG, JPG, JPEG, BMP, TIF, and TIFF found recursively in the specified folder.
-   Perform object detection using SAHI slicing.
-   Create a FiftyOne dataset for visualization.
-   Export the predictions to `predictions_coco.json` (COCO JSON file) and `predictions.xlsx` (Excel workbook).

## Function Documentation

### `visualize_folder_in_fiftyone`

```python
def visualize_folder_in_fiftyone(
    image_path: str,
    model_type: str = "yolov8",
    model_path: str = "../yolo_weights/yolo11n_PPCL_640_20250204.pt",
    model_confidence_threshold: float = 0.5,
    image_size: int = 640,
    overlab_ratio: float = 0.5,
    preform_standard_pred: bool = True,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOU",
    postprocess_match_threshold: float = 0.2,
    postprocess_class_agnostic: bool = True,
):
```

**Purpose:**

-   Processes all images in the specified folder using a SAHI slicing prediction pipeline.
-   Visualizes the predictions in the FiftyOne web app.
-   Exports the results to a COCO JSON file and an Excel workbook.

**Parameters:**

-   `image_path` (str): Folder path containing images.
-   `model_type` (str): Model type used by SAHI (default `"yolov8"`).
-   `model_path` (str): Path to the pre-trained model file.
-   `model_confidence_threshold` (float): Confidence threshold for filtering predictions.
-   `image_size` (int): Size (in pixels) to which images are resized for inference slicing.
-   `overlab_ratio` (float): Overlap ratio used during slicing.
-   `preform_standard_pred` (bool): Flag to select SAHI's standard prediction format.
-   `postprocess_type` (str): Postprocessing algorithm to use (e.g., `"GREEDYNMM"` or `"NMS"`).
-   `postprocess_match_metric` (str): Metric for postprocessing matching (e.g., `"IOU"`).
-   `postprocess_match_threshold` (float): Threshold for matching during postprocessing.
-   `postprocess_class_agnostic` (bool): If `True`, performs class-agnostic postprocessing.

**Workflow:**

-   Loads the detection model using SAHI.
-   Recursively scans the input folder for image files.
-   For each image:
    -   Reads image dimensions.
    -   Runs SAHI sliced inference.
    -   Converts and normalizes bounding box coordinates.
    -   Adds a sample with predictions to the FiftyOne dataset.
    -   Prepares per-image detection data for Excel export.
-   Exports the results:
    -   Calls `save_predictions_to_coco()` to write a COCO JSON file.
    -   Calls `save_predictions_to_excel()` to create an Excel workbook.
-   Launches the FiftyOne app to display the dataset.

---

### `save_predictions_to_coco`

```python
def save_predictions_to_coco(dataset: fo.Dataset, output_path: str):
```

**Purpose:**

-   Converts the FiftyOne dataset’s predictions into the COCO JSON format and writes the result to a specified file.

**Parameters:**

-   `dataset` (fo.Dataset): FiftyOne dataset containing the predictions.
-   `output_path` (str): Output file path for the JSON file (e.g., `"predictions_coco.json"`).

**Workflow:**

-   Iterates over each sample in the dataset.
-   Extracts image dimensions and file names.
-   Processes each detection, mapping labels to unique category IDs and converting normalized bounding boxes back to absolute pixel coordinates.
-   Constructs the `images`, `annotations`, and `categories` sections of the COCO JSON structure.
-   Writes the JSON to the output file.

---

### `save_predictions_to_excel`

```python
def save_predictions_to_excel(excel_data: dict, output_path: str):
```

**Purpose:**

-   Exports the detection results to an Excel workbook.
-   Each image’s detections are written to a separate worksheet named after the image (with restrictions applied for safe sheet naming).

**Parameters:**

-   `excel_data` (dict): A dictionary where keys are sheet names and values are lists of detection dictionaries.
-   `output_path` (str): Output file path for the Excel workbook (e.g., `"predictions.xlsx"`).

**Workflow:**

-   Uses `pandas.ExcelWriter` (with `openpyxl`) as a context manager to create and write to an Excel file.
-   Iterates over the dictionary and writes each list of detections to its corresponding worksheet.

---

## Example

Below is a complete example of how to run the tool:

```python
if __name__ == "__main__":
    visualize_folder_in_fiftyone(
        image_path="test/fiftyone_test",
        model_type="yolov8",
        model_path="yolo_weights/yolo11n_PPCL_640_20250204.pt",
        model_confidence_threshold=0.5,
        image_size=640,
        overlab_ratio=0.1,
        preform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.2,
        postprocess_class_agnostic=True,
    )
```

**What Happens:**

-   The script scans `test/fiftyone_test` for images.
-   It runs object detection on each image using the specified YOLOv8 model.
-   The detection results are displayed in the FiftyOne app.
-   Results are exported to `predictions_coco.json` (in COCO format) and `predictions.xlsx` (with each image’s detections on a separate sheet).
-   The FiftyOne app is launched to display the dataset.
