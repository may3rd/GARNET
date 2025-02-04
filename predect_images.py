""" 
    This script demonstrates how to use SAHI to predict objects in images and save the predictions in JSON and CSV format.
"""

import os
import cv2
import json
import pandas as pd
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from collections import defaultdict
import numpy as np

# Helper class to convert numpy types to JSON-serializable types
class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Define the function to predict images and save the predictions in JSON and CSV format
def predict_images(
    image_path: str = "test/images",  # Replace with your image folder path 
    output_path: str = "",  # Replace with your JSON output folder
    output_csv_path: str = "",  # Replace with your CSV output path
    model_type: str = "yolov8onnx",  # Replace with your model type (e.g., "yolov5", "mmdet", etc.)
    model_path: str = "yolo_weights/yolo11n_PPCL_640_20250203.onnx",  # Replace with your SAHI-compatible model path
    model_config_path: str = "datasets/yaml/data.yaml",  # Replace with your model config path
    model_confidence_threshold: float = 0.7,  # Confidence threshold for predictions
    image_size: int = 640,  # Resize images to this size before running inference
    overlab_ratio: float = 0.5,  # Overlap ratio for slicing
    preform_standard_pred: bool = True,  # Use SAHI's standard prediction format (True) or the format used in the SAHI paper (False)
    postprocess_type: str = "NMM", # Prostprocessing algorithm to use (None, "GREEDYNMM", "NMS")
    postprocess_match_metric: str = "IOU",  # Match metric for NMS postprocessing (IOS, IOU)
    postprocess_match_threshold: float = 0.2, # Match threshold for NMS postprocessing
    postprocess_class_agnostic: bool = True, # Class agnostic NMS for NMS postprocessing
    verbose: int = 0,  # Verbosity level for SAHI's prediction
    auto_slice_resolution: bool = True,  # Whether to automatically adjust slice resolution based on image size and model input size
    slice_export_prefix = None,  # Prefix for sliced image export
    slice_export_dir = None,  # Directory for sliced image export
):

    # Step 1: Set up paths and initialize variables
    
    # Create output folders if they don't exist
    if (not output_path or output_path == ""):
        output_path = os.path.join(image_path, "results")

    if (not output_csv_path or output_csv_path == ""):
        output_csv_path = os.path.join(output_path, "predictions.csv")
        
    # Create output folders if they don't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create sliced image export folder if specified
    if slice_export_dir is not None:
        os.makedirs(slice_export_dir, exist_ok=True)

    # Step 2: Initialize SAHI detection model

    # if model_type == "yolov8onnx" load category mapping from data.yaml
    if "yolov8onnx" == model_type:
        import onnx
        import ast
        model = onnx.load(model_path)
        props = { p.key: p.value for p in model.metadata_props }
        names = ast.literal_eval(props['names'])
        category_mapping = { str(key): value for key, value in names.items() }
    else:
        category_mapping = None
        
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        config_path=model_config_path,
        confidence_threshold=model_confidence_threshold,
        category_mapping=category_mapping,
        # device="cpu",  # Use "cuda" for GPU
    )

    # Step 3: Initialize data structures
    all_predictions = []
    class_counts_per_image = defaultdict(lambda: defaultdict(int))
    total_class_counts = defaultdict(int)

    # Step 4: Loop through images in the folder
    # Sort the images to ensure consistent ordering for reproducibility
    image_names = sorted(os.listdir(image_path))
    total_images = len(image_names)

    for image_name in image_names:
        if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            total_images -= 1
            continue  # Skip non-image files
    
    # Create an Excel writer object
    excel_output_path = os.path.join(output_path, "predictions.xlsx")
    
    with pd.ExcelWriter(excel_output_path, engine="openpyxl") as writer:
        image_count = 0
        for image_name in image_names:
            
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue  # Skip non-image files
            
            image_count += 1
            print(f"Predicting objects in image {image_count} of {total_images}: {image_name}")
            
            # Load image
            current_image_path = os.path.join(image_path, image_name)
            image = cv2.imread(current_image_path)

            # Step 5: Perform SAHI prediction
            result = get_sliced_prediction(
                image,
                detection_model,
                slice_height=image_size,  # Adjust slice height as needed
                slice_width=image_size,  # Adjust slice width as needed
                overlap_height_ratio=overlab_ratio,  # Adjust overlap ratio as needed
                overlap_width_ratio=overlab_ratio,  # Adjust overlap ratio as needed
                perform_standard_pred=preform_standard_pred,
                postprocess_type=postprocess_type,
                postprocess_match_metric=postprocess_match_metric,
                postprocess_match_threshold=postprocess_match_threshold,
                postprocess_class_agnostic=postprocess_class_agnostic,
                auto_slice_resolution=auto_slice_resolution,
                slice_export_prefix=slice_export_prefix,
                slice_dir=slice_export_dir,
                verbose=verbose,
            )
            
            # Save the prediction result for visualization
            result.export_visuals(
                export_dir=output_path,
                file_name=image_name,
                text_size=1,
                rect_th=3,
                hide_conf=True,
                hide_labels=True
            )

            # Step 6: Save predictions in JSON format
            predictions = []
            for detection in result.object_prediction_list:
                bbox = detection.bbox.to_voc_bbox()  # Get bounding box in [xmin, ymin, xmax, ymax] format
                class_name = detection.category.name
                confidence = detection.score.value

                predictions.append({
                    "image_name": image_name,
                    "class_name": class_name,
                    "confidence": np.float32(confidence),
                    "bbox": bbox,
                })

                # Update class counts
                class_counts_per_image[image_name][class_name] += 1
                total_class_counts[class_name] += 1

            # Save JSON for the current image, with numpy types converted to JSON-serializable types
            json_output_path = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}.json")
            with open(json_output_path, "w") as f:
                json.dump(predictions, f, indent=4, cls=NumpyTypeEncoder)

            # Create a Dataframe for the current image and save it to Excel
            image_df = pd.DataFrame(predictions)
            
            # Write the dataframe to a separate sheet in the Excel sheet
            sheet_name = os.path.splitext(image_name)[0]
            image_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Append predictions to the list for CSV export
            all_predictions.extend(predictions)

        # Step 7: Export predictions to CSV
        df = pd.DataFrame(all_predictions)
        df.to_csv(output_csv_path, index=False)

    # Step 8: Print class counts
    print("Class counts per image:")
    for image_name, counts in class_counts_per_image.items():
        print(f"Image: {image_name}")
        # sort by class name
        counts = dict(sorted(counts.items()))
        for class_name, count in counts.items():
            print(f"  {class_name}: {count}")

    print("\nTotal class counts:")
    # sort by class name
    total_class_counts = dict(sorted(total_class_counts.items()))
    for class_name, count in total_class_counts.items():
        print(f"  {class_name}: {count}")
        
if __name__ == "__main__":
    predict_images()
    