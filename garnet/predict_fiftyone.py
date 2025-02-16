import os
import glob
import json
import pandas as pd
import fiftyone as fo
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import cv2
import numpy as np
import Settings
import easyocr
from PIL import Image

# Initialize global components once
reader = easyocr.Reader(["en"])
allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"'
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
settings = Settings.Settings()
SYMBOL_WITH_TEXT = set(settings.SYMBOL_WITH_TEXT)

# Disable FiftyOne analytics
os.environ["FIFTYONE_DISABLE_ANALYTICS"] = "True"

def visualize_folder_in_fiftyone(
    image_path: str,
    model_type: str = "yolov8",
    model_path: str = "../yolo_weights/yolo11n_PPCL_640_20250204.pt",
    output_path: str = "predictions",
    model_confidence_threshold: float = 0.5,
    image_size: int = 640,
    overlab_ratio: float = 0.5,
    preform_standard_pred: bool = True,
    postprocess_type: str = "GREEDYNMM",
    postprocess_match_metric: str = "IOU",
    postprocess_match_threshold: float = 0.2,
    postprocess_class_agnostic: bool = True,
    class_list: list = [],
    rect_th: int = 2,
    text_size: float = 0.3,
    hide_conf: bool = True,
    hide_labels: bool = True,
):
    """Optimized version of the original function with key improvements."""
    
        
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
        confidence_threshold=model_confidence_threshold,
        category_mapping=category_mapping,
    )

    # Dataset setup
    dataset_name = "PID_SAHI_Folder_Predictions"
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(name=dataset_name)

    # Find and sort images
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        image_paths.extend(glob.glob(os.path.join(image_path, "**", ext), recursive=True))
    
    if not image_paths:
        print("No images found.")
        return

    image_paths.sort()
    total_images = len(image_paths)
    excel_data = {}

    for idx, img_path in enumerate(image_paths):
        print(f"Processing {idx+1}/{total_images}: {img_path}")
        
        # Single image load with OpenCV
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
            
        # SAHI prediction
        result = get_sliced_prediction(
            image=img_rgb,
            detection_model=detection_model,
            perform_standard_pred=preform_standard_pred,
            slice_height=image_size,
            slice_width=image_size,
            overlap_height_ratio=overlab_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
            verbose=2,
        )

        # Visualization export
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # If class_list is provided, filter the predictions before visualization
        if class_list:
            filtered_preductions = [
                obj for obj in result.object_prediction_list if obj.category.name in class_list
            ]
            
            # Visulize only the filtered predictions
            visualize_object_predictions(
                image=img_rgb,
                object_prediction_list=filtered_preductions,
                text_size=text_size,
                rect_th=rect_th,
                hide_conf=hide_conf,
                hide_labels=hide_labels,
                output_dir=output_path,
                file_name=image_name,
            )
        else:
            result.export_visuals(
                export_dir=output_path,
                file_name=image_name,
                text_size=text_size,
                rect_th=rect_th,
                hide_conf=hide_conf,
                hide_labels=hide_labels
            )

        # Process detections
        detections = []
        excel_rows = []
        
        for pred in result.object_prediction_list:
            bbox = pred.bbox.to_xywh()
            x, y, width, height = map(int, bbox)
            
            # FiftyOne formatting
            norm_bbox = [
                bbox[0]/w,
                bbox[1]/h,
                bbox[2]/w,
                bbox[3]/h,
            ]
            detections.append(fo.Detection(
                label=pred.category.name,
                bounding_box=norm_bbox,
                confidence=pred.score.value,
            ))

            # OCR processing
            if pred.category.name in SYMBOL_WITH_TEXT:
                cropped = img_cv[y:y+height, x:x+width]
                processed = cv2.bitwise_not(
                    cv2.dilate(
                        cv2.bitwise_not(cropped),
                        kernel, 
                        iterations=1
                    )
                )
                ocr_result = reader.readtext(
                    processed,
                    decoder="wordbeamsearch",
                    batch_size=8,  # Increased batch size
                    paragraph=True,
                    detail=0,
                    text_threshold=0.7,
                    allowlist=allowlist,
                )
                text = " ".join(ocr_result) if ocr_result else "" # type: ignore
            else:
                text = ""

            excel_rows.append({
                "category id": pred.category.id,
                "label": pred.category.name,
                "confidence": pred.score.value,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "extracted_text": text,
                "ocr_confidence": 1.0 if text else 0.0,
            })

        # Dataset update
        sample = fo.Sample(filepath=img_path)
        sample.metadata = fo.ImageMetadata(width=w, height=h)
        sample["predictions"] = fo.Detections(detections=detections)
        dataset.add_sample(sample)

        # Excel data preparation
        excel_rows.sort(key=lambda x: (x["category id"], x["confidence"]))
        sheet_name = os.path.basename(img_path)[:31].replace(":", "_").replace("/", "_")
        excel_data[sheet_name] = excel_rows

    # Save outputs
    save_predictions_to_coco(dataset, os.path.join(output_path, "predictions_coco.json"))
    save_predictions_to_excel(excel_data, os.path.join(output_path, "predictions.xlsx"))
    
    session = fo.launch_app(dataset) # type: ignore
    session.wait()
    return

# Rest of the functions (save_predictions_to_coco, save_predictions_to_excel) remain the same


def save_predictions_to_coco(dataset: fo.Dataset, output_path: str):
    """
    Convert the dataset predictions to COCO format and save to a JSON file.
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Build a mapping from category label to id
    category_to_id = {}
    next_cat_id = 1
    annotation_id = 1
    image_id = 1

    for sample in dataset:
        # Use the sample metadata for width and height
        if sample.metadata is not None:
            width = sample.metadata.width
            height = sample.metadata.height
        else:
            # Fallback: load image to get dimensions
            with Image.open(sample.filepath) as img:
                width, height = img.size

        file_name = os.path.basename(sample.filepath)
        coco["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
        })

        # Process predictions for this sample
        if "predictions" in sample:
            for det in sample.predictions.detections:
                # Map the label to a category id
                if det.label not in category_to_id:
                    category_to_id[det.label] = next_cat_id
                    next_cat_id += 1

                # Convert normalized bbox back to absolute pixel coordinates
                x = det.bounding_box[0] * width
                y = det.bounding_box[1] * height
                w = det.bounding_box[2] * width
                h = det.bounding_box[3] * height

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_to_id[det.label],
                    "bbox": [x, y, w, h],
                    "score": det.confidence,
                    "iscrowd": 0,
                })
                annotation_id += 1

        image_id += 1

    # Build the categories list
    for label, cat_id in category_to_id.items():
        coco["categories"].append({
            "id": cat_id,
            "name": label,
        })

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=4)
    print(f"COCO JSON saved to {output_path}")


def save_predictions_to_excel(excel_data: dict, output_path: str):
    """
    Save predictions stored in excel_data (a dict of sheet_name -> list of dicts)
    to an Excel workbook.
    """
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, rows in excel_data.items():
            # Create a DataFrame for the sheet
            df = pd.DataFrame(rows)
            # Write the DataFrame to a sheet in the Excel workbook
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Excel file saved to {output_path}")
    

if __name__ == "__main__":
    # Example usage:
    visualize_folder_in_fiftyone(
        image_path="test/pttep/test",
        model_type="yolov8onnx",
        model_path="yolo_weights/yolo11s_PTTEP_640_20250207.onnx",
        output_path="predictions/pttep/test",
        model_confidence_threshold=0.5,
        image_size=640,
        overlab_ratio=0.2,
        preform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.1,
        postprocess_class_agnostic=True,
        class_list=["pressure relief valve", "control valve", "shutdown valve"],
        rect_th=4
    )
