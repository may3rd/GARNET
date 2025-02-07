import os
import glob
import json
import pandas as pd
import fiftyone as fo
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

def visualize_folder_in_fiftyone(
    image_path: str,
    model_type: str = "yolov8",  # e.g., "yolov5", "mmdet", etc.
    model_path: str = "../yolo_weights/yolo11n_PPCL_640_20250204.pt",
    model_confidence_threshold: float = 0.5,
    image_size: int = 640,
    overlab_ratio: float = 0.5,
    preform_standard_pred: bool = True,
    postprocess_type: str = "GREEDYNMM",  # Options: None, "GREEDYNMM", "NMS", etc.
    postprocess_match_metric: str = "IOU",  # Options: "IOS", "IOU", etc.
    postprocess_match_threshold: float = 0.2,
    postprocess_class_agnostic: bool = True,
):
    """
    Process all images in a folder with SAHI slicing predictions, visualize in FiftyOne,
    and save prediction results in COCO JSON and Excel formats.
    """
    # Load the model with the provided parameters.
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=model_confidence_threshold,
    )
    
    dataset_name = "PID_SAHI_Folder_Predictions"
    
    # Check if dataset already exists and delete if necessary
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    # Create a new FiftyOne dataset
    dataset = fo.Dataset(name=dataset_name)
    
    # Define image extensions to consider
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    
    # Recursively find image files in the folder
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_path, f"**/*{ext}"), recursive=True))
    
    if not image_paths:
        print("No images found in the folder.")
        return

    # Sort the image paths alphabetically
    image_paths.sort()
    
    # This dictionary will hold predictions per image for Excel output
    excel_data = {}

    # Process each image
    for img_path in image_paths:
        # Open image to get dimensions
        with Image.open(img_path) as img:
            image_width, image_height = img.size

        # Run SAHI sliced inference with additional parameters.
        result = get_sliced_prediction(
            image=img_path,
            detection_model=detection_model,
            perform_standard_pred=preform_standard_pred,
            slice_height=image_size,
            slice_width=image_size,
            overlap_height_ratio=overlab_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
        )
        
        # Save the prediction result for visualization
        # Create a safe file name for the image
        image_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = "predictions"
        result.export_visuals(
            export_dir=output_path,
            file_name=image_name,
            text_size=1,
            rect_th=3,
            hide_conf=True,
            hide_labels=True
        )

        # Convert predictions to FiftyOne format
        detections = []
        # Prepare list of predictions for Excel output
        excel_rows = []
        for pred in result.object_prediction_list:
            # Convert bbox from (x, y, w, h) and normalize coordinates
            bbox = pred.bbox.to_xywh()
            # Normalize coordinates with respect to image dimensions
            norm_bbox = [
                bbox[0] / image_width,
                bbox[1] / image_height,
                bbox[2] / image_width,
                bbox[3] / image_height,
            ]
            detections.append(
                fo.Detection(
                    label=pred.category.name,
                    bounding_box=norm_bbox,
                    confidence=pred.score.value,
                )
            )
            # Also prepare data for Excel (convert normalized bbox back to pixel coordinates)
            pixel_bbox = [
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
            ]
            
            excel_rows.append({
                "category id": pred.category.id,
                "label": pred.category.name,
                "confidence": pred.score.value,
                "x": pixel_bbox[0],
                "y": pixel_bbox[1],
                "width": pixel_bbox[2],
                "height": pixel_bbox[3],
            })

        # Sort excel_rows by category id and confidence
        excel_rows.sort(key=lambda x: (x["category id"], x["confidence"]))
        
        # Create a sample, add metadata for image dimensions, and predictions
        sample = fo.Sample(filepath=img_path)
        sample.metadata = fo.ImageMetadata(width=image_width, height=image_height)
        sample["predictions"] = fo.Detections(detections=detections)
        dataset.add_sample(sample)

        # Save predictions for this image to excel_data using a safe sheet name
        sheet_name = os.path.splitext(os.path.basename(img_path))[0]
        # Sheet names must be <= 31 characters and not contain certain characters
        sheet_name = sheet_name[:31].replace(":", "_").replace("/", "_")
        excel_data[sheet_name] = excel_rows

    # Save predictions to COCO JSON file
    save_predictions_to_coco(dataset, output_path="predictions/predictions_coco.json")

    # Save predictions to an Excel file with each image's predictions in a separate sheet
    save_predictions_to_excel(excel_data, output_path="predictions/predictions.xlsx")
    
    # Launch the FiftyOne app to visualize the dataset
    session = fo.launch_app(dataset)
    session.wait()


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
        image_path="test/pttep",
        model_type="yolov8",
        model_path="yolo_weights/yolo11s_PTTEP_640_20250207.pt",
        model_confidence_threshold=0.5,
        image_size=640,
        overlab_ratio=0.1,
        preform_standard_pred=True,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.2,
        postprocess_class_agnostic=True,
    )
