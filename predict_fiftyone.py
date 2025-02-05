import os
import glob
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
    Process all images in a folder with SAHI slicing predictions and visualize in FiftyOne.
    """
    # Load the model with the provided parameters.
    detection_model = AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=model_path,
        confidence_threshold=model_confidence_threshold,
    )
    
    # Create a new FiftyOne dataset
    dataset = fo.Dataset(name="PID_SAHI_Folder_Predictions")
    
    # Define image extensions to consider
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    
    # Recursively find image files in the folder
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_path, f"**/*{ext}"), recursive=True))
    
    if not image_paths:
        print("No images found in the folder.")
        return

    # Process each image
    for image_path in image_paths:
        # Open image to get dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        # Run SAHI sliced inference with additional parameters.
        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=image_size,
            slice_width=image_size,
            overlap_height_ratio=overlab_ratio,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
        )

        # Convert predictions to FiftyOne format
        detections = []
        for pred in result.object_prediction_list:
            # Convert bbox from (x, y, w, h)
            bbox = pred.bbox.to_xywh()
            # Normalize coordinates with respect to image dimensions
            bbox[0] /= image_width
            bbox[1] /= image_height
            bbox[2] /= image_width
            bbox[3] /= image_height

            detections.append(
                fo.Detection(
                    label=pred.category.name,
                    bounding_box=bbox,
                    confidence=pred.score.value,
                )
            )

        # Create a sample and add predictions
        sample = fo.Sample(filepath=image_path)
        sample["predictions"] = fo.Detections(detections=detections)
        dataset.add_sample(sample)
    
    # Launch the FiftyOne app to visualize the dataset
    session = fo.launch_app(dataset)
    session.wait()

# Example usage:
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
