import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLO from ultralytics


def refine_bounding_box(image, bbox, class_name):
    """
    Refines a bounding box based on connected lines in a P&ID.

    Args:
        image: The input image (NumPy array, grayscale or BGR).
        bbox: The bounding box (x, y, width, height), where (x, y) is the top-left corner.
        class_name: The predicted class name (string).

    Returns:
        A tuple: (refined_bbox, changes_made), where:
          - refined_bbox is the adjusted bounding box (x, y, w, h) or None if no refinement was possible.
          - changes_made is a boolean indicating whether the bounding box was actually changed.
    """

    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)  # make them integer
    changes_made = False

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # --- Thresholding (critical for line detection) ---
    _, thresh = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY_INV
    )  # Invert for line detection (lines become white)

    def check_line_continuity(thresh_image, start_x, start_y, dx, dy, max_steps, min_line_length):
        """
        Checks for line continuity from a starting point in a given direction.
        """
        x, y = start_x, start_y
        steps = 0
        for _ in range(max_steps):
            if not (0 <= x < thresh_image.shape[1] and 0 <= y < thresh_image.shape[0]):
                break  # Out of bounds

            if thresh_image[y, x] == 255:  # Found a white pixel (part of the line)
                steps += 1
            else:
                break  # no continue line

            x += dx
            y += dy
        return steps > min_line_length

    # --- Refinement Parameters (Tune these!) ---
    line_check_margin = 5  # How far from the edge to start checking
    max_steps = 20  # Maximum steps to trace a line
    min_line_length = 5  # Minimum line length to be considered a connection
    shrink_amount = 2  # Amount to shrink the bounding box per step (pixels)

    refined_bbox = list(bbox)  # Create a mutable copy
    # --- Check Top Edge ---
    for i in range(line_check_margin, w - line_check_margin):
        if check_line_continuity(thresh, x + i, y + line_check_margin, 0, -1, max_steps, min_line_length):
            refined_bbox[1] += shrink_amount  # Move top edge down
            refined_bbox[3] -= shrink_amount  # Reduce height
            changes_made = True

    # --- Check Bottom Edge ---
    for i in range(line_check_margin, w - line_check_margin):
        if check_line_continuity(thresh, x + i, y + h - line_check_margin, 0, 1, max_steps, min_line_length):
            refined_bbox[3] -= shrink_amount  # Reduce height
            changes_made = True

    # --- Check Left Edge ---
    for i in range(line_check_margin, h - line_check_margin):
        if check_line_continuity(thresh, x + line_check_margin, y + i, -1, 0, max_steps, min_line_length):
            refined_bbox[0] += shrink_amount  # move the x to right
            refined_bbox[2] -= shrink_amount  # reduce the width
            changes_made = True

    # --- Check Right Edge ---
    for i in range(line_check_margin, h - line_check_margin):
        if check_line_continuity(thresh, x + w - line_check_margin, y + i, 1, 0, max_steps, min_line_length):
            refined_bbox[2] -= shrink_amount  # reduce the width
            changes_made = True

    # Ensure the refined bounding box is valid.
    refined_bbox[2] = max(0, refined_bbox[2])  # Width cannot be negative
    refined_bbox[3] = max(0, refined_bbox[3])  # Height cannot be negative

    return tuple(refined_bbox), changes_made


def get_yolov8_predictions(image_path, model_path, conf_thres=0.5):
    """
    Gets YOLOv8 predictions for an image.

    Args:
        image_path: Path to the input image.
        model_path: Path to the YOLOv8 model (.pt file).
        conf_thres: Confidence threshold for filtering predictions.

    Returns:
        A list of tuples: (bbox, class_name, confidence), where:
          - bbox is (x, y, width, height)
          - class_name is the predicted class name (string)
          - confidence is the prediction confidence (float)
    """

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Run inference
    results = model(image, conf=conf_thres)  # Perform object detection

    predictions = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get bounding box coordinates
            w = x2 - x1
            h = y2 - y1
            bbox = (float(x1), float(y1), float(w), float(h))  # Convert to (x, y, w, h)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]  # Get class name from model
            
            predictions.append((bbox, class_name, confidence))
    return predictions, image



def main():
    image_path = 'test/!test01.png'  # Replace with your image path
    model_path = 'yolo_weights/yolo11s_PPCL_640_20250207.pt'  # Replace with your YOLOv8 model path

    # Get YOLOv8 predictions
    try:
        predictions, image = get_yolov8_predictions(image_path, model_path)
    except ValueError as e:
        print(e)
        return

    # Refine bounding boxes and visualize
    for bbox, class_name, confidence in predictions:
        refined_bbox, changed = refine_bounding_box(image.copy(), bbox, class_name)  # Refine
        print(f"Original bbox: {bbox}, Refined bbox: {refined_bbox}, Changed: {changed}, Class: {class_name}, Confidence: {confidence}")

        x, y, w, h = bbox
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)  # Original in red

        if changed:
            x_r, y_r, w_r, h_r = refined_bbox
            cv2.rectangle(image, (int(x_r), int(y_r)), (int(x_r + w_r), int(y_r + h_r)), (0, 255, 0), 2)  # Refined in green


    cv2.imshow('Refined Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()