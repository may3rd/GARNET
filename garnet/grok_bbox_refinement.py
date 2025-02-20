import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

class BoundingBoxRefiner:
    def __init__(self, image_path: str, white_threshold: int = 200, min_content_size: int = 5):
        """Initialize the refiner with image and parameters"""
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")
        self.white_threshold = white_threshold  # Threshold for considering a pixel "white"
        self.min_content_size = min_content_size  # Minimum size of content to stop refinement
        self.target_classes = ['gate valve', 'reducer', 'instrument tag', 'check valve', 'pump', 'globe valve'
                               , 'ball valve', 'butterfly valve', 'plug valve', 'control valve', 'strainer']

    def is_white_or_sparse(self, region: np.ndarray) -> bool:
        """Check if a region is mostly white or contains only sparse small content"""
        # Count non-white pixels (assuming black content on white background)
        non_white_pixels = np.sum(region < self.white_threshold)
        
        # If very few non-white pixels, consider it empty
        if non_white_pixels == 0:
            return True
        
        # Check if the content is just small sparse pixels
        if non_white_pixels < self.min_content_size:
            return True
        
        return False

    def refine_bbox(self, bbox: Tuple[int, int, int, int], class_name: str) -> Tuple[int, int, int, int]:
        """Refine a single bounding box by checking pixel-by-pixel"""
        if class_name not in self.target_classes:
            return bbox

        x_min, y_min, x_max, y_max = bbox
        
        # Keep refining until all sides encounter significant content
        while True:
            changed = False

            # Check left side
            if x_max - x_min > self.min_content_size:  # Ensure box isn't too small
                left_edge = self.image[y_min:y_max, max(0, x_min):x_min+1]
                if self.is_white_or_sparse(left_edge):
                    x_min += 1
                    changed = True

            # Check right side
            if x_max - x_min > self.min_content_size:
                right_edge = self.image[y_min:y_max, x_max-1:x_max]
                if self.is_white_or_sparse(right_edge):
                    x_max -= 1
                    changed = True

            # Check top side
            if y_max - y_min > self.min_content_size:
                top_edge = self.image[max(0, y_min):y_min+1, x_min:x_max]
                if self.is_white_or_sparse(top_edge):
                    y_min += 1
                    changed = True

            # Check bottom side
            if y_max - y_min > self.min_content_size:
                bottom_edge = self.image[y_max-1:y_max, x_min:x_max]
                if self.is_white_or_sparse(bottom_edge):
                    y_max -= 1
                    changed = True

            # Stop if no changes were made (all sides hit content)
            if not changed:
                break

        # Ensure valid coordinates
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        return (x_min, y_min, x_max, y_max)

    def process_detections(self, detections: List[Dict]) -> List[Dict]:
        """Process all detections from YOLOv8"""
        refined_detections = []
        
        for detection in detections:
            refined_bbox = self.refine_bbox(detection['bbox'], detection['class'])
            refined_detection = detection.copy()
            refined_detection['bbox'] = refined_bbox
            refined_detections.append(refined_detection)
        
        return refined_detections

    def visualize_results(self, original_detections: List[Dict], refined_detections: List[Dict], output_path: str):
        """Visualize original and refined bounding boxes"""
        color_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        for det in original_detections:
            if det['class'] in self.target_classes:
                x_min, y_min, x_max, y_max = det['bbox']
                cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
                cv2.putText(color_img, f"{det['class']} ({det['confidence']:.2f})", 
                           (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        for det in refined_detections:
            if det['class'] in self.target_classes:
                x_min, y_min, x_max, y_max = det['bbox']
                cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                cv2.putText(color_img, f"{det['class']} ({det['confidence']:.2f})", 
                           (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imwrite(output_path, color_img)

def main():
    image_path = 'test/!test02.png'
    model_path = 'yolo_weights/best.pt'
    output_path = 'output_refined.png'
    
    white_threshold = 200
    min_content_size = 3

    try:
        model = YOLO(model_path)
        results = model(image_path)
    except Exception as e:
        print(f"Error with YOLO processing: {e}")
        return

    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detections.append({
                'class': model.names[int(box.cls)],
                'bbox': [int(x) for x in box.xyxy[0].tolist()],
                'confidence': float(box.conf)
            })

    print("Original detections:")
    for det in detections:
        print(f"Class: {det['class']}, Bbox: {det['bbox']}, Confidence: {det['confidence']:.2f}")

    try:
        refiner = BoundingBoxRefiner(image_path, white_threshold=white_threshold, min_content_size=min_content_size)
        refined_detections = refiner.process_detections(detections)

        print("\nRefined detections:")
        for det in refined_detections:
            print(f"Class: {det['class']}, Bbox: {det['bbox']}, Confidence: {det['confidence']:.2f}")

        refiner.visualize_results(detections, refined_detections, output_path)
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"Error in refinement process: {e}")

if __name__ == "__main__":
    main()