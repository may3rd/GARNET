import cv2
import numpy as np
from typing import List, Dict, Tuple
import os
from multiprocessing import Pool, cpu_count
from functools import partial

class BoundingBoxTemplateDetector:
    def __init__(self, image_path: str, template_dir: str = 'matching_templates', output_path: str = 'predictions/template_detection_visualization.png'):
        """Initialize with image and template directory"""
        self.output_path = output_path
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")
        self.template_classes = ['ball valve', 'gate valve', 'reducer', 'check valve', 'strainer', 'spectacle blind']
        
        # Load multiple templates per class
        self.templates = {}
        for cls in self.template_classes:
            self.templates[cls] = []
            cls_dir = os.path.join(template_dir, cls)
            if os.path.exists(cls_dir):
                for filename in os.listdir(cls_dir):
                    if filename.endswith('.png'):
                        template_path = os.path.join(cls_dir, filename)
                        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                        if template is not None:
                            self.templates[cls].append((filename, template))
                if not self.templates[cls]:
                    print(f"Warning: No templates found for {cls}")
            else:
                print(f"Warning: Directory {cls_dir} not found")

    def get_transformed_templates(self, template: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate flipped and rotated versions of a template"""
        transforms = {}
        rotation_angles = [0, 90, 180, 270]
        transforms['original'] = template
        transforms['h_flip'] = cv2.flip(template, 1)
        transforms['v_flip'] = cv2.flip(template, 0)
        for angle in rotation_angles:
            if angle != 0:
                center = (template.shape[1] // 2, template.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(template, matrix, (template.shape[1], template.shape[0]))
                transforms[f'rot_{angle}'] = rotated
        return transforms

    def compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def combine_overlapping_detections(self, detections: List[Dict], iou_threshold: float = 0.9) -> List[Dict]:
        """Combine overlapping detections using NMS"""
        if not detections:
            return []
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        selected_detections = []
        for det in sorted_detections:
            is_overlapping = False
            for selected in selected_detections:
                iou = self.compute_iou(det['bbox'], selected['bbox'])
                if iou > iou_threshold:
                    is_overlapping = True
                    break
            if not is_overlapping:
                selected_detections.append(det)
        return selected_detections

    def match_template_worker(self, args: Tuple[str, str, np.ndarray, float]) -> List[Dict]:
        """
        Worker function for parallel template matching.
        Args:
            args: Tuple of (class_name, transform_name, template, min_confidence)
        Returns:
            List of detections for this template
        """
        cls, transform_name, template, min_confidence = args
        detections = []
        
        # Perform template matching
        result = cv2.matchTemplate(self.image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= min_confidence)
        
        for pt in zip(*locations[::-1]):  # y, x
            x_min, y_min = pt
            x_max = x_min + template.shape[1]
            y_max = y_min + template.shape[0]
            detection = {
                'class': cls,
                'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                'confidence': float(result[int(y_min), int(x_min)])
            }
            detections.append(detection)
        
        return detections

    def detect_with_templates(self, min_confidence: float = 0.7) -> List[Dict]:
        """Detect symbols using parallel template matching"""
        color_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        all_detections = []

        # Prepare tasks for parallel processing
        tasks = []
        for cls in self.template_classes:
            for template_name, template in self.templates[cls]:
                template_variants = self.get_transformed_templates(template)
                for transform_name, transformed_template in template_variants.items():
                    tasks.append((cls, transform_name, transformed_template, min_confidence))

        # Use multiprocessing pool
        num_processes = min(cpu_count(), len(tasks))  # Use available CPUs or number of tasks
        print(f"Running template matching with {num_processes} processes...")
        with Pool(processes=num_processes) as pool:
            # Map tasks to worker function
            results = pool.map(self.match_template_worker, tasks)

        # Aggregate results
        for result in results:
            all_detections.extend(result)

        # Combine overlapping detections
        combined_detections = self.combine_overlapping_detections(all_detections, iou_threshold=0.85)

        # Visualization
        for det in combined_detections:
            x_min, y_min, x_max, y_max = det['bbox']
            cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.putText(color_img, f"{det['class']} ({det['confidence']:.2f})",
                        (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if combined_detections:
            cv2.imwrite(self.output_path, color_img)

        return combined_detections

def main():
    image_path = 'test/ppcl/Test-00001.jpg'
    template_dir = 'matching_templates'
    output_path = 'predictions/template_detection_visualization.png'

    try:
        detector = BoundingBoxTemplateDetector(image_path, template_dir, output_path)
        detections = detector.detect_with_templates(min_confidence=0.7)

        print("Template-based detections (after combining overlaps):")
        for det in detections:
            print(f"Class: {det['class']}, Bbox: {det['bbox']}, Confidence: {det['confidence']:.2f}")
        
        print(f"\nResults saved to {output_path}")

    except Exception as e:
        print(f"Error in template detection process: {e}")

if __name__ == "__main__":
    main()