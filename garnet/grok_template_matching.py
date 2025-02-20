import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import os
import easyocr
import re

class BoundingBoxTemplateRefiner:
    def __init__(self, image_path: str, template_dir: str = 'matching_templates'):
        """Initialize with image and template directory"""
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")
        self.template_classes = ['gate valve', 'reducer', 'check valve', 'strainer', 'spectacle blind']  # Added 'spectacle blind'
        self.circle_classes = ['instrument tag', 'instrument dcs']
        self.text_classes = ['line number', 'instrument tag', 'instrument dcs']
        self.rotation_angles = [0, 90, 180, 270]

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

        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA

    def get_transformed_templates(self, template: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate flipped and rotated versions of a template"""
        transforms = {}
        transforms['original'] = template
        transforms['h_flip'] = cv2.flip(template, 1)
        transforms['v_flip'] = cv2.flip(template, 0)
        for angle in self.rotation_angles:
            if angle != 0:
                center = (template.shape[1] // 2, template.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(template, matrix, (template.shape[1], template.shape[0]))
                transforms[f'rot_{angle}'] = rotated
        return transforms

    def refine_template_bbox(self, bbox: Tuple[int, int, int, int], class_name: str) -> Tuple[int, int, int, int]:
        """Refine bbox using multiple templates for template classes"""
        x_min, y_min, x_max, y_max = bbox
        padding = max(max(t[1].shape) for t in self.templates[class_name]) // 2
        region = self.image[
            max(0, y_min - padding):min(self.image.shape[0], y_max + padding),
            max(0, x_min - padding):min(self.image.shape[1], x_max + padding)
        ]

        best_score = -1.0
        best_bbox = bbox
        best_template_info = None

        for template_name, template in self.templates[class_name]:
            template_variants = self.get_transformed_templates(template)
            for transform_name, transformed_template in template_variants.items():
                if transformed_template.shape[0] > region.shape[0] or transformed_template.shape[1] > region.shape[1]:
                    continue
                result = cv2.matchTemplate(region, transformed_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    t_h, t_w = transformed_template.shape
                    new_x_min = max(0, x_min - padding + max_loc[0])
                    new_y_min = max(0, y_min - padding + max_loc[1])
                    new_x_max = min(self.image.shape[1], new_x_min + t_w)
                    new_y_max = min(self.image.shape[0], new_y_min + t_h)
                    best_bbox = (new_x_min, new_y_min, new_x_max, new_y_max)
                    best_template_info = (template_name, transform_name)

        if best_template_info:
            print(f"Best match for {class_name}: Template {best_template_info[0]}, Transform: {best_template_info[1]}, Score: {best_score:.2f}")
        else:
            print(f"No valid template match for {class_name}, keeping original bbox")
        return best_bbox

    def refine_circle_bbox(self, bbox: Tuple[int, int, int, int], class_name: str) -> Tuple[int, int, int, int]:
        """Refine bbox for circle-based symbols using Hough Circle Transform"""
        x_min, y_min, x_max, y_max = bbox
        padding = 10
        region = self.image[
            max(0, y_min - padding):min(self.image.shape[0], y_max + padding),
            max(0, x_min - padding):min(self.image.shape[1], x_max + padding)
        ]

        blurred = cv2.GaussianBlur(region, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        circles = cv2.HoughCircles(
            edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
            param1=50, param2=30, minRadius=5, maxRadius=max(region.shape) // 2
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            best_circle = max(circles, key=lambda x: x[2])
            cx, cy, r = best_circle
            new_x_min = max(0, x_min - padding + cx - r)
            new_y_min = max(0, y_min - padding + cy - r)
            new_x_max = min(self.image.shape[1], x_min - padding + cx + r)
            new_y_max = min(self.image.shape[0], y_min - padding + cy + r)
            
            print(f"Circle detected for {class_name}: Center ({cx}, {cy}), Radius {r}")
            return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))

        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest_contour)
            new_x_min = max(0, x_min - padding + cx)
            new_y_min = max(0, y_min - padding + cy)
            new_x_max = min(self.image.shape[1], new_x_min + cw)
            new_y_max = min(self.image.shape[0], new_y_min + ch)
            print(f"Contour fallback for {class_name}: Rect ({new_x_min}, {new_y_min}, {new_x_max}, {new_y_max})")
            return (int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max))

        print(f"No circle or contour detected for {class_name}, keeping original bbox")
        return bbox

    def extract_text(self, bbox: Tuple[int, int, int, int], class_name: str) -> str:
        """Extract text from the bounding box region using EasyOCR with class-specific handling"""
        x_min, y_min, x_max, y_max = bbox
        region = self.image[y_min:y_max, x_min:x_max]
        region_bgr = cv2.cvtColor(region, cv2.COLOR_GRAY2BGR)

        if class_name == 'line number':
            # Check rotations for line number
            results = self.reader.readtext(region_bgr, rotation_info=[0, 90, 180, 270])
            if not results:
                return "No text detected"
            best_result = max(results, key=lambda x: x[2])  # x[2] is confidence
            text = best_result[1].strip()
            print(f"Line number text: '{text}' (Confidence: {best_result[2]:.2f}, Rotation: {best_result[0]})")
            return text

        elif class_name in ['instrument tag', 'instrument dcs']:
            # No rotation, assume upright text
            results = self.reader.readtext(region_bgr, rotation_info=None)
            if not results:
                return "No text detected"
            
            if len(results) >= 2:
                # Sort by y-coordinate for two-line format
                sorted_results = sorted(results, key=lambda x: x[0][0][1])  # Top-left y-coord
                upper_text = sorted_results[0][1].strip()
                lower_text = sorted_results[1][1].strip()

                # Clean up lower text to remove special characters
                lower_text = re.sub(r'[^0-9A-Z]', '', lower_text)  # Keep only 0-9 and A-Z

                # Validate format: upper (2-3 chars), lower (2-5 digits + optional letter)
                upper_valid = bool(re.match(r'^[A-Z]{2,3}$', upper_text))
                lower_valid = bool(re.match(r'^\d{2,5}[A-Z]?$', lower_text))

                if upper_valid and lower_valid:
                    combined_text = f"{upper_text} {lower_text}"
                    print(f"{class_name} text: '{combined_text}' (Upper: {sorted_results[0][2]:.2f}, Lower: {sorted_results[1][2]:.2f})")
                    return combined_text
            
            # Fallback: Clean and combine all detected text
            text = ' '.join([re.sub(r'[^0-9A-Z]', '', res[1].strip()) for res in results])
            print(f"{class_name} fallback text: '{text}' (Detected lines: {len(results)})")
            return text

        return "No text detected"

    def refine_bbox(self, bbox: Tuple[int, int, int, int], class_name: str) -> Tuple[int, int, int, int]:
        """Refine bbox based on class type"""
        if class_name in self.template_classes and self.templates[class_name]:
            return self.refine_template_bbox(bbox, class_name)
        elif class_name in self.circle_classes:
            return self.refine_circle_bbox(bbox, class_name)
        return bbox

    def process_detections(self, detections: List[Dict]) -> List[Dict]:
        """Process all detections and extract text where applicable"""
        refined_detections = []
        for detection in detections:
            refined_bbox = self.refine_bbox(detection['bbox'], detection['class'])
            refined_detection = detection.copy()
            refined_detection['bbox'] = refined_bbox
            
            if detection['class'] in self.text_classes:
                text = self.extract_text(refined_bbox, detection['class'])
                refined_detection['text'] = text
            
            refined_detections.append(refined_detection)
        return refined_detections

    def visualize_results(self, original_detections: List[Dict], refined_detections: List[Dict], output_path: str):
        """Visualize original and refined bounding boxes with extracted text"""
        color_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for det in original_detections:
            x_min, y_min, x_max, y_max = det['bbox']
            cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
            cv2.putText(color_img, f"{det['class']} ({det['confidence']:.2f})", 
                        (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        for det in refined_detections:
            x_min, y_min, x_max, y_max = det['bbox']
            cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            label = f"{det['class']} ({det['confidence']:.2f})"
            if 'text' in det:
                label += f" - {det['text']}"
            cv2.putText(color_img, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, color_img)

def main():
    image_path = 'test/!test02.png'
    model_path = 'yolo_weights/best.pt'
    template_dir = 'matching_templates'
    output_path = 'predictions/output_refined.png'

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
        refiner = BoundingBoxTemplateRefiner(image_path, template_dir)
        refined_detections = refiner.process_detections(detections)

        print("\nRefined detections:")
        for det in refined_detections:
            text_info = f", Text: '{det['text']}'" if 'text' in det else ""
            print(f"Class: {det['class']}, Bbox: {det['bbox']}, Confidence: {det['confidence']:.2f}{text_info}")

        refiner.visualize_results(detections, refined_detections, output_path)
        print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        print(f"Error in refinement process: {e}")

if __name__ == "__main__":
    main()