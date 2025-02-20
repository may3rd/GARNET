import cv2
import numpy as np
from typing import List, Dict
import os

class BoundingBoxTemplateDetector:
    def __init__(self, image_path: str, template_dir: str = 'matching_templates'):
        """Initialize with image and template directory"""
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")
        self.template_classes = ['gate valve', 'reducer', 'check valve', 'strainer', 'spectacle blind']
        
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

    def detect_with_templates(self, min_confidence: float = 0.7) -> List[Dict]:
        """Detect symbols using template matching across the entire image"""
        detections = []
        color_img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)  # For visualization

        for cls in self.template_classes:
            for template_name, template in self.templates[cls]:
                template_variants = self.get_transformed_templates(template)
                for transform_name, transformed_template in template_variants.items():
                    # Use multi-scale matching if needed (simplified here with fixed scale)
                    result = cv2.matchTemplate(self.image, transformed_template, cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= min_confidence)
                    
                    # Extract bounding boxes
                    for pt in zip(*locations[::-1]):  # y, x
                        x_min, y_min = pt
                        x_max = x_min + transformed_template.shape[1]
                        y_max = y_min + transformed_template.shape[0]
                        
                        # Create detection dictionary
                        detection = {
                            'class': cls,
                            'bbox': [int(x_min), int(y_min), int(x_max), int(y_max)],
                            'confidence': float(result[y_min, x_min])  # Use match score as confidence
                        }
                        detections.append(detection)
                        
                        # Visualize (optional)
                        cv2.rectangle(color_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                        cv2.putText(color_img, f"{cls} ({detection['confidence']:.2f})", 
                                    (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Save visualization
        if detections:
            cv2.imwrite('template_detection_visualization.png', color_img)
        
        return detections

def main():
    # Define paths
    image_path = 'test/001.png'  # Update with your P&ID image path
    template_dir = 'matching_templates'
    output_path = 'template_detection_visualization.png'

    try:
        # Initialize detector
        detector = BoundingBoxTemplateDetector(image_path, template_dir)
        
        # Perform template-based detection
        detections = detector.detect_with_templates(min_confidence=0.7)
        
        # Print detections
        print("Template-based detections:")
        for det in detections:
            print(f"Class: {det['class']}, Bbox: {det['bbox']}, Confidence: {det['confidence']:.2f}")
        
        print(f"\nResults saved to {output_path}")

    except Exception as e:
        print(f"Error in template detection process: {e}")

if __name__ == "__main__":
    main()