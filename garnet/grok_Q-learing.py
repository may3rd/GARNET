import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict
import random

class BoundingBoxRLRefiner:
    def __init__(self, image_path: str, learning_rate: float = 0.1, discount_factor: float = 0.9, epsilon: float = 0.1):
        """Initialize the RL refiner"""
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError(f"Could not load image at {image_path}")
        self.target_classes = ['gate valve', 'reducer', 'instrument tag', 'check valve']
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # State-action value table
        self.white_threshold = 200  # Threshold for white background

    def get_state(self, bbox: Tuple[int, int, int, int]) -> str:
        """Convert bounding box to a discrete state based on edge content"""
        x_min, y_min, x_max, y_max = bbox
        edges = {
            'left': np.mean(self.image[y_min:y_max, max(0, x_min):x_min+1]) > self.white_threshold,
            'right': np.mean(self.image[y_min:y_max, x_max-1:x_max]) > self.white_threshold,
            'top': np.mean(self.image[max(0, y_min):y_min+1, x_min:x_max]) > self.white_threshold,
            'bottom': np.mean(self.image[y_max-1:y_max, x_min:x_max]) > self.white_threshold
        }
        return str(edges)  # Simple state representation

    def get_actions(self) -> List[str]:
        """Define possible actions"""
        return ['left_in', 'right_in', 'top_in', 'bottom_in', 'stay']

    def apply_action(self, bbox: Tuple[int, int, int, int], action: str) -> Tuple[int, int, int, int]:
        """Apply an action to the bounding box"""
        x_min, y_min, x_max, y_max = bbox
        if action == 'left_in' and x_max - x_min > 5:
            x_min += 1
        elif action == 'right_in' and x_max - x_min > 5:
            x_max -= 1
        elif action == 'top_in' and y_max - y_min > 5:
            y_min += 1
        elif action == 'bottom_in' and y_max - y_min > 5:
            y_max -= 1
        return (x_min, y_min, x_max, y_max)

    def get_reward(self, bbox: Tuple[int, int, int, int], prev_bbox: Tuple[int, int, int, int]) -> float:
        """Calculate reward based on box fit"""
        x_min, y_min, x_max, y_max = bbox
        # Check if edges are on content (non-white)
        left_edge = np.mean(self.image[y_min:y_max, max(0, x_min):x_min+1]) < self.white_threshold
        right_edge = np.mean(self.image[y_min:y_max, x_max-1:x_max]) < self.white_threshold
        top_edge = np.mean(self.image[max(0, y_min):y_min+1, x_min:x_max]) < self.white_threshold
        bottom_edge = np.mean(self.image[y_max-1:y_max, x_min:x_max]) < self.white_threshold
        
        # Reward for tight fit: all edges on content
        if left_edge and right_edge and top_edge and bottom_edge:
            return 10.0  # High reward for perfect fit
        # Penalty for moving away from content or leaving empty space
        elif (x_max - x_min) < 5 or (y_max - y_min) < 5:
            return -5.0  # Penalty for too small
        elif not (left_edge or right_edge or top_edge or bottom_edge):
            return -1.0  # Penalty for empty space
        return 0.1  # Small reward for progress

    def refine_bbox(self, bbox: Tuple[int, int, int, int], class_name: str, max_steps: int = 50) -> Tuple[int, int, int, int]:
        """Refine a bounding box using Q-learning"""
        if class_name not in self.target_classes:
            return bbox

        current_bbox = bbox
        for _ in range(max_steps):
            state = self.get_state(current_bbox)
            actions = self.get_actions()

            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                action = random.choice(actions)  # Explore
            else:
                # Exploit: choose best action from Q-table
                if state not in self.q_table:
                    self.q_table[state] = {a: 0.0 for a in actions}
                action = max(self.q_table[state], key=self.q_table[state].get)

            # Apply action and get new state
            next_bbox = self.apply_action(current_bbox, action)
            next_state = self.get_state(next_bbox)
            reward = self.get_reward(next_bbox, current_bbox)

            # Update Q-table
            if state not in self.q_table:
                self.q_table[state] = {a: 0.0 for a in actions}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0.0 for a in actions}
            
            current_q = self.q_table[state][action]
            next_max_q = max(self.q_table[next_state].values())
            new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
            self.q_table[state][action] = new_q

            # Update current box
            if reward == 10.0 or next_bbox == current_bbox:  # Stop if perfect or no change
                break
            current_bbox = next_bbox

        return current_bbox

    def process_detections(self, detections: List[Dict]) -> List[Dict]:
        """Process all detections"""
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
        refiner = BoundingBoxRLRefiner(image_path)
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