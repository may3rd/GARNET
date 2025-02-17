import cv2
import numpy as np
import torch
import gym
from gym import spaces
from ultralytics import YOLO
from stable_baselines3 import PPO

# Load YOLO Model
yolo_model = YOLO("yolov8n.pt")  # Replace with your trained model if needed

def detect_symbols_yolo(image):
    """
    Detect symbols in a P&ID image using YOLOv8.
    Returns list of bounding boxes (x_min, y_min, x_max, y_max).
    """
    results = yolo_model(image)
    bboxes = []
    for result in results:
        for box in result.boxes.xyxy:  # Bounding boxes
            bboxes.append(tuple(map(int, box.tolist())))
    return bboxes

def refine_bbox_classical(image, bbox):
    """
    Uses classical image processing to refine the bounding box using edge detection and contours.
    """
    x_min, y_min, x_max, y_max = bbox
    roi = image[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = bbox
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        refined_x_min = x_min + x
        refined_y_min = y_min + y
        refined_x_max = refined_x_min + w
        refined_y_max = refined_y_min + h
        best_bbox = (refined_x_min, refined_y_min, refined_x_max, refined_y_max)

    return best_bbox

# Reinforcement Learning Environment for Bounding Box Optimization
class BoundingBoxEnv(gym.Env):
    def __init__(self, image, initial_bboxes):
        super(BoundingBoxEnv, self).__init__()
        self.image = image
        self.bboxes = initial_bboxes
        self.index = 0  # Current bounding box to refine
        
        # Actions: move left, right, up, down, increase/decrease size
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=image.shape, dtype=np.uint8)

    def step(self, action):
        x_min, y_min, x_max, y_max = self.bboxes[self.index]

        # Define actions: 0=left, 1=right, 2=up, 3=down, 4=expand, 5=shrink
        if action == 0: x_min -= 2
        elif action == 1: x_max += 2
        elif action == 2: y_min -= 2
        elif action == 3: y_max += 2
        elif action == 4: x_max += 2; y_max += 2  # Expand
        elif action == 5: x_min += 2; y_min += 2  # Shrink

        # Ensure the bounding box stays within image bounds
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(self.image.shape[1], x_max), min(self.image.shape[0], y_max)
        
        # Compute reward: IOU between refined box and classical segmentation
        refined_bbox = refine_bbox_classical(self.image, (x_min, y_min, x_max, y_max))
        iou = self._calculate_iou(refined_bbox, (x_min, y_min, x_max, y_max))
        reward = iou * 10  # Higher IOU = better refinement

        self.bboxes[self.index] = refined_bbox
        done = self.index == len(self.bboxes) - 1  # Finish after refining all boxes

        return self.image, reward, done, {}

    def reset(self):
        self.index = 0
        return self.image

    def _calculate_iou(self, boxA, boxB):
        xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = inter_area / float(boxA_area + boxB_area - inter_area) if inter_area > 0 else 0
        return iou

# Train RL Model for Bounding Box Optimization
def train_rl(image, initial_bboxes):
    env = BoundingBoxEnv(image, initial_bboxes)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1000)
    return model

# Process an image with YOLO + Classical Segmentation + RL
def process_image(image_path):
    image = cv2.imread(image_path)
    
    # Step 1: YOLO Detection
    yolo_bboxes = detect_symbols_yolo(image)

    # Step 2: Classical Segmentation Refinement
    refined_bboxes = [refine_bbox_classical(image, bbox) for bbox in yolo_bboxes]

    # Step 3: RL Optimization
    trained_model = train_rl(image, refined_bboxes)
    optimized_bboxes = refined_bboxes

    # Draw final bounding boxes
    for bbox in optimized_bboxes:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green: Final Refined Box

    cv2.imshow("Optimized Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
process_image("path_to_pid_image.jpg")
