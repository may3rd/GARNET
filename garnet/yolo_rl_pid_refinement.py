import cv2
import numpy as np
import torch
import multiprocessing
import gymnasium as gym
from gymnasium import spaces
from ultralytics import YOLO
from stable_baselines3 import PPO

# Load YOLO Model
yolo_model = YOLO("yolo_weights/yolo11s_PPCL_640_20250207.pt")  # Replace with your trained model if needed

def detect_symbols_yolo(image):
    """
    Detect symbols in a P&ID image using YOLOv8.
    Returns list of bounding boxes (x_min, y_min, x_max, y_max).
    """
    torch.set_num_threads(multiprocessing.cpu_count())  # Use all available CPU threads for YOLO
    results = yolo_model(image, device="mps")
    cls_names = []
    bboxes = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):  # Bounding boxes
            bboxes.append(tuple(map(int, box.tolist())))
            cls_names.append(yolo_model.names[int(cls)])
    return bboxes, cls_names

def refine_bbox_classical(image, bbox):
    """
    Uses classical image processing to refine bounding box by detecting symbols inside it.
    """
    x_min, y_min, x_max, y_max = bbox

    # Ensure bounding box is within image bounds
    h, w = image.shape[:2]
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)

    # Extract ROI (Region of Interest)
    roi = image[y_min:y_max, x_min:x_max]

    # 🔴 FIX: Ensure ROI is not empty
    if roi is None or roi.size == 0:
        # print(f"❌ Error: Empty ROI detected for bounding box {bbox}. Skipping refinement.")
        return bbox  # Return original bounding box if ROI is invalid

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box around the largest detected contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Adjust bounding box to image scale
        refined_x_min = x_min + x
        refined_y_min = y_min + y
        refined_x_max = refined_x_min + w
        refined_y_max = refined_y_min + h

        return (refined_x_min, refined_y_min, refined_x_max, refined_y_max)

    return bbox  # Return original bounding box if no contours found


# Reinforcement Learning Environment for Bounding Box Optimization
class BoundingBoxEnv(gym.Env):
    def __init__(self, image, initial_bboxes):
        super(BoundingBoxEnv, self).__init__()
        self.image = image
        self.bboxes = initial_bboxes
        self.index = 0  # Current bounding box to refine

        # Actions: move left, right, up, down, expand, shrink
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=255, shape=image.shape, dtype=np.uint8)

    def step(self, action):
        x_min, y_min, x_max, y_max = self.bboxes[self.index]

        # Actions: 0=left, 1=right, 2=up, 3=down, 4=expand, 5=shrink
        if action == 0: x_min -= 2
        elif action == 1: x_max += 2
        elif action == 2: y_min -= 2
        elif action == 3: y_max += 2
        elif action == 4: x_max += 2; y_max += 2  # Expand
        elif action == 5: x_min += 2; y_min += 2  # Shrink

        # Ensure bounding box stays within image bounds
        x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(self.image.shape[1], x_max), min(self.image.shape[0], y_max)
        
        # Compute reward: IoU between refined box and classical segmentation
        refined_bbox = refine_bbox_classical(self.image, (x_min, y_min, x_max, y_max))
        iou = self._calculate_iou(refined_bbox, (x_min, y_min, x_max, y_max))
        reward = iou * 10  # Higher IoU = better refinement

        self.bboxes[self.index] = refined_bbox

        # 🔴 FIX: Separate `done` into `terminated` and `truncated`
        terminated = self.index == len(self.bboxes) - 1  # Done if all boxes are refined
        truncated = False  # Set `truncated=True` if using max steps

        return self.image, reward, terminated, truncated, {}
    

    def reset(self, seed=None, options=None):
        """Resets the environment and ensures compatibility with Stable-Baselines3."""
        self.index = 0

        # Return both observation and info dictionary
        return self.image, {}
    
    def _calculate_iou(self, boxA, boxB):
        """Calculate IoU (Intersection over Union) between two bounding boxes."""
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

    # 🔴 FIX: Ensure the image is loaded properly
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check file path and format.")

    # Step 1: YOLO Detection
    print("🔵 Step 1: YOLO Detection")
    yolo_bboxes, cls_names = detect_symbols_yolo(image)
    
    # Save the image with YOLO detections
    new_img = image.copy()
    for bbox in yolo_bboxes:
        cv2.rectangle(new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)  # Red: YOLO Detected Boxes
    cv2.imwrite("test/yolo_detections.png", new_img)
    
    # Step 2: Classical Segmentation Refinement
    print("🔵 Step 2: Classical Segmentation Refinement")
    refined_bboxes = [refine_bbox_classical(image, bbox) for bbox in yolo_bboxes]

    # Save image to show refined boxes
    new_img = image.copy()
    for bbox in refined_bboxes:
        cv2.rectangle(new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green: Refined Boxes
    cv2.imwrite("test/refined_boxes.png", new_img)
    
    # Step 3: RL Optimization
    # print("🔵 Step 3: RL Optimization")
    # trained_model = train_rl(image, refined_bboxes)

    # Draw final bounding boxes
    # new_img = image.copy()
    # for bbox in refined_bboxes:
    #     cv2.rectangle(new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)  # Green: Final Refined Boxes
    # cv2.imwrite("test/optimized_boxes.png", new_img)

    # cv2.imshow("Optimized Bounding Boxes", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage
process_image("test/!test02.png")
