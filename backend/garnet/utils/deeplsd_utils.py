import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc
import torch
import h5py

from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines

def define_torch_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (slower but functional)")
        
    return device

def load_deeplsd_model(device, ckpt_path='DeepLSD/weights/deeplsd_md.tar', conf=None):
    """
    Loads the DeepLSD model with specified configuration and weights.
    """
    if conf is None:
        conf = {
            'detect_lines': True,
            'line_detection_params': {
                'merge': True,
                'filtering': True,
                'grad_thresh': 3,
                'grad_nfa': True,
            }
        }
    
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()
    return net, conf

def detect_lines(model, gray_img_array, device):
    """
    Detects lines in an image using the provided DeepLSD model.
    gray_img_array: A grayscale NumPy array (H, W) representing the image.
    """
    inputs = {'image': torch.tensor(gray_img_array, dtype=torch.float, device=device)[None, None] / 255.}
    with torch.no_grad():
        out = model(inputs)
        pred_lines = out['lines'][0]
    return gray_img_array, pred_lines

def draw_and_save_lines(original_gray_img, lines, output_path):
    """
    Draws detected lines on the original grayscale image and saves the result.
    """
    img_with_lines = cv2.cvtColor(original_gray_img, cv2.COLOR_GRAY2BGR)
    for line in lines:
        p1, p2 = line.astype(int)
        x1, y1 = p1
        x2, y2 = p2
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imwrite(output_path, img_with_lines)
    print(f"Saved {len(lines)} lines to {output_path}")

def export_lines_to_json(lines, output_filepath):
    """
    Exports detected lines to a JSON file.
    """
    import json
    lines_list = lines.tolist() # Convert numpy array to list
    with open(output_filepath, 'w') as f:
        json.dump(lines_list, f, indent=4)
    print(f"Exported {len(lines)} lines to {output_filepath}")

import math

def combine_close_lines(lines, dist_threshold=10, angle_threshold=5):
    """
    Combines lines that are close to each other and are approximately collinear.
    lines: A list of line segments, where each line is [[x1, y1], [x2, y2]].
    dist_threshold: Maximum distance between endpoints to consider combining.
    angle_threshold: Maximum angle difference (in degrees) for collinearity.
    """
    if not lines:
        return []

    combined_lines = []
    used_indices = set()

    for i, line1 in enumerate(lines):
        if i in used_indices:
            continue
        
        # Extract coordinates from the line format [[x1, y1], [x2, y2]]
        x1, y1 = line1[0]
        x2, y2 = line1[1]

        p1_start = np.array([x1, y1])
        p1_end = np.array([x2, y2])
        
        # Calculate the vector for line1
        v1 = p1_end - p1_start
        len_v1 = np.linalg.norm(v1)
        if len_v1 == 0: # Handle zero-length lines
            continue
        v1_unit = v1 / len_v1

        current_combined_line = list(line1)
        
        for j, line2 in enumerate(lines):
            if i == j or j in used_indices:
                continue

            # Extract coordinates from the line format [[x1, y1], [x2, y2]]
            x1_2, y1_2 = line2[0]
            x2_2, y2_2 = line2[1]

            p2_start = np.array([x1_2, y1_2])
            p2_end = np.array([x2_2, y2_2])

            # Calculate the vector for line2
            v2 = p2_end - p2_start
            len_v2 = np.linalg.norm(v2)
            if len_v2 == 0: # Handle zero-length lines
                continue
            v2_unit = v2 / len_v2

            # Check for collinearity (angle between vectors)
            dot_product = np.dot(v1_unit, v2_unit)
            angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Consider both directions for collinearity
            if angle_deg > 90:
                angle_deg = 180 - angle_deg

            if angle_deg < angle_threshold:
                # Check if endpoints are close
                distances = [
                    np.linalg.norm(p1_start - p2_start),
                    np.linalg.norm(p1_start - p2_end),
                    np.linalg.norm(p1_end - p2_start),
                    np.linalg.norm(p1_end - p2_end)
                ]
                
                if min(distances) < dist_threshold:
                    # Combine lines
                    all_points = np.array([
                        current_combined_line[0][0], current_combined_line[0][1],
                        current_combined_line[1][0], current_combined_line[1][1],
                        line2[0][0], line2[0][1],
                        line2[1][0], line2[1][1]
                    ]).reshape(-1, 2)
                    
                    # Find the two most extreme points to form the new line
                    # Project points onto the line defined by v1_unit
                    projections = np.dot(all_points - p1_start, v1_unit)
                    min_proj_idx = np.argmin(projections)
                    max_proj_idx = np.argmax(projections)

                    new_start = p1_start + projections[min_proj_idx] * v1_unit
                    new_end = p1_start + projections[max_proj_idx] * v1_unit

                    current_combined_line = [[new_start[0], new_start[1]], [new_end[0], new_end[1]]]
                    used_indices.add(j)
        
        combined_lines.append(current_combined_line)
        used_indices.add(i) # Mark the initial line as used after processing its combinations

    return combined_lines


if __name__ == "__main__":
    # Example usage:
    device = define_torch_device()
    
    # Load the model
    model, conf = load_deeplsd_model(device)
    
    # Detect lines in an example image
    image_to_process = 'output/stage5_skeleton_inverted.png'
    gray_img, detected_lines = detect_lines(model, image_to_process, device)
    
    # Draw and save the lines
    output_image_path = 'out/output.jpg'
    draw_and_save_lines(gray_img, np.array(detected_lines), output_image_path)

    # Export detected lines to JSON
    json_output_path = 'out/detected_lines.json'
    export_lines_to_json(detected_lines, json_output_path)

    # Combine close lines and draw them
    combined_lines = combine_close_lines(detected_lines.tolist()) # Convert numpy array to list for the function
    combined_output_image_path = 'out/combined_lines.jpg'
    draw_and_save_lines(gray_img, np.array(combined_lines), combined_output_image_path)
    print(f"Combined {len(detected_lines)} lines into {len(combined_lines)} lines.")
