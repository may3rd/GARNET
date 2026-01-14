import numpy as np
from typing import List, Dict, Any, Tuple
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image
from sahi.annotation import BoundingBox

def convert_outputs_to_detections(
    outputs: List[np.ndarray],
    output_names: List[str],
    image_width: int,
    image_height: int,
    ratio_w: float,
    ratio_h: float,
    confidence_threshold: float,
    class_names: List[str],
    shift_amount: List[int] = [0, 0]
) -> List[ObjectPrediction]:
    """
    Convert ONNX model outputs to SAHI ObjectPrediction list.
    
    Args:
        outputs: List of output numpy arrays from the model.
        output_names: Names of the output tensors.
        image_width: Width of the original image (slice).
        image_height: Height of the original image (slice).
        ratio_w: Width scaling ratio (original / input).
        ratio_h: Height scaling ratio (original / input).
        confidence_threshold: Score threshold.
        class_names: List of class names mapping to class IDs.
        shift_amount: [x, y] shift amount (not used if returning for SAHI, 
                      as SAHI handles shifting if we return coords relative to slice).
                      
    Returns:
        List of ObjectPrediction.
    """
    
    boxes = None
    scores = None
    classes = None
    
    name_to_data = {name: data for name, data in zip(output_names, outputs)}
    
    # Try to find standard Custom Vision output names
    if 'detected_boxes' in name_to_data:
        boxes = name_to_data['detected_boxes'][0] # [num_det, 4]
        scores = name_to_data['detected_scores'][0] # [num_det]
        classes = name_to_data['detected_classes'][0] # [num_det]
    else:
        # Fallback: Heuristic matching by shape
        # Boxes: [batch, num, 4] or [num, 4]
        # Scores: [batch, num] or [num]
        # Classes: [batch, num] or [num]
        
        candidates_2d = []
        candidates_1d = []
        
        for data in outputs:
            data = data.squeeze()
            if data.ndim == 2 and data.shape[1] == 4:
                candidates_2d.append(data)
            elif data.ndim == 1:
                candidates_1d.append(data)
                
        if len(candidates_2d) == 1:
            boxes = candidates_2d[0]
        
        if len(candidates_1d) >= 2:
            # Differentiate scores and classes
            # Classes are likely integers, scores are floats 0-1
            first = candidates_1d[0]
            second = candidates_1d[1]
            
            is_first_int = np.issubdtype(first.dtype, np.integer) or (np.mod(first, 1) == 0).all()
            is_second_int = np.issubdtype(second.dtype, np.integer) or (np.mod(second, 1) == 0).all()
            
            if is_first_int and not is_second_int:
                classes = first
                scores = second
            elif not is_first_int and is_second_int:
                scores = first
                classes = second
            else:
                # If both look like scores or both look like ints, default to order?
                # Custom Vision usually: boxes, classes, scores (alphabetical?) or boxes, scores, classes
                # Let's assume scores are float and have values < 1 usually?
                # This is risky without names.
                # Assuming standard order if names fail is tricky.
                # Let's assign if we found exactly 2 1d arrays.
                scores = first
                classes = second

    object_prediction_list = []
    
    if boxes is None or scores is None or classes is None:
        return []
        
    for i, score in enumerate(scores):
        if score < confidence_threshold:
            continue
            
        class_id = int(classes[i])
        if class_names and class_id < len(class_names):
            category_name = class_names[class_id]
        else:
            category_name = str(class_id)
            
        box = boxes[i]
        
        # Determine coordinates
        x_min, y_min, x_max, y_max = box
        
        # Check if normalized [0, 1]
        # If all boxes in the batch are <= 1.0, likely normalized.
        # But a single box might be small.
        # However, Custom Vision usually exports normalized boxes.
        # We can assume normalized if max value <= 1.0 + epsilon
        is_normalized = boxes.max() <= 1.05 
        
        if is_normalized:
            # Scale by original dimensions
            x_min *= image_width
            x_max *= image_width
            y_min *= image_height
            y_max *= image_height
        else:
            # Absolute coordinates in target (resized) image
            # Scale by ratio
            x_min *= ratio_w
            x_max *= ratio_w
            y_min *= ratio_h
            y_max *= ratio_h
            
        # Ensure valid bbox
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width, x_max)
        y_max = min(image_height, y_max)
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width <= 0 or height <= 0:
            continue
            
        object_prediction = ObjectPrediction(
            bbox=[x_min, y_min, x_max, y_max],
            category_id=class_id,
            category_name=category_name,
            score=float(score),
            shift_amount=shift_amount
        )
        object_prediction_list.append(object_prediction)
        
    return object_prediction_list