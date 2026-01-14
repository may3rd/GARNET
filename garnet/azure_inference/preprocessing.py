import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Union, Optional, List
import logging

logger = logging.getLogger(__name__)

def get_input_shape(model_path: str) -> Tuple[int, int, int, int]:
    """
    Dynamically determine input shape from ONNX model.
    
    Args:
        model_path: Path to the ONNX model file.
        
    Returns:
        Tuple of (batch_size, channels, height, width).
    """
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        input_shape = session.get_inputs()[0].shape
        # Handle dynamic axes (often -1 or 'batch_size')
        if isinstance(input_shape[0], str) or input_shape[0] == -1:
             # Default to batch size 1 if dynamic
             input_shape = (1, *input_shape[1:])
        
        logger.debug(f"Model input shape: {input_shape}")
        return tuple(input_shape)
    except Exception as e:
        logger.error(f"Failed to get input shape from model: {e}")
        # Fallback default
        return (1, 3, 640, 640)

def preprocess_image(
    image: Union[str, np.ndarray], 
    target_size: Tuple[int, int],
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Tuple[np.ndarray, float, float]:
    """
    Preprocess image for Custom Vision ONNX model inference.
    
    Args:
        image: Input image (path or numpy array).
        target_size: Target (height, width) for the model. Note: Model often expects (H, W).
                     However, usually specified as (W, H) in user configs. 
                     We assume target_size is (Width, Height) to match OpenCV resize, 
                     but we should check if we need to swap for model input.
                     Let's assume input tuple is (Width, Height).
        mean: Normalization mean.
        std: Normalization std.
        
    Returns:
        Tuple of (preprocessed_image_batch, ratio_w, ratio_h).
        preprocessed_image_batch: [1, 3, H, W] float32 array.
    """
    # 1. Load image if path
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        # Assume input is RGB if it comes from SAHI/PIL, but OpenCV reads BGR.
        # SAHI usually passes RGB numpy arrays if loaded via PIL or its own utils.
        # If passed from main.py using cv2.imdecode (BGR) -> cv2.cvtColor (RGB), it is RGB.
        # We assume RGB input here.
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # 2. Handle Grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    original_h, original_w = img.shape[:2]
    target_w, target_h = target_size

    # 3. Resize
    # Use standard resize. 
    # NOTE: Some models use letterboxing (padding). Custom Vision typically uses resizing.
    # We will compute ratios to scale boxes back later.
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # 4. Normalize
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - np.array(mean)) / np.array(std)
    
    # 5. Transpose to CHW
    img_chw = img_norm.transpose(2, 0, 1)
    
    # 6. Add batch dimension
    img_batch = np.expand_dims(img_chw, axis=0)
    
    ratio_w = original_w / target_w
    ratio_h = original_h / target_h
    
    return img_batch.astype(np.float32), ratio_w, ratio_h
