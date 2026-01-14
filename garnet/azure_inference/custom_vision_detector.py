import logging
from typing import List, Optional, Any
import numpy as np
import onnxruntime as ort
from sahi import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list

from .config import CustomVisionConfig
from .preprocessing import preprocess_image
from .postprocessing import convert_outputs_to_detections

logger = logging.getLogger(__name__)

class CustomVisionSAHIDetector(DetectionModel):
    def __init__(self, config: CustomVisionConfig):
        """
        Init Custom Vision SAHI Detector.
        
        Args:
            config: CustomVisionConfig object.
        """
        self.config = config
        
        # Initialize base class
        super().__init__(
            model_path=config.model_path,
            confidence_threshold=config.confidence_threshold,
            category_mapping={str(i): name for i, name in enumerate(config.class_names)},
            image_size=config.input_size[0], # Assuming square or primary dim
            load_at_init=True,
        )

    def load_model(self):
        """
        Load ONNX model using ONNXRuntime.
        """
        try:
            logger.info(f"Loading Custom Vision model from {self.model_path}")
            # Use CPU by default, extend to GPU if available/configured
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
                
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"Model loaded successfully. Input: {self.input_name}, Outputs: {self.output_names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def perform_inference(self, image: np.ndarray):
        """
        Prediction with Custom Vision model.
        
        Args:
            image: np.ndarray of shape (H, W, 3) in RGB (SAHI conventions).
            
        Returns:
            DetectionResult
        """
        # Preprocess
        # image is numpy array (H, W, 3)
        
        target_size = self.config.input_size
        
        input_tensor, ratio_w, ratio_h = preprocess_image(
            image, 
            target_size=target_size,
            mean=self.config.mean,
            std=self.config.std
        )
        
        # Inference
        outputs = self.session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Postprocess
        # Get original dimensions from image
        height, width = image.shape[:2]
        
        object_prediction_list = convert_outputs_to_detections(
            outputs=outputs,
            output_names=self.output_names,
            image_width=width,
            image_height=height,
            ratio_w=ratio_w,
            ratio_h=ratio_h,
            confidence_threshold=self.confidence_threshold,
            class_names=self.config.class_names
        )
        
        from sahi.prediction import PredictionResult
        return PredictionResult(
            object_prediction_list=object_prediction_list,
            image=image
        )
