from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class CustomVisionConfig:
    """Configuration for Azure Custom Vision ONNX model."""
    model_path: str
    class_names: List[str]
    input_size: Tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.5
    # Normalization params (ImageNet defaults)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
