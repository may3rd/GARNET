# Azure Custom Vision ONNX Inference for GARNET

This module provides a wrapper for integrating Azure Custom Vision ONNX models with the SAHI (Sliced Aided Hyper Inference) framework in GARNET.

## Features
- **Seamless Integration**: Works with SAHI's slicing and prediction pipeline.
- **Custom Vision Support**: Handles specific preprocessing and output formats of Custom Vision ONNX exports.
- **Configurable**: Easy configuration via `CustomVisionConfig`.
- **Production Ready**: Includes error handling, logging, and unit tests.

## Usage

### 1. Basic Inference (Python Script)

```python
from garnet.azure_inference import CustomVisionSAHIDetector, CustomVisionConfig
import cv2

# Define configuration
config = CustomVisionConfig(
    model_path="path/to/model.onnx",
    class_names=["valve", "pump"],
    input_size=(640, 640),
    confidence_threshold=0.5
)

# Initialize detector
detector = CustomVisionSAHIDetector(config)

# Load image
image = cv2.imread("test_pid.jpg")

# Run inference
result = detector.perform_inference(image)

# Access results
print(f"Detected {len(result.object_prediction_list)} objects")
for obj in result.object_prediction_list:
    print(f"{obj.category.name}: {obj.score.value} at {obj.bbox}")
```

### 2. SAHI Integration

```python
from sahi.predict import get_sliced_prediction

result = get_sliced_prediction(
    "test_pid.jpg",
    detection_model=detector,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

### 3. Web Application (GARNET)

1.  **Place Model**: Copy your `.onnx` model file to `yolo_weights/` (or configured model path).
2.  **Create Config**: Create a YAML file in `datasets/yaml/` describing the classes. Format:
    ```yaml
    names:
      0: valve
      1: pump
      ...
    ```
    Ensure the order matches the Custom Vision project class IDs.
3.  **Run App**: Start the GARNET server (`python main.py`).
4.  **Select Model**:
    -   In the web interface, select **Model type**: `azure_custom_vision`.
    -   Select your **Weight file** (.onnx).
    -   Select your **Config file** (.yaml).
5.  **Submit**: Run inference.

## Utilities

### Inspect ONNX Model
Use the inspection tool to check input/output node names and shapes of your ONNX model.

```bash
python -m garnet.azure_inference.inspect_onnx path/to/model.onnx
```

## Configuration Details
- **Input Size**: The wrapper resizes input images to `input_size` (default 640x640) before inference. This should match the Custom Vision model's expected input (usually 320x320 or 640x640).
- **Normalization**: Default ImageNet normalization (Mean: `[0.485, 0.456, 0.406]`, Std: `[0.229, 0.224, 0.225]`) is applied.
- **Output Parsing**: The wrapper automatically attempts to identify `detected_boxes`, `detected_scores`, and `detected_classes` outputs.

## Development
Run tests with:
```bash
python -m unittest garnet/tests/test_custom_vision_detector.py
```
