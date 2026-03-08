from __future__ import annotations

from pathlib import Path
from typing import Optional


BACKEND_DIR = Path(__file__).resolve().parents[1]
YOLO_WEIGHTS_DIR = BACKEND_DIR / "yolo_weights"
PREFERRED_DEFAULT_ULTRALYTICS_WEIGHT = "yolo26n_PPCL_640_20260227.pt"


def list_weight_files() -> list[str]:
    paths = sorted(YOLO_WEIGHTS_DIR.glob("*.onnx")) + sorted(YOLO_WEIGHTS_DIR.glob("*.pt"))
    return [str(path.relative_to(BACKEND_DIR)) for path in paths]


def pick_default_weight_file(model_type: str) -> Optional[str]:
    weight_files = list_weight_files()
    if not weight_files:
        return None
    if model_type == "ultralytics":
        preferred = f"yolo_weights/{PREFERRED_DEFAULT_ULTRALYTICS_WEIGHT}"
        if preferred in weight_files:
            return preferred
        for item in weight_files:
            if item.endswith(".pt"):
                return item
    return weight_files[0]
