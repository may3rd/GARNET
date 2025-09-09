"""
P&ID pipeline: YOLO11 + SAHI → PaddleOCR → mask-for-DeepLSD
- Detect symbols with YOLO11 using SAHI tiling
- Detect/recognize text with PaddleOCR
- Create a mask that covers symbols and text (with padding + dilation)
- Output: JSON of detections + masked image ready for DeepLSD

Notes:
- Adjust CLASS_MAP and class name matching to your training config.
- For very large sheets, consider chunking per page region first.
"""

from ultralytics import YOLO
# from sahi.models.yolov8 import Yolov8DetectionModel
from sahi import AutoDetectionModel, DetectionModel
from sahi.predict import get_sliced_prediction
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = "yolo_weights/yolo11s_PPCL_640_20250207.pt"  # your YOLO11 weights
DEVICE     = "cuda:0"  # or "cpu"
IMG_PATH   = "pid_image.png"
OUT_DIR    = "out"

# SAHI slicing (good default for large drawings)
SLICE_W, SLICE_H = 640, 640
OVERLAP_W, OVERLAP_H = 0.25, 0.25
CONF_THRES = 0.25
NMS_IOU    = 0.6

# Masking behavior
SYMBOL_PAD = 6       # px to expand each symbol box before masking
TEXT_PAD   = 4       # px to expand each text box before masking
DILATE_K   = 3       # kernel size to dilate the mask; set 0/1 to disable
MASK_COLOR = 255     # 255=white, or use inpaint for fancier fill

# OCR settings
OCR_LANG = "en"      # "en" or "en_number" work well for tags
OCR_USE_CLS = True

# Classes you trained (names must match your YOLO model’s .names)
CLASS_MAP = [
    "gate_valve", "ball_valve", "check_valve", "globe valve", "butterfly valve",
    "pressure relief valve",
    "instrument tag", "instrument dsc", "instrument logic",
    "page connection", "connection", "utility connection",
    "flange", "tee", "elbow", "reducer",
    "pump", "compressor"
]

# Which classes to mask before DeepLSD (usually all symbols)
MASK_SYMBOL_CLASSES = set(CLASS_MAP)  # or a subset if you like


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class YoloDet:
    class_name: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]  # x1,y1,x2,y2

@dataclass
class OcrDet:
    text: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]


# ---------------------------
# Helpers
# ---------------------------
def load_yolo11_and_wrap_for_sahi(model_path: str, device: str):
    """
    SAHI treats YOLO11 like YOLOv8 under the hood.
    """
    # ensure model loads (optional, but good to verify)
    _ = YOLO(model_path)
    
    det_model: DetectionModel = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=CONF_THRES,
        # device=device,
        image_size=SLICE_W,
    )
    return det_model

def run_yolo_sahi(image_path: str, det_model) -> List[YoloDet]:
    result = get_sliced_prediction(
        image=image_path,
        detection_model=det_model,
        slice_height=SLICE_H,
        slice_width=SLICE_W,
        overlap_height_ratio=OVERLAP_H,
        overlap_width_ratio=OVERLAP_W,
        postprocess_type="NMS",  # or "GREEDYNMM"
        postprocess_match_threshold=NMS_IOU,
        verbose=0,
    )
    out: List[YoloDet] = []
    for obj in result.object_prediction_list:
        cls_id = int(obj.category.id)
        cls_name = obj.category.name
        score = float(obj.score.value)
        x1, y1, x2, y2 = [int(v) for v in obj.bbox.to_xyxy()]
        out.append(YoloDet(class_name=cls_name, score=score, bbox_xyxy=(x1, y1, x2, y2)))
    return out

def run_paddle_ocr(image_path: str, lang: str = OCR_LANG, use_cls: bool = OCR_USE_CLS) -> List[OcrDet]:
    ocr = PaddleOCR(use_angle_cls=use_cls, lang=lang)
    res = ocr.ocr(image_path, cls=use_cls)
    ocr_out: List[OcrDet] = []
    if not res or not res[0]:
        return ocr_out
    for line in res[0]:
        # line: [ [[x,y],...4pts], (text, conf) ]
        poly = line[0]
        text, conf = line[1]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        ocr_out.append(OcrDet(text=text, score=float(conf), bbox_xyxy=(x1, y1, x2, y2)))
    return ocr_out

def expand_box(x1, y1, x2, y2, pad, W, H):
    return max(0, x1 - pad), max(0, y1 - pad), min(W - 1, x2 + pad), min(H - 1, y2 + pad)

def build_mask(img: np.ndarray,
               symbol_dets: List[YoloDet],
               text_dets: List[OcrDet],
               symbol_pad: int = SYMBOL_PAD,
               text_pad: int = TEXT_PAD,
               dilate_k: int = DILATE_K,
               mask_color: int = MASK_COLOR) -> np.ndarray:
    """
    Returns a masked image (white rectangles drawn over symbols & text).
    For inpainting instead, create a binary mask and call cv2.inpaint.
    """
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    # Mask symbols
    for d in symbol_dets:
        if d.class_name in MASK_SYMBOL_CLASSES:
            x1, y1, x2, y2 = expand_box(*d.bbox_xyxy, symbol_pad, W, H)
            mask[y1:y2+1, x1:x2+1] = 255

    # Mask text
    for t in text_dets:
        x1, y1, x2, y2 = expand_box(*t.bbox_xyxy, text_pad, W, H)
        mask[y1:y2+1, x1:x2+1] = 255

    # Optional dilation to cover thin leaders around text/symbols
    if dilate_k and dilate_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_k, dilate_k))
        mask = cv2.dilate(mask, k, iterations=1)

    # Apply mask (simple white paint-over). If you prefer inpainting:
    # masked = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    masked = img.copy()
    masked[mask == 255] = mask_color

    return masked

def save_json(path: str, data: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------
# Main
# ---------------------------
def main(img_path: str = IMG_PATH):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Load image
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"Cannot read image: {img_path}"

    # A) Symbols via YOLO11+SAHI
    det_model = load_yolo11_and_wrap_for_sahi(MODEL_PATH, DEVICE)
    yolo_dets = run_yolo_sahi(img_path, det_model)

    # B) Text via PaddleOCR
    ocr_dets = run_paddle_ocr(img_path, OCR_LANG, OCR_USE_CLS)

    # C) Mask image (symbols + text) for DeepLSD
    masked = build_mask(img_bgr, yolo_dets, ocr_dets)

    # D) Save artifacts
    base = Path(img_path).stem
    cv2.imwrite(f"{OUT_DIR}/{base}_masked.png", masked)

    # Optional: save detections for audit/debug
    yolo_json = [asdict(d) for d in yolo_dets]
    ocr_json  = [asdict(t) for t in ocr_dets]
    save_json(f"{OUT_DIR}/{base}_yolo_dets.json", {"detections": yolo_json})
    save_json(f"{OUT_DIR}/{base}_ocr_dets.json",  {"detections": ocr_json})

    print(f"[OK] Masked image → {OUT_DIR}/{base}_masked.png")
    print(f"[OK] YOLO dets → {OUT_DIR}/{base}_yolo_dets.json")
    print(f"[OK] OCR dets  → {OUT_DIR}/{base}_ocr_dets.json")


if __name__ == "__main__":
    main()