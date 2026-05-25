#!/usr/bin/env python3
"""Batch port detection on PPCL test images 02-09.

For each image: run SAHI object detection (stage4) → detect ports via VLM →
save overlay showing detected port direction.

Usage: cd backend && python3 batch_port_detect.py
"""

from __future__ import annotations

import cv2
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from garnet.visual_primitives.prompts import PORT_FINDER_SYSTEM, PORT_FINDER_USER

load_dotenv(Path.home() / ".env")
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "google/gemini-2.5-pro-preview-05-06"

TEST_DIR = Path("test/ppcl")
OUTPUT_DIR = Path("output/port_detected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load SAHI model once
_MODEL_PATH = Path("yolo_weights/yolo26n_PPCL_640_20260227.pt")
_sahi_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=str(_MODEL_PATH),
    confidence_threshold=0.5,
    image_size=640,
)


def encode_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return __import__("base64").b64encode(buf.getvalue()).decode("utf-8")


def call_vlm(image_b64, system_prompt, user_text, max_tokens=32):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=payload, timeout=60,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    if content is None:
        return None
    return content.strip()


def detect_port(image, bbox, padding=200):
    """Detect which edge of a page connection the pipe attaches to."""
    h, w = image.shape[:2]
    x1 = max(0, bbox["x_min"] - padding)
    y1 = max(0, bbox["y_min"] - padding)
    x2 = min(w, bbox["x_max"] + padding)
    y2 = min(h, bbox["y_max"] + padding)

    crop = image[y1:y2, x1:x2]
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    b64 = encode_b64(pil)

    raw = call_vlm(b64, PORT_FINDER_SYSTEM, PORT_FINDER_USER, max_tokens=256)
    if raw is None:
        return None

    # Parse: EDGE FRACTION
    parts = raw.split()
    if len(parts) == 1:
        edge = parts[0].upper()
        frac = 0.5
    elif len(parts) >= 2:
        edge = parts[0].upper()
        try:
            frac = float(parts[1])
        except ValueError:
            frac = 0.5
    else:
        print(f"    WARN: unparseable VLM response: {raw}")
        return None

    # Map edge → pixel position
    bw = bbox["x_max"] - bbox["x_min"]
    bh = bbox["y_max"] - bbox["y_min"]
    cx = bbox["x_min"] + int(bw / 2)
    cy = bbox["y_min"] + int(bh / 2)

    if edge == "RIGHT":
        px = bbox["x_max"]
        py = bbox["y_min"] + int(bh * frac)
        direction = "RIGHT"
    elif edge == "LEFT":
        px = bbox["x_min"]
        py = bbox["y_min"] + int(bh * frac)
        direction = "LEFT"
    elif edge == "BOTTOM":
        px = bbox["x_min"] + int(bw * frac)
        py = bbox["y_max"]
        direction = "DOWN"
    elif edge == "TOP":
        px = bbox["x_min"] + int(bw * frac)
        py = bbox["y_min"]
        direction = "UP"
    else:
        print(f"    WARN: unknown edge: {edge}")
        return None

    return {"x": px, "y": py, "direction": direction, "edge": edge, "frac": frac, "raw": raw}


def draw_overlay(image, objects, ports):
    """Draw page connection bboxes and port direction arrows."""
    colors = {
        "UP": (0, 180, 0),
        "DOWN": (180, 0, 180),
        "LEFT": (0, 120, 230),
        "RIGHT": (230, 80, 0),
    }
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for obj in objects:
        cls = obj.get("class_name", "")
        if cls not in ("page connection", "utility connection"):
            continue
        pc_id = obj.get("id", "unknown")
        b = obj["bbox"]
        # Bbox: green for page, blue for utility
        bbox_color = (0, 180, 0) if cls == "page connection" else (0, 100, 200)
        draw.rectangle(
            [b["x_min"], b["y_min"], b["x_max"], b["y_max"]],
            outline=bbox_color, width=3,
        )

        port = ports.get(pc_id)
        if port:
            color = colors.get(port["direction"], (180, 0, 180))
            px, py = port["x"], port["y"]
            r = 10
            draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline="white", width=2)

            # Arrow
            arrow_len = 50
            dx = {"UP": 0, "DOWN": 0, "LEFT": -arrow_len, "RIGHT": arrow_len}.get(port["direction"], 0)
            dy = {"UP": -arrow_len, "DOWN": arrow_len, "LEFT": 0, "RIGHT": 0}.get(port["direction"], 0)
            draw.line([px, py, px + dx, py + dy], fill=color, width=4)

            # Label
            label = f"{pc_id} {port['direction']}"
            draw.text((b["x_min"] + 2, b["y_min"] - 20), label, fill=(0, 100, 0))

    return overlay


def process_image(image_path, output_dir):
    """Run stage4 detection + port detection on one image."""
    name = Path(image_path).stem
    print(f"\n--- {name} ---")

    # Step 1: Run SAHI detection
    print(f"  Running SAHI detection (slicing)...")
    stage4_path = output_dir / f"{name}_stage4.json"
    if stage4_path.exists():
        objects = json.loads(stage4_path.read_text())
        if isinstance(objects, dict):
            objects = objects.get("objects", objects)
        print(f"  Loaded existing stage4: {len(objects)} objects")
    else:
        try:
            result = get_sliced_prediction(
                image=str(image_path),
                detection_model=_sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_type="GREEDYNMM",
                postprocess_match_metric="IOS",
                postprocess_match_threshold=0.1,
                perform_standard_pred=False,
            )
            sahi_objects = result.object_prediction_list
            objects = []
            for i, obj in enumerate(sahi_objects):
                bbox = obj.bbox
                objects.append({
                    "id": f"obj_{i:06d}",
                    "class_name": obj.category.name,
                    "confidence": obj.score.value,
                    "bbox": {
                        "x_min": int(bbox.minx), "y_min": int(bbox.miny),
                        "x_max": int(bbox.maxx), "y_max": int(bbox.maxy),
                    },
                })
            stage4_path.write_text(json.dumps({"objects": objects}, indent=2))
            print(f"  SAHI done: {len(objects)} objects")
        except Exception as e:
            print(f"  SAHI failed: {e}")
            import traceback; traceback.print_exc()
            return

    # Step 2: Find connection objects (page + utility)
    CONN_CLASSES = {"page connection", "utility connection"}
    conns = [o for o in objects if o.get("class_name") in CONN_CLASSES]
    print(f"  Connections: {len(conns)} ({Counter(o['class_name'] for o in conns)})")

    if not conns:
        print("  No connections — skipping")
        return

    # Step 3: Load image
    img = cv2.imread(str(image_path))
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Step 4: Detect ports for each connection
    ports = {}
    for c in conns:
        pc_id = c.get("id", "unknown")
        print(f"  Detecting port for {pc_id} ({c['class_name']})...")
        port = detect_port(img, c["bbox"])
        if port is None:
            print(f"    VLM failed for {pc_id} — skipping")
            continue
        ports[pc_id] = port
        print(f"    → {port['direction']} ({port['edge']} {port['frac']:.2f}) [{port['raw']}]")
        time.sleep(0.5)  # rate limit

    # Step 5: Draw overlay
    overlay = draw_overlay(pil_img, objects, ports)
    overlay_path = output_dir / f"{name}_port_overlay.png"
    overlay.save(str(overlay_path))
    print(f"  Overlay saved: {overlay_path}")

    # Step 6: Save port data
    ports_path = output_dir / f"{name}_ports.json"
    ports_path.write_text(json.dumps(ports, indent=2))
    print(f"  Port data saved: {ports_path}")


def main():
    images = sorted([
        p for p in TEST_DIR.glob("Test-*.jpg")
        if not p.name.startswith("Test-00001")
    ])
    print(f"Processing {len(images)} images: {[p.name for p in images]}")

    t0 = time.perf_counter()
    for img_path in images:
        process_image(img_path, OUTPUT_DIR)
    elapsed = time.perf_counter() - t0
    print(f"\nDone: {len(images)} images in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
