#!/usr/bin/env python
"""Simple SAHI + YOLO object detection for one P&ID raster → JSON + overlay PNG.

Standalone: no GARNET pipeline, no OCR, no tracing. Tiles the image with SAHI and runs a
YOLO (Ultralytics) weight over the slices, then writes a flat JSON object list and an
overlay image (`<out-stem>_overlay.png`) with every detected box drawn and labelled.

Detection classes come from the weight file, not from a dataset YAML.

Usage (from anywhere):
    python garnet/detect_objects.py --image sheet.png
    python garnet/detect_objects.py --image sheet.png --weight backend/yolo_weights/yolo26n_OLE2_1024_20260327.pt \
        --image-size 1024 --overlap-ratio 0.25 --conf-th 0.5 --out out/objects.json

Run with the repo venv: /Users/maetee/Code/GARNET/.venv/bin/python
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHT = "backend/yolo_weights/yolo26n_PPCL_640_20260227.pt"


def _resolve(path: str) -> Path:
    """Resolve a user path against cwd first, then the repo root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    repo_candidate = REPO_ROOT / candidate
    return repo_candidate if repo_candidate.exists() else candidate


def detect(
    image_path: Path,
    *,
    weight_path: Path,
    image_size: int,
    overlap_ratio: float,
    conf_th: float,
    postprocess_type: str,
    postprocess_match_metric: str,
    postprocess_match_threshold: float,
    device: str | None,
) -> tuple[dict, "object"]:
    """Run sliced detection. Returns (payload, image_bgr)."""
    import cv2
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image_bgr.shape[:2]

    model_type = "ultralytics"
    kwargs: dict = {
        "model_type": model_type,
        "model_path": str(weight_path),
        "confidence_threshold": conf_th,
        "image_size": image_size,
    }
    if device:
        kwargs["device"] = device
    model = AutoDetectionModel.from_pretrained(**kwargs)

    result = get_sliced_prediction(
        image=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        detection_model=model,
        slice_height=image_size,
        slice_width=image_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=postprocess_match_threshold,
        # SAHI silently swaps the postprocess to NMS/IOU below conf 0.1, which would make the
        # recorded postprocess fields a lie; keep whatever the caller asked for.
        force_postprocess_type=True,
        verbose=0,
    )

    objects = []
    for idx, det in enumerate(result.object_prediction_list, start=1):
        x_min, y_min, x_max, y_max = det.bbox.to_xyxy()
        objects.append(
            {
                "id": f"obj_{idx:06d}",
                "class_name": det.category.name,
                "confidence": round(float(det.score.value), 4),
                "bbox": {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max),
                },
                # Contract A provenance: lets this JSON seed stage4_objects.json unchanged.
                "source_model": model_type,
                "source_weight": str(weight_path),
            }
        )

    payload = {
        "image_id": image_path.name,
        "pass_type": "sheet",
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "weight": str(weight_path),
        # The device SAHI actually resolved to — "auto" (None) picks mps here but cpu on a
        # CUDA-less Linux box, and an unavailable --device silently falls back to cpu.
        "device": str(model.device),
        "image_size": image_size,
        "overlap_ratio": overlap_ratio,
        "conf_th": conf_th,
        "postprocess_type": postprocess_type,
        "postprocess_match_metric": postprocess_match_metric,
        "postprocess_match_threshold": postprocess_match_threshold,
        "object_count": len(objects),
        "class_counts": dict(sorted(Counter(o["class_name"] for o in objects).items())),
        "objects": objects,
    }
    return payload, image_bgr


def _class_color(name: str) -> tuple[int, int, int]:
    """Stable BGR colour per class name (hashlib, not hash() — that is salted per process)."""
    digest = hashlib.md5(name.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def _collides_with_image(reserved: list[Path], image_path: Path) -> Path | None:
    """First reserved output path that aliases the input raster, if any.

    `detect()` reads the raster up front, so an output path pointing back at it would destroy
    the sheet only after a full inference run. Compare with `samefile` so hard links, symlinks
    and `./x.jpg` vs `x.jpg` all count.
    """
    for path in reserved:
        if path.exists() and path.samefile(image_path):
            return path
    return None


def draw_overlay(image_bgr, objects: list[dict]) -> "object":
    """Draw a coloured box + class/confidence label for each detected object."""
    import cv2

    overlay = image_bgr.copy()
    height, width = image_bgr.shape[:2]
    scale = max(0.5, min(1.5, width / 2000))
    pad = max(2, int(round(3 * scale)))
    thickness = 2

    for obj in objects:
        bb = obj["bbox"]
        x1, y1, x2, y2 = bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"]
        color = _class_color(obj["class_name"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        label = f"{obj['class_name']} {obj['confidence']:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        # Label above the box (below it for a box touching the top edge), then clamp the
        # rectangle into the canvas — a bottom-edge box would otherwise draw it off-image.
        ly = y1 - pad if y1 - th - 2 * pad >= 0 else y2 + th + pad
        top = min(max(ly - th - pad, 0), max(0, height - (th + base + pad)))
        left = min(x1, max(0, width - (tw + 2 * pad)))
        cv2.rectangle(overlay, (left, top), (left + tw + 2 * pad, top + th + base + pad), color, -1)
        cv2.putText(
            overlay,
            label,
            (left + pad, top + th + pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SAHI + YOLO object detection on a P&ID raster → JSON + overlay PNG.")
    parser.add_argument("--image", required=True, help="Input image (png/jpg/tif).")
    parser.add_argument("--weight", default=DEFAULT_WEIGHT, help=f"YOLO weight file. Default: {DEFAULT_WEIGHT}")
    parser.add_argument("--image-size", type=int, default=640, help="SAHI slice size in px. Default: 640")
    parser.add_argument("--overlap-ratio", type=float, default=0.2, help="SAHI slice overlap. Default: 0.2")
    parser.add_argument("--conf-th", type=float, default=0.8, help="Confidence threshold. Default: 0.8")
    parser.add_argument(
        "--postprocess-type",
        default="GREEDYNMM",
        # LSNMS is excluded deliberately: SAHI raises ModuleNotFoundError without the extra
        # `lsnms` package (not a dependency here) and rejects its only useful metric.
        choices=["GREEDYNMM", "NMM", "NMS"],
        help="SAHI postprocess. Default: GREEDYNMM",
    )
    parser.add_argument("--match-metric", default="IOS", choices=["IOS", "IOU"], help="SAHI match metric. Default: IOS")
    parser.add_argument("--match-threshold", type=float, default=0.1, help="SAHI match threshold. Default: 0.1")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cpu / mps / cuda:0. Default: auto")
    parser.add_argument("--out", default=None, help="Output JSON path. Default: output/<stem>_objects.json")
    parser.add_argument("--no-overlay", action="store_true", help="Skip writing the overlay PNG.")
    args = parser.parse_args(argv)

    if args.image_size < 1:
        print("error: --image-size must be >= 1", file=sys.stderr)
        return 1
    if not 0.0 <= args.overlap_ratio < 1.0:
        print("error: --overlap-ratio must be in [0, 1)", file=sys.stderr)
        return 1
    if not 0.0 <= args.conf_th <= 1.0:
        print("error: --conf-th must be in [0, 1]", file=sys.stderr)
        return 1
    if not 0.0 <= args.match_threshold <= 1.0:
        print("error: --match-threshold must be in [0, 1]", file=sys.stderr)
        return 1

    image_path = _resolve(args.image)
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1
    weight_path = _resolve(args.weight)
    if not weight_path.is_file():
        print(f"error: weight not found: {weight_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out).expanduser() if args.out else REPO_ROOT / "output" / f"{image_path.stem}_objects.json"
    overlay_path = out_path.with_name(out_path.stem + "_overlay.png")
    reserved = [out_path] + ([] if args.no_overlay else [overlay_path])
    colliding = _collides_with_image(reserved, image_path)
    if colliding is not None:
        print(f"error: refusing to overwrite the input image: {colliding}", file=sys.stderr)
        return 1

    payload, image_bgr = detect(
        image_path,
        weight_path=weight_path,
        image_size=args.image_size,
        overlap_ratio=args.overlap_ratio,
        conf_th=args.conf_th,
        postprocess_type=args.postprocess_type,
        postprocess_match_metric=args.match_metric,
        postprocess_match_threshold=args.match_threshold,
        device=args.device,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{payload['object_count']} objects → {out_path}")

    if not args.no_overlay:
        import cv2

        if not cv2.imwrite(str(overlay_path), draw_overlay(image_bgr, payload["objects"])):
            print(f"error: failed to write overlay: {overlay_path}", file=sys.stderr)
            return 1
        print(f"overlay → {overlay_path}")

    for name, count in payload["class_counts"].items():
        print(f"  {count:5d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
