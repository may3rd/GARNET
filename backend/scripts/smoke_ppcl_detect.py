"""Smoke test: run YOLO+SAHI detection on test/ppcl/*.jpg.

Saves per-image artifacts into backend/run/detect:
  <image_id>_objects.json
  <image_id>_summary.json
  <image_id>_overlay.png

Usage (from backend/):
  ../.venv/bin/python scripts/smoke_ppcl_detect.py \
      --images test/ppcl --out run/detect \
      --weight yolo_weights/yolo26n_PPCL_640_20260227.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from garnet.object_detection_sahi import DetectionSahiConfig, run_object_detection_sahi


def main() -> int:
    parser = argparse.ArgumentParser(description="PPCL detection smoke test")
    parser.add_argument("--images", default="test/ppcl", help="Directory of *.jpg images")
    parser.add_argument("--out", default="run/detect", help="Output artifact directory")
    parser.add_argument(
        "--weight",
        default="yolo_weights/yolo26n_PPCL_640_20260227.pt",
        help="YOLO weight file path",
    )
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence threshold")
    parser.add_argument("--image-size", type=int, default=640, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    args = parser.parse_args()

    image_dir = Path(args.images).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DetectionSahiConfig(
        weight_path=args.weight,
        conf_th=args.conf,
        image_size=args.image_size,
        overlap_ratio=args.overlap,
    )

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        print(f"No *.jpg images found in {image_dir}")
        return 1

    all_summaries = []
    for image_path in images:
        image_id = image_path.stem
        print(f"[smoke] {image_path.name} ...")
        result = run_object_detection_sahi(image_path, image_id, cfg=cfg)

        objects_payload = result["objects_payload"]
        summary = result["summary"]
        overlay = result["overlay_image"]

        with (out_dir / f"{image_id}_objects.json").open("w") as f:
            json.dump(objects_payload, f, indent=2)
        with (out_dir / f"{image_id}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        import cv2

        cv2.imwrite(str(out_dir / f"{image_id}_overlay.png"), overlay)

        all_summaries.append(summary)
        print(f"   -> {summary['object_count']} objects; classes={summary['class_counts']}")

    with (out_dir / "_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weight": args.weight,
                "conf_threshold": args.conf,
                "image_size": args.image_size,
                "overlap_ratio": args.overlap,
                "image_count": len(images),
                "total_objects": sum(s["object_count"] for s in all_summaries),
                "per_image": all_summaries,
            },
            f,
            indent=2,
        )
    print(f"[smoke] done. Artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
