"""Smoke test: run the full pid_extractor pipeline on a subset of test/ppcl images.

Saves per-image stage artifacts into backend/run/detect/<image_id>/.

Usage (from backend/, with backend on PYTHONPATH):
  PYTHONPATH=. ../.venv/bin/python scripts/smoke_ppcl_pipeline.py \
      --images test/ppcl/Test-00001.jpg test/ppcl/Test-00005.jpg test/ppcl/Test-00009.jpg \
      --out run/detect --weight yolo_weights/yolo26n_PPCL_640_20260227.pt \
      --ocr-route easyocr --stop-after 11

Entries may carry an explicit drawing/sheet ID as path=sheet_id; the ID is
passed to the pipeline as document_id (used by page-connector labeling and
multi-sheet merge matching). Without it the file stem is used.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.getLogger("garnet.pid_extractor").setLevel(logging.WARNING)

from garnet.pid_extractor import PIDPipeline, PipelineConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="pid_extractor pipeline smoke test")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths, optionally as path=sheet_id")
    parser.add_argument("--out", default="run/detect", help="Output artifact root directory")
    parser.add_argument(
        "--weight",
        default="yolo_weights/yolo26n_PPCL_640_20260227.pt",
        help="Stage 4 YOLO weight file",
    )
    parser.add_argument("--ocr-route", default="easyocr", help="OCR route")
    parser.add_argument("--stop-after", type=int, default=11, help="Run up to this stage")
    args = parser.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for image_arg in args.images:
        path_part, separator, sheet_id = image_arg.rpartition("=")
        if not separator:
            path_part, sheet_id = image_arg, ""
        image_path = Path(path_part).resolve()
        if not image_path.exists():
            print(f"SKIP: {image_path} (not found)")
            continue
        image_id = sheet_id.strip() or image_path.stem
        out_dir = out_root / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg = PipelineConfig(
            ocr_route=args.ocr_route,
            detection_weight_path=args.weight,
        )
        pipe = PIDPipeline(str(image_path), output_dir=str(out_dir), cfg=cfg, document_id=image_id or None)

        print(f"[smoke] {image_path.name} -> {out_dir} (stop_after={args.stop_after}) ...")
        t0 = time.time()
        try:
            pipe.run(stop_after=args.stop_after)
            elapsed = round(time.time() - t0, 2)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - report and continue
            elapsed = round(time.time() - t0, 2)
            status = f"error: {type(exc).__name__}: {exc}"
            print(f"   ERROR: {status}")

        artifacts = sorted(p.name for p in out_dir.glob("*") if p.is_file())
        results.append(
            {
                "image": image_path.name,
                "image_id": image_id,
                "output_dir": str(out_dir),
                "elapsed_s": elapsed,
                "status": status,
                "artifact_count": len(artifacts),
                "artifacts": artifacts,
            }
        )
        print(f"   -> {status} in {elapsed}s, {len(artifacts)} artifacts")

    with (out_root / "_pipeline_smoke_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "weight": args.weight,
                "ocr_route": args.ocr_route,
                "stop_after": args.stop_after,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"[smoke] done. Summary: {out_root / '_pipeline_smoke_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
