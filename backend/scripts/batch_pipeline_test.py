#!/usr/bin/env python3
"""
batch_pipeline_test.py — run full pipeline on multiple images, saving outputs
to GARNET project folder.
"""
from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path

# Suppress pipeline log noise
logging.getLogger("garnet.pid_extractor").setLevel(logging.WARNING)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from garnet.pid_extractor import PIDPipeline, PipelineConfig

OUTPUT_ROOT = Path("/Users/maetee/Documents/1. PROJECTS/claude-workspace/projects/gcme/1_active/garnet/output")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Test images: mix of simple and complex
TEST_IMAGES = [
    ("ppcl", BACKEND_ROOT / "test" / "ppcl" / "Test-00001.jpg"),
    ("ppcl", BACKEND_ROOT / "test" / "ppcl" / "Test-00004.jpg"),
    ("ppcl", BACKEND_ROOT / "test" / "ppcl" / "Test-00007.jpg"),
    ("ppcl", BACKEND_ROOT / "test" / "ppcl" / "Test-00009.jpg"),
    ("pttep", BACKEND_ROOT / "test" / "pttep" / "images" / "PLCPP2" / "001.png"),
    ("pttep", BACKEND_ROOT / "test" / "pttep" / "images" / "PLCPP2" / "010.png"),
    ("pttep", BACKEND_ROOT / "test" / "pttep" / "images" / "PLCPP2" / "149.png"),
    ("test", BACKEND_ROOT / "test" / "!test01.png"),
]


def run_one(label: str, image_path: Path) -> dict:
    out_dir = OUTPUT_ROOT / label / "default"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(ocr_route="easyocr")

    # Change to out_dir so PIDPipeline's relative "output" path resolves correctly.
    # Without this, the pipeline writes to backend/output/ (relative to CWD) instead
    # of the intended OUTPUT_ROOT subdirectory.
    import os as _os
    _os.chdir(str(out_dir))

    pipe = PIDPipeline(str(image_path), out_dir="output", cfg=cfg)

    t0 = time.time()
    pipe.run()  # Full pipeline through all stages
    elapsed = time.time() - t0

    # Collect key artifacts
    result = {
        "image": image_path.name,
        "elapsed_s": round(elapsed, 2),
        "output_dir": str(out_dir),
    }

    # Stage 5 pipe mask summary
    pm_summary = out_dir / "stage5_pipe_mask_summary.json"
    if pm_summary.exists():
        result["pipe_mask_summary"] = json.loads(pm_summary.read_text())

    graph_summary = out_dir / "stage7_graph_summary.json"
    if graph_summary.exists():
        result["graph_summary"] = json.loads(graph_summary.read_text())

    export_summary = out_dir / "stage10_process_export_summary.json"
    if export_summary.exists():
        result["process_export_summary"] = json.loads(export_summary.read_text())

    return result


def main():
    results = []
    for prefix, img_path in TEST_IMAGES:
        if not img_path.exists():
            print(f"SKIP: {img_path.name} (not found)")
            continue

        label = f"{prefix}/{img_path.stem}"

        print(f"\n{'='*60}")
        print(f"  CURRENT     {img_path.name}")
        print(f"{'='*60}")
        try:
            r = run_one(label, img_path)
            results.append(r)
            status = f"OK  {r['elapsed_s']:.0f}s"
            print(f"  -> {status}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({
                "image": img_path.name,
                "error": str(e),
                "output_dir": str(OUTPUT_ROOT / label / "default"),
            })

    # Write master report
    report_path = OUTPUT_ROOT / "batch_pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "total_images": len(results),
            "results": results,
        }, f, indent=2, default=str)
    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    ok = sum(1 for r in results if "error" not in r)
    fail = sum(1 for r in results if "error" in r)
    print(f"Completed: {ok} OK, {fail} FAILED out of {len(results)}")


if __name__ == "__main__":
    sys.exit(main())
