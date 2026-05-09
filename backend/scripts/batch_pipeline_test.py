#!/usr/bin/env python3
"""
batch_pipeline_test.py — run full pipeline on multiple images with both default
and geometric Stage 5 methods, saving outputs to GARNET project folder.
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


def run_one(label: str, image_path: Path, geometric: bool) -> dict:
    out_dir = OUTPUT_ROOT / label / ("geometric" if geometric else "default")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(use_geometric_line_detection=geometric)
    pipe = PIDPipeline(str(image_path), out_dir=str(out_dir), cfg=cfg)

    t0 = time.time()
    pipe.run(stop_after=10)  # Through Stage 10d
    elapsed = time.time() - t0

    # Collect key artifacts
    result = {
        "image": image_path.name,
        "geometric": geometric,
        "elapsed_s": round(elapsed, 2),
        "output_dir": str(out_dir),
    }

    # Stage 5 geometric summary (if geometric)
    geo_summary = out_dir / "stage5_geometric_summary.json"
    if geo_summary.exists():
        result["stage5_summary"] = json.loads(geo_summary.read_text())

    # Stage 5 pipe mask summary (both)
    pm_summary = out_dir / "stage5_pipe_mask_summary.json"
    if pm_summary.exists():
        result["pipe_mask_summary"] = json.loads(pm_summary.read_text())

    # Stage 6 sealing summary
    seal_summary = out_dir / "stage6_pipe_mask_sealed_summary.json"
    if seal_summary.exists():
        result["sealing_summary"] = json.loads(seal_summary.read_text())

    # Stage 10 continuity
    cont = out_dir / "stage10_continuity_result.json"
    if cont.exists():
        result["continuity_result"] = json.loads(cont.read_text())

    # Stage 10 gap summary
    gap = out_dir / "stage10_gap_summary.json"
    if gap.exists():
        result["gap_summary"] = json.loads(gap.read_text())

    # Stage 9 cluster summary
    cluster = out_dir / "stage9_node_cluster_summary.json"
    if cluster.exists():
        result["node_cluster_summary"] = json.loads(cluster.read_text())

    return result


def main():
    results = []
    for prefix, img_path in TEST_IMAGES:
        if not img_path.exists():
            print(f"SKIP: {img_path.name} (not found)")
            continue

        base = prefix / "geometric" if False else prefix  # just for display
        label = f"{prefix}/{img_path.stem}"

        for geometric in [True]:  # only geometric for now
            mode = "GEOMETRIC" if geometric else "default"
            print(f"\n{'='*60}")
            print(f"  {mode:10s}  {img_path.name}")
            print(f"{'='*60}")
            try:
                r = run_one(label, img_path, geometric)
                results.append(r)
                status = "OK"
                if "stage5_summary" in r:
                    s = r["stage5_summary"]
                    status += f"  segs={s.get('final_segments', '?')} H={s.get('horizontal_count', '?')} V={s.get('vertical_count', '?')}"
                status += f"  {r['elapsed_s']:.0f}s"
                print(f"  → {status}")
            except Exception as e:
                print(f"  → FAILED: {e}")
                results.append({
                    "image": img_path.name,
                    "geometric": geometric,
                    "error": str(e),
                    "output_dir": str(OUTPUT_ROOT / label / ("geometric" if geometric else "default")),
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
