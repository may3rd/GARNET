#!/usr/bin/env python3
"""
Fixed evaluation harness for GARNET graph-loop autoresearch.

This evaluator is scoped to the Stage 10-13 graph extraction results.
Only the graph extraction files are intended to change during the loop.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from garnet.pid_extractor import PipelineConfig, PIDPipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES = [
    REPO_ROOT / "autoresearch/test_images/Test-00001.jpg",
    REPO_ROOT / "autoresearch/test_images/Test-00003.jpg",
    REPO_ROOT / "autoresearch/test_images/Test-00005.jpg",
    REPO_ROOT / "autoresearch/test_images/Test-00008.jpg",
]


def graph_loop_score(stage12: dict, stage13: dict, connection_summary: dict) -> float:
    edge_components = float(stage12.get("edge_component_count", 0))
    unresolved_terminals = float(stage13.get("unresolved_terminal_edge_count", 0))
    review_queue = float(stage13.get("review_queue_count", 0))
    provisional_edges = float(stage12.get("edge_count", 0) - stage12.get("accepted_attachment_count", 0))
    connection_attachment_count = float(stage12.get("accepted_connection_attachment_count", 0))
    seeded_continuations = float(connection_summary.get("connection_seeded_continuation_count", 0))

    score = (
        edge_components * 1.5
        + unresolved_terminals * 2.0
        + review_queue * 0.2
        + provisional_edges * 0.1
        - seeded_continuations * 0.5
    )
    if connection_attachment_count > 0:
        score -= min(connection_attachment_count, seeded_continuations) * 0.25
    return round(max(score, 0.0), 2)


def evaluate_one(img_path: Path, cfg: PipelineConfig, out_root: Path) -> dict:
    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipeline = PIDPipeline(str(img_path), out_dir=str(out_dir), cfg=cfg)
    pipeline.run(stop_after=13)
    elapsed = round(time.time() - t0, 1)

    stage12 = json.loads((out_dir / "stage12_graph_summary.json").read_text())
    stage13 = json.loads((out_dir / "stage13_graph_qa_summary.json").read_text())
    connection_summary = json.loads((out_dir / "stage12_edge_connection_summary.json").read_text())

    score = graph_loop_score(stage12, stage13, connection_summary)
    return {
        "image": img_path.name,
        "score": score,
        "edge_components": stage12.get("edge_component_count", -1),
        "unresolved_terminals": stage13.get("unresolved_terminal_edge_count", -1),
        "review_queue": stage13.get("review_queue_count", -1),
        "connection_seeded_continuation_count": connection_summary.get("connection_seeded_continuation_count", -1),
        "time_sec": elapsed,
    }


def main() -> None:
    out_root = REPO_ROOT / "autoresearch" / "graph_loop" / "tmp_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig()
    per_image = [evaluate_one(path, cfg, out_root) for path in TEST_IMAGES]
    avg_score = round(sum(item["score"] for item in per_image) / max(len(per_image), 1), 2)
    total_seconds = round(sum(item["time_sec"] for item in per_image), 1)

    print("---")
    print(f"avg_score:     {avg_score}")
    print(f"total_seconds: {total_seconds}")
    for item in per_image:
        print(
            f"  {item['image']:20s} score={item['score']:8.1f} "
            f"edge_comp={item['edge_components']:5d} "
            f"term={item['unresolved_terminals']:5d} "
            f"review={item['review_queue']:5d} "
            f"conn_seed={item['connection_seeded_continuation_count']:4d} "
            f"time={item['time_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()
