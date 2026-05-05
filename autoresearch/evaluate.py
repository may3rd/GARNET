#!/usr/bin/env python3
"""
Fixed evaluation harness for GARNET autoresearch.

DO NOT MODIFY THIS FILE — it is the ground-truth evaluator.
Only PipelineConfig in pid_extractor.py is fair game.

Usage:
    python autoresearch/evaluate.py
    python autoresearch/evaluate.py --config-overrides '{"node_cluster_eps": 8.0}'
    python autoresearch/evaluate.py --image autoresearch/test_images/Test-00005.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-relative path setup so the script works from the repo root.
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from garnet.pid_extractor import PipelineConfig, PIDPipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Test images — relative to repo root.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_IMAGES = [
    REPO_ROOT / "autoresearch/test_images/Test-00001.jpg",
    REPO_ROOT / "autoresearch/test_images/Test-00003.jpg",
    REPO_ROOT / "autoresearch/test_images/Test-00005.jpg",
    REPO_ROOT / "autoresearch/test_images/Test-00008.jpg",
]

# ---------------------------------------------------------------------------
# Metric: graph quality score (lower = better)
# ---------------------------------------------------------------------------
def graph_quality_score(qa: dict) -> float:
    """
    Composite quality score from Stage 13 QA summary.

    Targets for an ideal P&ID graph:
    - 1 connected component (entire pipe network is connected)
    - 0 isolated nodes
    - 0 unresolved crossings
    - 0 unresolved terminal edges
    - minimal review queue items relative to graph size

    Lower is better (mirrors autoresearch's val_bpb convention).
    """
    components = qa.get("connected_component_count", 1)
    isolated = qa.get("isolated_node_count", 0)
    unresolved_crossings = qa.get("unresolved_crossing_count", 0)
    unresolved_terminals = qa.get("unresolved_terminal_edge_count", 0)
    review_queue = qa.get("review_queue_count", 0)

    # Edge count comes from stage12 summary; fall back to review_queue as proxy.
    total_edges = max(review_queue, 1)

    score = (
        (components - 1) * 2.0        # extra disconnected components
        + isolated * 1.5               # orphan nodes
        + unresolved_crossings * 3.0   # topology ambiguity (high penalty)
        + unresolved_terminals * 2.0   # open ends at equipment
        + review_queue / total_edges * 5.0  # unresolved items as fraction
    )
    return max(score, 0.0)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------
def evaluate_one(img_path: Path, cfg: PipelineConfig, out_root: Path) -> dict:
    """Run full pipeline on a single image, return per-image metrics."""
    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pipeline = PIDPipeline(str(img_path), out_dir=str(out_dir), cfg=cfg)
    pipeline.run(stop_after=13)
    elapsed = round(time.time() - t0, 1)

    qa_path = out_dir / "stage13_graph_qa_summary.json"
    g12_path = out_dir / "stage12_graph_summary.json"

    qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    g12 = json.loads(g12_path.read_text()) if g12_path.exists() else {}

    score = graph_quality_score(qa)

    return {
        "image": img_path.name,
        "score": round(score, 2),
        "components": qa.get("connected_component_count", -1),
        "isolated": qa.get("isolated_node_count", -1),
        "unresolved_crossings": qa.get("unresolved_crossing_count", -1),
        "unresolved_terminals": qa.get("unresolved_terminal_edge_count", -1),
        "review_queue": qa.get("review_queue_count", -1),
        "nodes": g12.get("node_count", -1),
        "edges": g12.get("edge_count", -1),
        "time_sec": elapsed,
    }


def evaluate(cfg: PipelineConfig, images: list[Path]) -> dict:
    """Run pipeline on all test images, return aggregate metrics."""
    out_root = REPO_ROOT / "autoresearch" / "tmp_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    per_image = []
    total_time = 0.0
    for img_path in images:
        result = evaluate_one(img_path, cfg, out_root)
        per_image.append(result)
        total_time += result["time_sec"]

    scores = [r["score"] for r in per_image]
    avg_score = sum(scores) / max(len(scores), 1)

    return {
        "avg_score": round(avg_score, 2),
        "total_time_sec": round(total_time, 1),
        "per_image": per_image,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GARNET autoresearch evaluator")
    parser.add_argument(
        "--config-overrides",
        type=str,
        default="{}",
        help='JSON dict of PipelineConfig overrides, e.g. \'{"node_cluster_eps": 8.0}\'',
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Evaluate a single image instead of the full test set",
    )
    args = parser.parse_args()

    overrides = json.loads(args.config_overrides)
    cfg = PipelineConfig(**overrides)

    if args.image:
        images = [Path(args.image)]
    else:
        images = DEFAULT_TEST_IMAGES

    results = evaluate(cfg, images)

    # Machine-readable output (parseable by grep)
    print("---")
    print(f"avg_score:     {results['avg_score']}")
    print(f"total_seconds: {results['total_time_sec']}")
    for r in results["per_image"]:
        print(
            f"  {r['image']:20s}  score={r['score']:8.1f}  "
            f"comp={r['components']:4d}  iso={r['isolated']:4d}  "
            f"x={r['unresolved_crossings']:3d}  term={r['unresolved_terminals']:4d}  "
            f"review={r['review_queue']:5d}  nodes={r['nodes']:5d}  "
            f"edges={r['edges']:5d}  time={r['time_sec']:.1f}s"
        )


if __name__ == "__main__":
    main()
