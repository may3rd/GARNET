#!/usr/bin/env python3
"""
Merge multiple P&ID sheets using the multi-sheet merge engine.

Usage:
    python -m tools.merge_sheets job_id_1 job_id_2 [job_id_3 ...]
    python -m tools.merge_sheets --dir /path/to/output/pipeline_jobs job_id_1 job_id_2

Each job_id must have a ``stage12b_graph_v1.json`` artifact.
Output is written to ``merge_result.json`` in the current directory.

Example:
    python -m tools.merge_sheets ppcl-Test-00008 ppcl-Test-00009
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as module
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from garnet.pipe_sheet_merge import resolve_merge_pairs


def load_graph_v1(job_id: str, jobs_dir: Path) -> dict | None:
    """Load stage12b_graph_v1.json for a job, or None if not found."""
    graph_path = jobs_dir / job_id / "stage12b_graph_v1.json"
    if not graph_path.exists():
        return None
    with open(graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple P&ID sheets via the multi-sheet merge engine."
    )
    parser.add_argument(
        "job_ids",
        nargs="+",
        help="Pipeline job IDs to merge (each must have stage12b_graph_v1.json)",
    )
    parser.add_argument(
        "--dir",
        dest="jobs_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output" / "pipeline_jobs",
        metavar="DIR",
        help=f"Directory containing pipeline job folders (default: auto-detected)",
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        type=Path,
        default=Path("merge_result.json"),
        metavar="FILE",
        help="Output file path (default: merge_result.json)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    args = parser.parse_args()

    jobs_dir = args.jobs_dir
    graphs: list[dict] = []
    missing: list[str] = []

    for job_id in args.job_ids:
        g = load_graph_v1(job_id, jobs_dir)
        if g is None:
            missing.append(job_id)
        else:
            doc_id = g.get("document", {}).get("doc_id") or job_id
            if not g.get("document", {}).get("doc_id"):
                g.setdefault("document", {})["doc_id"] = doc_id
            graphs.append(g)

    if missing:
        print(
            f"WARNING: stage12b_graph_v1.json not found for: {', '.join(missing)}",
            file=sys.stderr,
        )

    if not graphs:
        print("ERROR: No valid graph payloads found. Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"Merging {len(graphs)} sheet(s)...", file=sys.stderr)
    result = resolve_merge_pairs(graphs)
    result_dict = result.to_dict()

    output_path = args.output_file
    with open(output_path, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        else:
            json.dump(result_dict, f, ensure_ascii=False)

    # Summary
    print(f"Done. Output: {output_path}", file=sys.stderr)
    print(
        f"  cross-sheet edges: {len(result_dict['cross_sheet_edges'])}",
        file=sys.stderr,
    )
    print(
        f"  merge issues:     {len(result_dict['merge_issues'])}",
        file=sys.stderr,
    )
    for issue in result_dict["merge_issues"]:
        print(
            f"    [{issue['type']}] {issue['issue_id']}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
