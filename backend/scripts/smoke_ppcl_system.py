"""Smoke test: multi-page pipeline system end-to-end via the live API.

Uploads PPCL sheets as a pipeline system (sheet IDs confirmed up front),
resumes every page through stage 11 (skipping the HITL gates), waits for the
strict connector merge, verifies the system-graph payload consumed by the
connector-review UI (resolved edges, issue badges, connector map positions),
then submits an accept-all connector review and verifies the graph
regenerates with a bumped revision.

Usage (from backend/):
  ../.venv/bin/python scripts/smoke_ppcl_system.py \
      --sheets test/ppcl/Test-00001.jpg=25-0002 test/ppcl/Test-00002.jpg=25-0003 \
      --api http://127.0.0.1:8090 --ocr-route ocrmac
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import requests

print = functools.partial(print, flush=True)  # noqa: A001 - unbuffered progress for polled runs

GATE_STATUSES = {"awaiting_object_review", "awaiting_trace_review", "awaiting_graph_review"}
SETTLED_PAGE_STATUSES = GATE_STATUSES | {"completed"}


def fail(message: str) -> None:
    print(f"[smoke-system] FAIL: {message}")
    sys.exit(1)


def poll_system(api: str, system_id: str, deadline: float, interval: float = 5.0) -> dict:
    last_line = ""
    while time.time() < deadline:
        response = requests.get(f"{api}/api/pipeline/systems/{system_id}", timeout=30)
        response.raise_for_status()
        system = response.json()
        pages_summary = ", ".join(
            f"{page['sheet_id']}={page['status']}" for page in system["pages"]
        )
        line = f"system={system['status']} merge={system['merge'].get('status')} pages: {pages_summary}"
        if line != last_line:
            print(f"[smoke-system] {line}")
            last_line = line
        page_statuses = {page["status"] for page in system["pages"]}
        if "failed" in page_statuses or system["status"] == "failed":
            return system
        if page_statuses <= SETTLED_PAGE_STATUSES:
            return system
        time.sleep(interval)
    fail(f"timed out waiting for system {system_id}")


def connector_rows(graph: dict) -> list[dict]:
    rows = []
    for sheet in graph.get("sheets", []):
        nodes_by_id = {node["id"]: node for node in sheet["graph_v1"].get("nodes", [])}
        tile = (sheet["graph_v1"].get("tiling") or {}).get("tile") or {}
        width, height = tile.get("tile_width"), tile.get("tile_height")
        for edge in sheet["graph_v1"].get("edges", []):
            connector = edge.get("off_page_connector")
            if not isinstance(connector, dict):
                continue
            node_id = next(
                (str(node_id) for node_id in (edge.get("src"), edge.get("dst")) if str(node_id).startswith("connection::")),
                None,
            )
            node = nodes_by_id.get(node_id or "")
            bbox = (node or {}).get("bbox")
            center = ((node or {}).get("geometry") or {}).get("center")
            x = bbox["x"] + bbox["w"] / 2 if bbox else (center or {}).get("x")
            y = bbox["y"] + bbox["h"] / 2 if bbox else (center or {}).get("y")
            rows.append(
                {
                    "connector_id": f"{sheet['sheet_id']}::{edge['id']}",
                    "sheet_id": sheet["sheet_id"],
                    "connector_key": str(connector.get("connector_key") or ""),
                    "target_sheet_id": str(connector.get("target_sheet_reference") or connector.get("reference_value") or ""),
                    "x": x,
                    "y": y,
                    "tile_width": width,
                    "tile_height": height,
                }
            )
    return rows


def verify_graph_payload(graph: dict, stage: str) -> None:
    rows = connector_rows(graph)
    if not rows:
        fail(f"{stage}: no off-page connector edges in system graph")
    sheet_ids = {sheet["sheet_id"] for sheet in graph.get("sheets", [])}
    for row in rows:
        if row["x"] is None or row["y"] is None:
            fail(f"{stage}: connector {row['connector_id']} has no node position (bbox/geometry.center)")
        if not row["tile_width"] or not row["tile_height"]:
            fail(f"{stage}: sheet {row['sheet_id']} missing tiling.tile dimensions for position scaling")
        nx, ny = row["x"] / row["tile_width"], row["y"] / row["tile_height"]
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            fail(f"{stage}: connector {row['connector_id']} normalizes outside the sheet: ({nx:.3f}, {ny:.3f})")
    resolved_ids = {
        terminal.get("connector_id")
        for edge in graph.get("cross_sheet_edges", [])
        for terminal in edge.get("terminals", [])
        if terminal.get("connector_id")
    }
    known_ids = {row["connector_id"] for row in rows}
    unknown = resolved_ids - known_ids
    if unknown:
        fail(f"{stage}: resolved edges reference unknown connectors: {sorted(unknown)}")
    for edge in graph.get("cross_sheet_edges", []):
        sheets = set(edge.get("sheets") or [])
        if not sheets <= sheet_ids:
            fail(f"{stage}: resolved edge {edge.get('id')} references unknown sheets {sheets - sheet_ids}")
    print(
        f"[smoke-system] {stage}: {len(rows)} connectors, "
        f"{len(graph.get('cross_sheet_edges', []))} resolved, "
        f"{len(graph.get('merge_issues', []))} issues"
    )
    for edge in graph.get("cross_sheet_edges", []):
        print(f"[smoke-system]   resolved: {edge.get('connector_key')} ({', '.join(sorted(set(edge.get('sheets') or [])))}) via {edge.get('match_method')}")
    for issue in graph.get("merge_issues", []):
        connector_ids = [item.get("connector_id") for item in issue.get("connectors", [])]
        print(f"[smoke-system]   issue: {issue['type']} -> {connector_ids}")


def main() -> int:
    parser = argparse.ArgumentParser(description="pipeline system end-to-end smoke test")
    parser.add_argument("--sheets", nargs="+", required=True, help="Image paths as path=sheet_id pairs (2-50)")
    parser.add_argument("--api", default="http://127.0.0.1:8090", help="Base URL of the running backend")
    parser.add_argument("--ocr-route", default="ocrmac", help="OCR route for every page")
    parser.add_argument("--weight", default="yolo_weights/yolo26n_PPCL_640_20260227.pt", help="Stage 4 weight file")
    parser.add_argument("--reviewer", default="smoke", help="Reviewer name recorded with the connector review")
    parser.add_argument("--timeout", type=float, default=2400.0, help="Overall deadline in seconds")
    args = parser.parse_args()

    pairs: list[tuple[Path, str]] = []
    for item in args.sheets:
        path, separator, sheet_id = item.rpartition("=")
        if not separator or not sheet_id.strip():
            fail(f"--sheets entries must look like path=sheet_id, got: {item}")
        image = Path(path)
        if not image.is_file():
            fail(f"sheet image not found: {image}")
        pairs.append((image, sheet_id.strip()))
    if not 2 <= len(pairs) <= 50:
        fail("pipeline systems require 2 to 50 pages")

    files = [("files", (image.name, image.read_bytes(), "image/jpeg")) for image, _sheet_id in pairs]
    payload = {
        "sheet_ids": [sheet_id for _image, sheet_id in pairs],
        "ocr_route": args.ocr_route,
        "weight_file": args.weight,
    }
    print(f"[smoke-system] creating system: {[sheet_id for _i, sheet_id in pairs]}")
    response = requests.post(
        f"{args.api}/api/pipeline/systems",
        files=files,
        data=payload,
        timeout=120,
    )
    if response.status_code != 200:
        fail(f"system creation failed ({response.status_code}): {response.text[:400]}")
    system = response.json()
    system_id = system["system_id"]
    print(f"[smoke-system] system {system_id}")

    system = poll_system(args.api, system_id, time.time() + args.timeout)
    if system["status"] == "failed":
        failed = [page for page in system["pages"] if page["status"] == "failed"]
        fail(f"page(s) failed: {[page['sheet_id'] for page in failed]}")

    for page in system["pages"]:
        if page["status"] == "completed":
            continue
        print(f"[smoke-system] resuming {page['sheet_id']} from stage5_pipe_mask (stop_after=11)")
        resume = requests.post(
            f"{args.api}/api/pipeline/jobs/{page['job_id']}/resume-from/stage5_pipe_mask",
            params={"stop_after": 11},
            timeout=30,
        )
        if resume.status_code != 200:
            fail(f"resume failed for {page['sheet_id']} ({resume.status_code}): {resume.text[:300]}")

    system = poll_system(args.api, system_id, time.time() + args.timeout)
    if system["status"] == "failed":
        fail("system failed during stage 5-11 resume")

    graph_response = requests.get(f"{args.api}/api/pipeline/systems/{system_id}/graph", timeout=60)
    if graph_response.status_code != 200:
        fail(f"system graph not ready ({graph_response.status_code}): {graph_response.text[:300]}")
    graph = graph_response.json()
    verify_graph_payload(graph, "after-merge")

    rows = connector_rows(graph)
    overrides = [
        {
            "connector_id": row["connector_id"],
            "connector_key": row["connector_key"],
            "target_sheet_id": row["target_sheet_id"] if row["target_sheet_id"] in {sid for _i, sid in pairs} else None,
            "review_state": "accepted",
        }
        for row in rows
    ]
    review = requests.put(
        f"{args.api}/api/pipeline/systems/{system_id}/connector-review",
        json={"connector_overrides": overrides, "manual_pairs": [], "reviewer": args.reviewer},
        timeout=120,
    )
    if review.status_code != 200:
        fail(f"connector review failed ({review.status_code}): {review.text[:400]}")
    reviewed = review.json()
    revision = (reviewed.get("connector_review") or {}).get("revision")
    if revision != 1:
        fail(f"expected connector review revision 1, got {revision}")
    print(f"[smoke-system] connector review saved (revision {revision}, status {reviewed['status']})")

    graph_response = requests.get(f"{args.api}/api/pipeline/systems/{system_id}/graph", timeout=60)
    if graph_response.status_code != 200:
        fail(f"graph not regenerated after review ({graph_response.status_code})")
    verify_graph_payload(graph_response.json(), "after-review")

    print(f"[smoke-system] OK: system {system_id} finished at status {reviewed['status']}")
    print(f"[smoke-system] graph download: {args.api}/api/pipeline/systems/{system_id}/graph")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except requests.RequestException as exc:
        fail(f"request error: {exc}")
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON response: {exc}")