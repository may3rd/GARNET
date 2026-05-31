"""
run_continuity_checker_stage.py
Integrates pipe continuity checker into the GARNET pipeline.
Runs as part of Stage 13 (graph QA) and produces a dedicated
stage14_continuity_summary.json + overlay visualization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


def _draw_violations_overlay(
    image_bgr: Any,
    violations: list[dict[str, Any]],
) -> Any:
    overlay = image_bgr.copy()
    color_map = {
        "error": (0, 0, 255),     # red
        "warning": (0, 255, 255),  # yellow
    }
    for v in violations:
        pos = v.get("position") or {}
        x = int(round(float(pos.get("x", 0))))
        y = int(round(float(pos.get("y", 0))))
        color = color_map.get(v.get("severity", "warning"), (0, 255, 255))
        radius = 8 if v.get("severity") == "error" else 5
        cv2.circle(overlay, (x, y), radius, color, -1)
        # Label
        edge_ids = v.get("edge_ids", [])
        label = f"R{v['rule']}"
        cv2.putText(overlay, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        # Draw lines to affected edge endpoints
        for edge_id in edge_ids[:2]:  # max 2
            pass  # edge endpoints already shown via position markers
    return overlay


def run_continuity_checker_stage(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
    equipment_attachments_payload: dict[str, Any] | None = None,
    connection_attachments_payload: dict[str, Any] | None = None,
    image_bgr: Any,
) -> dict[str, Any]:
    from garnet.pipe_continuity_checker import check_continuity

    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])

    # Parse equipment attachments
    equip_attachments = None
    if equipment_attachments_payload:
        equip_attachments = equipment_attachments_payload.get("accepted", [])

    conn_attachments = None
    if connection_attachments_payload:
        conn_attachments = connection_attachments_payload.get("accepted", [])

    result = check_continuity(
        nodes=nodes,
        edges=edges,
        equipment_attachments=equip_attachments,
        connection_attachments=conn_attachments,
    )

    violations_out = [v.to_dict() for v in result.violations]
    overlay_img = _draw_violations_overlay(image_bgr, violations_out)

    return {
        "continuity_result": result.to_dict(),
        "violations": violations_out,
        "overlay_image": overlay_img,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "total_violations": len(violations_out),
            "errors": sum(1 for v in violations_out if v["severity"] == "error"),
            "warnings": sum(1 for v in violations_out if v["severity"] == "warning"),
            "rule_breakdown": result.to_dict()["summary"]["rule_counts"],
            "stats": result.stats,
            "source_artifacts": [
                "stage12_graph.json",
                "stage12_equipment_attachments.json",
                "stage12_connection_attachments.json",
            ],
        },
    }