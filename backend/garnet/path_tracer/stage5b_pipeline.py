"""Stage 5b pipe-tracing pipeline mixin.

The CV walk logic lives in `cv_pipe_tracer.py`; this mixin owns the
artifact-level Stage 5b orchestration, branch candidate discovery, overlays,
and equipment-port preparation used by `PIDPipeline`.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from garnet.path_tracer.cv_pipe_tracer import CVPipeTracer

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

logger = logging.getLogger("pid")

EQUIPMENT_LABELS = {
    "vessel",
    "column",
    "pump",
    "compressor",
    "blower",
    "heat exchanger",
    "heat_exchanger",
    "tank",
    "reactor",
    "mixer",
    "pot",
    "knockout drum",
    "knockout_drum",
    "filter",
    "cooler",
    "heater",
    "injection pump",
    "injection_pump",
}


def normalize_for_save(img: np.ndarray) -> np.ndarray:
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


class Stage5bPipelineMixin:
    # ------------------------------------------------------------------
    # VLM port detection (replaces heuristic get_connection_ports)
    # ------------------------------------------------------------------

    def _snap_port_to_pipe_centerline(
        self,
        pipe_mask: np.ndarray,
        port: tuple[int, int, str],
        search_radius: int = 12,
    ) -> tuple[int, int, str]:
        """Move a bbox-edge port onto the center of the local pipe thickness."""
        x, y, direction = port
        direction = direction.upper()
        h, w = pipe_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return (int(x), int(y), direction)

        def _closest_run(values: list[int], current: int) -> Optional[tuple[int, int]]:
            if not values:
                return None
            values = sorted(set(values))
            runs: list[tuple[int, int]] = []
            start = prev = values[0]
            for value in values[1:]:
                if value == prev + 1:
                    prev = value
                    continue
                runs.append((start, prev))
                start = prev = value
            runs.append((start, prev))
            return min(
                runs,
                key=lambda r: (
                    0
                    if r[0] <= current <= r[1]
                    else min(abs(current - r[0]), abs(current - r[1]))
                ),
            )

        if direction in ("UP", "DOWN"):
            cols = [
                cx for cx in range(max(0, x - search_radius), min(w, x + search_radius + 1))
                if pipe_mask[y, cx] > 0
            ]
            run = _closest_run(cols, x)
            if run:
                x = (run[0] + run[1]) // 2
        else:
            rows = [
                cy for cy in range(max(0, y - search_radius), min(h, y + search_radius + 1))
                if pipe_mask[cy, x] > 0
            ]
            run = _closest_run(rows, y)
            if run:
                y = (run[0] + run[1]) // 2
        return (int(x), int(y), direction)

    def _snap_ports_to_pipe_centerlines(
        self,
        ports: dict[str, list],
        pipe_mask: np.ndarray,
    ) -> dict[str, list[tuple[int, int, str]]]:
        snapped: dict[str, list[tuple[int, int, str]]] = {}
        for obj_id, port_list in ports.items():
            snapped[obj_id] = []
            for port in port_list:
                if len(port) != 3:
                    continue
                px, py, direction = port
                snapped[obj_id].append(
                    self._snap_port_to_pipe_centerline(
                        pipe_mask,
                        (int(px), int(py), str(direction)),
                    )
                )
        return snapped

    def _point_near_segment(
        self,
        x: int,
        y: int,
        seg: dict[str, Any],
        tolerance: int = 5,
    ) -> bool:
        x1 = int(seg["x1"])
        y1 = int(seg["y1"])
        x2 = int(seg["x2"])
        y2 = int(seg["y2"])
        if seg["direction"] in ("LEFT", "RIGHT"):
            return (
                min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
                and abs(y - y1) <= tolerance
            )
        return (
            min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
            and abs(x - x1) <= tolerance
        )

    def _point_inside_any_bbox(
        self,
        x: int,
        y: int,
        objects: list[dict[str, Any]],
        margin: int = 4,
    ) -> bool:
        for obj in objects:
            bbox = obj.get("bbox")
            if not bbox:
                continue
            if (
                bbox["x_min"] - margin <= x <= bbox["x_max"] + margin
                and bbox["y_min"] - margin <= y <= bbox["y_max"] + margin
            ):
                return True
        return False

    def _branch_already_traced(
        self,
        x: int,
        y: int,
        branch_direction: str,
        all_results: dict[str, dict],
        source_obj_id: str,
        source_segment_index: int,
        tolerance: int = 5,
    ) -> bool:
        dx, dy = {
            "UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0),
        }[branch_direction]
        probe_x = x + dx * 12
        probe_y = y + dy * 12
        for obj_id, result in all_results.items():
            for seg_index, seg in enumerate(result.get("segments", [])):
                if obj_id == source_obj_id and seg_index == source_segment_index:
                    continue
                opposite = {
                    "UP": "DOWN",
                    "DOWN": "UP",
                    "LEFT": "RIGHT",
                    "RIGHT": "LEFT",
                }
                if seg.get("direction") not in (branch_direction, opposite[branch_direction]):
                    continue
                if self._point_near_segment(probe_x, probe_y, seg, tolerance=tolerance):
                    return True
        return False

    def _has_orthogonal_branch_run(
        self,
        pipe_mask: np.ndarray,
        x: int,
        y: int,
        direction: str,
        min_run: int = 25,
        max_exact_gap: int = 3,
    ) -> bool:
        deltas = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }
        dx, dy = deltas[direction]
        h, w = pipe_mask.shape
        exact_hits = 0
        current_gap = 0
        max_gap = 0
        for step in range(1, min_run + 1):
            px = x + dx * step
            py = y + dy * step
            ok = 0 <= px < w and 0 <= py < h and pipe_mask[py, px] > 0
            if ok:
                exact_hits += 1
                current_gap = 0
            else:
                current_gap += 1
                max_gap = max(max_gap, current_gap)

        # A real branch may have small raster gaps, but it should not drift like
        # a diagonal leader line or symbol stroke.
        return exact_hits >= int(round(min_run * 0.72)) and max_gap <= max_exact_gap

    def _inline_object_on_axis(
        self,
        x: int,
        y: int,
        direction: str,
        inline_symbols: list[dict[str, Any]],
        max_distance: int = 35,
        axis_margin: int = 3,
    ) -> Optional[dict[str, Any]]:
        best: Optional[tuple[int, dict[str, Any]]] = None
        for obj in inline_symbols:
            bbox = obj.get("bbox")
            if not bbox:
                continue
            if direction == "UP":
                if not (bbox["x_min"] - axis_margin <= x <= bbox["x_max"] + axis_margin):
                    continue
                distance = y - int(bbox["y_max"])
            elif direction == "DOWN":
                if not (bbox["x_min"] - axis_margin <= x <= bbox["x_max"] + axis_margin):
                    continue
                distance = int(bbox["y_min"]) - y
            elif direction == "LEFT":
                if not (bbox["y_min"] - axis_margin <= y <= bbox["y_max"] + axis_margin):
                    continue
                distance = x - int(bbox["x_max"])
            else:
                if not (bbox["y_min"] - axis_margin <= y <= bbox["y_max"] + axis_margin):
                    continue
                distance = int(bbox["x_min"]) - x

            if distance < -axis_margin or distance > max_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, obj)
        return best[1] if best else None

    def _has_inline_bridge_branch_run(
        self,
        pipe_mask: np.ndarray,
        x: int,
        y: int,
        direction: str,
        inline_symbols: list[dict[str, Any]],
        min_run: int = 25,
    ) -> bool:
        deltas = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }
        dx, dy = deltas[direction]
        h, w = pipe_mask.shape

        def has_pipe_near(px: int, py: int, band: int = 1) -> bool:
            if direction in ("LEFT", "RIGHT"):
                return any(
                    0 <= py + off < h and 0 <= px < w and pipe_mask[py + off, px] > 0
                    for off in range(-band, band + 1)
                )
            return any(
                0 <= px + off < w and 0 <= py < h and pipe_mask[py, px + off] > 0
                for off in range(-band, band + 1)
            )

        lead_run = 0
        for step in range(1, 10):
            if has_pipe_near(x + dx * step, y + dy * step):
                lead_run += 1
            else:
                break
        if lead_run < 3:
            return False

        inline_obj = self._inline_object_on_axis(x, y, direction, inline_symbols)
        if not inline_obj:
            return False

        bbox = inline_obj["bbox"]
        if direction == "UP":
            exit_x, exit_y = x, int(bbox["y_min"]) - 1
        elif direction == "DOWN":
            exit_x, exit_y = x, int(bbox["y_max"]) + 1
        elif direction == "LEFT":
            exit_x, exit_y = int(bbox["x_min"]) - 1, y
        else:
            exit_x, exit_y = int(bbox["x_max"]) + 1, y

        resumed = False
        resume_x = exit_x
        resume_y = exit_y
        for offset in range(0, 16):
            px = exit_x + dx * offset
            py = exit_y + dy * offset
            if has_pipe_near(px, py):
                resume_x, resume_y = px, py
                resumed = True
                break
        if not resumed:
            return False

        after_hits = 0
        for step in range(0, 16):
            if has_pipe_near(resume_x + dx * step, resume_y + dy * step):
                after_hits += 1
        span = max(abs(resume_x - x), abs(resume_y - y))
        return span >= min_run and after_hits >= 5

    def _has_branch_candidate_run(
        self,
        pipe_mask: np.ndarray,
        x: int,
        y: int,
        direction: str,
        inline_symbols: list[dict[str, Any]],
        min_run: int = 25,
    ) -> bool:
        from garnet.path_tracer.cv_pipe_tracer import _has_connected_side_pipe

        if _has_connected_side_pipe(pipe_mask, x, y, direction, min_run=min_run):
            return self._has_orthogonal_branch_run(pipe_mask, x, y, direction, min_run=min_run)
        return self._has_inline_bridge_branch_run(
            pipe_mask,
            x,
            y,
            direction,
            inline_symbols,
            min_run=min_run,
        )

    def _cluster_branch_candidates(
        self,
        raw_candidates: list[dict[str, Any]],
        radius: int = 8,
    ) -> list[dict[str, Any]]:
        clusters: list[dict[str, Any]] = []
        for candidate in raw_candidates:
            match = None
            for cluster in clusters:
                if cluster["branch_direction"] != candidate["branch_direction"]:
                    continue
                if (
                    abs(cluster["x"] - candidate["x"]) <= radius
                    and abs(cluster["y"] - candidate["y"]) <= radius
                ):
                    match = cluster
                    break
            if match is None:
                clusters.append({**candidate, "members": [candidate]})
                continue
            match["members"].append(candidate)
            members = match["members"]
            match["x"] = int(round(sum(m["x"] for m in members) / len(members)))
            match["y"] = int(round(sum(m["y"] for m in members) / len(members)))
            if candidate["status"] == "queued":
                match["status"] = "queued"
                match["reason"] = candidate["reason"]
            if candidate.get("node_obj_id") and not match.get("node_obj_id"):
                match["node_obj_id"] = candidate["node_obj_id"]
                match["reason"] = candidate.get("reason", match["reason"])
        return clusters

    def _detect_stage5b_branch_candidates(
        self,
        pipe_mask: np.ndarray,
        all_results: dict[str, dict],
        inline_symbols: list[dict[str, Any]],
        node_symbols: Optional[list[dict[str, Any]]] = None,
        equipment_objects: Optional[list[dict[str, Any]]] = None,
        sample_step: int = 5,
        min_branch_run: int = 25,
    ) -> list[dict[str, Any]]:
        from garnet.path_tracer.cv_pipe_tracer import (
            TURN_LEFT,
            TURN_RIGHT,
        )

        tee_points = [
            (int(result["terminal_x"]), int(result["terminal_y"]))
            for result in all_results.values()
            if result.get("terminal_type") == "tee_junction"
        ]
        turn_points = [
            (int(turn["x"]), int(turn["y"]))
            for result in all_results.values()
            for turn in result.get("turns", [])
        ]
        existing_points = tee_points + turn_points

        raw_candidates: list[dict[str, Any]] = []
        for obj_id, result in all_results.items():
            for seg_index, seg in enumerate(result.get("segments", [])):
                direction = seg["direction"]
                branch_dirs = (TURN_LEFT[direction], TURN_RIGHT[direction])
                x1 = int(seg["x1"])
                y1 = int(seg["y1"])
                x2 = int(seg["x2"])
                y2 = int(seg["y2"])
                if direction in ("LEFT", "RIGHT"):
                    length = abs(x2 - x1)
                else:
                    length = abs(y2 - y1)
                if length < min_branch_run:
                    continue
                dx, dy = {
                    "UP": (0, -1),
                    "DOWN": (0, 1),
                    "LEFT": (-1, 0),
                    "RIGHT": (1, 0),
                }[direction]

                for dist in range(min_branch_run, max(min_branch_run, length - min_branch_run) + 1, sample_step):
                    if direction in ("LEFT", "RIGHT"):
                        x = x1 + dx * dist
                        y = int(round(y1 + ((y2 - y1) * dist / length)))
                    else:
                        x = int(round(x1 + ((x2 - x1) * dist / length)))
                        y = y1 + dy * dist
                    if self._point_inside_any_bbox(x, y, inline_symbols, margin=0):
                        continue

                    for branch_direction in branch_dirs:
                        if not self._has_branch_candidate_run(
                            pipe_mask,
                            x,
                            y,
                            branch_direction,
                            inline_symbols,
                            min_run=min_branch_run,
                        ):
                            continue

                        status = "queued"
                        reason = "untraced_branch"
                        if self._point_inside_any_bbox(x, y, equipment_objects or [], margin=2):
                            status = "rejected_inside_equipment"
                            reason = "candidate_inside_equipment_bbox"
                        elif any(abs(x - tx) <= 10 and abs(y - ty) <= 10 for tx, ty in tee_points):
                            status = "done_existing_tee"
                            reason = "near_existing_tee_terminal"
                        elif any(abs(x - tx) <= 8 and abs(y - ty) <= 8 for tx, ty in turn_points):
                            status = "done_existing_turn"
                            reason = "near_existing_turn"
                        elif self._branch_already_traced(
                            x, y, branch_direction, all_results, obj_id, seg_index
                        ):
                            status = "done_already_traced"
                            reason = "branch_direction_covered_by_existing_segment"

                        raw_candidates.append({
                            "x": int(x),
                            "y": int(y),
                            "branch_direction": branch_direction,
                            "source_trace_id": obj_id,
                            "source_segment_index": seg_index,
                            "source_direction": direction,
                            "status": status,
                            "reason": reason,
                        })

        for node in node_symbols or []:
            bbox = node.get("bbox")
            if not bbox:
                continue
            x = int(round((bbox["x_min"] + bbox["x_max"]) / 2))
            y = int(round((bbox["y_min"] + bbox["y_max"]) / 2))
            if any(abs(x - px) <= 10 and abs(y - py) <= 10 for px, py in existing_points):
                continue
            if self._point_inside_any_bbox(x, y, equipment_objects or [], margin=2):
                continue

            matched_segment: Optional[tuple[str, int, dict[str, Any]]] = None
            for obj_id, result in all_results.items():
                for seg_index, seg in enumerate(result.get("segments", [])):
                    if self._point_near_segment(x, y, seg, tolerance=6):
                        matched_segment = (obj_id, seg_index, seg)
                        break
                if matched_segment:
                    break
            if not matched_segment:
                continue

            source_obj_id, source_segment_index, source_seg = matched_segment
            source_direction = str(source_seg["direction"])
            for branch_direction in (TURN_LEFT[source_direction], TURN_RIGHT[source_direction]):
                if not self._has_branch_candidate_run(
                    pipe_mask,
                    x,
                    y,
                    branch_direction,
                    inline_symbols,
                    min_run=min_branch_run,
                ):
                    continue
                status = "queued"
                reason = "node_object_branch"
                if self._branch_already_traced(
                    x,
                    y,
                    branch_direction,
                    all_results,
                    source_obj_id,
                    source_segment_index,
                ):
                    status = "done_already_traced"
                    reason = "branch_direction_covered_by_existing_segment"
                raw_candidates.append({
                    "x": x,
                    "y": y,
                    "branch_direction": branch_direction,
                    "source_trace_id": source_obj_id,
                    "source_segment_index": source_segment_index,
                    "source_direction": source_direction,
                    "status": status,
                    "reason": reason,
                    "node_obj_id": node.get("id", ""),
                })

        candidates = self._cluster_branch_candidates(raw_candidates)
        for candidate in candidates:
            result = all_results.get(str(candidate.get("source_trace_id", "")), {})
            segments = result.get("segments", [])
            seg_index = candidate.get("source_segment_index")
            if not isinstance(seg_index, int) or seg_index < 0 or seg_index >= len(segments):
                candidate["status"] = "rejected_not_on_source_trace"
                candidate["reason"] = "missing_source_segment"
                continue
            if not self._point_near_segment(
                int(candidate["x"]),
                int(candidate["y"]),
                segments[seg_index],
                tolerance=6,
            ):
                candidate["status"] = "rejected_not_on_source_trace"
                candidate["reason"] = "candidate_off_source_centerline"
        for index, candidate in enumerate(candidates, start=1):
            candidate["id"] = f"branch_{index:06d}"
            candidate["member_count"] = len(candidate.pop("members", []))
        return candidates

    def _draw_stage5b_branch_candidate_overlay(
        self,
        image: np.ndarray,
        candidates: list[dict[str, Any]],
        base_results: Optional[dict[str, dict]] = None,
        branch_results: Optional[dict[str, dict]] = None,
    ) -> np.ndarray:
        import cv2 as _cv2

        overlay = image.copy()
        self._draw_stage5b_result_paths(
            overlay,
            base_results or {},
            line_color=(0, 170, 0),
            terminal_color=(0, 120, 0),
            thickness=2,
            label_terminals=False,
        )
        self._draw_stage5b_result_paths(
            overlay,
            branch_results or {},
            line_color=(0, 0, 255),
            terminal_color=(0, 0, 255),
            thickness=3,
            label_terminals=True,
        )
        colors = {
            "queued": (0, 0, 255),
            "done_existing_tee": (0, 180, 0),
            "done_existing_turn": (0, 180, 0),
            "done_already_traced": (0, 180, 0),
            "done_branch_connection": (0, 180, 0),
            "done_used_by_branch": (0, 180, 0),
            "pending_branch_connection": (0, 180, 180),
        }
        deltas = {
            "UP": (0, -16), "DOWN": (0, 16),
            "LEFT": (-16, 0), "RIGHT": (16, 0),
        }
        for candidate in candidates:
            x = int(candidate["x"])
            y = int(candidate["y"])
            status = candidate.get("status", "queued")
            color = colors.get(status, (128, 128, 128))
            _cv2.circle(overlay, (x, y), 7, color, -1)
            _cv2.circle(overlay, (x, y), 7, (255, 255, 255), 1)
            dx, dy = deltas.get(candidate.get("branch_direction", ""), (0, 0))
            if dx or dy:
                _cv2.arrowedLine(
                    overlay,
                    (x, y),
                    (x + dx, y + dy),
                    color,
                    2,
                    tipLength=0.35,
                )
            label = candidate.get("id", "").replace("branch_", "b")
            _cv2.putText(
                overlay,
                label,
                (x + 9, y - 9),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
        return overlay

    def _draw_stage5b_result_paths(
        self,
        overlay: np.ndarray,
        results: dict[str, dict],
        line_color: tuple[int, int, int],
        terminal_color: tuple[int, int, int],
        thickness: int,
        label_terminals: bool,
    ) -> None:
        import cv2 as _cv2

        for trace_id, data in (results or {}).items():
            if not data.get("segments"):
                continue
            if data.get("status") not in (None, "ok", "traced"):
                continue
            for seg in data.get("segments", []):
                _cv2.line(
                    overlay,
                    (int(seg["x1"]), int(seg["y1"])),
                    (int(seg["x2"]), int(seg["y2"])),
                    line_color,
                    thickness,
                )
            tx = int(data.get("terminal_x", 0))
            ty = int(data.get("terminal_y", 0))
            if tx or ty:
                _cv2.circle(overlay, (tx, ty), 4 + thickness, terminal_color, -1)
                if label_terminals:
                    _cv2.putText(
                        overlay,
                        str(trace_id).replace("branch_", "b"),
                        (tx + 8, ty + 14),
                        _cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        terminal_color,
                        1,
                    )

    def _draw_stage5b_branch_trace_overlay(
        self,
        image: np.ndarray,
        branch_results: dict[str, dict],
    ) -> np.ndarray:
        import cv2 as _cv2

        overlay = image.copy()
        for branch_id, data in (branch_results or {}).items():
            if data.get("status") != "traced":
                continue
            for seg in data.get("segments", []):
                _cv2.line(
                    overlay,
                    (int(seg["x1"]), int(seg["y1"])),
                    (int(seg["x2"]), int(seg["y2"])),
                    (0, 0, 255),
                    3,
                )
            tx = int(data.get("terminal_x", 0))
            ty = int(data.get("terminal_y", 0))
            _cv2.circle(overlay, (tx, ty), 6, (0, 0, 255), -1)
            _cv2.putText(
                overlay,
                f"{branch_id}:branch",
                (tx + 8, ty + 14),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )
        return overlay

    def _safe_stage5b_trace_image_name(self, trace_id: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
            for ch in str(trace_id)
        ).strip("._")
        return safe or "trace"

    def _draw_single_stage5b_trace_path(
        self,
        image: np.ndarray,
        trace_id: str,
        data: dict[str, Any],
        *,
        path_color: tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        import cv2 as _cv2

        overlay = image.copy()
        segments = data.get("segments") or []
        for seg in segments:
            p1 = (int(seg["x1"]), int(seg["y1"]))
            p2 = (int(seg["x2"]), int(seg["y2"]))
            _cv2.line(overlay, p1, p2, path_color, 7, _cv2.LINE_AA)
            _cv2.circle(overlay, p1, 5, path_color, -1, _cv2.LINE_AA)
            _cv2.circle(overlay, p2, 5, path_color, -1, _cv2.LINE_AA)

        for idx, turn in enumerate(data.get("turns") or [], start=1):
            pt = (int(turn["x"]), int(turn["y"]))
            _cv2.circle(overlay, pt, 13, (0, 180, 255), 3, _cv2.LINE_AA)
            _cv2.putText(
                overlay,
                f"T{idx}:{turn.get('new_dir', '')}",
                (pt[0] + 12, pt[1] - 12),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2,
                _cv2.LINE_AA,
            )

        port = data.get("port") or {}
        if port.get("x") is not None and port.get("y") is not None:
            start = (int(port["x"]), int(port["y"]))
            _cv2.circle(overlay, start, 14, (0, 180, 0), -1, _cv2.LINE_AA)
            _cv2.putText(
                overlay,
                f"{trace_id} start",
                (start[0] + 14, start[1] - 14),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 180, 0),
                2,
                _cv2.LINE_AA,
            )

        if data.get("terminal_x") is not None and data.get("terminal_y") is not None:
            end = (int(data["terminal_x"]), int(data["terminal_y"]))
            _cv2.circle(overlay, end, 16, (255, 0, 255), -1, _cv2.LINE_AA)
            _cv2.putText(
                overlay,
                f"{trace_id}:{data.get('terminal_type', '')}",
                (end[0] + 16, end[1] - 16),
                _cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 255),
                2,
                _cv2.LINE_AA,
            )

        banner = (
            f"{trace_id} only | {len(segments)} segs | "
            f"terminal={data.get('terminal_type')} | "
            f"length={data.get('trace_length_px', 0)} px"
        )
        _cv2.rectangle(
            overlay,
            (20, 20),
            (min(overlay.shape[1] - 20, 1700), 72),
            (255, 255, 255),
            -1,
        )
        _cv2.rectangle(
            overlay,
            (20, 20),
            (min(overlay.shape[1] - 20, 1700), 72),
            path_color,
            2,
        )
        _cv2.putText(
            overlay,
            banner,
            (35, 55),
            _cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            path_color,
            2,
            _cv2.LINE_AA,
        )
        return overlay

    def _write_stage5b_individual_trace_images(
        self,
        image: np.ndarray,
        all_results: dict[str, dict],
        branch_results: dict[str, dict],
    ) -> int:
        if cv2 is None and Image is None:  # pragma: no cover
            raise RuntimeError("No image backend available")

        trace_dir = self.out_dir / "stage5b_traced_path"
        if trace_dir.exists():
            shutil.rmtree(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        traces: list[tuple[str, dict[str, Any], tuple[int, int, int]]] = []
        traces.extend((trace_id, data, (0, 160, 0)) for trace_id, data in sorted(all_results.items()))
        traces.extend((trace_id, data, (0, 0, 255)) for trace_id, data in sorted(branch_results.items()))
        for trace_id, data, color in traces:
            if data.get("status") == "skipped":
                continue
            if not data.get("segments"):
                continue
            overlay = self._draw_single_stage5b_trace_path(
                image,
                trace_id,
                data,
                path_color=color,
            )
            filename = self._safe_stage5b_trace_image_name(trace_id) + ".png"
            path = trace_dir / filename
            out = normalize_for_save(overlay)
            if cv2 is not None:
                cv2.imwrite(str(path), out)
            elif Image is not None:  # pragma: no cover
                Image.fromarray(out).save(str(path))
            written += 1

        self._register_artifact("stage5b_traced_path")
        logger.info("saved %d individual Stage 5b trace images to %s", written, trace_dir)
        return written

    def _find_matching_stage5b_branch_candidate(
        self,
        candidate: dict[str, Any],
        candidates: list[dict[str, Any]],
        radius: int = 10,
    ) -> Optional[dict[str, Any]]:
        for existing in candidates:
            if existing.get("branch_direction") != candidate.get("branch_direction"):
                continue
            if (
                abs(int(existing["x"]) - int(candidate["x"])) <= radius
                and abs(int(existing["y"]) - int(candidate["y"])) <= radius
            ):
                return existing
        return None

    def _stage5b_branch_result_as_trace_source(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "port": result.get("port", {}),
            "terminal_type": result.get("terminal_type"),
            "terminal_x": result.get("terminal_x"),
            "terminal_y": result.get("terminal_y"),
            "terminal_obj_id": result.get("terminal_obj_id"),
            "segments": result.get("segments", []),
            "turns": result.get("turns", []),
            "hits": result.get("hits", []),
            "trace_length_px": result.get("trace_length_px", 0),
            "status": result.get("status"),
        }

    def _stage5b_branch_start_reached_by_prior_result(
        self,
        candidate: dict[str, Any],
        branch_results: dict[str, dict],
        tolerance: int = 12,
    ) -> Optional[tuple[str, dict]]:
        node_obj_id = str(candidate.get("node_obj_id") or "")
        branch_direction = str(candidate.get("branch_direction") or "").upper()
        opposite = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }
        if branch_direction not in opposite:
            return None

        cx = int(candidate.get("x", 0))
        cy = int(candidate.get("y", 0))
        for prior_id, prior in branch_results.items():
            if prior.get("status") != "traced":
                continue
            terminal_matches = False
            if node_obj_id and str(prior.get("terminal_obj_id") or "") == node_obj_id:
                terminal_matches = True
            elif (
                prior.get("terminal_x") is not None
                and prior.get("terminal_y") is not None
                and abs(int(prior["terminal_x"]) - cx) <= tolerance
                and abs(int(prior["terminal_y"]) - cy) <= tolerance
            ):
                terminal_matches = True
            if not terminal_matches:
                continue
            segments = prior.get("segments") or []
            if not segments:
                continue
            last_dir = str(segments[-1].get("direction") or "").upper()
            if last_dir == opposite[branch_direction]:
                return prior_id, prior
        return None

    def _find_stage5b_existing_path_intersection(
        self,
        candidate: dict[str, Any],
        segments: list[dict[str, Any]],
        existing_results: dict[str, dict],
        min_distance_from_start: int = 30,
        tolerance: int = 5,
        min_straight_continuation: int = 25,
    ) -> Optional[tuple[int, int, int, int, str]]:
        """Find the first point where a branch reaches an already traced path."""
        deltas = {
            "UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0),
        }
        source_trace_id = str(candidate.get("source_trace_id", ""))
        source_segment_index = candidate.get("source_segment_index")
        distance_from_start = 0
        for seg_index, seg in enumerate(segments):
            direction = str(seg.get("direction", "")).upper()
            dx, dy = deltas.get(direction, (0, 0))
            if dx == 0 and dy == 0:
                continue
            x1 = int(seg["x1"])
            y1 = int(seg["y1"])
            seg_len = int(seg.get("length_px") or max(abs(int(seg["x2"]) - x1), abs(int(seg["y2"]) - y1)))
            for step in range(0, seg_len + 1):
                path_distance = distance_from_start + step
                if path_distance <= min_distance_from_start:
                    continue
                px = x1 + dx * step
                py = y1 + dy * step
                for existing_id, existing in existing_results.items():
                    for existing_seg_index, existing_seg in enumerate(existing.get("segments", [])):
                        if (
                            str(existing_id) == source_trace_id
                            and existing_seg_index == source_segment_index
                            and path_distance <= min_distance_from_start * 2
                        ):
                            continue
                        if self._point_near_segment(px, py, existing_seg, tolerance=tolerance):
                            remaining_on_seg = max(0, seg_len - step)
                            if remaining_on_seg >= min_straight_continuation:
                                continue
                            return seg_index, px, py, path_distance, str(existing_id)
            distance_from_start += seg_len
        return None

    def _truncate_stage5b_branch_at_intersection(
        self,
        segments: list[dict[str, Any]],
        trace_length: int,
        intersection: tuple[int, int, int, int, str],
    ) -> tuple[list[dict[str, Any]], int, int, int, str]:
        seg_index, ix, iy, path_distance, existing_id = intersection
        truncated = [dict(seg) for seg in segments[:seg_index + 1]]
        if truncated:
            last = truncated[-1]
            direction = str(last.get("direction", "")).upper()
            if direction in ("UP", "DOWN"):
                ix = int(last["x1"])
            elif direction in ("LEFT", "RIGHT"):
                iy = int(last["y1"])
            last["x2"] = ix
            last["y2"] = iy
            last["length_px"] = max(abs(ix - int(last["x1"])), abs(iy - int(last["y1"])))
        return truncated, path_distance, ix, iy, existing_id

    def _promote_stage5b_paired_node_terminal_if_attached(
        self,
        result: dict[str, Any],
        paired_candidate: dict[str, Any],
        *,
        endpoint_tolerance: int = 12,
        axis_tolerance: int = 3,
        max_axis_extension: int = 20,
    ) -> bool:
        if not paired_candidate.get("node_obj_id"):
            return False
        segments = result.get("segments") or []
        if not segments:
            return False

        last = segments[-1]
        terminal_x = int(paired_candidate["x"])
        terminal_y = int(paired_candidate["y"])
        end_x = int(last["x2"])
        end_y = int(last["y2"])
        distance = max(abs(end_x - terminal_x), abs(end_y - terminal_y))

        direction = str(last.get("direction") or "").upper()
        same_axis = (
            direction in ("LEFT", "RIGHT")
            and abs(end_y - terminal_y) <= axis_tolerance
        ) or (
            direction in ("UP", "DOWN")
            and abs(end_x - terminal_x) <= axis_tolerance
        )
        axis_extension = (
            abs(end_x - terminal_x)
            if direction in ("LEFT", "RIGHT")
            else abs(end_y - terminal_y)
        )
        attached = distance <= endpoint_tolerance or (
            same_axis and axis_extension <= max_axis_extension
        )
        if not attached:
            return False

        result["terminal_type"] = "tee_junction"
        result["terminal_obj_id"] = str(paired_candidate["node_obj_id"])
        result["terminal_x"] = terminal_x
        result["terminal_y"] = terminal_y

        if not same_axis:
            return True

        old_len = int(last.get("length_px", 0))
        if direction in ("LEFT", "RIGHT"):
            last["x2"] = terminal_x
            last["y2"] = int(last["y1"])
        else:
            last["x2"] = int(last["x1"])
            last["y2"] = terminal_y
        new_len = max(
            abs(int(last["x2"]) - int(last["x1"])),
            abs(int(last["y2"]) - int(last["y1"])),
        )
        last["length_px"] = new_len
        result["trace_length_px"] = int(result.get("trace_length_px", 0)) + new_len - old_len
        return True

    def _stage5b_branch_connection_attached_to_candidate(
        self,
        result: dict[str, Any],
        candidate: dict[str, Any],
        tolerance: int = 12,
    ) -> bool:
        if result.get("terminal_type") != "branch_connection":
            return False
        if str(result.get("terminal_obj_id") or "") != str(candidate.get("id") or ""):
            return False
        terminal_x = result.get("terminal_x")
        terminal_y = result.get("terminal_y")
        if terminal_x is None or terminal_y is None:
            return False
        return (
            abs(int(terminal_x) - int(candidate.get("x", 0))) <= tolerance
            and abs(int(terminal_y) - int(candidate.get("y", 0))) <= tolerance
        )

    def _trace_stage5b_branch_candidates(
        self,
        pipe_mask: np.ndarray,
        image: np.ndarray,
        candidates: list[dict[str, Any]],
        page_connections: list[dict[str, Any]],
        instrument_tags: list[dict[str, Any]],
        equipment: list[dict[str, Any]],
        inline_symbols: list[dict[str, Any]],
        node_symbols: list[dict[str, Any]],
        visited: np.ndarray,
        terminal_candidates: Optional[list[dict[str, Any]]] = None,
        existing_trace_sources: Optional[dict[str, dict]] = None,
    ) -> dict[str, dict]:
        from garnet.path_tracer.cv_pipe_tracer import (
            CVPipeTracer,
            TURN_LEFT,
            TURN_RIGHT,
            _has_connected_side_pipe,
        )

        deltas = {
            "UP": (0, -1), "DOWN": (0, 1),
            "LEFT": (-1, 0), "RIGHT": (1, 0),
        }
        h_mask, w_mask = pipe_mask.shape
        branch_results: dict[str, dict] = {}
        all_terminal_candidates = terminal_candidates or candidates
        candidate_by_id = {
            str(candidate["id"]): candidate
            for candidate in all_terminal_candidates
        }
        used_branch_pairs: dict[str, str] = {}
        pending_branch_pairs: dict[str, str] = {}
        for candidate in candidates:
            branch_id = candidate["id"]
            if branch_id in used_branch_pairs and branch_id not in pending_branch_pairs:
                candidate["status"] = "done_used_by_branch"
                candidate["reason"] = "paired_branch_already_traced"
                candidate["paired_branch_id"] = used_branch_pairs[branch_id]
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": "done_used_by_branch",
                    "paired_branch_id": used_branch_pairs[branch_id],
                    "candidate": candidate,
                }
                continue
            if candidate.get("status") != "queued":
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": candidate.get("status"),
                    "candidate": candidate,
                }
                continue

            direction = str(candidate["branch_direction"]).upper()
            dx, dy = deltas[direction]
            start_x = int(candidate["x"]) + dx * 3
            start_y = int(candidate["y"]) + dy * 3
            found_start = False
            for offset in range(3, 31):
                tx = int(candidate["x"]) + dx * offset
                ty = int(candidate["y"]) + dy * offset
                if 0 <= tx < w_mask and 0 <= ty < h_mask and pipe_mask[ty, tx] > 0:
                    start_x, start_y = tx, ty
                    found_start = True
                    break
            if not found_start:
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": "no_pipe_at_branch_start",
                    "candidate": candidate,
                }
                continue
            prior_reach = self._stage5b_branch_start_reached_by_prior_result(
                candidate,
                branch_results,
            )
            if prior_reach is not None:
                prior_id, _prior = prior_reach
                candidate["status"] = "done_used_by_branch"
                candidate["reason"] = "source_node_reached_by_prior_branch"
                candidate["paired_branch_id"] = prior_id
                branch_results[branch_id] = {
                    "status": "skipped",
                    "skip_reason": "source_node_reached_by_prior_branch",
                    "paired_branch_id": prior_id,
                    "candidate": candidate,
                }
                continue

            branch_terminals = []
            for other in all_terminal_candidates:
                other_id = other.get("id")
                if other_id == branch_id:
                    continue
                ox = int(other["x"])
                oy = int(other["y"])
                branch_terminals.append({
                    "id": other_id,
                    "class_name": "branch_candidate",
                    "bbox": {
                        "x_min": ox - 8,
                        "y_min": oy - 8,
                        "x_max": ox + 8,
                        "y_max": oy + 8,
                    },
                })
            source_node_id = str(candidate.get("node_obj_id", ""))
            junction_markers = [
                node for node in node_symbols
                if str(node.get("id", "")) != source_node_id
            ]
            tracer = CVPipeTracer(
                pipe_mask=pipe_mask,
                image=image,
                page_connections=page_connections + branch_terminals,
                instrument_tags=instrument_tags,
                equipment_objects=equipment,
                junction_markers=junction_markers,
                visited_mask=visited,
            )
            tracer.set_inline_symbols(inline_symbols)
            result = tracer.trace(
                start_x,
                start_y,
                direction,
                source_obj_id=branch_id,
            )
            terminal_inside_exact_bbox = False
            terminal_margin_gap: Optional[int] = None
            if result.terminal_type == "instrument_tag" and result.terminal_obj_id:
                terminal_obj = next(
                    (tag for tag in instrument_tags if tag.get("id") == result.terminal_obj_id),
                    None,
                )
                terminal_bbox = terminal_obj.get("bbox") if terminal_obj else None
                if terminal_bbox:
                    terminal_inside_exact_bbox = (
                        terminal_bbox["x_min"] <= result.terminal_x <= terminal_bbox["x_max"]
                        and terminal_bbox["y_min"] <= result.terminal_y <= terminal_bbox["y_max"]
                    )
                    if result.segments:
                        last_dir = result.segments[-1].direction
                        if last_dir == "UP" and result.terminal_y > terminal_bbox["y_max"]:
                            terminal_margin_gap = result.terminal_y - int(terminal_bbox["y_max"])
                        elif last_dir == "DOWN" and result.terminal_y < terminal_bbox["y_min"]:
                            terminal_margin_gap = int(terminal_bbox["y_min"]) - result.terminal_y
                        elif last_dir == "LEFT" and result.terminal_x > terminal_bbox["x_max"]:
                            terminal_margin_gap = result.terminal_x - int(terminal_bbox["x_max"])
                        elif last_dir == "RIGHT" and result.terminal_x < terminal_bbox["x_min"]:
                            terminal_margin_gap = int(terminal_bbox["x_min"]) - result.terminal_x

            recover_side_turn = (
                result.terminal_type == "dead_end"
                or (
                    result.terminal_type == "instrument_tag"
                    and not terminal_inside_exact_bbox
                    and terminal_margin_gap is not None
                    and terminal_margin_gap <= 10
                )
            )
            extra_turns = []
            extra_result = None
            recovery_point = None
            if recover_side_turn and result.segments:
                last_seg = result.segments[-1]
                last_dir = last_seg.direction
                sdx, sdy = deltas[last_dir]
                for backtrack in range(0, 21):
                    bx = result.terminal_x - sdx * backtrack
                    by = result.terminal_y - sdy * backtrack
                    for turn_dir in (TURN_LEFT[last_dir], TURN_RIGHT[last_dir]):
                        if not _has_connected_side_pipe(pipe_mask, bx, by, turn_dir, min_run=25):
                            continue
                        tdx, tdy = deltas[turn_dir]
                        cont_x = bx + tdx * 3
                        cont_y = by + tdy * 3
                        found_cont = False
                        for offset in range(3, 31):
                            tx = bx + tdx * offset
                            ty = by + tdy * offset
                            if 0 <= tx < w_mask and 0 <= ty < h_mask and pipe_mask[ty, tx] > 0:
                                cont_x, cont_y = tx, ty
                                found_cont = True
                                break
                        if not found_cont:
                            continue
                        recovery_tracer = CVPipeTracer(
                            pipe_mask=pipe_mask,
                            image=image,
                            page_connections=page_connections + branch_terminals,
                            instrument_tags=instrument_tags,
                            equipment_objects=equipment,
                            junction_markers=junction_markers,
                            visited_mask=visited,
                        )
                        recovery_tracer.set_inline_symbols(inline_symbols)
                        extra_result = recovery_tracer.trace(
                            cont_x,
                            cont_y,
                            turn_dir,
                            source_obj_id=branch_id,
                        )
                        recovery_point = (bx, by, turn_dir)
                        extra_turns.append({"x": bx, "y": by, "new_dir": turn_dir})
                        break
                    if extra_result is not None:
                        break

            segments = [
                {
                    "x1": s.x1, "y1": s.y1,
                    "x2": s.x2, "y2": s.y2,
                    "direction": s.direction,
                    "length_px": s.length_px,
                }
                for s in result.segments
            ]
            trace_length = result.trace_length_px
            terminal_type = result.terminal_type
            terminal_x = result.terminal_x
            terminal_y = result.terminal_y
            terminal_obj_id = result.terminal_obj_id
            turns = [
                {"x": tx, "y": ty, "new_dir": td}
                for tx, ty, td in result.turns
            ]
            hits = [
                {"class": h.class_name, "x": h.x, "y": h.y}
                for h in result.hits
            ]
            if extra_result is not None and recovery_point is not None and segments:
                bx, by, turn_dir = recovery_point
                last = segments[-1]
                old_len = int(last["length_px"])
                new_len = max(abs(bx - int(last["x1"])), abs(by - int(last["y1"])))
                last["x2"] = bx
                last["y2"] = by
                last["length_px"] = new_len
                trace_length += new_len - old_len
                turns.extend(extra_turns)
                turns.extend(
                    {"x": tx, "y": ty, "new_dir": td}
                    for tx, ty, td in extra_result.turns
                )
                hits.extend(
                    {"class": h.class_name, "x": h.x, "y": h.y}
                    for h in extra_result.hits
                )
                extra_segments = [
                    {
                        "x1": s.x1, "y1": s.y1,
                        "x2": s.x2, "y2": s.y2,
                        "direction": s.direction,
                        "length_px": s.length_px,
                    }
                    for s in extra_result.segments
                ]
                segments.extend(extra_segments)
                trace_length += extra_result.trace_length_px
                terminal_type = extra_result.terminal_type
                terminal_x = extra_result.terminal_x
                terminal_y = extra_result.terminal_y
                terminal_obj_id = extra_result.terminal_obj_id

            if not str(terminal_obj_id or "").startswith("branch_"):
                existing_paths = dict(existing_trace_sources or {})
                existing_paths.update(branch_results)
                intersection = self._find_stage5b_existing_path_intersection(
                    candidate,
                    segments,
                    existing_paths,
                )
                if intersection is not None:
                    segments, trace_length, terminal_x, terminal_y, hit_trace_id = (
                        self._truncate_stage5b_branch_at_intersection(
                            segments,
                            trace_length,
                            intersection,
                        )
                    )
                    terminal_type = "tee_junction"
                    terminal_obj_id = hit_trace_id

            if (terminal_obj_id or "").startswith("branch_"):
                terminal_type = "branch_connection"
                paired_branch_id = str(terminal_obj_id)
                paired_candidate = candidate_by_id.get(paired_branch_id)
                terminal_payload = {
                    "terminal_type": terminal_type,
                    "terminal_x": terminal_x,
                    "terminal_y": terminal_y,
                    "terminal_obj_id": terminal_obj_id,
                    "segments": segments,
                    "trace_length_px": trace_length,
                }
                attached_to_paired_candidate = (
                    paired_candidate is not None
                    and self._stage5b_branch_connection_attached_to_candidate(
                        terminal_payload,
                        paired_candidate,
                    )
                )
                if attached_to_paired_candidate:
                    used_branch_pairs[branch_id] = paired_branch_id
                    used_branch_pairs[paired_branch_id] = branch_id
                    candidate["status"] = "done_branch_connection"
                    candidate["reason"] = "traced_to_branch_candidate"
                    candidate["paired_branch_id"] = paired_branch_id
                if paired_candidate is not None:
                    if attached_to_paired_candidate:
                        paired_candidate["paired_branch_id"] = branch_id
                        if paired_branch_id in branch_results:
                            paired_candidate["status"] = "done_used_by_branch"
                            paired_candidate["reason"] = "reverse_branch_trace_preferred"
                        else:
                            paired_candidate["status"] = "pending_branch_connection"
                            paired_candidate["reason"] = "paired_branch_candidate_pending_reverse_check"
                            pending_branch_pairs[paired_branch_id] = branch_id
                    elif candidate.get("status") == "queued":
                        candidate["reason"] = "branch_connection_not_attached_to_candidate"
                    if attached_to_paired_candidate:
                        if self._promote_stage5b_paired_node_terminal_if_attached(
                            terminal_payload,
                            paired_candidate,
                        ):
                            terminal_type = terminal_payload["terminal_type"]
                            terminal_obj_id = terminal_payload["terminal_obj_id"]
                            terminal_x = terminal_payload["terminal_x"]
                            terminal_y = terminal_payload["terminal_y"]
                            trace_length = terminal_payload["trace_length_px"]
            branch_results[branch_id] = {
                "status": "traced",
                "candidate": candidate,
                "port": {"x": start_x, "y": start_y, "direction": direction},
                "terminal_type": terminal_type,
                "terminal_x": terminal_x,
                "terminal_y": terminal_y,
                "terminal_obj_id": terminal_obj_id,
                "segments": segments,
                "turns": turns,
                "hits": hits,
                "trace_length_px": trace_length,
            }
            self._extend_stage5b_result_to_terminal(branch_results[branch_id])
            branch_results[branch_id]["turns"] = self._rebuild_stage5b_turns_from_segments(
                branch_results[branch_id].get("segments") or []
            )
            if branch_id in pending_branch_pairs and terminal_type == "branch_connection":
                previous_branch_id = pending_branch_pairs.pop(branch_id)
                previous_result = branch_results.get(previous_branch_id)
                if previous_result is not None:
                    previous_candidate = candidate_by_id.get(previous_branch_id)
                    if previous_candidate is not None:
                        previous_candidate["status"] = "done_used_by_branch"
                        previous_candidate["reason"] = "reverse_branch_trace_preferred"
                        previous_candidate["paired_branch_id"] = branch_id
                    previous_result["status"] = "skipped"
                    previous_result["skip_reason"] = "done_used_by_branch"
                    previous_result["paired_branch_id"] = branch_id
                    previous_result["preferred_branch_id"] = branch_id
                candidate["status"] = "done_branch_connection"
                candidate["reason"] = "reverse_branch_trace_preferred"
                candidate["paired_branch_id"] = previous_branch_id
            if terminal_type == "branch_connection" and terminal_obj_id in branch_results:
                previous_result = branch_results.get(str(terminal_obj_id))
                paired_candidate = candidate_by_id.get(str(terminal_obj_id))
                attached_to_paired_candidate = (
                    paired_candidate is not None
                    and self._stage5b_branch_connection_attached_to_candidate(
                        branch_results[branch_id],
                        paired_candidate,
                    )
                )
                if previous_result is not None:
                    if attached_to_paired_candidate:
                        previous_result["status"] = "skipped"
                        previous_result["skip_reason"] = "done_used_by_branch"
                        previous_result["paired_branch_id"] = branch_id
                        previous_result["preferred_branch_id"] = branch_id
                if paired_candidate is not None and attached_to_paired_candidate:
                    paired_candidate["status"] = "done_used_by_branch"
                    paired_candidate["reason"] = "reverse_branch_trace_preferred"
                    paired_candidate["paired_branch_id"] = branch_id

        for pending_branch_id, traced_branch_id in pending_branch_pairs.items():
            pending_candidate = candidate_by_id.get(pending_branch_id)
            if pending_candidate is not None and pending_candidate.get("status") == "pending_branch_connection":
                pending_candidate["status"] = "done_branch_connection"
                pending_candidate["reason"] = "paired_branch_candidate_traced"
                pending_candidate["paired_branch_id"] = traced_branch_id
            pending_result = branch_results.get(pending_branch_id)
            if pending_result is not None and pending_result.get("status") == "skipped":
                pending_result["skip_reason"] = "done_used_by_branch"
                pending_result["paired_branch_id"] = traced_branch_id
                pending_result["preferred_branch_id"] = traced_branch_id
        return branch_results

    def _trace_stage5b_branches_iterative(
        self,
        pipe_mask: np.ndarray,
        image: np.ndarray,
        base_results: dict[str, dict],
        page_connections: list[dict[str, Any]],
        instrument_tags: list[dict[str, Any]],
        equipment: list[dict[str, Any]],
        inline_symbols: list[dict[str, Any]],
        node_symbols: list[dict[str, Any]],
        visited: np.ndarray,
        max_iterations: int = 5,
        candidate_overlay_base: Optional[np.ndarray] = None,
    ) -> tuple[list[dict[str, Any]], dict[str, dict], list[dict[str, Any]]]:
        trace_sources = dict(base_results)
        all_candidates: list[dict[str, Any]] = []
        all_branch_results: dict[str, dict] = {}
        iteration_summaries: list[dict[str, Any]] = []

        for iteration in range(1, max_iterations + 1):
            detected_candidates = self._detect_stage5b_branch_candidates(
                pipe_mask=pipe_mask,
                all_results=trace_sources,
                inline_symbols=inline_symbols,
                node_symbols=node_symbols,
                equipment_objects=equipment,
            )
            new_candidates: list[dict[str, Any]] = []
            for detected in detected_candidates:
                existing = self._find_matching_stage5b_branch_candidate(
                    detected,
                    all_candidates,
                )
                if existing is not None:
                    if (
                        str(existing.get("status", "")).startswith("rejected_")
                        and detected.get("status") == "queued"
                    ):
                        existing.update({
                            key: value
                            for key, value in detected.items()
                            if key not in {"id", "member_count"}
                        })
                        existing["iteration"] = iteration
                    continue

                candidate = {
                    key: value
                    for key, value in detected.items()
                    if key != "id"
                }
                candidate["id"] = f"branch_{len(all_candidates) + 1:06d}"
                candidate["iteration"] = iteration
                all_candidates.append(candidate)
                new_candidates.append(candidate)

            queued_candidates = [
                candidate
                for candidate in new_candidates
                if candidate.get("status") == "queued"
            ]
            if not new_candidates:
                iteration_summaries.append({
                    "iteration": iteration,
                    "new_candidates": 0,
                    "queued": 0,
                    "traced": 0,
                    "new_trace_sources": 0,
                    "stop_reason": "no_new_candidates",
                })
                if self.cfg.debug_artifacts and candidate_overlay_base is not None:
                    candidate_overlay = self._draw_stage5b_branch_candidate_overlay(
                        candidate_overlay_base,
                        new_candidates,
                        base_results=base_results,
                        branch_results=all_branch_results,
                    )
                    self._save_img("stage5b_branch_candidates_overlay", candidate_overlay)
                    self._save_img(f"stage5b_branch_candidates_iter_{iteration:02d}_overlay", candidate_overlay)
                break
            if not queued_candidates:
                iteration_summaries.append({
                    "iteration": iteration,
                    "new_candidates": len(new_candidates),
                    "queued": 0,
                    "traced": 0,
                    "new_trace_sources": 0,
                    "stop_reason": "no_new_queued_candidates",
                })
                if self.cfg.debug_artifacts and candidate_overlay_base is not None:
                    candidate_overlay = self._draw_stage5b_branch_candidate_overlay(
                        candidate_overlay_base,
                        new_candidates,
                        base_results=base_results,
                        branch_results=all_branch_results,
                    )
                    self._save_img("stage5b_branch_candidates_overlay", candidate_overlay)
                    self._save_img(f"stage5b_branch_candidates_iter_{iteration:02d}_overlay", candidate_overlay)
                break

            prior_branch_results = dict(all_branch_results)
            branch_results = self._trace_stage5b_branch_candidates(
                pipe_mask=pipe_mask,
                image=image,
                candidates=queued_candidates,
                page_connections=page_connections,
                instrument_tags=instrument_tags,
                equipment=equipment,
                inline_symbols=inline_symbols,
                node_symbols=node_symbols,
                visited=visited,
                terminal_candidates=all_candidates,
                existing_trace_sources=trace_sources,
            )
            all_branch_results.update(branch_results)

            new_trace_sources = 0
            traced_count = 0
            for branch_id, result in branch_results.items():
                if result.get("status") != "traced" or not result.get("segments"):
                    continue
                traced_count += 1
                trace_sources[branch_id] = self._stage5b_branch_result_as_trace_source(result)
                new_trace_sources += 1

            iteration_summaries.append({
                "iteration": iteration,
                "new_candidates": len(new_candidates),
                "queued": len(queued_candidates),
                "traced": traced_count,
                "new_trace_sources": new_trace_sources,
                "stop_reason": "max_iterations" if iteration == max_iterations else None,
            })
            if self.cfg.debug_artifacts and candidate_overlay_base is not None:
                candidate_overlay = self._draw_stage5b_branch_candidate_overlay(
                    candidate_overlay_base,
                    new_candidates,
                    base_results=base_results,
                    branch_results=prior_branch_results,
                )
                self._save_img("stage5b_branch_candidates_overlay", candidate_overlay)
                self._save_img(f"stage5b_branch_candidates_iter_{iteration:02d}_overlay", candidate_overlay)
            if new_trace_sources == 0:
                iteration_summaries[-1]["stop_reason"] = "no_new_trace_sources"
                break

        return all_candidates, all_branch_results, iteration_summaries

    def _align_stage5b_branch_node_terminals(
        self,
        branch_results: dict[str, dict],
        node_symbols: list[dict[str, Any]],
    ) -> None:
        """Move saved tee endpoints to the node bbox center after discovery.

        This keeps the iterative branch search using the raw traced geometry,
        while the final JSON/overlay lands node terminals on the visible tee dot.
        """
        node_by_id = {
            str(node.get("id", "")): node
            for node in node_symbols
            if node.get("bbox")
        }
        for result in branch_results.values():
            if result.get("status") != "traced":
                continue
            if result.get("terminal_type") != "tee_junction":
                continue
            terminal_obj_id = str(result.get("terminal_obj_id") or "")
            node = node_by_id.get(terminal_obj_id)
            if not node:
                continue
            bbox = node["bbox"]
            terminal_x = (int(bbox["x_min"]) + int(bbox["x_max"])) // 2
            terminal_y = (int(bbox["y_min"]) + int(bbox["y_max"])) // 2
            result["terminal_x"] = terminal_x
            result["terminal_y"] = terminal_y
            segments = result.get("segments") or []
            if not segments:
                continue
            last = segments[-1]
            same_axis = (
                last["direction"] in ("LEFT", "RIGHT")
                and abs(int(last["y2"]) - terminal_y) <= 3
            ) or (
                last["direction"] in ("UP", "DOWN")
                and abs(int(last["x2"]) - terminal_x) <= 3
            )
            if not same_axis:
                continue
            old_len = int(last.get("length_px", 0))
            last["x2"] = terminal_x
            last["y2"] = terminal_y
            new_len = max(
                abs(int(last["x2"]) - int(last["x1"])),
                abs(int(last["y2"]) - int(last["y1"])),
            )
            last["length_px"] = new_len
            result["trace_length_px"] = int(result.get("trace_length_px", 0)) + new_len - old_len

    def _extend_stage5b_result_to_terminal(self, result: dict[str, Any]) -> None:
        segments = result.get("segments") or []
        if not segments:
            return
        terminal_x = result.get("terminal_x")
        terminal_y = result.get("terminal_y")
        if terminal_x is None or terminal_y is None:
            return
        last = segments[-1]
        same_axis = (
            last["direction"] in ("LEFT", "RIGHT")
            and abs(int(last["y2"]) - int(terminal_y)) <= 3
        ) or (
            last["direction"] in ("UP", "DOWN")
            and abs(int(last["x2"]) - int(terminal_x)) <= 3
        )
        if not same_axis:
            return
        if int(last["x2"]) == int(terminal_x) and int(last["y2"]) == int(terminal_y):
            return
        old_len = int(last.get("length_px", 0))
        if last["direction"] in ("LEFT", "RIGHT"):
            last["x2"] = int(terminal_x)
            last["y2"] = int(last["y1"])
        else:
            last["x2"] = int(last["x1"])
            last["y2"] = int(terminal_y)
        new_len = max(
            abs(int(last["x2"]) - int(last["x1"])),
            abs(int(last["y2"]) - int(last["y1"])),
        )
        last["length_px"] = new_len
        result["trace_length_px"] = int(result.get("trace_length_px", 0)) + new_len - old_len

    def _rebuild_stage5b_turns_from_segments(
        self,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, int | str]]:
        turns: list[dict[str, int | str]] = []
        for prev_seg, next_seg in zip(segments, segments[1:]):
            prev_dir = str(prev_seg.get("direction") or "")
            next_dir = str(next_seg.get("direction") or "")
            if not next_dir or next_dir == prev_dir:
                continue
            turns.append(
                {
                    "x": int(prev_seg.get("x2", 0)),
                    "y": int(prev_seg.get("y2", 0)),
                    "new_dir": next_dir,
                }
            )
        return turns

    def _find_existing_stage5b_terminal_trace(
        self,
        all_results: dict[str, dict],
        obj_id: str,
        point: Optional[tuple[int, int]] = None,
        point_tolerance: int = 12,
    ) -> Optional[tuple[str, dict]]:
        is_equipment_source = str(obj_id).startswith("equip_")
        for existing_trace_id, existing in all_results.items():
            terminal_matches_obj = (
                not is_equipment_source
                and str(existing.get("terminal_obj_id", "")) == str(obj_id)
            )
            terminal_matches_point = False
            if point is not None:
                tx = existing.get("terminal_x")
                ty = existing.get("terminal_y")
                terminal_matches_point = (
                    tx is not None
                    and ty is not None
                    and abs(int(tx) - int(point[0])) <= point_tolerance
                    and abs(int(ty) - int(point[1])) <= point_tolerance
                )
            if not (terminal_matches_obj or terminal_matches_point):
                continue
            if existing.get("status") not in {None, "ok"}:
                continue
            if not existing.get("segments"):
                continue
            return existing_trace_id, existing
        return None

    def _align_stage5b_result_to_near_node(
        self,
        result: dict[str, Any],
        node_symbols: list[dict[str, Any]],
        max_distance: int = 20,
    ) -> None:
        if result.get("terminal_type") != "tee_junction":
            return
        if result.get("terminal_obj_id"):
            return
        tx = result.get("terminal_x")
        ty = result.get("terminal_y")
        if tx is None or ty is None:
            return
        nearest = None
        nearest_dist = max_distance + 1
        for node in node_symbols:
            bbox = node.get("bbox")
            if not bbox:
                continue
            cx = (int(bbox["x_min"]) + int(bbox["x_max"])) // 2
            cy = (int(bbox["y_min"]) + int(bbox["y_max"])) // 2
            dist = max(abs(int(tx) - cx), abs(int(ty) - cy))
            if dist < nearest_dist:
                nearest = (str(node.get("id", "")), cx, cy)
                nearest_dist = dist
        if nearest is None:
            return
        node_id, cx, cy = nearest
        result["terminal_obj_id"] = node_id
        result["terminal_x"] = cx
        result["terminal_y"] = cy
        segments = result.get("segments") or []
        if not segments:
            return
        last = segments[-1]
        old_len = int(last.get("length_px", 0))
        last["x2"] = cx
        last["y2"] = cy
        new_len = max(
            abs(int(last["x2"]) - int(last["x1"])),
            abs(int(last["y2"]) - int(last["y1"])),
        )
        last["length_px"] = new_len
        result["trace_length_px"] = int(result.get("trace_length_px", 0)) + new_len - old_len

    def _reverse_stage5b_trace_result(
        self,
        *,
        obj_id: str,
        port_index: int,
        port: tuple[int, int, str],
        existing_trace_id: str,
        existing: dict[str, Any],
    ) -> dict[str, Any]:
        opposite = {
            "UP": "DOWN",
            "DOWN": "UP",
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }
        segments = []
        for seg in reversed(existing.get("segments") or []):
            old_dir = str(seg.get("direction", "")).upper()
            segments.append({
                "x1": int(seg.get("x2", 0)),
                "y1": int(seg.get("y2", 0)),
                "x2": int(seg.get("x1", 0)),
                "y2": int(seg.get("y1", 0)),
                "direction": opposite.get(old_dir, old_dir),
                "length_px": int(seg.get("length_px", 0)),
            })

        turns = [
            {
                "x": int(prev_seg["x2"]),
                "y": int(prev_seg["y2"]),
                "new_dir": str(next_seg.get("direction", "")),
            }
            for prev_seg, next_seg in zip(segments, segments[1:])
        ]

        source_port = existing.get("port") or {}
        terminal_x = int(source_port.get("x", segments[-1]["x2"] if segments else port[0]))
        terminal_y = int(source_port.get("y", segments[-1]["y2"] if segments else port[1]))
        terminal_obj_id = str(existing.get("source_obj_id") or "").split(":")[0]
        terminal_type = "equipment" if terminal_obj_id.startswith("equip_") else "page_connection"

        return {
            "source_obj_id": obj_id,
            "port_index": port_index,
            "port": {"x": int(port[0]), "y": int(port[1]), "direction": str(port[2])},
            "terminal_type": terminal_type,
            "terminal_x": terminal_x,
            "terminal_y": terminal_y,
            "terminal_obj_id": terminal_obj_id,
            "segments": segments,
            "turns": turns,
            "hits": list(existing.get("hits") or []),
            "trace_length_px": int(existing.get("trace_length_px") or 0),
            "status": "reused_existing_trace",
            "reused_trace_id": existing_trace_id,
        }

    def _skip_stage5b_existing_trace_result(
        self,
        *,
        obj_id: str,
        port_index: int,
        port: tuple[int, int, str],
        existing_trace_id: str,
        existing: dict[str, Any],
        reason: str = "source_reached_by_existing_trace",
    ) -> dict[str, Any]:
        return {
            "source_obj_id": obj_id,
            "port_index": port_index,
            "port": {"x": int(port[0]), "y": int(port[1]), "direction": str(port[2])},
            "terminal_type": existing.get("terminal_type"),
            "terminal_x": existing.get("terminal_x"),
            "terminal_y": existing.get("terminal_y"),
            "terminal_obj_id": existing.get("terminal_obj_id"),
            "segments": [],
            "turns": [],
            "hits": [],
            "trace_length_px": 0,
            "status": "skipped_existing_trace",
            "skip_reason": reason,
            "reused_trace_id": existing_trace_id,
        }

    def _find_existing_stage5b_source_port_trace(
        self,
        all_results: dict[str, dict],
        obj_id: str,
        point: tuple[int, int],
        direction: str,
        point_tolerance: int = 35,
    ) -> Optional[tuple[str, dict]]:
        for existing_trace_id, existing in all_results.items():
            if str(existing.get("source_obj_id", "")) != str(obj_id):
                continue
            if str((existing.get("port") or {}).get("direction", "")).upper() != direction.upper():
                continue
            if existing.get("status") != "skipped_existing_trace":
                continue
            port = existing.get("port") or {}
            px = port.get("x")
            py = port.get("y")
            if px is None or py is None:
                continue
            if abs(int(px) - int(point[0])) <= point_tolerance and abs(int(py) - int(point[1])) <= point_tolerance:
                return existing_trace_id, existing
        return None

    def _detect_port_cv(
        self, image: np.ndarray, bbox: dict[str, int], track_len: int = 60
    ) -> Optional[tuple[int, int, str]]:
        """Detect pipe port using edge scanning + outward tracking (no VLM).

        Scans all 4 edges of the bbox for non-background pixel clusters,
        then tracks outward perpendicular to each edge. The edge+cluster
        with the longest contiguous non-background run wins.

        Returns (x, y, direction) or None if no pipe found.
        """
        import cv2 as _cv2

        x1, y1 = bbox["x_min"], bbox["y_min"]
        x2, y2 = bbox["x_max"], bbox["y_max"]
        h_img, w_img = image.shape[:2]

        # Convert to grayscale for pixel checks
        if len(image.shape) == 3:
            gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Background threshold (median of a strip outside the bbox)
        bg_strip_y = max(0, y1 - 10)
        bg_strip = gray[bg_strip_y:y1, x1:x2] if y1 > 10 else gray[0:10, x1:x2]
        bg_val = float(np.median(bg_strip)) if bg_strip.size > 0 else 200

        # Define edges: (name, start_xy, end_xy, step_direction, track_dx, track_dy)
        edges = [
            ("TOP",    (x1, y1), (x2, y1), (1, 0),   (0, -1)),
            ("BOTTOM", (x1, y2), (x2, y2), (1, 0),   (0,  1)),
            ("LEFT",   (x1, y1), (x1, y2), (0, 1),   (-1, 0)),
            ("RIGHT",  (x2, y1), (x2, y2), (0, 1),   (1,  0)),
        ]

        best_score = 0
        best_result = None

        for edge_name, start, end, step_dir, track_dir in edges:
            sx, sy = start
            ex, ey = end
            length = max(abs(ex - sx), abs(ey - sy))

            for i in range(length):
                px = sx + i * step_dir[0]
                py = sy + i * step_dir[1]
                if px < 0 or px >= w_img or py < 0 or py >= h_img:
                    continue

                # Check if this pixel is pipe (dark)
                pixel_val = gray[py, px]
                if pixel_val > bg_val * 0.7:  # near background → skip
                    continue

                # Found a dark pixel — track outward
                track_count = 0
                tx, ty = px, py
                for _ in range(track_len):
                    tx += track_dir[0]
                    ty += track_dir[1]
                    if tx < 0 or tx >= w_img or ty < 0 or ty >= h_img:
                        break
                    if gray[ty, tx] < bg_val * 0.6:
                        track_count += 1
                    else:
                        break  # hit background

                if track_count > best_score:
                    best_score = track_count
                    direction = {"TOP": "UP", "BOTTOM": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}[edge_name]
                    best_result = (px, py, direction)

                # Skip the rest of this cluster
                while i + 1 < length:
                    nx = sx + (i + 1) * step_dir[0]
                    ny = sy + (i + 1) * step_dir[1]
                    if nx < 0 or nx >= w_img or ny < 0 or ny >= h_img:
                        i += 1
                        continue
                    if gray[ny, nx] > bg_val * 0.7:
                        break
                    i += 1

        if best_result and best_score >= 3:
            px, py, direction = best_result
            # Snap to bbox edge
            if direction == "UP":
                py = y1
            elif direction == "DOWN":
                py = y2
            elif direction == "LEFT":
                px = x1
            elif direction == "RIGHT":
                px = x2
            return (int(px), int(py), direction)

        return None

    def _detect_equipment_ports_cv(
        self, image: np.ndarray, bbox: dict[str, int], track_len: int = 60
    ) -> list[tuple[int, int, str]]:
        """Detect ALL pipe attachment ports on an equipment bbox.

        Unlike page connections (single pipe), equipment can have multiple
        nozzles — inlet, outlet, drain, vent, etc.  Returns all valid ports
        found on any edge of the bbox.

        Returns list of (x, y, direction) — empty if no pipes found.
        """
        import cv2 as _cv2

        x1, y1 = bbox["x_min"], bbox["y_min"]
        x2, y2 = bbox["x_max"], bbox["y_max"]
        h_img, w_img = image.shape[:2]

        if len(image.shape) == 3:
            gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        bg_strip_y = max(0, y1 - 10)
        bg_strip = gray[bg_strip_y:y1, x1:x2] if y1 > 10 else gray[0:10, x1:x2]
        bg_val = float(np.median(bg_strip)) if bg_strip.size > 0 else 200

        edges = [
            ("TOP",    (x1, y1), (x2, y1), (1, 0),   (0, -1)),
            ("BOTTOM", (x1, y2), (x2, y2), (1, 0),   (0,  1)),
            ("LEFT",   (x1, y1), (x1, y2), (0, 1),   (-1, 0)),
            ("RIGHT",  (x2, y1), (x2, y2), (0, 1),   (1,  0)),
        ]
        direction_map = {"TOP": "UP", "BOTTOM": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}

        all_ports: list[tuple[int, int, str]] = []

        for edge_name, start, end, step_dir, track_dir in edges:
            sx, sy = start
            ex, ey = end
            length = max(abs(ex - sx), abs(ey - sy))

            i = 0
            while i < length:
                px = sx + i * step_dir[0]
                py = sy + i * step_dir[1]
                if px < 0 or px >= w_img or py < 0 or py >= h_img:
                    i += 1
                    continue

                pixel_val = gray[py, px]
                if pixel_val > bg_val * 0.7:
                    i += 1
                    continue

                # Track outward
                track_count = 0
                tx, ty = px, py
                for _ in range(track_len):
                    tx += track_dir[0]
                    ty += track_dir[1]
                    if tx < 0 or tx >= w_img or ty < 0 or ty >= h_img:
                        break
                    if gray[ty, tx] < bg_val * 0.6:
                        track_count += 1
                    else:
                        break

                if track_count >= 3:
                    direction = direction_map[edge_name]
                    out_x, out_y = px, py
                    if direction == "UP":
                        out_y = y1
                    elif direction == "DOWN":
                        out_y = y2
                    elif direction == "LEFT":
                        out_x = x1
                    elif direction == "RIGHT":
                        out_x = x2
                    all_ports.append((int(out_x), int(out_y), direction))

                # Skip 15 px ahead — equipment nozzles are well-separated
                i += 15
                continue

        return all_ports

    def _compute_connection_ports(
        self, objects: list[dict[str, Any]]
    ) -> dict[str, list[tuple[int, int, str]]]:
        """Detect pipe ports on page connection symbols using CV edge scanning."""
        ports: dict[str, list[tuple[int, int, str]]] = {}
        image = self._ensure_image_loaded()

        conn_objects = [
            o for o in objects
            if o.get("class_name") in (
                "page_connection",
                "page connection",
                "connection",
                "utility connection",
                "page connection symbol",
            )
        ]
        if not conn_objects:
            all_classes = {o.get("class_name", "?") for o in objects}
            logger.info("No connection objects found. Stage4 classes: %s", sorted(all_classes))
            return ports

        logger.info("CV port detection: %d connection objects", len(conn_objects))
        for obj in conn_objects:
            obj_id = obj["id"]
            result = self._detect_port_cv(image, obj["bbox"])
            if result:
                px, py, direction = result
                ports[obj_id] = [(px, py, direction)]
                logger.info("  %s -> %s (%d,%d) [CV]", obj_id, direction, px, py)
            else:
                logger.warning("  %s -> CV failed, skipping", obj_id)

        logger.info(
            "Connection port detection: %d/%d ports found",
            len(ports),
            len(conn_objects),
        )

        # --- Equipment port detection from Stage 3 HITL bboxes, LabelMe fallback ---
        equipment_bboxes = self._load_equipment_bboxes_for_stage5b()
        if equipment_bboxes:
            logger.info("Equipment port detection: %d equipment bboxes", len(equipment_bboxes))
            for equipment_obj in equipment_bboxes:
                eq_id = str(equipment_obj["id"])
                label = str(equipment_obj.get("class_name", "equipment"))
                bbox = equipment_obj["bbox"]
                eq_ports = self._detect_equipment_ports_cv(image, bbox)
                if eq_ports:
                    ports[eq_id] = eq_ports
                    port_str = ", ".join(f"{d}({x},{y})" for x, y, d in eq_ports)
                    logger.info(
                        "  %s (%s) -> %d ports: %s",
                        eq_id,
                        label,
                        len(eq_ports),
                        port_str,
                    )
                else:
                    logger.info("  %s (%s) -> no ports detected", eq_id, label)

        equip_port_ids = {k for k in ports if k.startswith("equip_")}
        conn_port_ids = {k for k in ports if not k.startswith("equip_")}
        logger.info("Total ports: %d objects (%d conn + %d equip)",
                 len(ports), len(conn_port_ids), len(equip_port_ids))
        return ports

    # ---------- Stage 5b: CV Pipe Tracing ----------
    def stage5b_pipe_trace(self) -> None:
        """Trace pipes from each connection port to their terminals using CV.

        Uses the Stage 5 pipe mask and Stage 5 connection ports to walk from
        each object/equipment port. The tracer detects straight runs, turns,
        inline pass-through objects, PSV-style orthogonal exits, ray-cast jumps
        over small symbols/gaps, and terminal types such as page connections,
        instrument tags, equipment, tee junctions, sheet edges, and dead ends.

        After port tracing, tee-branch candidates are found from saved paths and
        traced iteratively until no new branch nodes are discovered or the loop
        limit is reached.

        Main artifacts include:
        - stage5_connection_ports.json
        - stage5b_trace_results.json
        - stage5b_branch_candidates.json
        - stage5b_branch_trace_results.json
        - stage5b_trace_overlay.png
        - stage5b_branch_trace_overlay.png

        Debug-only artifacts include:
        - stage5b_branch_candidates_overlay.png
        - stage5b_branch_candidates_iter_XX_overlay.png
        - stage5b_traced_path/*.png
        """
        from garnet.path_tracer.cv_pipe_tracer import CVPipeTracer

        import cv2 as _cv2
        import time as _time

        pipe_mask = _cv2.imread(
            str(self.out_dir / "stage5_pipe_mask.png"), _cv2.IMREAD_GRAYSCALE
        )
        if pipe_mask is None:
            logger.error("Cannot load stage5_pipe_mask.png")
            return

        # Morphological close: bridge 1-2px gaps at corners and text edges
        # without merging distinct parallel pipes (3x3 kernel is safe).
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
        pipe_mask = _cv2.morphologyEx(pipe_mask, _cv2.MORPH_CLOSE, kernel)

        # Compute connection ports if not already cached
        ports_path = self.out_dir / "stage5_connection_ports.json"
        if not ports_path.exists():
            logger.info("Computing connection ports (first run)")
            objects_all = self._load_json_artifact("stage4_objects").get("objects", [])
            ports = self._compute_connection_ports(objects_all)
        else:
            ports = self._load_json_artifact("stage5_connection_ports")

        if not ports:
            logger.warning("No connection ports found — skipping pipe trace")
            return

        ports = self._snap_ports_to_pipe_centerlines(ports, pipe_mask)
        self._save_json("stage5_connection_ports", ports)

        objects = self._load_json_artifact("stage4_objects").get("objects", [])
        image = self._ensure_image_loaded()

        # Separate stage4 objects by type
        page_connections = [
            o for o in objects
            if o.get("class_name") in ("page_connection", "page connection",
                                        "connection", "utility connection",
                                        "page connection symbol")
        ]
        instrument_tags = [
            o for o in objects
            if o.get("class_name") in (
                "instrument tag", "instrument dcs", "instrument logic",
            )
        ]
        equipment = [
            o for o in objects
            if o.get("class_name") in (
                "vessel", "column", "pump", "compressor", "blower",
                "heat_exchanger", "tank", "reactor", "knockout_drum",
                "filter", "cooler", "heater",
            )
        ]

        reviewed_equipment = self._load_equipment_bboxes_for_stage5b()
        equipment.extend(reviewed_equipment)
        if reviewed_equipment:
            logger.info("Added %d Stage 3/fallback equipment bboxes for tracer terminals", len(reviewed_equipment))

        # Inline symbols (valves, reducers, etc.)
        inline_classes = {
            "gate_valve", "globe_valve", "check_valve", "ball_valve",
            "butterfly_valve", "control_valve", "pressure_relief_valve",
            "reducer", "spectacle_blind", "strainer",
            "gate valve", "globe valve", "check valve", "ball valve",
            "butterfly valve", "control valve", "pressure relief valve",
            "spectacle blind",
        }
        inline_symbols = [
            o for o in objects
            if o.get("class_name") in inline_classes
        ]
        node_symbols = [
            o for o in objects
            if o.get("class_name") == "node"
        ]

        # Extend pipe mask into equipment and instrument bboxes
        # so the tracer can walk into terminal objects instead of
        # stopping at the mask edge.
        _all_terminals = equipment + instrument_tags
        if _all_terminals:
            pipe_mask = self._extend_mask_to_terminals(
                pipe_mask, _all_terminals, max_gap=80,
            )
            logger.info(
                "Extended pipe mask toward %d terminals",
                len(_all_terminals),
            )

        # Trace from each port
        visited = np.zeros_like(pipe_mask)
        all_results: dict[str, dict] = {}

        total_ports = sum(len(port_list) for port_list in ports.values())
        logger.info("CV pipe trace: %d objects, %d ports", len(ports), total_ports)
        t0 = _time.monotonic()

        port_items = sorted(
            ports.items(),
            key=lambda item: (1 if str(item[0]).startswith("equip_") else 0, str(item[0])),
        )

        for obj_id, port_list in port_items:
            for port_index, (px, py, direction) in enumerate(port_list, start=1):
                trace_id = obj_id if len(port_list) == 1 else f"{obj_id}:port_{port_index:02d}"
                is_page_connection_source = any(pc.get("id") == obj_id for pc in page_connections)
                existing_terminal = (
                    None
                    if is_page_connection_source
                    else self._find_existing_stage5b_terminal_trace(
                        all_results,
                        obj_id,
                        point=(int(px), int(py)),
                    )
                )
                if existing_terminal is None and str(obj_id).startswith("equip_"):
                    existing_terminal = self._find_existing_stage5b_source_port_trace(
                        all_results,
                        obj_id,
                        point=(int(px), int(py)),
                        direction=str(direction),
                    )
                if existing_terminal is not None:
                    existing_trace_id, existing = existing_terminal
                    all_results[trace_id] = self._skip_stage5b_existing_trace_result(
                        obj_id=obj_id,
                        port_index=port_index,
                        port=(px, py, direction),
                        existing_trace_id=existing_trace_id,
                        existing=existing,
                    )
                    logger.info(
                        "  %s -> reused %s (%d px, %d segs)",
                        trace_id,
                        existing_trace_id,
                        all_results[trace_id]["trace_length_px"],
                        len(all_results[trace_id]["segments"]),
                    )
                    continue
                tracer = CVPipeTracer(
                    pipe_mask=pipe_mask,
                    image=image,
                    page_connections=page_connections,
                    instrument_tags=instrument_tags,
                    equipment_objects=equipment,
                    junction_markers=node_symbols,
                    visited_mask=visited,
                )
                tracer.set_inline_symbols(inline_symbols)

                result = tracer.trace(px, py, direction, source_obj_id=obj_id)
                terminal_type = result.terminal_type

                all_results[trace_id] = {
                    "source_obj_id": obj_id,
                    "port_index": port_index,
                    "port": {"x": px, "y": py, "direction": direction},
                    "terminal_type": terminal_type,
                    "terminal_x": result.terminal_x,
                    "terminal_y": result.terminal_y,
                    "terminal_obj_id": result.terminal_obj_id,
                    "segments": [
                        {
                            "x1": s.x1, "y1": s.y1,
                            "x2": s.x2, "y2": s.y2,
                            "direction": s.direction,
                            "length_px": s.length_px,
                        }
                        for s in result.segments
                    ],
                    "turns": [
                        {"x": tx, "y": ty, "new_dir": td}
                        for tx, ty, td in result.turns
                    ],
                    "hits": [
                        {"class": h.class_name, "x": h.x, "y": h.y}
                        for h in result.hits
                    ],
                    "trace_length_px": result.trace_length_px,
                    "status": result.status,
                }
                self._extend_stage5b_result_to_terminal(all_results[trace_id])
                self._align_stage5b_result_to_near_node(all_results[trace_id], node_symbols)
                logger.info(
                    "  %s -> %s (%d px, %d segs)",
                    trace_id, result.terminal_type,
                    result.trace_length_px, len(result.segments),
                )

        elapsed = _time.monotonic() - t0
        logger.info("CV pipe trace done: %d traces in %.1fs", len(all_results), elapsed)

        self._save_json("stage5b_trace_results", all_results)
        for stale_overlay in self.out_dir.glob("stage5b_branch_candidates_iter_*_overlay.png"):
            stale_overlay.unlink()
        if not self.cfg.debug_artifacts:
            stale_candidate_overlay = self.out_dir / "stage5b_branch_candidates_overlay.png"
            if stale_candidate_overlay.exists():
                stale_candidate_overlay.unlink()
            stale_trace_dir = self.out_dir / "stage5b_traced_path"
            if stale_trace_dir.exists():
                shutil.rmtree(stale_trace_dir)
        branch_candidates, branch_results, branch_iterations = self._trace_stage5b_branches_iterative(
            pipe_mask=pipe_mask,
            image=image,
            base_results=all_results,
            page_connections=page_connections,
            instrument_tags=instrument_tags,
            equipment=equipment,
            inline_symbols=inline_symbols,
            node_symbols=node_symbols,
            visited=visited,
            candidate_overlay_base=image if self.cfg.debug_artifacts else None,
        )
        self._align_stage5b_branch_node_terminals(branch_results, node_symbols)
        self._save_json("stage5b_branch_candidates", {
            "candidates": branch_candidates,
            "iterations": branch_iterations,
            "summary": {
                "total": len(branch_candidates),
                "queued": len([c for c in branch_candidates if c.get("status") == "queued"]),
                "done": len([c for c in branch_candidates if str(c.get("status", "")).startswith("done_")]),
                "rejected": len([c for c in branch_candidates if str(c.get("status", "")).startswith("rejected_")]),
                "iteration_count": len(branch_iterations),
            },
        })
        self._save_json("stage5b_branch_trace_results", {
            "branches": branch_results,
            "iterations": branch_iterations,
            "summary": {
                "total": len(branch_results),
                "traced": len([r for r in branch_results.values() if r.get("status") == "traced"]),
                "skipped": len([r for r in branch_results.values() if r.get("status") == "skipped"]),
                "iteration_count": len(branch_iterations),
            },
        })

        # Draw overlay
        overlay = image.copy()
        colors = {
            "page_connection": (0, 180, 0),
            "instrument_tag": (0, 120, 0),
            "equipment": (0, 100, 180),
            "tee_junction": (180, 0, 180),
            "sheet_edge": (100, 100, 100),
            "dead_end": (0, 0, 200),
            "max_steps": (200, 100, 0),
        }
        # Build object lookup for source bbox drawing
        id_to_obj = {o["id"]: o for o in objects}

        # Human-readable P&ID labels
        _CLASS_LABEL_MAP = {
            "gate_valve": "Gate Valve",
            "globe_valve": "Globe Valve",
            "check_valve": "Check Valve",
            "ball_valve": "Ball Valve",
            "butterfly_valve": "Butterfly Valve",
            "control_valve": "Control Valve",
            "pressure_relief_valve": "Pressure Relief Valve",
            "reducer": "Reducer",
            "spectacle_blind": "Spectacle Blind",
            "strainer": "Strainer",
            "pump": "Pump",
            "vessel": "Vessel",
            "column": "Column",
            "heat_exchanger": "Heat Exchanger",
            "tank": "Tank",
            "compressor": "Compressor",
            "blower": "Blower",
            "filter": "Filter",
            "cooler": "Cooler",
            "heater": "Heater",
            "reactor": "Reactor",
            "knockout_drum": "Knockout Drum",
            "page connection": "Page Conn.",
            "connection": "Connection",
            "utility connection": "Utility Conn.",
            "instrument tag": "Instr. Tag",
            "instrument dcs": "Instr. (DCS)",
            "instrument logic": "Instr. (Logic)",
            "line number": "Line No.",
            "arrow": "Flow Arrow",
            "node": "Node",
            "sampling point": "Samp. Point",
        }

        for obj_id, data in all_results.items():
            # --- Draw source object bbox ---
            source_obj_id = str(data.get("source_obj_id") or obj_id)
            src_obj = id_to_obj.get(source_obj_id)
            if src_obj:
                src_bbox = src_obj["bbox"]
                _cv2.rectangle(
                    overlay,
                    (src_bbox["x_min"], src_bbox["y_min"]),
                    (src_bbox["x_max"], src_bbox["y_max"]),
                    (0, 200, 0), 2,
                )
                # Human-readable class label above bbox
                cls_raw = src_obj.get("class_name", "?")
                cls_label = _CLASS_LABEL_MAP.get(cls_raw, cls_raw.replace("_", " ").title())
                label_text = f"{cls_label} ({source_obj_id})"
                _cv2.putText(
                    overlay, label_text,
                    (src_bbox["x_min"], src_bbox["y_min"] - 6),
                    _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2,
                )

            # Draw trace path
            for seg in data["segments"]:
                _cv2.line(
                    overlay,
                    (seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]),
                    (0, 200, 0), 2,
                )
                # Arrow at midpoint
                mx = (seg["x1"] + seg["x2"]) // 2
                my = (seg["y1"] + seg["y2"]) // 2
                _cv2.circle(overlay, (mx, my), 3, (0, 160, 0), -1)

            # Port marker (start point)
            px, py = data["port"]["x"], data["port"]["y"]
            _cv2.circle(overlay, (px, py), 5, (0, 255, 0), -1)
            _cv2.circle(overlay, (px, py), 5, (255, 255, 255), 1)

            # Terminal marker
            tx, ty = data["terminal_x"], data["terminal_y"]
            ttype = data.get("terminal_type", "unknown")
            color = colors.get(ttype, (128, 128, 128))
            _cv2.circle(overlay, (tx, ty), 8, color, -1)
            _cv2.circle(overlay, (tx, ty), 8, (255, 255, 255), 1)

            # Label
            label = f"{obj_id}:{ttype}"
            _cv2.putText(
                overlay, label,
                (tx + 12, ty - 12),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        # Draw equipment ports (cyan markers from stage5_connection_ports)
        self._draw_equipment_port_markers(overlay, ports)

        # Draw equipment bboxes from Stage 3 HITL artifact or LabelMe fallback.
        self._draw_equipment_ground_truth(overlay)

        self._save_img("stage5b_trace_overlay", overlay)
        if self.cfg.debug_artifacts:
            branch_overlay = self._draw_stage5b_branch_candidate_overlay(
                overlay,
                branch_candidates,
                branch_results=branch_results,
            )
            self._save_img("stage5b_branch_candidates_overlay", branch_overlay)
        branch_trace_overlay = self._draw_stage5b_branch_trace_overlay(
            overlay,
            branch_results,
        )
        self._save_img("stage5b_branch_trace_overlay", branch_trace_overlay)
        if self.cfg.debug_artifacts:
            self._write_stage5b_individual_trace_images(
                image,
                all_results,
                branch_results,
            )


