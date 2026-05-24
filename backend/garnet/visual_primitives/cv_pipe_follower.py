"""CV Pipe Follower — traces pipeline paths on the binary pipe mask.

Walks along connected pipe-mask pixels from a starting point.  Detects
hits on YOLO stage-4 objects and branches at junctions.  Pure CV — no VLM
calls needed for line following.

Used by Agent 2 (hybrid mode) as a fast alternative to VLM step-by-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CVTraceStep:
    """One step in a CV-guided pipeline trace."""

    kind: str  # "move", "hit", "junction", "terminal"
    direction: str = ""  # UP/DOWN/LEFT/RIGHT
    distance_px: int = 0
    x: int = 0  # absolute image pixel x at this step
    y: int = 0  # absolute image pixel y at this step
    hit_object: Optional[dict[str, Any]] = None  # YOLO object dict if hit
    junction_branches: list[str] = field(default_factory=list)


@dataclass
class CVTraceResult:
    """Result of tracing one pipeline segment with CV."""

    anchor_id: str = ""
    start_x: int = 0
    start_y: int = 0
    start_direction: str = ""
    steps: list[CVTraceStep] = field(default_factory=list)
    terminal_kind: str = "unknown"  # "equipment", "page_connection", "sheet_edge", "dead_end"
    terminal_object: Optional[dict[str, Any]] = None
    terminal_x: int = 0
    terminal_y: int = 0
    total_length_px: int = 0
    path_mask: Optional[np.ndarray] = None  # binary mask of traversed path


# ---------------------------------------------------------------------------
# Line follower
# ---------------------------------------------------------------------------


class CVPipeFollower:
    """Traces pipe lines on a binary mask using pixel-walking.

    The follower tracks connected white pixels from a starting point and
    detects when the path enters a known object bounding box or reaches
    a junction / dead end.
    """

    def __init__(
        self,
        pipe_mask: np.ndarray,
        stage4_objects: list[dict[str, Any]],
        image_w: int,
        image_h: int,
        step_size: int = 5,
        window_size: int = 40,
    ):
        self.mask = (pipe_mask > 0).astype(np.uint8) * 255
        self.objects = stage4_objects
        self.image_w = image_w
        self.image_h = image_h
        self.step_size = step_size
        self.window_size = window_size

        # Build spatial index for YOLO objects
        self._object_index: list[tuple[int, int, int, int, dict]] = []
        for obj in stage4_objects:
            b = obj["bbox"]
            self._object_index.append((b["x_min"], b["y_min"], b["x_max"], b["y_max"], obj))

        # Track global visited mask to avoid re-walking
        self.visited_mask = np.zeros_like(self.mask, dtype=np.uint8)

        # Direction vectors
        self._dir_vec: dict[str, tuple[int, int]] = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0),
        }

    # ------------------------------------------------------------------
    # Trace
    # ------------------------------------------------------------------

    def trace(
        self,
        start_x: int,
        start_y: int,
        direction: str,
        anchor_id: str = "",
        max_steps: int = 500,
    ) -> CVTraceResult:
        """Trace a pipe line from a starting point.

        Walks along pipe-mask pixels, detecting YOLO objects and junctions.
        Returns a CVTraceResult with all steps and terminal info.
        """
        result = CVTraceResult(
            anchor_id=anchor_id,
            start_x=start_x,
            start_y=start_y,
            start_direction=direction,
        )

        x, y = start_x, start_y
        cur_dir = direction

        # Jump forward into the mask if not on it
        x, y = self._snap_to_mask(x, y, cur_dir)
        if x < 0:
            result.terminal_kind = "no_pipe_found"
            return result

        path_mask = np.zeros_like(self.mask, dtype=np.uint8)
        prev_x, prev_y = x, y  # track previous position for line marking

        for step_i in range(max_steps):
            # Mark visited — solid line from previous position
            cv2.line(self.visited_mask, (prev_x, prev_y), (x, y), (255,), 5)
            cv2.line(path_mask, (prev_x, prev_y), (x, y), (255,), 5)
            prev_x, prev_y = x, y

            # --- Object check ---
            hit_obj = self._object_at(x, y)
            if hit_obj:
                cls = hit_obj.get("class_name", "")
                if self._is_terminal(hit_obj):
                    result.steps.append(CVTraceStep(
                        kind="hit", x=x, y=y, hit_object=hit_obj))
                    result.terminal_kind = cls
                    result.terminal_object = hit_obj
                    result.terminal_x, result.terminal_y = x, y
                    result.path_mask = path_mask
                    return result

                if self._is_inline(hit_obj):
                    result.steps.append(CVTraceStep(
                        kind="hit", x=x, y=y, hit_object=hit_obj))
                    resume = self._jump_past_object(x, y, cur_dir, hit_obj)
                    if resume:
                        result.steps.append(CVTraceStep(
                            kind="move", direction=cur_dir,
                            distance_px=0, x=resume[0], y=resume[1]))
                        x, y = resume
                        continue
                    result.terminal_kind = "dead_end"
                    result.terminal_x, result.terminal_y = x, y
                    result.path_mask = path_mask
                    return result
                else:
                    result.steps.append(CVTraceStep(
                        kind="hit", x=x, y=y, hit_object=hit_obj))

            # --- Find next direction ---
            next_dir, is_junction, branches = self._find_next_direction(
                x, y, cur_dir
            )

            if next_dir is None:
                if self._at_sheet_edge(x, y):
                    result.terminal_kind = "sheet_edge"
                else:
                    result.terminal_kind = "dead_end"
                result.terminal_x, result.terminal_y = x, y
                result.path_mask = path_mask
                return result

            if is_junction and branches:
                result.steps.append(CVTraceStep(
                    kind="junction", x=x, y=y,
                    junction_branches=list(branches)))
                # Continue straight through junction — main line rule
                # (END at junction is determined by downstream terminal, not here)

            # --- Move ---
            dx, dy = self._dir_vec[next_dir]
            distance = self._walk_distance(x, y, next_dir)
            if distance < self.step_size:
                distance = self.step_size

            x += dx * distance
            y += dy * distance
            x = max(0, min(self.image_w - 1, x))
            y = max(0, min(self.image_h - 1, y))

            step = CVTraceStep(
                kind="move", direction=next_dir,
                distance_px=distance, x=x, y=y,
            )
            result.steps.append(step)
            result.total_length_px += distance
            cur_dir = next_dir

        result.terminal_kind = "max_steps"
        result.terminal_x, result.terminal_y = x, y
        result.path_mask = path_mask
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snap_to_mask(self, x: int, y: int, direction: str) -> tuple[int, int]:
        """Find nearest mask pixel in the given direction."""
        dx, dy = self._dir_vec.get(direction, (1, 0))
        for dist in range(0, 80, 2):
            nx = x + dx * dist
            ny = y + dy * dist
            if 0 <= nx < self.image_w and 0 <= ny < self.image_h:
                if self.mask[ny, nx] > 0 and self.visited_mask[ny, nx] == 0:
                    return (nx, ny)
        return (-1, -1)

    def _find_next_direction(
        self, x: int, y: int, current_dir: str
    ) -> tuple[Optional[str], bool, list[str]]:
        """Determine next walking direction by scanning local window.

        Returns (next_direction, is_junction, branch_directions).

        P&ID rules:
        - Pipes run horizontal or vertical.
        - At a junction, the main line continues straight (same axis).
        - Turns are 90-degree at corners.
        """
        half = self.window_size // 2
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(self.image_w, x + half)
        y2 = min(self.image_h, y + half)

        window = self.mask[y1:y2, x1:x2].copy()
        visited_win = self.visited_mask[y1:y2, x1:x2]
        window[visited_win > 0] = 0

        cx = x - x1
        cy = y - y1

        hits: dict[str, int] = {}
        for dname, (dx, dy) in self._dir_vec.items():
            count = 0
            px, py = cx, cy
            for _ in range(half):
                px += dx
                py += dy
                if 0 <= px < window.shape[1] and 0 <= py < window.shape[0]:
                    if window[py, px] > 0:
                        count += 1
            hits[dname] = count

        threshold = 2  # lower threshold catches thin lines at corners
        valid = {d: c for d, c in hits.items() if c >= threshold}

        if not valid:
            return (None, False, [])

        opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

        if current_dir in valid:
            other_branches = [d for d in valid if d != current_dir and d != opposite[current_dir]]
            is_junction = len(other_branches) >= 1
            return (current_dir, is_junction, other_branches)

        best = max(valid, key=lambda d: valid[d])
        other_branches = [d for d in valid if d != best and d != opposite[best]]
        is_junction = len(other_branches) >= 1
        return (best, is_junction, other_branches)

    def _walk_distance(self, x: int, y: int, direction: str) -> int:
        """How far can we walk in this direction before hitting a gap or visited pixel?"""
        dx, dy = self._dir_vec[direction]
        max_dist = 0
        for d in range(self.step_size, self.window_size * 2, self.step_size):
            nx = x + dx * d
            ny = y + dy * d
            if not (0 <= nx < self.image_w and 0 <= ny < self.image_h):
                return max(d - self.step_size, self.step_size)
            if self.mask[ny, nx] == 0 or self.visited_mask[ny, nx] > 0:
                return max(d - self.step_size, self.step_size)
            max_dist = d
        return max(max_dist, self.step_size)

    def _object_at(self, x: int, y: int) -> Optional[dict[str, Any]]:
        """Check if (x,y) falls inside any YOLO-detected object bbox."""
        for x1, y1, x2, y2, obj in self._object_index:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return obj
        return None

    def _is_terminal(self, obj: dict[str, Any]) -> bool:
        """Check if a YOLO object is a pipeline terminal (equipment, connection)."""
        cls = obj.get("class_name", "")
        terminals = {
            "pump", "heat exchanger", "tank", "vessel", "column",
            "compressor", "blower", "fan",
            "page connection", "connection", "utility connection",
            "instrument tag",   # PC pipes terminate at instrument labels
            "nozzle",           # vessel nozzle connections
        }
        return cls in terminals

    def _is_inline(self, obj: dict[str, Any]) -> bool:
        """Check if a YOLO object is an inline component (valve, reducer, etc.)."""
        cls = obj.get("class_name", "")
        inline = {
            "valve", "control valve", "check valve", "ball valve",
            "gate valve", "globe valve", "butterfly valve",
            "instrument", "indicator", "transmitter", "controller",
            "solenoid", "actuator", "reducer", "strainer",
            "spectacle blind", "sampling point",
        }
        return cls in inline

    def _jump_past_object(
        self, x: int, y: int, direction: str, obj: dict[str, Any]
    ) -> Optional[tuple[int, int]]:
        """Jump to the far side of an inline object and find where pipe resumes.

        Returns (x, y) of resume point or None if can't bridge.
        """
        bbox = obj["bbox"]
        dir_dx = {"RIGHT": 1, "LEFT": -1, "UP": 0, "DOWN": 0}
        dir_dy = {"RIGHT": 0, "LEFT": 0, "UP": -1, "DOWN": 1}
        dx = dir_dx.get(direction, 1)
        dy = dir_dy.get(direction, 0)

        # Start from far side of object
        if direction == "RIGHT":
            sx = bbox["x_max"] + 3
            sy = y
        elif direction == "LEFT":
            sx = bbox["x_min"] - 3
            sy = y
        elif direction == "DOWN":
            sx = x
            sy = bbox["y_max"] + 3
        else:  # UP
            sx = x
            sy = bbox["y_min"] - 3

        # Scan for mask pixels in the same direction
        for offset in range(0, 200, 5):
            nx = sx + dx * offset
            ny = sy + dy * offset
            if not (0 <= nx < self.image_w and 0 <= ny < self.image_h):
                return None
            if self.mask[ny, nx] > 0 and self.visited_mask[ny, nx] == 0:
                return (nx, ny)
        return None

    def _at_sheet_edge(self, x: int, y: int, margin: int = 15) -> bool:
        """Check if position is at sheet boundary."""
        return (
            x <= margin or x >= self.image_w - margin
            or y <= margin or y >= self.image_h - margin
        )

    def clear_visited(self):
        """Reset the visited mask — use between independent trace groups."""
        self.visited_mask = np.zeros_like(self.mask, dtype=np.uint8)
