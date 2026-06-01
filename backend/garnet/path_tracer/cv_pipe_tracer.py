"""CV pipe tracer — traces pipe lines from port to terminal using the pipe mask.

Algorithm:
  START at port (px, py, direction)
  LOOP:
    1. Walk straight on pipe mask
    2. Detect valid continuations (straight / turn L / turn R)
    3. Detect inline objects (valve, reducer) → mark, jump over
    4. Detect terminals (page connection, equipment, tag, tee, sheet edge, dead end)
    5. Report segments, turns, hits, terminal

Pure CV — no VLM calls. Fast, deterministic, zero cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TraceToken(Enum):
    STRAIGHT = auto()
    TURN = auto()
    HIT = auto()      # inline object (valve, reducer, etc.)
    TERMINAL = auto()


class TerminalType(Enum):
    PAGE_CONNECTION = "page_connection"
    INSTRUMENT_TAG = "instrument_tag"
    EQUIPMENT = "equipment"
    TEE_JUNCTION = "tee_junction"
    SHEET_EDGE = "sheet_edge"
    DEAD_END = "dead_end"
    NO_PIPE = "no_pipe"


@dataclass
class TraceSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    direction: str  # UP, DOWN, LEFT, RIGHT
    length_px: int


@dataclass
class InlineHit:
    class_name: str  # valve, reducer, spectacle_blind, etc.
    x: int
    y: int
    bbox: Optional[dict[str, int]] = None


@dataclass
class TraceResult:
    segments: list[TraceSegment] = field(default_factory=list)
    turns: list[tuple[int, int, str]] = field(default_factory=list)  # (x, y, new_dir)
    hits: list[InlineHit] = field(default_factory=list)
    terminal_type: Optional[str] = None  # TerminalType value
    terminal_x: int = 0
    terminal_y: int = 0
    terminal_obj_id: Optional[str] = None  # matching stage4 object ID
    trace_length_px: int = 0
    status: str = "ok"  # ok, no_pipe, abandoned


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

DIRECTION_DELTA = {
    "UP":    (0, -1),
    "DOWN":  (0, 1),
    "LEFT":  (-1, 0),
    "RIGHT": (1, 0),
}

TURN_LEFT = {
    "UP": "LEFT", "LEFT": "DOWN",
    "DOWN": "RIGHT", "RIGHT": "UP",
}

TURN_RIGHT = {
    "UP": "RIGHT", "RIGHT": "DOWN",
    "DOWN": "LEFT", "LEFT": "UP",
}

OPPOSITE = {
    "UP": "DOWN", "DOWN": "UP",
    "LEFT": "RIGHT", "RIGHT": "LEFT",
}


def _is_pipe(mask: np.ndarray, x: int, y: int) -> bool:
    """Check if (x,y) is pipe pixel (value > 0)."""
    h, w = mask.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    return bool(mask[y, x])


def _is_pipe_band(mask: np.ndarray, x: int, y: int,
                  direction: str, band_width: int = 3) -> bool:
    """Check for pipe in a band perpendicular to travel direction.

    For horizontal traces (LEFT/RIGHT), checks y-band_width to y+band_width.
    For vertical traces (UP/DOWN), checks x-band_width to x+band_width.
    Tolerates 1-3px line offsets common in scanned P&IDs.
    """
    h, w = mask.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    if direction in ("LEFT", "RIGHT"):
        for dy in range(-band_width, band_width + 1):
            ny = y + dy
            if 0 <= ny < h and mask[ny, x] > 0:
                return True
    else:  # UP, DOWN
        for dx in range(-band_width, band_width + 1):
            nx = x + dx
            if 0 <= nx < w and mask[y, nx] > 0:
                return True
    return False


def _has_line_of_sight(mask: np.ndarray, x: int, y: int,
                       direction: str, distance: int = 8) -> bool:
    """Check if pipe exists for at least `distance` pixels in `direction`."""
    dx, dy = DIRECTION_DELTA[direction]
    for i in range(1, distance + 1):
        if not _is_pipe(mask, x + i * dx, y + i * dy):
            return False
    return True


def _has_line_of_sight_axis_band(mask: np.ndarray, x: int, y: int,
                                 direction: str, distance: int = 8,
                                 band_width: int = 1) -> bool:
    """Axis-constrained LOS with small perpendicular tolerance."""
    dx, dy = DIRECTION_DELTA[direction]
    for i in range(1, distance + 1):
        nx = x + i * dx
        ny = y + i * dy
        if not _is_pipe_band(mask, nx, ny, direction, band_width=band_width):
            return False
    return True


def _has_line_of_sight_band_narrow(mask: np.ndarray, x: int, y: int,
                                   direction: str, distance: int = 8) -> bool:
    """Narrow turn-only LOS check to reduce side-path false positives."""
    return _has_line_of_sight_axis_band(mask, x, y, direction, distance, band_width=1)


def _has_line_of_sight_axis_exact(mask: np.ndarray, x: int, y: int,
                                  direction: str, distance: int = 8) -> bool:
    """Strict LOS check for turns: no perpendicular band tolerance."""
    return _has_line_of_sight_axis_band(mask, x, y, direction, distance, band_width=0)


def _has_connected_side_pipe(mask: np.ndarray, x: int, y: int,
                             direction: str, min_run: int = 10) -> bool:
    """Detect a side branch connected at the current point.

    This is stricter than a banded turn check: it requires pipe pixels adjacent
    to the current point and a continuous run away from it. That keeps flange
    strokes from being treated as real pipe turns.
    """
    dx, dy = DIRECTION_DELTA[direction]
    h, w = mask.shape

    for lateral in (-1, 0, 1):
        if direction in ("LEFT", "RIGHT"):
            sx = x + dx
            sy = y + lateral
        else:
            sx = x + lateral
            sy = y + dy
        if not (0 <= sx < w and 0 <= sy < h) or mask[sy, sx] == 0:
            continue

        run = 0
        for step in range(1, min_run + 1):
            px = x + dx * step
            py = y + dy * step
            if direction in ("LEFT", "RIGHT"):
                ok = any(
                    0 <= py + off < h and 0 <= px < w and mask[py + off, px] > 0
                    for off in (-1, 0, 1)
                )
            else:
                ok = any(
                    0 <= px + off < w and 0 <= py < h and mask[py, px + off] > 0
                    for off in (-1, 0, 1)
                )
            if not ok:
                break
            run += 1
        if run >= min_run:
            return True
    return False


def _check_bbox_hit(x: int, y: int, bbox: dict[str, int], margin: int = 4) -> bool:
    """Check if point is within or near a bbox."""
    return (bbox["x_min"] - margin <= x <= bbox["x_max"] + margin and
            bbox["y_min"] - margin <= y <= bbox["y_max"] + margin)


def _is_inside_bbox_exact(x: int, y: int, bbox: dict[str, int]) -> bool:
    return bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]


# ---------------------------------------------------------------------------
# Main tracer
# ---------------------------------------------------------------------------

class CVPipeTracer:
    """Trace one pipe path from port to terminal."""

    def __init__(
        self,
        pipe_mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        page_connections: Optional[list[dict]] = None,
        instrument_tags: Optional[list[dict]] = None,
        equipment_objects: Optional[list[dict]] = None,
        junction_markers: Optional[list[dict]] = None,
        visited_mask: Optional[np.ndarray] = None,
        max_steps: int = 5000,
        min_step: int = 5,
        straight_min_step: int = 10,
        turn_min_step: int = 3,
        lookahead: int = 30,
    ):
        self.mask = pipe_mask
        self.image = image
        self.h, self.w = pipe_mask.shape
        self.max_steps = max_steps
        self.min_step = min_step
        self.straight_min_step = straight_min_step
        self.turn_min_step = turn_min_step
        self.lookahead = lookahead

        # Terminal candidates
        self.page_connections = page_connections or []
        self.instrument_tags = instrument_tags or []
        self.equipment_objects = equipment_objects or []
        self.junction_markers = junction_markers or []

        # Visited mask (shared across traces to avoid re-walking)
        self.visited = visited_mask if visited_mask is not None else np.zeros_like(pipe_mask)

        # Inline symbol bboxes (valves, reducers, etc.) — from stage4
        self._inline_symbols: list[dict] = []  # set via set_inline_symbols()

    def _center_of_nearest_run(self, values: list[int], target: int) -> Optional[int]:
        if not values:
            return None
        runs: list[list[int]] = []
        current: list[int] = []
        for value in sorted(values):
            if not current or value == current[-1] + 1:
                current.append(value)
                continue
            runs.append(current)
            current = [value]
        if current:
            runs.append(current)
        best_run = min(
            runs,
            key=lambda run: (
                0 if run[0] <= target <= run[-1] else min(abs(target - run[0]), abs(target - run[-1])),
                abs(((run[0] + run[-1]) / 2.0) - target),
            ),
        )
        return int(round((best_run[0] + best_run[-1]) / 2.0))

    def _line_support_score(self, x: int, y: int, direction: str, radius: int = 8) -> int:
        """Count local pipe pixels along the current travel axis."""
        score = 0
        if direction in ("UP", "DOWN"):
            for cy in range(y - radius, y + radius + 1):
                if 0 <= cy < self.h and 0 <= x < self.w and self.mask[cy, x] > 0:
                    score += 1
        else:
            for cx in range(x - radius, x + radius + 1):
                if 0 <= cx < self.w and 0 <= y < self.h and self.mask[y, cx] > 0:
                    score += 1
        return score

    def _center_of_best_axis_support(
        self,
        candidates: list[int],
        target: int,
        score_at,
    ) -> Optional[int]:
        """Return midpoint of the nearest best-supported contiguous candidate run."""
        if not candidates:
            return None

        scored = [(value, score_at(value)) for value in sorted(set(candidates))]
        if not scored:
            return None
        best_score = max(score for _, score in scored)
        if best_score <= 0:
            return None

        strong_values = [value for value, score in scored if score == best_score]
        return self._center_of_nearest_run(strong_values, target)

    def _snap_to_centerline(self, x: int, y: int, direction: str) -> tuple[int, int]:
        """Snap to the centerline using support along the current travel axis.

        A perpendicular row/column midpoint is unstable at elbows because the
        orthogonal leg contaminates the local stroke width.  Scoring candidates
        by same-axis continuity keeps vertical walks on vertical strokes and
        horizontal walks on horizontal strokes.
        """
        if direction in ("UP", "DOWN"):
            cols = [cx for cx in range(x - 8, x + 9)
                    if 0 <= cx < self.w and 0 <= y < self.h and self.mask[y, cx] > 0]
            center = self._center_of_best_axis_support(
                cols,
                x,
                lambda cx: self._line_support_score(cx, y, direction),
            )
            if center is not None:
                return (center, y)
        else:
            rows = [cy for cy in range(y - 8, y + 9)
                    if 0 <= cy < self.h and 0 <= x < self.w and self.mask[cy, x] > 0]
            center = self._center_of_best_axis_support(
                rows,
                y,
                lambda cy: self._line_support_score(x, cy, direction),
            )
            if center is not None:
                return (x, center)
        return (x, y)

    def _enter_turn_leg(self, x: int, y: int, turn_dir: str) -> tuple[int, int]:
        dx, dy = DIRECTION_DELTA[turn_dir]
        probe_x = x
        probe_y = y
        for step in range(1, self.min_step + 1):
            nx = x + dx * step
            ny = y + dy * step
            if _is_pipe_band(self.mask, nx, ny, turn_dir, band_width=1):
                probe_x, probe_y = nx, ny
                break
        return self._snap_to_centerline(probe_x, probe_y, turn_dir)

    def _turn_segment_start(
        self,
        turn_x: int,
        turn_y: int,
        entered_x: int,
        entered_y: int,
        turn_dir: str,
    ) -> tuple[int, int]:
        """Keep close-corner inline turns visually connected to the turn point.

        A valve immediately after an elbow may have sparse mask pixels.  The
        walker still needs to enter at the first stable pipe pixel, but the
        result segment should start at the elbow so the traced path does not
        show an artificial jump over the lower half of the inline symbol.
        """
        if max(abs(entered_x - turn_x), abs(entered_y - turn_y)) <= self.min_step:
            return entered_x, entered_y
        return self._snap_to_centerline(turn_x, turn_y, turn_dir)


    def set_inline_symbols(self, objects: list[dict]) -> None:
        """Set stage4 objects that are inline symbols (valve, reducer, etc.)."""
        inline_classes = {
            "gate_valve", "globe_valve", "check_valve", "ball_valve",
            "butterfly_valve", "control_valve", "pressure_relief_valve",
            "reducer", "spectacle_blind", "strainer",
            "arrow",
            "gate valve", "globe valve", "check valve", "ball valve",
            "butterfly valve", "control valve", "pressure relief valve",
            "spectacle blind",
        }
        self._inline_symbols = [
            o for o in objects
            if o.get("class_name", "") in inline_classes
        ]

    def _terminal_bbox_by_type(self, terminal_type: str, terminal_obj_id: Optional[str]) -> Optional[dict[str, int]]:
        if terminal_type == TerminalType.PAGE_CONNECTION.value:
            for pc in self.page_connections:
                if pc.get("id", "") == (terminal_obj_id or ""):
                    return pc.get("bbox")
            return None
        if terminal_type == TerminalType.INSTRUMENT_TAG.value:
            for tag in self.instrument_tags:
                if tag.get("id", "") == (terminal_obj_id or ""):
                    return tag.get("bbox")
            return None
        if terminal_type == TerminalType.EQUIPMENT.value:
            for eq in self.equipment_objects:
                if eq.get("id", "") == (terminal_obj_id or ""):
                    return eq.get("bbox")
            return None
        return None

    def _is_inline_target(self, x: int, y: int) -> bool:
        """True if point is inside/near an inline symbol bbox."""
        for sym in self._inline_symbols:
            bbox = sym.get("bbox")
            if bbox and _check_bbox_hit(x, y, bbox, margin=2):
                return True
        return False

    def _has_connected_side_path(
        self,
        x: int,
        y: int,
        direction: str,
        min_run: int,
        inline_probe_px: int = 60,
    ) -> bool:
        """Side path is connected by pipe run or by a short pipe stub into inline."""
        if _has_connected_side_pipe(self.mask, x, y, direction, min_run):
            return True

        dx, dy = DIRECTION_DELTA[direction]
        stub_run = 0
        for step in range(1, inline_probe_px + 1):
            px = x + dx * step
            py = y + dy * step
            if _is_pipe_band(self.mask, px, py, direction, band_width=1):
                stub_run += 1
                continue
            if self._is_inline_target(px, py) and stub_run >= self.turn_min_step:
                return True
            if stub_run == 0:
                continue
            break
        return False

    def _retreat_from_terminal_bbox(self, x: int, y: int, direction: str,
                                    terminal_type: str, terminal_obj_id: Optional[str]) -> tuple[int, int]:
        """If current point is inside terminal bbox, step back to its boundary."""
        bbox = self._terminal_bbox_by_type(terminal_type, terminal_obj_id)
        if not bbox:
            return x, y
        if not _is_inside_bbox_exact(x, y, bbox):
            return x, y

        dx, dy = DIRECTION_DELTA[direction]
        for _ in range(200):
            px = x - dx
            py = y - dy
            if _is_inside_bbox_exact(px, py, bbox):
                x, y = px, py
                continue
            break
        return x, y

    def _append_segment(self, result: TraceResult, x1: int, y1: int,
                        x2: int, y2: int, direction: str) -> None:
        if direction in ("UP", "DOWN"):
            cx = int(round((x1 + x2) / 2.0))
            x1 = cx
            x2 = cx
        elif direction in ("LEFT", "RIGHT"):
            cy = int(round((y1 + y2) / 2.0))
            y1 = cy
            y2 = cy
        seg_len = max(abs(x2 - x1), abs(y2 - y1))
        if seg_len >= self.min_step:
            result.segments.append(TraceSegment(
                x1=x1, y1=y1,
                x2=x2, y2=y2, direction=direction,
                length_px=seg_len,
            ))
            result.trace_length_px += seg_len

    def _anchor_close_turn_segments(self, result: TraceResult, max_gap: int = 60) -> None:
        """Pull the first post-turn segment back to the elbow for close inline gaps."""
        if not result.turns or not result.segments:
            return

        for turn_x, turn_y, turn_dir in result.turns:
            for segment in result.segments:
                if segment.direction != turn_dir:
                    continue
                if turn_dir in ("UP", "DOWN"):
                    if abs(segment.x1 - turn_x) > 20 and abs(segment.x2 - turn_x) > 20:
                        continue
                    near_y = segment.y1 if abs(segment.y1 - turn_y) <= abs(segment.y2 - turn_y) else segment.y2
                    gap = abs(near_y - turn_y)
                    if gap == 0 or gap > max_gap:
                        continue
                    if turn_dir == "UP" and near_y > turn_y:
                        continue
                    if turn_dir == "DOWN" and near_y < turn_y:
                        continue
                    x_axis = int(round((segment.x1 + segment.x2 + turn_x) / 3.0))
                    segment.x1 = x_axis
                    segment.x2 = x_axis
                    if abs(segment.y1 - turn_y) <= abs(segment.y2 - turn_y):
                        segment.y1 = turn_y
                    else:
                        segment.y2 = turn_y
                    break
                else:
                    if abs(segment.y1 - turn_y) > 20 and abs(segment.y2 - turn_y) > 20:
                        continue
                    near_x = segment.x1 if abs(segment.x1 - turn_x) <= abs(segment.x2 - turn_x) else segment.x2
                    gap = abs(near_x - turn_x)
                    if gap == 0 or gap > max_gap:
                        continue
                    if turn_dir == "LEFT" and near_x > turn_x:
                        continue
                    if turn_dir == "RIGHT" and near_x < turn_x:
                        continue
                    y_axis = int(round((segment.y1 + segment.y2 + turn_y) / 3.0))
                    segment.y1 = y_axis
                    segment.y2 = y_axis
                    if abs(segment.x1 - turn_x) <= abs(segment.x2 - turn_x):
                        segment.x1 = turn_x
                    else:
                        segment.x2 = turn_x
                    break

        result.trace_length_px = sum(
            max(abs(s.x2 - s.x1), abs(s.y2 - s.y1))
            for s in result.segments
        )
        for segment in result.segments:
            segment.length_px = max(
                abs(segment.x2 - segment.x1),
                abs(segment.y2 - segment.y1),
            )

    def _find_straight_raycast_candidate(
        self,
        x: int,
        y: int,
        direction: str,
        source_obj_id: str,
        ray_start: int = 20,
        ray_max: int = 50,
        ray_step: int = 2,
        allow_nearby_instrument: bool = False,
        relaxed_band: bool = False,
        target_run_px: Optional[int] = None,
        max_snap_shift_px: Optional[int] = None,
    ) -> Optional[tuple[int, int]]:
        dx, dy = DIRECTION_DELTA[direction]
        required_run_px = target_run_px if target_run_px is not None else self.turn_min_step
        for ray_dist in range(ray_start, ray_max, ray_step):
            rx = x + ray_dist * dx
            ry = y + ray_dist * dy
            sx, sy = self._snap_to_centerline(rx, ry, direction)
            same_axis_target = True
            if max_snap_shift_px is not None:
                if direction in ("UP", "DOWN"):
                    same_axis_target = abs(sx - x) <= max_snap_shift_px
                else:
                    same_axis_target = abs(sy - y) <= max_snap_shift_px
            if relaxed_band:
                is_pipe_target = (
                    same_axis_target
                    and _is_pipe_band(self.mask, rx, ry, direction, band_width=4)
                    and _has_line_of_sight_axis_band(
                        self.mask, sx, sy, direction, required_run_px, band_width=1
                    )
                )
            else:
                is_pipe_target = (
                    same_axis_target
                    and _is_pipe(self.mask, rx, ry)
                    and _has_line_of_sight_axis_band(
                        self.mask, rx, ry, direction, required_run_px, band_width=1
                    )
                )
                sx, sy = self._snap_to_centerline(rx, ry, direction)
            is_inline_target = self._is_inline_target(rx, ry)
            if not (is_pipe_target or is_inline_target):
                continue
            # Ray-cast is only for continuing pipe/inline objects, not terminal jumps.
            # Inline symbols are direct in-path objects, so they remain valid even
            # when an equipment bbox is nearby or overlapping the ray target.
            hit_terminal = self._check_terminals(rx, ry, direction, source_obj_id, look_ahead=0)
            if (
                hit_terminal
                and not is_inline_target
                and not self._terminal_is_pass_through(hit_terminal, direction)
            ):
                terminal_obj_id = hit_terminal[1] if len(hit_terminal) > 1 else None
                terminal_bbox = self._terminal_bbox_by_type(hit_terminal[0], terminal_obj_id)
                is_nearby_instrument = (
                    allow_nearby_instrument
                    and
                    hit_terminal[0] == TerminalType.INSTRUMENT_TAG.value
                    and terminal_bbox
                    and not _is_inside_bbox_exact(rx, ry, terminal_bbox)
                )
                if not is_nearby_instrument:
                    continue
            return sx, sy
        return None

    def _has_continuation_past_bbox(self, bbox: dict[str, int], direction: str) -> bool:
        dx, dy = DIRECTION_DELTA[direction]
        if direction == "UP":
            x = (bbox["x_min"] + bbox["x_max"]) // 2
            y = bbox["y_min"] - 2
        elif direction == "DOWN":
            x = (bbox["x_min"] + bbox["x_max"]) // 2
            y = bbox["y_max"] + 2
        elif direction == "LEFT":
            x = bbox["x_min"] - 2
            y = (bbox["y_min"] + bbox["y_max"]) // 2
        else:
            x = bbox["x_max"] + 2
            y = (bbox["y_min"] + bbox["y_max"]) // 2

        for offset in range(0, 80, 2):
            px = x + offset * dx
            py = y + offset * dy
            if _is_pipe_band(self.mask, px, py, direction, band_width=1):
                return True
        return False

    def _terminal_is_pass_through(self, terminal: tuple, direction: str) -> bool:
        terminal_type = terminal[0]
        terminal_obj_id = terminal[1] if len(terminal) > 1 else None
        if terminal_type != TerminalType.INSTRUMENT_TAG.value:
            return False
        bbox = self._terminal_bbox_by_type(terminal_type, terminal_obj_id)
        tag = next((t for t in self.instrument_tags if t.get("id", "") == (terminal_obj_id or "")), None)
        return bool(
            tag
            and tag.get("class_name") == "instrument dcs"
            and bbox
            and self._has_continuation_past_bbox(bbox, direction)
        )

    def _turn_hits_blocking_terminal(
        self,
        x: int,
        y: int,
        turn_dir: str,
        source_obj_id: str,
        distance: int = 70,
    ) -> bool:
        dx, dy = DIRECTION_DELTA[turn_dir]
        for step in range(1, distance + 1):
            tx = x + dx * step
            ty = y + dy * step
            terminal = self._check_terminals(tx, ty, turn_dir, source_obj_id, look_ahead=0)
            if terminal and not self._terminal_is_pass_through(terminal, turn_dir):
                return True
        return False

    def _turn_hits_terminal_type(
        self,
        x: int,
        y: int,
        turn_dir: str,
        source_obj_id: str,
        terminal_type: str,
        distance: int = 160,
    ) -> bool:
        dx, dy = DIRECTION_DELTA[turn_dir]
        for step in range(1, distance + 1):
            tx = x + dx * step
            ty = y + dy * step
            terminal = self._check_terminals(tx, ty, turn_dir, source_obj_id, look_ahead=0)
            if terminal and terminal[0] == terminal_type:
                return True
        return False

    def _is_page_connection_source(self, source_obj_id: str) -> bool:
        return any(pc.get("id", "") == source_obj_id for pc in self.page_connections)

    def _check_junction_marker(self, x: int, y: int, source_obj_id: str) -> Optional[tuple[str, int, int]]:
        for marker in self.junction_markers:
            marker_id = str(marker.get("id", ""))
            if marker_id == source_obj_id:
                continue
            bbox = marker.get("bbox")
            if bbox and _check_bbox_hit(x, y, bbox, margin=12):
                cx = (int(bbox["x_min"]) + int(bbox["x_max"])) // 2
                cy = (int(bbox["y_min"]) + int(bbox["y_max"])) // 2
                return marker_id, cx, cy
        return None

    def _find_branch_terminal_side_turn(
        self,
        x: int,
        y: int,
        direction: str,
        source_obj_id: str,
        probe_px: int = 8,
        terminal_distance: int = 90,
    ) -> Optional[tuple[int, int, str]]:
        if not source_obj_id.startswith("branch_"):
            return None
        dx, dy = DIRECTION_DELTA[direction]
        for offset in range(0, probe_px + 1):
            px = x + dx * offset
            py = y + dy * offset
            if not _is_pipe_band(self.mask, px, py, direction, band_width=1):
                continue
            for turn_dir in (TURN_LEFT[direction], TURN_RIGHT[direction]):
                if not _has_connected_side_pipe(self.mask, px, py, turn_dir, min_run=25):
                    continue
                tdx, tdy = DIRECTION_DELTA[turn_dir]
                for dist in range(5, terminal_distance + 1, 5):
                    tx = px + tdx * dist
                    ty = py + tdy * dist
                    terminal = self._check_terminals(tx, ty, turn_dir, source_obj_id, look_ahead=0)
                    if not terminal:
                        continue
                    terminal_id = terminal[1] if len(terminal) > 1 else ""
                    terminal_class = ""
                    if terminal[0] == TerminalType.PAGE_CONNECTION.value:
                        for item in self.page_connections:
                            if item.get("id") == terminal_id:
                                terminal_class = str(item.get("class_name", ""))
                                break
                    if terminal_id.startswith("branch_") or terminal_class == "node":
                        return (px, py, turn_dir)
                    break
        return None

    def _has_turn_gap(self, x: int, y: int, turn_dir: str, source_obj_id: str) -> bool:
        dx, dy = DIRECTION_DELTA[turn_dir]
        for dist in range(5, 60, 2):
            tx = x + dist * dx
            ty = y + dist * dy
            is_pipe_target = _is_pipe(self.mask, tx, ty) and _has_line_of_sight_axis_band(
                self.mask, tx, ty, turn_dir, self.turn_min_step, band_width=1
            )
            is_inline_target = self._is_inline_target(tx, ty)
            if is_inline_target and not _has_line_of_sight_axis_band(
                self.mask, x, y, turn_dir, self.turn_min_step, band_width=1
            ):
                continue
            if not (is_pipe_target or is_inline_target):
                continue
            terminal = self._check_terminals(tx, ty, turn_dir, source_obj_id, look_ahead=0)
            if terminal and not self._terminal_is_pass_through(terminal, turn_dir):
                continue
            return True
        return False

    def _find_nearby_turn_candidates(
        self,
        x: int,
        y: int,
        direction: str,
        source_obj_id: str,
        probe_px: int = 8,
    ) -> list[tuple[int, int, str]]:
        dx, dy = DIRECTION_DELTA[direction]
        left_dir = TURN_LEFT[direction]
        right_dir = TURN_RIGHT[direction]
        los_candidates: list[tuple[int, int, str]] = []
        gap_candidates: list[tuple[int, int, str]] = []
        seen_turns: set[tuple[int, int, str]] = set()

        for offset in range(0, probe_px + 1):
            px = x + offset * dx
            py = y + offset * dy
            if not _is_pipe_band(self.mask, px, py, direction, band_width=1):
                continue
            left_ok = _has_line_of_sight_band_narrow(self.mask, px, py, left_dir, self.min_step)
            right_ok = _has_line_of_sight_band_narrow(self.mask, px, py, right_dir, self.min_step)
            if left_ok and right_ok:
                continue
            for turn_dir, ok in (
                (left_dir, left_ok),
                (right_dir, right_ok),
            ):
                if not ok:
                    continue
                key = (px, py, turn_dir)
                if key in seen_turns:
                    continue
                los_candidates.append(key)
                seen_turns.add(key)
            left_gap = False if left_ok else self._has_turn_gap(px, py, left_dir, source_obj_id)
            right_gap = False if right_ok else self._has_turn_gap(px, py, right_dir, source_obj_id)
            for turn_dir, ok in (
                (left_dir, left_gap),
                (right_dir, right_gap),
            ):
                if not ok:
                    continue
                key = (px, py, turn_dir)
                if key in seen_turns:
                    continue
                gap_candidates.append(key)
                seen_turns.add(key)

        return los_candidates + gap_candidates

    def _nearest_candidate_point(self, x: int, y: int, candidates: list[tuple]) -> Optional[tuple]:
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c[0] - x) + abs(c[1] - y))

    def _has_bidirectional_turn_leg(self, x: int, y: int, turn_dir: str, distance: int = 5) -> bool:
        """Return true when a turn candidate is actually a tee-through leg."""
        opposite_dir = OPPOSITE[turn_dir]
        connected_turn_min = max(25, self.straight_min_step)
        if not (
            self._has_connected_side_path(x, y, turn_dir, connected_turn_min)
            and self._has_connected_side_path(x, y, opposite_dir, connected_turn_min)
        ):
            return False
        return (
            _has_line_of_sight_band_narrow(self.mask, x, y, turn_dir, distance)
            and _has_line_of_sight_band_narrow(self.mask, x, y, opposite_dir, distance)
        )

    def _connected_continuation_dirs(self, x: int, y: int, direction: str) -> set[str]:
        dirs: set[str] = set()
        if _has_line_of_sight_axis_band(self.mask, x, y, direction, self.min_step, band_width=1):
            dirs.add(direction)
        left_dir = TURN_LEFT[direction]
        right_dir = TURN_RIGHT[direction]
        connected_turn_min = max(25, self.straight_min_step)
        if self._has_connected_side_path(x, y, left_dir, connected_turn_min):
            dirs.add(left_dir)
        if self._has_connected_side_path(x, y, right_dir, connected_turn_min):
            dirs.add(right_dir)
        return dirs

    def _find_bidirectional_side_junction(
        self,
        x: int,
        y: int,
        direction: str,
        search_px: int = 8,
    ) -> Optional[tuple[int, int]]:
        """Find a nearby point where both side directions are connected.

        The walker can overshoot a black-dot tee by a few pixels before forward
        blocks.  Search backward and forward along the current line so a true
        bidirectional tee is not downgraded to a one-sided elbow.
        """
        dx, dy = DIRECTION_DELTA[direction]
        left_dir = TURN_LEFT[direction]
        right_dir = TURN_RIGHT[direction]
        connected_turn_min = max(25, self.straight_min_step)
        candidates: list[tuple[int, int]] = []
        for offset in range(-search_px, search_px + 1):
            px = x + dx * offset
            py = y + dy * offset
            if not _is_pipe_band(self.mask, px, py, direction, band_width=1):
                continue
            if (
                self._has_connected_side_path(px, py, left_dir, connected_turn_min)
                and self._has_connected_side_path(px, py, right_dir, connected_turn_min)
            ):
                candidates.append((px, py))
        if not candidates:
            return None
        return min(candidates, key=lambda p: abs(p[0] - x) + abs(p[1] - y))

    def _is_backtrack_turn(
        self,
        result: TraceResult,
        x: int,
        y: int,
        turn_dir: str,
        tolerance: int = 10,
    ) -> bool:
        opposite_dir = OPPOSITE.get(turn_dir)
        if opposite_dir is None:
            return False
        for segment in result.segments:
            if segment.direction != opposite_dir:
                continue
            if turn_dir in ("LEFT", "RIGHT"):
                if abs(segment.y1 - y) > tolerance and abs(segment.y2 - y) > tolerance:
                    continue
                seg_min = min(segment.x1, segment.x2)
                seg_max = max(segment.x1, segment.x2)
                if seg_min - tolerance <= x <= seg_max + tolerance:
                    return True
            else:
                if abs(segment.x1 - x) > tolerance and abs(segment.x2 - x) > tolerance:
                    continue
                seg_min = min(segment.y1, segment.y2)
                seg_max = max(segment.y1, segment.y2)
                if seg_min - tolerance <= y <= seg_max + tolerance:
                    return True
        return False

    def trace(self, start_x: int, start_y: int, start_dir: str,
              source_obj_id: str = "") -> TraceResult:
        """Trace from port to terminal. source_obj_id is excluded from terminal checks."""
        result = TraceResult()
        result.terminal_type = None

        x, y = start_x, start_y
        direction = start_dir.upper()
        if direction not in DIRECTION_DELTA:
            # Map alternate names
            alt_map = {"TOP": "UP", "BOTTOM": "DOWN"}
            direction = alt_map.get(direction, direction)
        dx, dy = DIRECTION_DELTA[direction]

        # Verify start point is on pipe
        if not _is_pipe(self.mask, x, y):
            # Try walking forward a few pixels to find pipe
            for step in range(1, 15):
                nx = x + step * dx
                ny = y + step * dy
                if _is_pipe(self.mask, nx, ny):
                    x, y = nx, ny
                    break
            else:
                result.status = "no_pipe"
                result.terminal_type = TerminalType.NO_PIPE.value
                return result

        # Ensure we start centered on the line before any walk.
        x, y = self._snap_to_centerline(x, y, direction)
        seg_start_x, seg_start_y = x, y
        steps = 0
        state_counts: dict[tuple[int, int, str], int] = {}
        exact_positions: set[tuple[int, int, str]] = set()
        exact_position_repeats: dict[tuple[int, int, str], int] = {}

        # Walk clear of source symbol before first terminal check
        warmup_steps = 20
        for _ in range(warmup_steps):
            x += dx
            y += dy
            if not _is_pipe(self.mask, x, y):
                x -= dx
                y -= dy
                break
            if 0 <= y < self.h and 0 <= x < self.w:
                self.visited[y, x] = 1
        x, y = self._snap_to_centerline(x, y, direction)

        while steps < self.max_steps:
            steps += 1

            # Mark visited
            if 0 <= y < self.h and 0 <= x < self.w:
                self.visited[y, x] = 1

            # Keep walker centered before evaluating/advancing.
            x, y = self._snap_to_centerline(x, y, direction)
            state_key = (int(round(x / 3)), int(round(y / 3)), direction)
            state_counts[state_key] = state_counts.get(state_key, 0) + 1
            exact_key = (x, y, direction)
            if exact_key in exact_positions:
                exact_position_repeats[exact_key] = exact_position_repeats.get(exact_key, 0) + 1
            else:
                exact_positions.add(exact_key)
                exact_position_repeats[exact_key] = 1
            if state_counts[state_key] > 3 or exact_position_repeats[exact_key] > 2:
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                result.terminal_type = TerminalType.DEAD_END.value
                result.terminal_x, result.terminal_y = x, y
                break

            # Check for inline symbols (valve, reducer) — these are
            # traversed through, not terminals.  Compute the exit position
            # on the far side of the inline object (or overlapping group).
            inline_hits = self._find_inline_overlap(x, y, direction)
            if inline_hits:
                for hit in inline_hits:
                    result.hits.append(InlineHit(
                        class_name=hit.get("class_name", "unknown"),
                        x=x, y=y,
                    ))
                psv_exit = (
                    self._compute_pressure_relief_exit(x, y, direction, inline_hits)
                    if self._is_pressure_relief_group(inline_hits)
                    else None
                )
                if psv_exit is not None:
                    far_x, far_y, exit_dir = psv_exit
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    result.turns.append((x, y, exit_dir))
                    x, y = self._snap_to_centerline(far_x, far_y, exit_dir)
                    direction = exit_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    seg_start_x, seg_start_y = x, y
                    continue
                # Exit position: far edge of the furthest overlapping inline obj
                far_x, far_y = self._compute_inline_exit(x, y, direction, inline_hits)
                if _is_pipe(self.mask, far_x, far_y) or _is_pipe_band(self.mask, far_x, far_y, direction):
                    x, y = far_x, far_y
                else:
                    # Fallback: jump by the object extent
                    extent = self._inline_group_extent(direction, inline_hits)
                    x += dx * (extent + 10)
                    y += dy * (extent + 10)
                continue

            # Look ahead — what's in front?
            forward_ok = _has_line_of_sight_axis_band(self.mask, x, y, direction, self.min_step, band_width=1)

            current_leg_len = max(abs(x - seg_start_x), abs(y - seg_start_y))
            if source_obj_id.startswith("branch_") and current_leg_len > 60:
                junction_obj_id = self._check_junction_marker(x, y, source_obj_id)
                if junction_obj_id:
                    left_dir = TURN_LEFT[direction]
                    right_dir = TURN_RIGHT[direction]
                    connected_turn_min = max(25, self.straight_min_step)
                    left_connected = self._has_connected_side_path(x, y, left_dir, connected_turn_min)
                    right_connected = self._has_connected_side_path(x, y, right_dir, connected_turn_min)
                    if left_connected and right_connected:
                        marker_id, marker_x, marker_y = junction_obj_id
                        self._append_segment(result, seg_start_x, seg_start_y, marker_x, marker_y, direction)
                        result.terminal_type = TerminalType.TEE_JUNCTION.value
                        result.terminal_x, result.terminal_y = marker_x, marker_y
                        result.terminal_obj_id = marker_id
                        break

            junction_obj_id = (
                self._check_junction_marker(x, y, source_obj_id)
                if result.turns
                and current_leg_len <= 200
                and not source_obj_id.startswith("equip_")
                else None
            )
            if (
                junction_obj_id
                and not forward_ok
                and len(self._connected_continuation_dirs(x, y, direction)) > 1
            ):
                marker_id, marker_x, marker_y = junction_obj_id
                self._append_segment(result, seg_start_x, seg_start_y, marker_x, marker_y, direction)
                result.terminal_type = TerminalType.TEE_JUNCTION.value
                result.terminal_x, result.terminal_y = marker_x, marker_y
                result.terminal_obj_id = marker_id
                break

            if forward_ok:
                side_turn = self._find_branch_terminal_side_turn(x, y, direction, source_obj_id)
                if side_turn is not None:
                    tx, ty, turn_dir = side_turn
                    self._append_segment(result, seg_start_x, seg_start_y, tx, ty, direction)
                    result.turns.append((tx, ty, turn_dir))
                    x = tx + DIRECTION_DELTA[turn_dir][0] * self.min_step
                    y = ty + DIRECTION_DELTA[turn_dir][1] * self.min_step
                    x, y = self._snap_to_centerline(x, y, turn_dir)
                    direction = turn_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    seg_start_x, seg_start_y = x, y
                    continue
                # Walk forward
                for _ in range(self.min_step):
                    x += dx
                    y += dy
                    if 0 <= y < self.h and 0 <= x < self.w:
                        self.visited[y, x] = 1
                x, y = self._snap_to_centerline(x, y, direction)
                continue

            terminal = self._check_terminals(x, y, direction, source_obj_id, look_ahead=0)
            if terminal and not self._terminal_is_pass_through(terminal, direction):
                raycast = self._find_straight_raycast_candidate(x, y, direction, source_obj_id)
                terminal_obj_id = terminal[1] if len(terminal) > 1 else None
                terminal_bbox = self._terminal_bbox_by_type(terminal[0], terminal_obj_id)
                if (
                    raycast is None
                    and terminal[0] == TerminalType.INSTRUMENT_TAG.value
                    and terminal_bbox
                    and not _is_inside_bbox_exact(x, y, terminal_bbox)
                ):
                    raycast = self._find_straight_raycast_candidate(
                        x,
                        y,
                        direction,
                        source_obj_id,
                        allow_nearby_instrument=True,
                        relaxed_band=True,
                    )
                if (
                    raycast is not None
                    and terminal[0] == TerminalType.INSTRUMENT_TAG.value
                    and terminal_bbox
                    and not _is_inside_bbox_exact(x, y, terminal_bbox)
                ):
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    x, y = raycast
                    seg_start_x, seg_start_y = x, y
                    continue
                result.terminal_type = terminal[0]
                result.terminal_obj_id = terminal[1] if len(terminal) > 1 else None
                x, y = self._retreat_from_terminal_bbox(x, y, direction, result.terminal_type, result.terminal_obj_id)
                terminal_inline_hits = self._find_inline_overlap(x, y, direction)
                for hit in terminal_inline_hits:
                    hit_key = (hit.get("class_name", "unknown"), x, y)
                    if not any((h.class_name, h.x, h.y) == hit_key for h in result.hits):
                        result.hits.append(InlineHit(
                            class_name=hit.get("class_name", "unknown"),
                            x=x, y=y,
                        ))
                result.terminal_x, result.terminal_y = x, y
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                break

            inline_hits = self._find_inline_overlap(x, y, direction)
            if inline_hits:
                for hit in inline_hits:
                    result.hits.append(InlineHit(
                        class_name=hit.get("class_name", "unknown"),
                        x=x, y=y,
                    ))
                psv_exit = (
                    self._compute_pressure_relief_exit(x, y, direction, inline_hits)
                    if self._is_pressure_relief_group(inline_hits)
                    else None
                )
                if psv_exit is not None:
                    far_x, far_y, exit_dir = psv_exit
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    result.turns.append((x, y, exit_dir))
                    x, y = self._snap_to_centerline(far_x, far_y, exit_dir)
                    direction = exit_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    seg_start_x, seg_start_y = x, y
                    continue
                far_x, far_y = self._compute_inline_exit(x, y, direction, inline_hits)
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                if _is_pipe(self.mask, far_x, far_y) or _is_pipe_band(self.mask, far_x, far_y, direction):
                    x, y = far_x, far_y
                else:
                    extent = self._inline_group_extent(direction, inline_hits)
                    x += dx * (extent + 10)
                    y += dy * (extent + 10)
                seg_start_x, seg_start_y = x, y
                continue

            turn_candidates = self._find_nearby_turn_candidates(x, y, direction, source_obj_id)
            left_dir = TURN_LEFT[direction]
            right_dir = TURN_RIGHT[direction]
            left_ok = _has_line_of_sight_band_narrow(self.mask, x, y, left_dir, self.min_step)
            right_ok = _has_line_of_sight_band_narrow(self.mask, x, y, right_dir, self.min_step)
            left_exact = _has_line_of_sight_axis_exact(
                self.mask, x, y, left_dir, self.straight_min_step
            )
            right_exact = _has_line_of_sight_axis_exact(
                self.mask, x, y, right_dir, self.straight_min_step
            )
            connected_turn_min = max(25, self.straight_min_step)
            left_connected = self._has_connected_side_path(x, y, left_dir, connected_turn_min)
            right_connected = self._has_connected_side_path(x, y, right_dir, connected_turn_min)
            strict_straight_raycast = False

            side_junction = self._find_bidirectional_side_junction(x, y, direction)
            if side_junction is not None:
                raycast = self._find_straight_raycast_candidate(x, y, direction, source_obj_id)
                if raycast is not None:
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    x, y = raycast
                    seg_start_x, seg_start_y = x, y
                    continue
                jx, jy = side_junction
                self._append_segment(result, seg_start_x, seg_start_y, jx, jy, direction)
                result.terminal_type = TerminalType.TEE_JUNCTION.value
                result.terminal_x, result.terminal_y = jx, jy
                break

            # Once a trace has already taken multiple elbows, a true
            # bidirectional side branch is a tee terminal.  Gap-only side
            # candidates can be nearby text/leader strokes, so require physical
            # side-pipe connectivity before promoting the point to a tee.
            turn_dirs = {c[2] for c in turn_candidates if c[2] in (left_dir, right_dir)}
            if len(result.turns) >= 2 and len(turn_dirs) > 1:
                connected_turn_dirs = set()
                if left_connected:
                    connected_turn_dirs.add(left_dir)
                if right_connected:
                    connected_turn_dirs.add(right_dir)
                if len(connected_turn_dirs) > 1:
                    raycast = self._find_straight_raycast_candidate(x, y, direction, source_obj_id)
                    if raycast is not None:
                        self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                        x, y = raycast
                        seg_start_x, seg_start_y = x, y
                        continue
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    result.terminal_type = TerminalType.TEE_JUNCTION.value
                    result.terminal_x, result.terminal_y = x, y
                    break
                turn_candidates = [c for c in turn_candidates if c[2] in connected_turn_dirs]
                strict_straight_raycast = True

            # A directly connected elbow should turn before any straight
            # ray-cast jump. Require a long adjacent side-pipe run so short
            # flange strokes do not become turns.
            raycast_kwargs = (
                {"target_run_px": self.straight_min_step, "max_snap_shift_px": 4}
                if strict_straight_raycast else {}
            )
            raycast = self._find_straight_raycast_candidate(
                x, y, direction, source_obj_id, **raycast_kwargs
            )
            exact_turn_dir = None
            if left_connected and not right_connected:
                exact_turn_dir = left_dir
            elif right_connected and not left_connected:
                exact_turn_dir = right_dir
            if exact_turn_dir is not None:
                if self._is_backtrack_turn(result, x, y, exact_turn_dir):
                    exact_turn_dir = None
                else:
                    turn_hits_blocking_terminal = self._turn_hits_blocking_terminal(
                        x, y, exact_turn_dir, source_obj_id
                    )
                    if raycast is None and turn_hits_blocking_terminal:
                        raycast = self._find_straight_raycast_candidate(
                            x,
                            y,
                            direction,
                            source_obj_id,
                            ray_start=5,
                            ray_max=50,
                            ray_step=1,
                            relaxed_band=True,
                        )
                    if (
                        self._is_page_connection_source(source_obj_id)
                        and self._turn_hits_terminal_type(
                            x,
                            y,
                            exact_turn_dir,
                            source_obj_id,
                            TerminalType.PAGE_CONNECTION.value,
                        )
                    ):
                        self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                        result.terminal_type = TerminalType.TEE_JUNCTION.value
                        result.terminal_x, result.terminal_y = x, y
                        break
                    if (
                        raycast is not None
                        and turn_hits_blocking_terminal
                    ):
                        self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                        x, y = raycast
                        seg_start_x, seg_start_y = x, y
                        continue
                    if self._has_bidirectional_turn_leg(x, y, exact_turn_dir):
                        self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                        result.terminal_type = TerminalType.TEE_JUNCTION.value
                        result.terminal_x, result.terminal_y = x, y
                        break
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    result.turns.append((x, y, exact_turn_dir))
                    direction = exact_turn_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    turn_x, turn_y = x, y
                    x, y = self._enter_turn_leg(turn_x, turn_y, direction)
                    seg_start_x, seg_start_y = self._turn_segment_start(
                        turn_x, turn_y, x, y, direction
                    )
                    continue
            candidate_turn_dirs = {c[2] for c in turn_candidates if c[2] in (left_dir, right_dir)}
            if len(candidate_turn_dirs) == 1 and direction == "UP":
                turn_dir = next(iter(candidate_turn_dirs))
                turn = self._nearest_candidate_point(
                    x,
                    y,
                    [c for c in turn_candidates if c[2] == turn_dir],
                )
                straight_missing_or_far = (
                    raycast is None
                    or max(abs(raycast[0] - x), abs(raycast[1] - y)) > 40
                )
                turn_target = (
                    self._find_straight_raycast_candidate(
                        turn[0],
                        turn[1],
                        turn_dir,
                        source_obj_id,
                        ray_start=5,
                        ray_max=80,
                        ray_step=1,
                        relaxed_band=True,
                    )
                    if turn is not None else None
                )
                if (
                    turn is not None
                    and turn_dir == left_dir
                    and straight_missing_or_far
                    and turn_target is not None
                    and _has_line_of_sight_axis_band(
                        self.mask,
                        turn_target[0],
                        turn_target[1],
                        turn_dir,
                        40,
                        band_width=1,
                    )
                    and not self._is_backtrack_turn(result, turn[0], turn[1], turn_dir)
                ):
                    tx, ty, turn_dir = turn
                    if self._has_bidirectional_turn_leg(tx, ty, turn_dir):
                        self._append_segment(result, seg_start_x, seg_start_y, tx, ty, direction)
                        result.terminal_type = TerminalType.TEE_JUNCTION.value
                        result.terminal_x, result.terminal_y = tx, ty
                        break
                    self._append_segment(result, seg_start_x, seg_start_y, tx, ty, direction)
                    result.turns.append((tx, ty, turn_dir))
                    direction = turn_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    x, y = self._enter_turn_leg(tx, ty, direction)
                    seg_start_x, seg_start_y = self._turn_segment_start(
                        tx, ty, x, y, direction
                    )
                    continue

            if raycast is not None:
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                x, y = raycast
                seg_start_x, seg_start_y = x, y
                continue

            relaxed_straight = self._find_straight_raycast_candidate(
                x,
                y,
                direction,
                source_obj_id,
                ray_start=5,
                ray_max=50,
                ray_step=1,
                relaxed_band=True,
                **raycast_kwargs,
            )
            if relaxed_straight is not None:
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                x, y = relaxed_straight
                seg_start_x, seg_start_y = x, y
                continue

            if left_ok and right_ok:
                extended_raycast = self._find_straight_raycast_candidate(
                    x, y, direction, source_obj_id, ray_max=50, ray_step=1
                )
                if extended_raycast is not None:
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    x, y = extended_raycast
                    seg_start_x, seg_start_y = x, y
                    continue
                left_raycast = self._find_straight_raycast_candidate(x, y, left_dir, source_obj_id)
                right_raycast = self._find_straight_raycast_candidate(x, y, right_dir, source_obj_id)
                if (left_raycast is not None) != (right_raycast is not None):
                    turn_dir = left_dir if left_raycast is not None else right_dir
                    turn_target = left_raycast if left_raycast is not None else right_raycast
                    if self._is_backtrack_turn(result, x, y, turn_dir):
                        turn_target = None
                if (left_raycast is not None) != (right_raycast is not None) and turn_target is not None:
                    self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                    if self._has_bidirectional_turn_leg(x, y, turn_dir):
                        result.terminal_type = TerminalType.TEE_JUNCTION.value
                        result.terminal_x, result.terminal_y = x, y
                        break
                    turn_x, turn_y = x, y
                    result.turns.append((turn_x, turn_y, turn_dir))
                    x, y = turn_target
                    direction = turn_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    if max(abs(x - turn_x), abs(y - turn_y)) <= 50:
                        seg_start_x, seg_start_y = self._turn_segment_start(
                            turn_x, turn_y, x, y, direction
                        )
                    else:
                        seg_start_x, seg_start_y = x, y
                    continue
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                if (
                    direction == "UP"
                    and self._is_page_connection_source(source_obj_id)
                    and _has_line_of_sight_axis_exact(self.mask, x, y, left_dir, self.straight_min_step)
                    and _has_line_of_sight_axis_exact(self.mask, x, y, right_dir, self.straight_min_step)
                ):
                    turn_dir = left_dir
                    result.turns.append((x, y, turn_dir))
                    direction = turn_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    x, y = self._enter_turn_leg(x, y, direction)
                    seg_start_x, seg_start_y = self._turn_segment_start(
                        result.turns[-1][0], result.turns[-1][1], x, y, direction
                    )
                    continue
                result.terminal_type = TerminalType.TEE_JUNCTION.value
                result.terminal_x, result.terminal_y = x, y
                break

            turn = self._nearest_candidate_point(x, y, turn_candidates)
            if turn is not None:
                tx, ty, turn_dir = turn
                if self._is_backtrack_turn(result, tx, ty, turn_dir):
                    turn = None
            if turn is not None:
                tx, ty, turn_dir = turn
                turn_leg_len = max(abs(tx - seg_start_x), abs(ty - seg_start_y))
                self._append_segment(result, seg_start_x, seg_start_y, tx, ty, direction)
                candidate_dirs = {c[2] for c in turn_candidates}
                if len(result.turns) == 1 and turn_leg_len < 200 and len(candidate_dirs) > 1:
                    if (
                        source_obj_id.startswith("equip_")
                        and turn_leg_len < 20
                        and result.trace_length_px > 200
                    ):
                        result.terminal_type = TerminalType.DEAD_END.value
                        result.terminal_x, result.terminal_y = tx, ty
                        break
                    result.terminal_type = TerminalType.TEE_JUNCTION.value
                    result.terminal_x, result.terminal_y = tx, ty
                    break
                if self._has_bidirectional_turn_leg(tx, ty, turn_dir):
                    result.terminal_type = TerminalType.TEE_JUNCTION.value
                    result.terminal_x, result.terminal_y = tx, ty
                    break
                entered_x = tx + DIRECTION_DELTA[turn_dir][0] * self.min_step
                entered_y = ty + DIRECTION_DELTA[turn_dir][1] * self.min_step
                x, y = self._snap_to_centerline(entered_x, entered_y, turn_dir)
                result.turns.append((tx, ty, turn_dir))
                direction = turn_dir
                dx, dy = DIRECTION_DELTA[direction]
                seg_start_x, seg_start_y = self._turn_segment_start(
                    tx, ty, x, y, direction
                )
                continue

            self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
            if self._is_sheet_edge(x, y, direction):
                result.terminal_type = TerminalType.SHEET_EDGE.value
                result.terminal_x, result.terminal_y = x, y
            else:
                result.terminal_type = TerminalType.DEAD_END.value
                result.terminal_x, result.terminal_y = x, y
            break

        if steps >= self.max_steps:
            result.terminal_type = "max_steps"
            result.terminal_x, result.terminal_y = x, y

        self._anchor_close_turn_segments(result)
        return result

    def _check_terminals(self, x: int, y: int, direction: str,
                         source_obj_id: str = "", look_ahead: int = 0) -> Optional[tuple]:
        """Check if position or area ahead is a terminal. Returns (type, obj_id) or None."""
        dx, dy = DIRECTION_DELTA.get(direction, (0, 0))
        current_page_margin = 2
        current_tag_margin = 4
        current_dcs_margin = 6
        current_equipment_margin = 4
        ahead_page_margin = 2
        ahead_equipment_margin = 4
        ahead_tag_margin = 2

        # --- First: check if we're *already inside* any terminal bbox ---
        # This catches cases where the mask extension draws the tracer
        # into a bbox but the directional scan would miss it.

        # Page connections (skip source)
        for pc in self.page_connections:
            pc_id = pc.get("id", "")
            if pc_id == source_obj_id:
                continue
            if _check_bbox_hit(x, y, pc["bbox"], margin=current_page_margin):
                return (TerminalType.PAGE_CONNECTION.value, pc_id)

        # Exact equipment hit wins over nearby labels.
        for eq in self.equipment_objects:
            eq_id = eq.get("id", "")
            if eq_id == source_obj_id:
                continue
            eq_bbox = eq["bbox"]
            if _is_inside_bbox_exact(x, y, eq_bbox):
                return (TerminalType.EQUIPMENT.value, eq_id)

        # Instrument tags can sit just off the pipe, but dense P&IDs need
        # small margins so nearby labels do not steal the current trace.
        for tag in self.instrument_tags:
            margin = current_dcs_margin if tag.get("class_name") == "instrument dcs" else current_tag_margin
            if _check_bbox_hit(x, y, tag["bbox"], margin=margin):
                return (TerminalType.INSTRUMENT_TAG.value, tag.get("id", ""))

        # Expanded equipment hit catches pipe entering large equipment bboxes
        # without letting nearby objects interfere with dense look-ahead.
        for eq in self.equipment_objects:
            eq_id = eq.get("id", "")
            if eq_id == source_obj_id:
                continue
            eq_bbox = eq["bbox"]
            if (eq_bbox["x_min"] - current_equipment_margin <= x <= eq_bbox["x_max"] + current_equipment_margin and
                eq_bbox["y_min"] - current_equipment_margin <= y <= eq_bbox["y_max"] + current_equipment_margin):
                return (TerminalType.EQUIPMENT.value, eq_id)

        # --- Second: directional scan ahead ---
        for offset in range(0, look_ahead, 5):
            tx = x + offset * dx
            ty = y + offset * dy

            # Check page connections (skip the source)
            for pc in self.page_connections:
                pc_id = pc.get("id", "")
                if pc_id == source_obj_id:
                    continue
                if _check_bbox_hit(tx, ty, pc["bbox"], margin=ahead_page_margin):
                    return (TerminalType.PAGE_CONNECTION.value, pc_id)

            # Check equipment with only a small bbox expansion.
            for eq in self.equipment_objects:
                eq_id = eq.get("id", "")
                if eq_id == source_obj_id:
                    continue
                eq_bbox = eq["bbox"]
                if (eq_bbox["x_min"] - ahead_equipment_margin <= tx <= eq_bbox["x_max"] + ahead_equipment_margin and
                    eq_bbox["y_min"] - ahead_equipment_margin <= ty <= eq_bbox["y_max"] + ahead_equipment_margin):
                    return (TerminalType.EQUIPMENT.value, eq_id)

            # Check instrument tags
            for tag in self.instrument_tags:
                if _check_bbox_hit(tx, ty, tag["bbox"], margin=ahead_tag_margin):
                    return (TerminalType.INSTRUMENT_TAG.value, tag.get("id", ""))

        return None

    def _check_inline(self, x: int, y: int) -> Optional[InlineHit]:
        """Check if current position is inside an inline symbol (valve, etc.)."""
        for sym in self._inline_symbols:
            bbox = sym["bbox"]
            if _check_bbox_hit(x, y, bbox):
                return InlineHit(
                    class_name=sym.get("class_name", "unknown"),
                    x=x, y=y, bbox=bbox,
                )
        return None

    def _find_inline_overlap(self, x: int, y: int,
                             direction: str) -> list[dict]:
        """Return ALL inline symbols whose bbox contains (x,y).

        For overlapping inline symbols, returns the entire overlapping group
        so the tracer can jump past the furthest extent.
        """
        hits = []
        for sym in self._inline_symbols:
            bbox = sym["bbox"]
            if _check_bbox_hit(x, y, bbox, margin=2):
                hits.append(sym)

        if len(hits) <= 1:
            return hits

        # Merge overlapping bboxes into groups
        def _overlap(a: dict, b: dict) -> bool:
            ab, bb = a["bbox"], b["bbox"]
            return not (
                ab["x_max"] < bb["x_min"] or bb["x_max"] < ab["x_min"]
                or ab["y_max"] < bb["y_min"] or bb["y_max"] < ab["y_min"]
            )

        groups = []
        used = set()
        for i, h in enumerate(hits):
            if i in used:
                continue
            group = [h]
            used.add(i)
            for j in range(i + 1, len(hits)):
                if j in used:
                    continue
                if any(_overlap(g, hits[j]) for g in group):
                    group.append(hits[j])
                    used.add(j)
            groups.append(group)

        # Return the group containing our hit — or all hits
        for g in groups:
            for h in g:
                if _check_bbox_hit(x, y, h["bbox"], margin=2):
                    return g
        return hits

    def _inline_group_extent(self, direction: str,
                             group: list[dict]) -> int:
        """Return the pixel extent of an overlapping inline group in the travel
        direction (width for LEFT/RIGHT, height for UP/DOWN)."""
        bboxes = [s["bbox"] for s in group]
        if direction in ("LEFT", "RIGHT"):
            x_min = min(b["x_min"] for b in bboxes)
            x_max = max(b["x_max"] for b in bboxes)
            return x_max - x_min
        else:
            y_min = min(b["y_min"] for b in bboxes)
            y_max = max(b["y_max"] for b in bboxes)
            return y_max - y_min

    def _compute_inline_exit(self, x: int, y: int, direction: str,
                             group: list[dict]) -> tuple[int, int]:
        """Compute exit coordinates just past the far edge of the inline group.

        For LEFT/RIGHT travel, exits at the same y with x just past the
        far bbox edge.  For UP/DOWN, exits at the same x with y just past.
        """
        margin = 6
        bboxes = [s["bbox"] for s in group]
        if direction == "LEFT":
            far_x = min(b["x_min"] for b in bboxes) - margin
            return (max(0, far_x), y)
        elif direction == "RIGHT":
            far_x = max(b["x_max"] for b in bboxes) + margin
            return (min(self.w - 1, far_x), y)
        elif direction == "UP":
            far_y = min(b["y_min"] for b in bboxes) - margin
            return (x, max(0, far_y))
        else:  # DOWN
            far_y = max(b["y_max"] for b in bboxes) + margin
            return (x, min(self.h - 1, far_y))

    def _is_pressure_relief_group(self, group: list[dict]) -> bool:
        return any(
            str(sym.get("class_name", "")).replace("_", " ").lower() == "pressure relief valve"
            for sym in group
        )

    def _score_pipe_exit(
        self,
        x: int,
        y: int,
        direction: str,
        distance: int = 80,
    ) -> tuple[int, int, int]:
        dx, dy = DIRECTION_DELTA[direction]
        score = 0
        first_pipe: Optional[tuple[int, int]] = None
        for step in range(0, distance):
            px = x + dx * step
            py = y + dy * step
            if _is_pipe_band(self.mask, px, py, direction, band_width=1):
                score += 1
                if first_pipe is None:
                    first_pipe = self._snap_to_centerline(px, py, direction)
        if first_pipe is None:
            first_pipe = (x, y)
        return (score, first_pipe[0], first_pipe[1])

    def _compute_pressure_relief_exit(
        self,
        x: int,
        y: int,
        direction: str,
        group: list[dict],
    ) -> Optional[tuple[int, int, str]]:
        """PSV ports are typically perpendicular: bottom plus left/right."""
        bboxes = [s["bbox"] for s in group]
        x_min = min(b["x_min"] for b in bboxes)
        x_max = max(b["x_max"] for b in bboxes)
        y_min = min(b["y_min"] for b in bboxes)
        y_max = max(b["y_max"] for b in bboxes)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        margin = 6

        if direction in ("UP", "DOWN"):
            candidates = [
                (max(0, x_min - margin), cy, "LEFT"),
                (min(self.w - 1, x_max + margin), cy, "RIGHT"),
            ]
        else:
            candidates = [
                (cx, min(self.h - 1, y_max + margin), "DOWN"),
                (cx, max(0, y_min - margin), "UP"),
            ]

        scored = [
            (*self._score_pipe_exit(px, py, exit_dir), exit_dir)
            for px, py, exit_dir in candidates
        ]
        score, pipe_x, pipe_y, exit_dir = max(scored, key=lambda item: item[0])
        if score < self.min_step:
            return None
        return (pipe_x, pipe_y, exit_dir)

    def _is_sheet_edge(self, x: int, y: int, direction: str) -> bool:
        """Check if we're at the image boundary."""
        margin = 10
        if direction == "UP" and y <= margin:
            return True
        if direction == "DOWN" and y >= self.h - margin:
            return True
        if direction == "LEFT" and x <= margin:
            return True
        if direction == "RIGHT" and x >= self.w - margin:
            return True
        return False
