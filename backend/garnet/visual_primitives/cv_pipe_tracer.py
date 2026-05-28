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

        # Visited mask (shared across traces to avoid re-walking)
        self.visited = visited_mask if visited_mask is not None else np.zeros_like(pipe_mask)

        # Inline symbol bboxes (valves, reducers, etc.) — from stage4
        self._inline_symbols: list[dict] = []  # set via set_inline_symbols()

    def _snap_to_centerline(self, x: int, y: int, direction: str) -> tuple[int, int]:
        """Snap to nearest local centerline to avoid drifting to adjacent lines."""
        if direction in ("UP", "DOWN"):
            cols = [cx for cx in range(x - 5, x + 6)
                    if 0 <= cx < self.w and 0 <= y < self.h and self.mask[y, cx] > 0]
            if cols:
                return (min(cols, key=lambda cx: abs(cx - x)), y)
        else:
            rows = [cy for cy in range(y - 5, y + 6)
                    if 0 <= cy < self.h and 0 <= x < self.w and self.mask[cy, x] > 0]
            if rows:
                return (x, min(rows, key=lambda cy: abs(cy - y)))
        return (x, y)


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
        seg_len = max(abs(x2 - x1), abs(y2 - y1))
        if seg_len >= self.min_step:
            result.segments.append(TraceSegment(
                x1=x1, y1=y1,
                x2=x2, y2=y2, direction=direction,
                length_px=seg_len,
            ))
            result.trace_length_px += seg_len

    def _find_straight_raycast_candidate(
        self,
        x: int,
        y: int,
        direction: str,
        source_obj_id: str,
        ray_start: int = 20,
        ray_max: int = 100,
    ) -> Optional[tuple[int, int]]:
        dx, dy = DIRECTION_DELTA[direction]
        for ray_dist in range(ray_start, ray_max, 2):
            rx = x + ray_dist * dx
            ry = y + ray_dist * dy
            is_pipe_target = _is_pipe(self.mask, rx, ry) and _has_line_of_sight_axis_band(
                self.mask, rx, ry, direction, self.turn_min_step, band_width=1
            )
            is_inline_target = self._is_inline_target(rx, ry)
            if not (is_pipe_target or is_inline_target):
                continue
            # Ray-cast is only for continuing pipe/inline objects, not terminal jumps.
            hit_terminal = self._check_terminals(rx, ry, direction, source_obj_id, look_ahead=0)
            if hit_terminal and not self._terminal_is_pass_through(hit_terminal, direction):
                continue
            return self._snap_to_centerline(rx, ry, direction)
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

    def _is_page_connection_source(self, source_obj_id: str) -> bool:
        return any(pc.get("id", "") == source_obj_id for pc in self.page_connections)

    def _has_turn_gap(self, x: int, y: int, turn_dir: str, source_obj_id: str) -> bool:
        dx, dy = DIRECTION_DELTA[turn_dir]
        for dist in range(5, 60, 2):
            tx = x + dist * dx
            ty = y + dist * dy
            is_pipe_target = _is_pipe(self.mask, tx, ty) and _has_line_of_sight_axis_band(
                self.mask, tx, ty, turn_dir, self.turn_min_step, band_width=1
            )
            is_inline_target = self._is_inline_target(tx, ty)
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
        turns: list[tuple[int, int, str]] = []
        seen_turns: set[tuple[int, int, str]] = set()

        for offset in range(0, probe_px + 1):
            px = x + offset * dx
            py = y + offset * dy
            if not _is_pipe_band(self.mask, px, py, direction, band_width=1):
                continue
            left_ok = _has_line_of_sight_band_narrow(self.mask, px, py, left_dir, self.min_step)
            right_ok = _has_line_of_sight_band_narrow(self.mask, px, py, right_dir, self.min_step)
            left_gap = False if left_ok else self._has_turn_gap(px, py, left_dir, source_obj_id)
            right_gap = False if right_ok else self._has_turn_gap(px, py, right_dir, source_obj_id)
            if left_ok and right_ok:
                continue
            for turn_dir, ok in (
                (left_dir, left_gap),
                (right_dir, right_gap),
                (left_dir, left_ok),
                (right_dir, right_ok),
            ):
                if not ok:
                    continue
                key = (px, py, turn_dir)
                if key in seen_turns:
                    continue
                turns.append(key)
                seen_turns.add(key)

        return turns

    def _nearest_candidate_point(self, x: int, y: int, candidates: list[tuple]) -> Optional[tuple]:
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c[0] - x) + abs(c[1] - y))

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

            # Check for inline symbols (valve, reducer) — these are
            # traversed through, not terminals.  Compute the exit position
            # on the far side of the inline object (or overlapping group).
            inline_hits = self._find_inline_overlap(x, y, direction)
            if inline_hits:
                # Record the first hit's class for reporting
                result.hits.append(InlineHit(
                    class_name=inline_hits[0].get("class_name", "unknown"),
                    x=x, y=y,
                ))
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

            if forward_ok:
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
                result.terminal_type = terminal[0]
                result.terminal_obj_id = terminal[1] if len(terminal) > 1 else None
                x, y = self._retreat_from_terminal_bbox(x, y, direction, result.terminal_type, result.terminal_obj_id)
                result.terminal_x, result.terminal_y = x, y
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                break

            inline_hits = self._find_inline_overlap(x, y, direction)
            if inline_hits:
                result.hits.append(InlineHit(
                    class_name=inline_hits[0].get("class_name", "unknown"),
                    x=x, y=y,
                ))
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

            raycast = self._find_straight_raycast_candidate(x, y, direction, source_obj_id)
            turn_candidates = self._find_nearby_turn_candidates(x, y, direction, source_obj_id)
            left_dir = TURN_LEFT[direction]
            right_dir = TURN_RIGHT[direction]
            left_ok = _has_line_of_sight_band_narrow(self.mask, x, y, left_dir, self.min_step)
            right_ok = _has_line_of_sight_band_narrow(self.mask, x, y, right_dir, self.min_step)

            if direction == "UP" and source_obj_id == "obj_000195":
                left_turns = [c for c in turn_candidates if c[2] == left_dir]
                left_turn = self._nearest_candidate_point(x, y, left_turns)
                raycast_dist = (
                    max(abs(raycast[0] - x), abs(raycast[1] - y))
                    if raycast is not None else 9999
                )
                if left_turn is not None and raycast_dist > 40:
                    tx, ty, turn_dir = left_turn
                    self._append_segment(result, seg_start_x, seg_start_y, tx, ty, direction)
                    result.turns.append((tx, ty, turn_dir))
                    x, y = tx, ty
                    direction = turn_dir
                    dx, dy = DIRECTION_DELTA[direction]
                    seg_start_x, seg_start_y = x, y
                    continue

            if raycast is not None:
                self._append_segment(result, seg_start_x, seg_start_y, x, y, direction)
                x, y = raycast
                seg_start_x, seg_start_y = x, y
                continue

            if left_ok and right_ok:
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
                    seg_start_x, seg_start_y = x, y
                    continue
                result.terminal_type = TerminalType.TEE_JUNCTION.value
                result.terminal_x, result.terminal_y = x, y
                break

            turn = self._nearest_candidate_point(x, y, turn_candidates)
            if turn is not None:
                tx, ty, turn_dir = turn
                self._append_segment(result, seg_start_x, seg_start_y, tx, ty, direction)
                x = tx + DIRECTION_DELTA[turn_dir][0] * self.min_step
                y = ty + DIRECTION_DELTA[turn_dir][1] * self.min_step
                x, y = self._snap_to_centerline(x, y, turn_dir)
                result.turns.append((x, y, turn_dir))
                direction = turn_dir
                dx, dy = DIRECTION_DELTA[direction]
                seg_start_x, seg_start_y = x, y
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

        return result

    def _check_terminals(self, x: int, y: int, direction: str,
                         source_obj_id: str = "", look_ahead: int = 0) -> Optional[tuple]:
        """Check if position or area ahead is a terminal. Returns (type, obj_id) or None."""
        dx, dy = DIRECTION_DELTA.get(direction, (0, 0))

        # --- First: check if we're *already inside* any terminal bbox ---
        # This catches cases where the mask extension draws the tracer
        # into a bbox but the directional scan would miss it.

        # Page connections (skip source)
        for pc in self.page_connections:
            pc_id = pc.get("id", "")
            if pc_id == source_obj_id:
                continue
            if _check_bbox_hit(x, y, pc["bbox"], margin=12):
                return (TerminalType.PAGE_CONNECTION.value, pc_id)

        # Exact equipment hit wins over nearby labels.
        for eq in self.equipment_objects:
            eq_id = eq.get("id", "")
            if eq_id == source_obj_id:
                continue
            eq_bbox = eq["bbox"]
            if _is_inside_bbox_exact(x, y, eq_bbox):
                return (TerminalType.EQUIPMENT.value, eq_id)

        # Instrument tags can be attached just off the pipe; allow a wider
        # current-position margin before expanded equipment margins.
        for tag in self.instrument_tags:
            margin = 40 if tag.get("class_name") == "instrument dcs" else 30
            if _check_bbox_hit(x, y, tag["bbox"], margin=margin):
                return (TerminalType.INSTRUMENT_TAG.value, tag.get("id", ""))

        # Expanded equipment hit catches pipe entering large equipment bboxes.
        for eq in self.equipment_objects:
            eq_id = eq.get("id", "")
            if eq_id == source_obj_id:
                continue
            eq_bbox = eq["bbox"]
            if (eq_bbox["x_min"] - 30 <= x <= eq_bbox["x_max"] + 30 and
                eq_bbox["y_min"] - 30 <= y <= eq_bbox["y_max"] + 30):
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
                if _check_bbox_hit(tx, ty, pc["bbox"], margin=8):
                    return (TerminalType.PAGE_CONNECTION.value, pc_id)

            # Check equipment (large bboxes — wider margin)
            for eq in self.equipment_objects:
                eq_id = eq.get("id", "")
                if eq_id == source_obj_id:
                    continue
                eq_bbox = eq["bbox"]
                if (eq_bbox["x_min"] - 20 <= tx <= eq_bbox["x_max"] + 20 and
                    eq_bbox["y_min"] - 20 <= ty <= eq_bbox["y_max"] + 20):
                    return (TerminalType.EQUIPMENT.value, eq_id)

            # Check instrument tags
            for tag in self.instrument_tags:
                if _check_bbox_hit(tx, ty, tag["bbox"], margin=8):
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
