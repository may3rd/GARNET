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


def _has_line_of_sight(mask: np.ndarray, x: int, y: int,
                       direction: str, distance: int = 8) -> bool:
    """Check if pipe exists for at least `distance` pixels in `direction`."""
    dx, dy = DIRECTION_DELTA[direction]
    for i in range(1, distance + 1):
        if not _is_pipe(mask, x + i * dx, y + i * dy):
            return False
    return True


def _check_bbox_hit(x: int, y: int, bbox: dict[str, int], margin: int = 4) -> bool:
    """Check if point is within or near a bbox."""
    return (bbox["x_min"] - margin <= x <= bbox["x_max"] + margin and
            bbox["y_min"] - margin <= y <= bbox["y_max"] + margin)


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
        lookahead: int = 30,
    ):
        self.mask = pipe_mask
        self.image = image
        self.h, self.w = pipe_mask.shape
        self.max_steps = max_steps
        self.min_step = min_step
        self.lookahead = lookahead

        # Terminal candidates
        self.page_connections = page_connections or []
        self.instrument_tags = instrument_tags or []
        self.equipment_objects = equipment_objects or []

        # Visited mask (shared across traces to avoid re-walking)
        self.visited = visited_mask if visited_mask is not None else np.zeros_like(pipe_mask)

        # Inline symbol bboxes (valves, reducers, etc.) — from stage4
        self._inline_symbols: list[dict] = []  # set via set_inline_symbols()

    def set_inline_symbols(self, objects: list[dict]) -> None:
        """Set stage4 objects that are inline symbols (valve, reducer, etc.)."""
        inline_classes = {
            "gate_valve", "globe_valve", "check_valve", "ball_valve",
            "butterfly_valve", "control_valve", "pressure_relief_valve",
            "reducer", "spectacle_blind", "strainer",
        }
        self._inline_symbols = [
            o for o in objects
            if o.get("class_name", "") in inline_classes
        ]

    def trace(self, start_x: int, start_y: int, start_dir: str,
              source_obj_id: str = "") -> TraceResult:
        """Trace from port to terminal. source_obj_id is excluded from terminal checks."""
        result = TraceResult()
        result.terminal_type = None

        x, y = start_x, start_y
        direction = start_dir
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

        while steps < self.max_steps:
            steps += 1

            # Mark visited
            if 0 <= y < self.h and 0 <= x < self.w:
                self.visited[y, x] = 1

            # Check terminal conditions at current position
            terminal = self._check_terminals(x, y, direction, source_obj_id)
            if terminal:
                result.terminal_type = terminal[0]
                result.terminal_x, result.terminal_y = x, y
                if len(terminal) > 2:
                    result.terminal_obj_id = terminal[2]

                # Save final segment
                seg_len = max(abs(x - seg_start_x), abs(y - seg_start_y))
                if seg_len >= self.min_step:
                    result.segments.append(TraceSegment(
                        x1=seg_start_x, y1=seg_start_y,
                        x2=x, y2=y, direction=direction,
                        length_px=seg_len,
                    ))
                    result.trace_length_px += seg_len
                # Don't count the terminal step as a step
                steps -= 1
                break

            # Check for inline symbols (valve, reducer)
            inline = self._check_inline(x, y)
            if inline:
                result.hits.append(inline)
                # Jump past inline object
                jump_px = 60
                nx = x + jump_px * dx
                ny = y + jump_px * dy
                if _is_pipe(self.mask, nx, ny):
                    x, y = nx, ny
                    continue

            # Look ahead — what's in front?
            forward_ok = _has_line_of_sight(self.mask, x, y, direction, self.min_step)

            if forward_ok:
                # Walk forward
                for _ in range(self.min_step):
                    x += dx
                    y += dy
                    if 0 <= y < self.h and 0 <= x < self.w:
                        self.visited[y, x] = 1
                continue

            # Forward blocked — check for turns
            left_dir = TURN_LEFT[direction]
            right_dir = TURN_RIGHT[direction]
            left_ok = _has_line_of_sight(self.mask, x, y, left_dir, self.min_step)
            right_ok = _has_line_of_sight(self.mask, x, y, right_dir, self.min_step)

            # Check for tee junction (both L and R have pipe)
            if left_ok and right_ok:
                # Save current segment
                seg_len = max(abs(x - seg_start_x), abs(y - seg_start_y))
                if seg_len >= self.min_step:
                    result.segments.append(TraceSegment(
                        x1=seg_start_x, y1=seg_start_y,
                        x2=x, y2=y, direction=direction,
                        length_px=seg_len,
                    ))
                    result.trace_length_px += seg_len

                result.terminal_type = TerminalType.TEE_JUNCTION.value
                result.terminal_x, result.terminal_y = x, y
                break

            if left_ok:
                # Save current segment
                seg_len = max(abs(x - seg_start_x), abs(y - seg_start_y))
                if seg_len >= self.min_step:
                    result.segments.append(TraceSegment(
                        x1=seg_start_x, y1=seg_start_y,
                        x2=x, y2=y, direction=direction,
                        length_px=seg_len,
                    ))
                    result.trace_length_px += seg_len

                # Turn
                x += DIRECTION_DELTA[left_dir][0] * self.min_step
                y += DIRECTION_DELTA[left_dir][1] * self.min_step
                result.turns.append((x, y, left_dir))
                direction = left_dir
                dx, dy = DIRECTION_DELTA[direction]
                seg_start_x, seg_start_y = x, y
                continue

            if right_ok:
                seg_len = max(abs(x - seg_start_x), abs(y - seg_start_y))
                if seg_len >= self.min_step:
                    result.segments.append(TraceSegment(
                        x1=seg_start_x, y1=seg_start_y,
                        x2=x, y2=y, direction=direction,
                        length_px=seg_len,
                    ))
                    result.trace_length_px += seg_len

                x += DIRECTION_DELTA[right_dir][0] * self.min_step
                y += DIRECTION_DELTA[right_dir][1] * self.min_step
                result.turns.append((x, y, right_dir))
                direction = right_dir
                dx, dy = DIRECTION_DELTA[direction]
                seg_start_x, seg_start_y = x, y
                continue

            # No forward, no turns — try ray-cast to bridge mask gap
            seg_len = max(abs(x - seg_start_x), abs(y - seg_start_y))
            if seg_len >= self.min_step:
                result.segments.append(TraceSegment(
                    x1=seg_start_x, y1=seg_start_y,
                    x2=x, y2=y, direction=direction,
                    length_px=seg_len,
                ))
                result.trace_length_px += seg_len

            # Check terminals at current position
            terminal = self._check_terminals(x, y, direction, source_obj_id, look_ahead=60)
            if terminal:
                result.terminal_type = terminal[0]
                result.terminal_x, result.terminal_y = x, y
                if len(terminal) > 2:
                    result.terminal_obj_id = terminal[2]
                break

            # Ray-cast: look ahead for next pipe pixel (bridge mask gaps)
            ray_max = 200
            found_pipe = False
            for ray_dist in range(5, ray_max, 2):
                rx = x + ray_dist * dx
                ry = y + ray_dist * dy
                if _is_pipe(self.mask, rx, ry):
                    # Found pipe — jump to it and continue
                    x, y = rx, ry
                    seg_start_x, seg_start_y = x, y
                    found_pipe = True
                    # Check terminals at jump point
                    terminal2 = self._check_terminals(x, y, direction, source_obj_id, look_ahead=60)
                    if terminal2:
                        result.terminal_type = terminal2[0]
                        result.terminal_x, result.terminal_y = x, y
                        if len(terminal2) > 2:
                            result.terminal_obj_id = terminal2[2]
                        # Can't break outer while from here, set flag
                        steps = self.max_steps  # force exit
                    break

            if found_pipe and steps < self.max_steps:
                continue

            # Check terminals with larger look-ahead
            terminal3 = self._check_terminals(x, y, direction, source_obj_id, look_ahead=200)
            if terminal3:
                result.terminal_type = terminal3[0]
                result.terminal_x, result.terminal_y = x, y
                if len(terminal3) > 2:
                    result.terminal_obj_id = terminal3[2]
            elif self._is_sheet_edge(x, y, direction):
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
                         source_obj_id: str = "", look_ahead: int = 40) -> Optional[tuple]:
        """Check if position or area ahead is a terminal. Returns (type, ...) or None."""
        dx, dy = DIRECTION_DELTA.get(direction, (0, 0))

        # Check at current position and ahead (pipe stops at mask edge,
        # terminal bbox may be just beyond)
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

            # Check instrument tags
            for tag in self.instrument_tags:
                if _check_bbox_hit(tx, ty, tag["bbox"], margin=8):
                    return (TerminalType.INSTRUMENT_TAG.value, tag.get("id", ""))

            # Check equipment (large bboxes — wider margin)
            for eq in self.equipment_objects:
                eq_bbox = eq["bbox"]
                if (eq_bbox["x_min"] - 20 <= tx <= eq_bbox["x_max"] + 20 and
                    eq_bbox["y_min"] - 20 <= ty <= eq_bbox["y_max"] + 20):
                    return (TerminalType.EQUIPMENT.value, eq.get("id", ""))

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
