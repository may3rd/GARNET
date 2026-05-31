"""Parses VLM responses from the pipeline tracer into structured TraceStep lists.

Handles the visual-primitive protocol: <|step|>, <|hit|>, <|term|> tokens
interleaved with free-form reasoning text.
"""

from __future__ import annotations

import re
from typing import Optional

from .schemas import TraceDirection, TraceStep, TraceTokenType

# ---------------------------------------------------------------------------
# Token patterns
# ---------------------------------------------------------------------------

# <|step|> DIRECTION DISTANCE
# Direction may optionally be followed by "pixels" or "px"
_STEP_RE = re.compile(
    r"<\|step\|>\s*(UP|DOWN|LEFT|RIGHT)\s+(\d+)\s*(?:px|pixels)?",
    re.IGNORECASE,
)

# <|hit|>CLASS<|box|>[[x1,y1,x2,y2]]<|/box|><|/hit|>
# Handles both flat [[x1,y1,x2,y2]] and point-pair [[x1,y1],[x2,y2]] formats
# Closing <|/hit|> tag is optional (VLMs sometimes omit it)
_HIT_RE = re.compile(
    r"<\|hit\|>([\w][\w\s]*[\w]?)\s*<\|box\|>"
    r"\[\[(\d+),\s*(\d+)(?:\],\s*\[(\d+),\s*(\d+)|\s*,\s*(\d+),\s*(\d+))\]\]"
    r"<\|/box\|>(?:\s*(?:tag=(\S+))?\s*<\|/hit\|>)?",
    re.IGNORECASE,
)

# <|term|>CLASS<|box|>[[x1,y1,x2,y2]]<|/box|><|/term|>
# Closing <|/term|> tag is optional
_TERM_RE = re.compile(
    r"<\|term\|>([\w][\w\s]*[\w]?)\s*<\|box\|>"
    r"\[\[(\d+),\s*(\d+)(?:\],\s*\[(\d+),\s*(\d+)|\s*,\s*(\d+),\s*(\d+))\]\]"
    r"<\|/box\|>(?:\s*(?:tag=(\S+))?\s*<\|/term\|>)?",
    re.IGNORECASE,
)

# Bare <|term|>CLASS<|/term|> (no box) — for crop_edge, no_pipe_found, tee_junction
_TERM_NOBOX_RE = re.compile(
    r"<\|term\|>([\w][\w\s]*[\w]?)\s*<\|/term\|>",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_trace_response(response: str) -> list[TraceStep]:
    """Extract ordered trace steps from a VLM tracing response.

    Tokens are extracted in the order they appear in the text.  Free-form
    reasoning between tokens is ignored.  Unknown token types are skipped.

    Returns an empty list if no tokens found.
    """
    steps: list[TraceStep] = []

    # We need to find tokens in order.  Since the three types can interleave,
    # we scan for the earliest occurrence of any token type, extract it,
    # then continue from after that match.
    pos = 0
    while pos < len(response):
        earliest_match: Optional[re.Match] = None
        earliest_pos: int = len(response)
        token_type: Optional[TraceTokenType] = None

        # Check all patterns from current position
        for pattern, ttype in [
            (_STEP_RE, TraceTokenType.STEP),
            (_HIT_RE, TraceTokenType.HIT),
            (_TERM_RE, TraceTokenType.TERM),
            (_TERM_NOBOX_RE, TraceTokenType.TERM),
        ]:
            m = pattern.search(response, pos)
            if m and m.start() < earliest_pos:
                earliest_match = m
                earliest_pos = m.start()
                token_type = ttype

        if earliest_match is None:
            break  # no more tokens

        # Extract based on type
        if token_type == TraceTokenType.STEP:
            direction_str = earliest_match.group(1).upper()
            distance = int(earliest_match.group(2))
            steps.append(
                TraceStep(
                    token_type=TraceTokenType.STEP,
                    direction=TraceDirection(direction_str),
                    distance_px=distance,
                )
            )

        elif token_type == TraceTokenType.HIT:
            symbol_class = earliest_match.group(1).strip().lower().replace(" ", "_")
            g2, g3, g4, g5, g6, g7 = [earliest_match.group(i) for i in range(2, 8)]
            # Handle both flat [[x1,y1,x2,y2]] and point-pair [[x1,y1],[x2,y2]]
            if g6 is not None and g7 is not None:
                # Flat format: groups 2,3,6,7 = x1,y1,x2,y2
                x1, y1, x2, y2 = int(g2), int(g3), int(g6), int(g7)
            else:
                # Point-pair format: groups 2,3,4,5 = x1,y1,x2,y2
                x1, y1, x2, y2 = int(g2), int(g3), int(g4), int(g5)
            tag = earliest_match.group(8)
            steps.append(
                TraceStep(
                    token_type=TraceTokenType.HIT,
                    symbol_class=symbol_class,
                    symbol_tag=tag,
                    symbol_bbox_view=[x1, y1, x2, y2],
                )
            )

        elif token_type == TraceTokenType.TERM:
            symbol_class = earliest_match.group(1).strip().lower().replace(" ", "_")
            # Check if it has a box
            if earliest_match.re is _TERM_RE:
                g2, g3, g4, g5, g6, g7 = [earliest_match.group(i) for i in range(2, 8)]
                if g6 is not None and g7 is not None:
                    x1, y1, x2, y2 = int(g2), int(g3), int(g6), int(g7)
                else:
                    x1, y1, x2, y2 = int(g2), int(g3), int(g4), int(g5)
                tag = earliest_match.group(8)
                steps.append(
                    TraceStep(
                        token_type=TraceTokenType.TERM,
                        symbol_class=symbol_class,
                        symbol_tag=tag,
                        symbol_bbox_view=[x1, y1, x2, y2],
                    )
                )
            else:
                # No box — crop_edge, no_pipe_found, etc.
                steps.append(
                    TraceStep(
                        token_type=TraceTokenType.TERM,
                        symbol_class=symbol_class,
                    )
                )

        pos = earliest_match.end()

    return steps


def has_terminal(steps: list[TraceStep]) -> bool:
    """Check if the step list ends with a TERM token."""
    return any(s.token_type == TraceTokenType.TERM for s in steps)


def last_terminal(steps: list[TraceStep]) -> Optional[TraceStep]:
    """Return the last TERM token in the step list, if any."""
    for s in reversed(steps):
        if s.token_type == TraceTokenType.TERM:
            return s
    return None


def total_trace_distance(steps: list[TraceStep]) -> int:
    """Sum the pixel distances of all STEP tokens."""
    return sum(s.distance_px or 0 for s in steps if s.token_type == TraceTokenType.STEP)
