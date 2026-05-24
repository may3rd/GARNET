"""VLM trace response parser - extracts structured tokens from VLM output.

Tokens: <|go|>DIR DIST, <|turn|>DIR, <|hit|>CLASS, <|jump|>DIR DIST, <|term|>CLASS
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import logging

log = logging.getLogger("vlm_trace_parser")


# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------


@dataclass
class TraceToken:
    """One parsed token from VLM output."""
    kind: str           # "go", "turn", "hit", "jump", "term"
    direction: str = "" # UP/DOWN/LEFT/RIGHT
    distance: int = 0   # pixels
    class_name: str = ""  # for hit/term
    raw: str = ""       # original text match


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_TOKEN_PATTERNS = [
    (re.compile(r"<\|go\|>\s*(\w+)(?:\s+(\d+))?", re.IGNORECASE), "go"),
    (re.compile(r"<\|turn\|>\s*(\w+)", re.IGNORECASE), "turn"),
    (re.compile(r"<\|hit\|>\s*(.+?)(?:\s*$)", re.IGNORECASE), "hit"),
    (re.compile(r"<\|jump\|>\s*(\w+)\s+(\d+)", re.IGNORECASE), "jump"),
    (re.compile(r"<\|term\|>\s*(.+?)(?:\s*$)", re.IGNORECASE), "term"),
]


def parse_trace_response(raw: str | None) -> list[TraceToken]:
    """Parse VLM response into ordered list of trace tokens.

    Returns empty list if VLM gave nothing usable.
    Falls back to single "go straight" if no structured tokens found.
    """
    if not raw:
        log.warning("empty VLM response")
        return []

    raw = raw.strip()
    tokens: list[TraceToken] = []

    for pattern, kind in _TOKEN_PATTERNS:
        for m in pattern.finditer(raw):
            token = TraceToken(kind=kind, raw=m.group(0))
            if kind in ("go", "jump"):
                token.direction = _normalize_dir(m.group(1))
                distance_str = m.group(2)
                token.distance = int(distance_str) if distance_str else 350  # default to crop size
            elif kind == "turn":
                token.direction = _normalize_dir(m.group(1))
            elif kind == "hit":
                token.class_name = m.group(1).strip().strip("<").rstrip()
            elif kind == "term":
                token.class_name = m.group(1).strip().strip("<").rstrip()
            tokens.append(token)

    if not tokens:
        log.warning("no structured tokens found in: %s", raw[:100])
        return []

    return tokens


def _normalize_dir(s: str) -> str:
    """Normalize direction string to UP/DOWN/LEFT/RIGHT."""
    s = s.upper().strip()
    if s in ("UP", "DOWN", "LEFT", "RIGHT"):
        return s
    if s in ("TOP", "U", "NORTH", "N"):
        return "UP"
    if s in ("BOTTOM", "B", "SOUTH", "S"):
        return "DOWN"
    if s in ("L", "WEST", "W"):
        return "LEFT"
    if s in ("R", "EAST", "E"):
        return "RIGHT"
    log.warning("unknown direction: %s, falling back to UP", s)
    return "UP"
