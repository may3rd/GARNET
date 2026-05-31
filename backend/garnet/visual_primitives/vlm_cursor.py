"""VLM Cursor - position tracking with visual marking for VLM-guided tracing.

The cursor tracks position and draws grey marks + path lines on a temp copy
of the P&ID image. VLM sees these marks and uses them to decide where to go next,
preventing backtracking into already-traced pipe segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

DIR_VEC: dict[str, tuple[int, int]] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}

OPPOSITE: dict[str, str] = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}

MARK_COLOR = (128, 128, 128)   # grey - visible but not confused with pipe black
MARK_RADIUS = 5
LINE_THICKNESS = 2


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


@dataclass
class VLMCursor:
    """Tracks position and builds visual path marks on the image."""

    x: int
    y: int
    direction: str
    path: list[tuple[int, int]] = field(default_factory=list)
    total_distance: int = 0

    def __post_init__(self):
        if not self.path:
            self.path = [(self.x, self.y)]

    # --- Mutations ---

    def move(self, direction: str, distance: int) -> tuple[int, int]:
        """Move cursor in direction by distance. Returns new (x, y)."""
        dx, dy = DIR_VEC[direction]
        self.x = max(0, self.x + dx * distance)
        self.y = max(0, self.y + dy * distance)
        self.direction = direction
        self.total_distance += distance
        self.path.append((self.x, self.y))
        return (self.x, self.y)

    def turn(self, direction: str):
        """Change direction at current position (corner)."""
        self.direction = direction

    def jump(self, destination: tuple[int, int], direction: str):
        """Jump to a new position (past valve/gap)."""
        self.x, self.y = destination
        self.direction = direction
        self.path.append((self.x, self.y))
        # Don't count gap distance — not actual pipe pixels

    # --- Drawing ---

    def draw_on(self, image: np.ndarray) -> np.ndarray:
        """Draw all path marks and lines onto a copy of the image."""
        marked = image.copy()

        # Path lines connecting all marks
        for i in range(1, len(self.path)):
            cv2.line(
                marked,
                self.path[i - 1],
                self.path[i],
                MARK_COLOR,
                LINE_THICKNESS,
            )

        # Draw dots at each mark position
        for px, py in self.path:
            cv2.circle(marked, (px, py), MARK_RADIUS, MARK_COLOR, -1)

        # Draw a slightly larger ring at the current (latest) position
        cv2.circle(marked, (self.x, self.y), MARK_RADIUS + 2, MARK_COLOR, 1)

        return marked


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


def crop_around_cursor(
    image: np.ndarray,
    cursor: VLMCursor,
    crop_size: int = 350,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a marked image centered on the cursor position.

    Returns (cropped_image, (x1, y1, x2, y2)) in source image coordinates.
    The crop is used to convert VLM-relative responses back to global coords.
    """
    h, w = image.shape[:2]
    half = crop_size // 2

    x1 = max(0, cursor.x - half)
    y1 = max(0, cursor.y - half)
    x2 = min(w, cursor.x + half)
    y2 = min(h, cursor.y + half)

    # Extend opposite edge if near boundary
    if cursor.x - half < 0:
        x2 = min(w, x2 + (-(cursor.x - half)))
    if cursor.y - half < 0:
        y2 = min(h, y2 + (-(cursor.y - half)))
    if cursor.x + half > w:
        x1 = max(0, x1 - (cursor.x + half - w))
    if cursor.y + half > h:
        y1 = max(0, y1 - (cursor.y + half - h))

    marked = cursor.draw_on(image)
    crop = marked[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)
