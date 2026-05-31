"""Cursor state management for step-by-step VLM pipeline tracing.

Tracks position, direction, visited paths, and handles image cropping with
visual markers (crosshair, visited overlay) for VLM guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Cursor state
# ---------------------------------------------------------------------------


@dataclass
class PipelineCursor:
    """Tracks position and state during VLM-guided pipeline tracing.

    The cursor represents the current position on a pipe line being traced.
    Each VLM-guided step advances the cursor along the pipe.  The visited
    path is drawn as a green overlay to prevent the VLM from backtracking.
    """

    image: np.ndarray  # original BGR image (from cv2.imread)
    x: int  # current cursor x in original image coordinates
    y: int  # current cursor y in original image coordinates
    direction: str  # UP, DOWN, LEFT, RIGHT — direction the pipe is going
    crop_size: int = 300  # crop square size in pixels
    margin: int = 20  # min distance from crop edge before recentering

    # Internal state
    _visited: list[tuple[int, int]] = field(default_factory=list)
    _path_pixels: list[tuple[int, int]] = field(default_factory=list)
    _entry_edge: Optional[str] = None

    def __post_init__(self) -> None:
        if not self._visited:
            self._visited = [(self.x, self.y)]

    # ------------------------------------------------------------------
    # Cropping
    # ------------------------------------------------------------------

    def crop_view(self) -> tuple[Image.Image, dict]:
        """Crop a crop_size×crop_size region centred on the cursor.

        Returns (PIL Image, metadata dict).
        Metadata includes the crop bounding box in original image coordinates
        so downstream code can convert view-relative coordinates back.
        """
        h, w = self.image.shape[:2]
        half = self.crop_size // 2

        x1 = max(0, self.x - half)
        y1 = max(0, self.y - half)
        x2 = min(w, x1 + self.crop_size)
        y2 = min(h, y1 + self.crop_size)

        # Clamp back if we hit the image edge
        if x2 - x1 < self.crop_size:
            x1 = max(0, x2 - self.crop_size)
        if y2 - y1 < self.crop_size:
            y1 = max(0, y2 - self.crop_size)

        crop_bgr = self.image[y1:y2, x1:x2]
        # BGR → RGB for PIL
        crop_rgb = crop_bgr[..., ::-1]
        pil_img = Image.fromarray(crop_rgb)

        meta: dict = {
            "crop_x1": int(x1),
            "crop_y1": int(y1),
            "crop_x2": int(x2),
            "crop_y2": int(y2),
            "crop_w": int(x2 - x1),
            "crop_h": int(y2 - y1),
            "cursor_x_view": int(self.x - x1),
            "cursor_y_view": int(self.y - y1),
        }
        return pil_img, meta

    # ------------------------------------------------------------------
    # Markers
    # ------------------------------------------------------------------

    def draw_cursor_marker(self, crop: Image.Image, meta: dict) -> Image.Image:
        """Draw a red crosshair and direction arrow at the cursor position.

        The crosshair is drawn on a copy so the original stays clean.
        """
        marked = crop.copy()
        draw = ImageDraw.Draw(marked)
        cx = meta["cursor_x_view"]
        cy = meta["cursor_y_view"]
        r = 12  # crosshair arm length

        # Red crosshair
        draw.line([(cx - r, cy), (cx + r, cy)], fill=(255, 0, 0), width=2)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=(255, 0, 0), width=2)

        # White outline for contrast
        draw.line([(cx - r - 1, cy), (cx + r + 1, cy)], fill=(255, 255, 255), width=1)
        draw.line([(cx, cy - r - 1), (cx, cy + r + 1)], fill=(255, 255, 255), width=1)

        # Direction arrow (drawn from cursor in the trace direction)
        arrow_len = 20
        offsets: dict[str, tuple[int, int]] = {
            "RIGHT": (arrow_len, 0),
            "LEFT": (-arrow_len, 0),
            "DOWN": (0, arrow_len),
            "UP": (0, -arrow_len),
        }
        dx, dy = offsets.get(self.direction, (arrow_len, 0))

        # Arrow shaft
        draw.line([(cx, cy), (cx + dx, cy + dy)], fill=(255, 0, 0), width=3)
        # Arrow head
        if self.direction in ("RIGHT", "LEFT"):
            sign = 1 if dx > 0 else -1
            draw.line(
                [(cx + dx, cy + dy), (cx + dx - sign * 6, cy + dy - 6)],
                fill=(255, 0, 0),
                width=2,
            )
            draw.line(
                [(cx + dx, cy + dy), (cx + dx - sign * 6, cy + dy + 6)],
                fill=(255, 0, 0),
                width=2,
            )
        else:
            sign = 1 if dy > 0 else -1
            draw.line(
                [(cx + dx, cy + dy), (cx + dx - 6, cy + dy - sign * 6)],
                fill=(255, 0, 0),
                width=2,
            )
            draw.line(
                [(cx + dx, cy + dy), (cx + dx + 6, cy + dy - sign * 6)],
                fill=(255, 0, 0),
                width=2,
            )

        return marked

    def draw_visited_path(self, crop: Image.Image, meta: dict) -> Image.Image:
        """Overlay visited path pixels as a semi-transparent green trail.

        This shows the VLM where it has already been to prevent loops.
        """
        marked = crop.copy()
        overlay = Image.new("RGBA", marked.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        x1, y1 = meta["crop_x1"], meta["crop_y1"]

        # Draw visited path as green dots
        for px, py in self._visited:
            vx = px - x1
            vy = py - y1
            if 0 <= vx < self.crop_size and 0 <= vy < self.crop_size:
                overlay_draw.ellipse(
                    [(vx - 2, vy - 2), (vx + 2, vy + 2)],
                    fill=(0, 255, 0, 180),
                )

        # Connect visited points with lines if they're sequential
        if len(self._visited) >= 2:
            for i in range(1, len(self._visited)):
                px0, py0 = self._visited[i - 1]
                px1, py1 = self._visited[i]
                vx0 = px0 - x1
                vy0 = py0 - y1
                vx1 = px1 - x1
                vy1 = py1 - y1
                if (
                    0 <= vx0 < self.crop_size
                    and 0 <= vy0 < self.crop_size
                    and 0 <= vx1 < self.crop_size
                    and 0 <= vy1 < self.crop_size
                ):
                    overlay_draw.line(
                        [(vx0, vy0), (vx1, vy1)],
                        fill=(0, 255, 0, 120),
                        width=2,
                    )

        marked = Image.alpha_composite(marked.convert("RGBA"), overlay).convert("RGB")
        return marked

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def advance(self, direction: str, distance_px: int) -> None:
        """Move cursor by *distance_px* in *direction*, recording path pixels.

        This is called after the VLM issues a <|step|> token.
        """
        self.direction = direction

        dx, dy = 0, 0
        if direction == "RIGHT":
            dx = 1
        elif direction == "LEFT":
            dx = -1
        elif direction == "DOWN":
            dy = 1
        elif direction == "UP":
            dy = -1

        for i in range(1, distance_px + 1):
            px = self.x + int(dx * i)
            py = self.y + int(dy * i)
            self._path_pixels.append((px, py))
            if (px, py) not in self._visited:
                self._visited.append((px, py))

        self.x += int(dx * distance_px)
        self.y += int(dy * distance_px)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def global_coords(self) -> tuple[int, int]:
        """Return cursor position in original image coordinates (pixels)."""
        return (self.x, self.y)

    def norm_coords(self, img_w: int, img_h: int) -> list[int]:
        """Return cursor position normalized to [0, 999]."""
        return [
            int(round(self.x / img_w * 999)),
            int(round(self.y / img_h * 999)),
        ]

    def view_to_global(self, vx: int, vy: int, meta: dict) -> tuple[int, int]:
        """Convert a view-relative coordinate back to original image pixels."""
        return (vx + meta["crop_x1"], vy + meta["crop_y1"])

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def visited_count(self) -> int:
        return len(self._visited)

    @property
    def path_length(self) -> int:
        return len(self._path_pixels)

    def is_near_edge(self, padding: int = 10) -> bool:
        """Check if cursor is near the image boundary."""
        h, w = self.image.shape[:2]
        return (
            self.x < padding
            or self.x > w - padding
            or self.y < padding
            or self.y > h - padding
        )
