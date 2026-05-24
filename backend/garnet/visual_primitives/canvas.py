"""Canvas loader, coordinate normalizer, and downsampler for the visual-primitives pipeline.

Coordinates use a normalized [0, 999] integer space mapped to raw canvas pixels.
All four agents work in this shared coordinate system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CanvasConfig:
    """Tunable parameters for canvas operations."""

    global_view_max_dim: int = 1200
    """Target max dimension for the downsampled global view. Adaptive to sheet size."""

    normalized_range: int = 999
    """Max value in the normalized [0, N] coordinate space."""


# ---------------------------------------------------------------------------
# Canvas metadata
# ---------------------------------------------------------------------------

@dataclass
class CanvasMeta:
    """Metadata extracted from a loaded P&ID canvas."""

    width: int
    height: int
    channels: int
    dpi: Optional[int]
    source_path: str


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_canvas(image_path: str | Path) -> tuple[np.ndarray, CanvasMeta]:
    """Load a P&ID image and return the full-resolution array + metadata.

    Supports PNG, JPG, TIFF via OpenCV.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image (unsupported format or corrupt): {path}")

    # Normalise to 3-channel BGR for downstream consistency.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    meta = CanvasMeta(
        width=img.shape[1],
        height=img.shape[0],
        channels=img.shape[2],
        dpi=None,  # OpenCV doesn't reliably read DPI from all formats
        source_path=str(path.resolve()),
    )
    return img, meta


# ---------------------------------------------------------------------------
# Coordinate normalizer
# ---------------------------------------------------------------------------

def normalize(
    pixel_x: int,
    pixel_y: int,
    canvas_width: int,
    canvas_height: int,
    *,
    cfg: Optional[CanvasConfig] = None,
) -> tuple[int, int]:
    """Map pixel coordinates -> [0, N] normalized integer space.

    Clamps to ensure output stays in [0, N].
    """
    c = cfg or CanvasConfig()
    nx = int(round(pixel_x / canvas_width * c.normalized_range))
    ny = int(round(pixel_y / canvas_height * c.normalized_range))
    return (
        max(0, min(c.normalized_range, nx)),
        max(0, min(c.normalized_range, ny)),
    )


def denormalize(
    norm_x: int,
    norm_y: int,
    canvas_width: int,
    canvas_height: int,
    *,
    cfg: Optional[CanvasConfig] = None,
) -> tuple[int, int]:
    """Map [0, N] normalized coordinates -> pixel coordinates."""
    c = cfg or CanvasConfig()
    px = int(round(norm_x / c.normalized_range * canvas_width))
    py = int(round(norm_y / c.normalized_range * canvas_height))
    return (px, py)


# ---------------------------------------------------------------------------
# Downsampled global view
# ---------------------------------------------------------------------------

def make_global_view(
    image: np.ndarray,
    *,
    cfg: Optional[CanvasConfig] = None,
) -> np.ndarray:
    """Downsample the full canvas to a size suitable for Agent 1 input.

    Uses adaptive sizing: the longer edge is clamped to global_view_max_dim,
    preserving the original aspect ratio with no letterboxing.
    """
    c = cfg or CanvasConfig()
    h, w = image.shape[:2]
    scale = c.global_view_max_dim / max(w, h)

    if scale >= 1.0:
        return image.copy()  # Already small enough; return as-is.

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pixel_to_norm_bbox(
    bbox: tuple[int, int, int, int],
    canvas_width: int,
    canvas_height: int,
    *,
    cfg: Optional[CanvasConfig] = None,
) -> list[int]:
    """Convert a pixel bbox (x1, y1, x2, y2) to normalized [0, N]."""
    x1, y1 = normalize(bbox[0], bbox[1], canvas_width, canvas_height, cfg=cfg)
    x2, y2 = normalize(bbox[2], bbox[3], canvas_width, canvas_height, cfg=cfg)
    return [x1, y1, x2, y2]


def norm_to_pixel_bbox(
    bbox: list[int],
    canvas_width: int,
    canvas_height: int,
    *,
    cfg: Optional[CanvasConfig] = None,
) -> tuple[int, int, int, int]:
    """Convert a normalized bbox [x1, y1, x2, y2] to pixel coordinates."""
    x1, y1 = denormalize(bbox[0], bbox[1], canvas_width, canvas_height, cfg=cfg)
    x2, y2 = denormalize(bbox[2], bbox[3], canvas_width, canvas_height, cfg=cfg)
    return (x1, y1, x2, y2)
