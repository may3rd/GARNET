"""Patch utilities — crop sub-images with reference grid overlays.

Used by Agent 2 (nozzle crops) and Agent 3 (sliding window). Agent 1 works on
the full downsampled view and does not use patches.
"""

from __future__ import annotations

import cv2
import numpy as np


def crop_patch(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    window_size: int = 500,
    *,
    overlay_grid: bool = True,
) -> np.ndarray:
    """Cut a square sub-patch from the image centered at pixel coordinates.

    The patch is exactly window_size × window_size. If the center is near an
    edge, the patch is clamped to the image boundary (may be smaller than
    requested).

    When overlay_grid=True, a lightweight 10×10 reference grid is drawn on the
    patch edges to help the VLM estimate relative coordinates inside the crop.

    Returns a BGR numpy array.
    """
    h, w = image.shape[:2]
    half = window_size // 2

    x1 = max(0, center_x - half)
    y1 = max(0, center_y - half)
    x2 = min(w, center_x + half)
    y2 = min(h, center_y + half)

    patch = image[y1:y2, x1:x2].copy()

    if overlay_grid and patch.size > 0:
        patch = _draw_reference_grid(patch, window_size)

    return patch


def crop_patch_normalized(
    image: np.ndarray,
    norm_x: int,
    norm_y: int,
    canvas_width: int,
    canvas_height: int,
    window_size: int = 500,
) -> np.ndarray:
    """Same as crop_patch but takes [0, 999] normalized coordinates.

    Converts to pixel coords before cropping.
    """
    from .canvas import denormalize

    px, py = denormalize(norm_x, norm_y, canvas_width, canvas_height)
    return crop_patch(image, px, py, window_size=window_size)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _draw_reference_grid(patch: np.ndarray, window_size: int) -> np.ndarray:
    """Draw a subtle 10×10 spatial reference grid on the crop edges.

    Grid cells are numbered 0-99 along each axis. Numbers appear only on the
    top and left borders. Tick marks on all four edges.
    """
    h, w = patch.shape[:2]
    cell_h = max(1, h // 10)
    cell_w = max(1, w // 10)

    # Very light gray for grid lines — visible but not obtrusive.
    color = (180, 180, 180)
    thickness = 1

    # Horizontal grid lines
    for i in range(1, 10):
        y = i * cell_h
        cv2.line(patch, (0, y), (w - 1, y), color, thickness)

    # Vertical grid lines
    for i in range(1, 10):
        x = i * cell_w
        cv2.line(patch, (x, 0), (x, h - 1), color, thickness)

    # Tick marks on edges (every cell)
    for i in range(11):
        y = min(i * cell_h, h - 1)
        x = min(i * cell_w, w - 1)
        # Top edge
        cv2.line(patch, (x, 0), (x, 4), color, 1)
        # Bottom edge
        cv2.line(patch, (x, h - 1), (x, h - 5), color, 1)
        # Left edge
        cv2.line(patch, (0, y), (4, y), color, 1)
        # Right edge
        cv2.line(patch, (w - 1, y), (w - 5, y), color, 1)

    return patch
