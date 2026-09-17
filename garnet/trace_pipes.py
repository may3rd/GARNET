#!/usr/bin/env python
"""Pipe tracing for a P&ID sheet, driven by the fixture JSONs in garnet/tests/input.

New, self-contained implementation. It borrows the *walking algorithm* from
`backend/garnet/path_tracer/cv_pipe_tracer.py` (CVPipeTracer) — the centreline-snap +
banded line-of-sight + turn-resolution + bbox-terminal + loop-guard loop, and the direction
tables — but nothing else. No PIDPipeline, no stage_manifest, no Stage5bPipelineMixin.

Inputs are the three fixtures:
    <stem>_objects.json             YOLO objects  -> inline pass-through symbols + terminals
    <stem>_equipment_bboxes.json    Contract B    -> terminal bboxes + trace start ports (nozzles)
    <stem>_line_number_boxes.json   Contract C    -> text suppression + per-trace line attachment

The pipe mask is built from the raster, suppressing text and object glyphs so the walker
follows pipes rather than lettering (the fixtures supply the boxes to suppress).

Outputs into --out:
    trace_results.json / trace_overlay.png            port traces
    branch_results.json / branch_overlay.png          tee-branch traces
    pipe_mask.png                                     the mask that was walked

Usage:
    python garnet/trace_pipes.py --stem Test-00001
    python garnet/trace_pipes.py --stem Test-00001 --indir garnet/tests/input --out output/trace_test01

Run with the repo venv: /Users/maetee/Code/GARNET/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Direction tables — borrowed from cv_pipe_tracer.py (same names and semantics)
# ---------------------------------------------------------------------------

DIRECTION_DELTA = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
TURN_LEFT = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
TURN_RIGHT = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# Terminal classes recognised from the fixtures.
PAGE_CONNECTION_CLASSES = {"page connection", "page_connection", "connection",
                           "utility connection", "page connection symbol"}
INSTRUMENT_CLASSES = {"instrument tag", "instrument dcs", "instrument logic"}
# In-line elements the walker passes through instead of stopping at.
INLINE_CLASSES = {"gate valve", "globe valve", "check valve", "ball valve", "butterfly valve",
                  "control valve", "pressure relief valve", "reducer", "spectacle blind",
                  "strainer", "three way valve"}
# Never suppressed from the mask: they carry no pipe body and the walker uses them as evidence.
MASK_KEEP_CLASSES = {"arrow", "node", "line number"}


# ---------------------------------------------------------------------------
# mask helpers — borrowed probe primitives
# ---------------------------------------------------------------------------


def _is_pipe(mask: np.ndarray, x: int, y: int) -> bool:
    h, w = mask.shape
    return 0 <= x < w and 0 <= y < h and bool(mask[y, x])


def _is_pipe_band(mask: np.ndarray, x: int, y: int, direction: str, band_width: int = 3) -> bool:
    """Pipe present in a band perpendicular to travel; tolerates 1-3px scan offsets."""
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h):
        return False
    if direction in ("LEFT", "RIGHT"):
        return any(0 <= y + d < h and mask[y + d, x] > 0 for d in range(-band_width, band_width + 1))
    return any(0 <= x + d < w and mask[y, x + d] > 0 for d in range(-band_width, band_width + 1))


def _has_los(mask: np.ndarray, x: int, y: int, direction: str,
             distance: int, band_width: int = 1) -> bool:
    dx, dy = DIRECTION_DELTA[direction]
    for i in range(1, distance + 1):
        if not _is_pipe_band(mask, x + i * dx, y + i * dy, direction, band_width):
            return False
    return True


def _has_side_run(mask: np.ndarray, x: int, y: int, direction: str, min_run: int) -> bool:
    """A side branch *connected* at this point: adjacent pipe plus a continuous run away."""
    dx, dy = DIRECTION_DELTA[direction]
    h, w = mask.shape
    for lateral in (-1, 0, 1):
        if direction in ("LEFT", "RIGHT"):
            sx, sy = x + dx, y + lateral
        else:
            sx, sy = x + lateral, y + dy
        if not (0 <= sx < w and 0 <= sy < h) or mask[sy, sx] == 0:
            continue
        run = 0
        for step in range(1, min_run + 1):
            px, py = x + dx * step, y + dy * step
            if direction in ("LEFT", "RIGHT"):
                ok = any(0 <= py + o < h and 0 <= px < w and mask[py + o, px] > 0 for o in (-1, 0, 1))
            else:
                ok = any(0 <= px + o < w and 0 <= py < h and mask[py, px + o] > 0 for o in (-1, 0, 1))
            if not ok:
                break
            run += 1
        if run >= min_run:
            return True
    return False


def _inside(x: int, y: int, bbox: dict, margin: int = 0) -> bool:
    return (bbox["x_min"] - margin <= x <= bbox["x_max"] + margin
            and bbox["y_min"] - margin <= y <= bbox["y_max"] + margin)


def _boxes_overlap(a: tuple, b: dict) -> bool:
    """True if box `a` (x1,y1,x2,y2) overlaps bbox `b`."""
    return not (a[2] < b["x_min"] or a[0] > b["x_max"] or a[3] < b["y_min"] or a[1] > b["y_max"])


# ---------------------------------------------------------------------------
# pipe mask
# ---------------------------------------------------------------------------


def ocr_text_boxes(image_bgr: np.ndarray, *, confidence_threshold: float = 0.0) -> list[dict]:
    """Detect text boxes with macOS Vision (ocrmac), in raster pixel coordinates.

    The three fixtures give only 22 line-number boxes, but a sheet carries hundreds of text
    regions (notes, data blocks, title block, instrument tags). Measured on Test-00001 those
    unsuppressed strokes account for ~83% of the raw mask, and the walker follows them off the
    pipes — which is exactly why the real pipeline suppresses ~557 OCR regions here.
    Returns [] when ocrmac is unavailable, so the script still runs without it.
    """
    try:
        from ocrmac import ocrmac
    except Exception:
        return []
    from PIL import Image

    h, w = image_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    try:
        results = ocrmac.OCR(pil, recognition_level="fast",
                             confidence_threshold=confidence_threshold).recognize()
    except Exception:
        return []
    boxes = []
    for item in results:
        # ocrmac returns (text, confidence, (x, y, w, h)) in NORMALIZED coords with y from bottom
        if not (isinstance(item, (list, tuple)) and len(item) >= 3):
            continue
        xywh = item[2]
        if not (isinstance(xywh, (list, tuple)) and len(xywh) == 4):
            continue
        nx, ny, nw, nh = (float(v) for v in xywh)
        x = int(nx * w)
        bw = int(nw * w)
        bh = int(nh * h)
        y = int((1.0 - ny - nh) * h)          # flip to top-left origin
        if bw > 0 and bh > 0:
            boxes.append({"x_min": max(0, x), "y_min": max(0, y),
                          "x_max": min(w, x + bw), "y_max": min(h, y + bh)})
    return boxes


def build_pipe_mask(image_bgr: np.ndarray, objects: list[dict], line_numbers: list[dict],
                    equipment: list[dict], *, ocr_padding: int = 1,
                    object_inset: int = 1, inline_inset: int = 12,
                    min_area: int = 16, blur_kernel: int = 5,
                    adaptive_block_size: int = 21, adaptive_c: int = 5,
                    equipment_inset: int = 0,
                    extra_text_boxes: list[dict] | None = None,
                    debug_stats: dict | None = None) -> np.ndarray:
    """Binarize the raster, then erase text and object glyphs so only pipework remains.

    Mirrors the intent of backend/garnet/pipe_mask.py but reads its suppression boxes from
    the fixtures rather than pipeline artifacts. Binarization parameters are the project's
    accepted baseline (PipelineConfig: blur 5, adaptive block 21, C 5), not arbitrary values —
    a looser threshold leaves text strokes and hatching that derail the walker.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(adaptive, otsu)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    h, w = mask.shape
    suppressed_line_numbers = 0

    def blank(x1, y1, x2, y2):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    # 1. Erase line-number text (fixture is pixel xywh; some labels are rotated).
    for ln in line_numbers:
        try:
            x, y = int(ln["Left"]), int(ln["Top"])
            bw, bh = int(ln["Width"]), int(ln["Height"])
        except (KeyError, TypeError, ValueError):
            continue
        blank(x - ocr_padding, y - ocr_padding,
              x + bw + ocr_padding, y + bh + ocr_padding)
        suppressed_line_numbers += 1

    # 1b. Erase everything OCR found (notes, data blocks, title block, tags). Without this the
    #     walker leaves the pipes and follows text — the fixtures alone are not enough.
    suppressed_ocr = 0
    for box in (extra_text_boxes or []):
        blank(box["x_min"] - ocr_padding, box["y_min"] - ocr_padding,
              box["x_max"] + ocr_padding, box["y_max"] + ocr_padding)
        suppressed_ocr += 1

    # 2. Erase object glyphs. Three regimes:
    #      in-line symbol  -> blank only the INTERIOR (inset), so the pipe stub ring survives
    #                         and the walker can pass through the symbol
    #      arrows / nodes  -> keep: they carry no pipe body and are topology evidence
    #      anything else   -> blank the whole glyph box (+ padding), so the walker never
    #                         follows instrument bubbles, notes or connectors
    #    Equipment interiors are NOT blanked by default (equipment_inset=0). Blanking them
    #    stops the walker running along a symbol outline, but it also severs the pipe stubs that
    #    meet the nozzles: measured on Test-00001, blanking dropped total traced length from
    #    10,670 px to 3,757 px and produced three "no_pipe" starts. Leaving the outlines in
    #    costs some branch quality (see the branch filter in main) but keeps the main network
    #    connected, which matters more. Set equipment_inset>0 to trade the other way.
    for item in equipment:
        if equipment_inset <= 0:
            break
        eb = item["bbox"]
        blank(eb["x_min"] + equipment_inset, eb["y_min"] + equipment_inset,
              eb["x_max"] - equipment_inset, eb["y_max"] - equipment_inset)
    for obj in objects:
        cls = str(obj.get("class_name", "")).strip().lower()
        if cls in MASK_KEEP_CLASSES:
            continue
        bb = obj.get("bbox") or {}
        if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bb):
            continue
        x1, y1, x2, y2 = (int(bb[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        if cls in INLINE_CLASSES:
            if inline_inset:
                blank(x1 + inline_inset, y1 + inline_inset, x2 - inline_inset, y2 - inline_inset)
        else:
            blank(x1 - ocr_padding, y1 - ocr_padding, x2 + ocr_padding, y2 + ocr_padding)

    # 3. Drop small specks, then join 1-2px corner gaps.
    #    NOTE: no shape-based text filter here. Measured on this sheet, pipe strokes are
    #    1-3px half-width and text strokes are 1.4-2px — indistinguishable by shape. The real
    #    pipeline removes text with ~557 OCR text regions; from these three fixtures we only
    #    have the line-number boxes plus the text-like object classes, so text suppression is
    #    necessarily partial. Morphological opening at 5x5 would remove 82% of the mask and
    #    with it the pipework, so it is deliberately not used.
    before = int((mask > 0).sum())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    removed = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
            removed += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # 4. Bridge each equipment nozzle to the nearest pipe pixel OUTSIDE its own bbox, so the
    #    walk can start at the nozzle and enter the network. Restricting to outside-the-box
    #    matters now that equipment interiors are blanked: otherwise the nearest ink is the
    #    symbol's own remaining border and the bridge just re-links the box to itself.
    bridges = 0
    if equipment:
        for item in equipment:
            eb = item["bbox"]
            for port in item.get("_ports", []):
                px, py = int(port["x"]), int(port["y"])
                # search outward from the nozzle for the nearest pipe pixel not inside the box
                found = None
                for radius in (2, 4, 6, 8, 12, 16, 22, 30, 40, 55, 70, 80):
                    for dx in range(-radius, radius + 1):
                        for dy in (-radius, radius):
                            qx, qy = px + dx, py + dy
                            if not (0 <= qx < w and 0 <= qy < h) or mask[qy, qx] == 0:
                                continue
                            # must be genuinely OUTSIDE the equipment symbol (margin=0), so the
                            # bridge reaches the connecting pipe rather than the symbol border
                            if _inside(qx, qy, eb, margin=0):
                                continue
                            found = (qx, qy)
                            break
                        if found:
                            break
                    if found:
                        break
                if found:
                    # Draw an L-shaped (axis-aligned) bridge, not a diagonal. Every trace
                    # segment must stay orthogonal so downstream consumers can treat
                    # direction as one of UP/DOWN/LEFT/RIGHT; a diagonal would break that.
                    qx, qy = found
                    if abs(qx - px) >= abs(qy - py):
                        cv2.line(mask, (px, py), (qx, py), 255, 3)
                        cv2.line(mask, (qx, py), (qx, qy), 255, 3)
                    else:
                        cv2.line(mask, (px, py), (px, qy), 255, 3)
                        cv2.line(mask, (px, qy), (qx, qy), 255, 3)
                    bridges += 1

    if debug_stats is not None:
        debug_stats.update({
            "mask_px_before_cleanup": before,
            "mask_px": int((mask > 0).sum()),
            "pct_of_sheet": round(float((mask > 0).mean()) * 100, 2),
            "components_removed": removed,
            "line_numbers_suppressed": suppressed_line_numbers,
            "ocr_boxes_suppressed": suppressed_ocr,
            "nozzle_bridges": bridges,
            "binarize": {"blur_kernel": blur_kernel,
                         "adaptive_block_size": adaptive_block_size,
                         "adaptive_c": adaptive_c,
                         "min_component_area": min_area},
        })
    return mask


# ---------------------------------------------------------------------------
# walker — the borrowed algorithm
# ---------------------------------------------------------------------------


@dataclass
class TraceResult:
    terminal_type: str | None = None
    terminal_x: int = 0
    terminal_y: int = 0
    terminal_obj_id: str | None = None
    segments: list[dict] = field(default_factory=list)
    turns: list[tuple[int, int, str]] = field(default_factory=list)
    status: str = "ok"
    trace_length_px: int = 0


class PipeWalker:
    """Walk one pipe path from a start point and direction to a terminal.

    Algorithm borrowed from CVPipeTracer.trace(): centreline snap -> loop guard -> inline
    pass-through -> banded forward LOS -> terminal check -> turn resolution -> raycast gap
    jump -> dead end. Reimplemented compactly and driven by fixture bounding boxes.
    """

    def __init__(self, mask: np.ndarray, *, terminals: dict[str, dict],
                 inline: list[dict], min_step: int = 3, max_steps: int = 6000,
                 centerline_radius_px: int = 8, lookahead_px: int = 3,
                 turn_run_min_px: int = 6, raycast_max_px: int = 50,
                 sheet_margin_px: int = 4, visited: np.ndarray | None = None) -> None:
        self.mask = mask
        self.h, self.w = mask.shape
        self.terminals = terminals          # obj_id -> {"bbox", "type", "class"}
        self._terminal_list = list(terminals.items())
        self.inline = inline
        self.min_step = min_step
        self.max_steps = max_steps
        self.centerline_radius_px = centerline_radius_px
        self.lookahead_px = lookahead_px
        self.turn_run_min_px = turn_run_min_px
        self.raycast_max_px = raycast_max_px
        self.sheet_margin_px = sheet_margin_px
        self.visited = visited if visited is not None else np.zeros_like(mask)

    # -- centreline snapping (adapted) -------------------------------------
    def _support(self, x: int, y: int, direction: str) -> int:
        r = self.centerline_radius_px
        if direction in ("UP", "DOWN"):
            return sum(1 for cy in range(y - r, y + r + 1) if _is_pipe(self.mask, x, cy))
        return sum(1 for cx in range(x - r, x + r + 1) if _is_pipe(self.mask, cx, y))

    @staticmethod
    def _centre_of_run(values: list[int], target: int) -> int | None:
        if not values:
            return None
        runs, cur = [], []
        for v in sorted(values):
            if not cur or v == cur[-1] + 1:
                cur.append(v)
                continue
            runs.append(cur)
            cur = [v]
        if cur:
            runs.append(cur)
        best = min(runs, key=lambda rn: (
            0 if rn[0] <= target <= rn[-1] else min(abs(target - rn[0]), abs(target - rn[-1])),
            abs((rn[0] + rn[-1]) / 2.0 - target)))
        return int(round((best[0] + best[-1]) / 2.0))

    def _snap(self, x: int, y: int, direction: str) -> tuple[int, int]:
        r = self.centerline_radius_px
        if direction in ("UP", "DOWN"):
            cands = [cx for cx in range(x - r, x + r + 1) if _is_pipe(self.mask, cx, y)]
            scored = [(c, self._support(c, y, direction)) for c in sorted(set(cands))]
        else:
            cands = [cy for cy in range(y - r, y + r + 1) if _is_pipe(self.mask, x, cy)]
            scored = [(c, self._support(x, c, direction)) for c in sorted(set(cands))]
        if not scored:
            return x, y
        best = max(s for _, s in scored)
        if best <= 0:
            return x, y
        strong = [c for c, s in scored if s == best]
        centre = self._centre_of_run(strong, x if direction in ("UP", "DOWN") else y)
        if centre is None:
            return x, y
        return (centre, y) if direction in ("UP", "DOWN") else (x, centre)

    # -- helpers -----------------------------------------------------------
    def _append_segment(self, res: TraceResult, x1: int, y1: int, x2: int, y2: int,
                        direction: str) -> None:
        """Append one or two ORTHOGONAL segments between the two points.

        Centreline snapping can shift a point sideways while travelling, which produced a
        diagonal single segment (and a `direction` that disagreed with the geometry). Splitting
        it keeps every segment axis-aligned, so `direction` is always truthful.
        """
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x1 == x2 and y1 == y2:
            return

        def emit(ax, ay, bx, by, dir_hint):
            if ax == bx and ay == by:
                return
            if ax == bx:
                d = "UP" if by < ay else "DOWN"
            elif ay == by:
                d = "LEFT" if bx < ax else "RIGHT"
            else:                                     # shouldn't happen; split by caller
                d = dir_hint
            res.segments.append({"x1": ax, "y1": ay, "x2": bx, "y2": by,
                                 "direction": d,
                                 "length_px": int(abs(bx - ax) + abs(by - ay))})
            res.trace_length_px += int(abs(bx - ax) + abs(by - ay))

        if x1 == x2 or y1 == y2:
            emit(x1, y1, x2, y2, direction)
        else:
            # take the longer axis first so the elbow sits near the original direction
            if abs(x2 - x1) >= abs(y2 - y1):
                emit(x1, y1, x2, y1, direction)
                emit(x2, y1, x2, y2, direction)
            else:
                emit(x1, y1, x1, y2, direction)
                emit(x1, y2, x2, y2, direction)

    def _terminal_at(self, x: int, y: int, source_obj_id: str,
                     margin: int = 4) -> tuple[str, str, dict] | None:
        for tid, info in self._terminal_list:
            if tid == source_obj_id:
                continue                     # the object we started from is not a terminal
            if _inside(x, y, info["bbox"], margin):
                return tid, info["type"], info["bbox"]
        return None

    def _run_length(self, x: int, y: int, direction: str, max_run: int) -> int:
        """How many pixels of pipe continue from (x,y) in `direction` (band tolerance 1)."""
        dx, dy = DIRECTION_DELTA[direction]
        run = 0
        for step in range(1, max_run + 1):
            if not _is_pipe_band(self.mask, x + dx * step, y + dy * step, direction, 1):
                break
            run += 1
        return run

    def is_pipe_axis(self, x: int, y: int, direction: str, min_run: int) -> bool:
        """Public check used by branch discovery: is there a genuine pipe run along `direction`?

        Text strokes and short flanges give a run of a few px; a real branch gives a sustained
        run. Branch discovery keyed on the raw band test alone produced branches lying on text.
        """
        return _is_pipe_band(self.mask, x, y, direction, 1) and \
            self._run_length(x, y, direction, min_run) >= min_run

    def _inline_exit(self, x: int, y: int, direction: str) -> tuple[int, int] | None:
        """If inside an in-line symbol, jump to its far edge along the travel axis."""
        for obj in self.inline:
            if not _inside(x, y, obj["bbox"]):
                continue
            bb = obj["bbox"]
            if direction in ("LEFT", "RIGHT"):
                nx = bb["x_min"] - 1 if direction == "LEFT" else bb["x_max"] + 1
                return int(nx), y
            ny = bb["y_min"] - 1 if direction == "UP" else bb["y_max"] + 1
            return x, int(ny)
        return None

    # -- the walk ----------------------------------------------------------
    def trace(self, start_x: int, start_y: int, direction: str,
              source_obj_id: str = "") -> TraceResult:
        res = TraceResult()
        x, y = int(start_x), int(start_y)

        # Enter the pipe: accept the given direction if pipe is there, else try the opposite.
        if not _is_pipe_band(self.mask, x, y, direction, 3):
            alt = OPPOSITE[direction]
            if _is_pipe_band(self.mask, x, y, alt, 3):
                direction = alt
            else:
                # nudge onto the nearest pipe pixel within a small window
                found = None
                for r in range(1, 12):
                    for dx in range(-r, r + 1):
                        for dy in (-r, r):
                            if _is_pipe(self.mask, x + dx, y + dy):
                                found = (x + dx, y + dy)
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    res.status = "no_pipe"
                    res.terminal_type = "no_pipe"
                    res.terminal_x, res.terminal_y = x, y
                    return res
                x, y = found
        x, y = self._snap(x, y, direction)
        seg_x, seg_y = x, y
        dx, dy = DIRECTION_DELTA[direction]

        steps = 0
        state_counts: dict[tuple[int, int, str], int] = {}
        while steps < self.max_steps:
            steps += 1
            if 0 <= y < self.h and 0 <= x < self.w:
                self.visited[y, x] = 1

            x, y = self._snap(x, y, direction)

            # loop guard: repeat the same coarse cell/pose too often -> dead end
            key = (int(round(x / 3)), int(round(y / 3)), direction)
            state_counts[key] = state_counts.get(key, 0) + 1
            if state_counts[key] > 3:
                self._append_segment(res, seg_x, seg_y, x, y, direction)
                res.terminal_type, res.terminal_x, res.terminal_y = "dead_end", x, y
                break

            # sheet edge
            if (x <= self.sheet_margin_px or x >= self.w - self.sheet_margin_px
                    or y <= self.sheet_margin_px or y >= self.h - self.sheet_margin_px):
                self._append_segment(res, seg_x, seg_y, x, y, direction)
                res.terminal_type, res.terminal_x, res.terminal_y = "sheet_edge", x, y
                break

            # in-line symbol -> pass through, don't stop
            jump = self._inline_exit(x, y, direction)
            if jump is not None:
                self._append_segment(res, seg_x, seg_y, x, y, direction)
                x, y = self._snap(jump[0], jump[1], direction)
                seg_x, seg_y = x, y
                dx, dy = DIRECTION_DELTA[direction]
                continue

            # Forward test. The real tracer's `forward_ok` only requires a short banded look
            # ahead; requiring a long run here is what made every trace stop 5-7px after a
            # turn leg. A short run is enough to keep walking; the turn/raycast logic below
            # copes when the pipe genuinely ends.
            forward = _has_los(self.mask, x, y, direction, self.lookahead_px, band_width=1)
            if forward:
                for _ in range(self.min_step):
                    x += dx
                    y += dy
                    if 0 <= y < self.h and 0 <= x < self.w:
                        self.visited[y, x] = 1
                x, y = self._snap(x, y, direction)
                continue

            # forward blocked -> is a terminal here?
            hit = self._terminal_at(x, y, source_obj_id)
            if hit is not None:
                tid, ttype, _ = hit
                self._append_segment(res, seg_x, seg_y, x, y, direction)
                res.terminal_type, res.terminal_x, res.terminal_y = ttype, x, y
                res.terminal_obj_id = tid
                break

            # Turn resolution. Three cases, in the order the real tracer tries them:
            #   1. exactly one connected side  -> turn into it
            #   2. BOTH sides connected (a tee / crossing while travelling) -> this is a junction.
            #      Pick the side that is a genuine branch: score by run length and prefer the
            #      perpendicular turn that actually continues, then stop as a tee_junction.
            #      (An XOR test alone returns nothing here and the walk dies at every tee,
            #      which is what the first version did.)
            #   3. neither side -> try a raycast gap jump below.
            left, right = TURN_LEFT[direction], TURN_RIGHT[direction]
            left_run = self._run_length(x, y, left, self.turn_run_min_px * 2)
            right_run = self._run_length(x, y, right, self.turn_run_min_px * 2)
            left_ok = left_run >= self.turn_run_min_px
            right_ok = right_run >= self.turn_run_min_px

            if left_ok and right_ok:
                # tee: record the junction and continue along the stronger side
                turn_dir = left if left_run >= right_run else right
                self._append_segment(res, seg_x, seg_y, x, y, direction)
                res.turns.append((int(x), int(y), turn_dir))
                tdx, tdy = DIRECTION_DELTA[turn_dir]
                nx, ny = x + tdx * self.min_step, y + tdy * self.min_step
                x, y = self._snap(nx, ny, turn_dir)
                direction = turn_dir
                dx, dy = tdx, tdy
                seg_x, seg_y = x, y
                continue

            if left_ok ^ right_ok:
                turn_dir = left if left_ok else right
                self._append_segment(res, seg_x, seg_y, x, y, direction)
                res.turns.append((int(x), int(y), turn_dir))
                tdx, tdy = DIRECTION_DELTA[turn_dir]
                nx, ny = x + tdx * self.min_step, y + tdy * self.min_step
                x, y = self._snap(nx, ny, turn_dir)
                direction = turn_dir
                dx, dy = tdx, tdy
                seg_x, seg_y = x, y
                continue

            # Raycast: bridge a gap only if the landing pixel has a real continuing run ahead,
            # and the snap stays on-axis. Accepting a bare pipe pixel instead would let the
            # walker jump onto any nearby stroke (including text) and derail.
            jumped = False
            rdx, rdy = DIRECTION_DELTA[direction]
            for gap in range(self.lookahead_px + 1, self.raycast_max_px + 1):
                px, py = x + rdx * gap, y + rdy * gap
                if not (0 <= px < self.w and 0 <= py < self.h):
                    break
                if not _is_pipe_band(self.mask, px, py, direction, band_width=2):
                    continue
                if self._run_length(px, py, direction, self.min_step * 2) < self.min_step:
                    continue                      # not a continuing pipe, just a stroke
                nx, ny = self._snap(px, py, direction)
                if direction in ("UP", "DOWN"):
                    if abs(nx - x) > 4:
                        continue
                elif abs(ny - y) > 4:
                    continue
                self._append_segment(res, seg_x, seg_y, x, y, direction)
                x, y = nx, ny
                seg_x, seg_y = x, y
                jumped = True
                break
            if jumped:
                continue

            self._append_segment(res, seg_x, seg_y, x, y, direction)
            res.terminal_type, res.terminal_x, res.terminal_y = "dead_end", x, y
            break

        if res.terminal_type is None:
            self._append_segment(res, seg_x, seg_y, x, y, direction)
            res.terminal_type, res.terminal_x, res.terminal_y = "max_steps", x, y
        return res


# ---------------------------------------------------------------------------
# fixture loading
# ---------------------------------------------------------------------------


def load_fixtures(indir: Path, stem: str) -> tuple[list[dict], list[dict], list[dict]]:
    def read(name, *alts):
        for n in (name, *alts):
            p = indir / n.format(stem=stem)
            if p.is_file():
                return json.loads(p.read_text())
        raise FileNotFoundError(f"no fixture for {name.format(stem=stem)} in {indir}")

    objects = read("{stem}_objects.json").get("objects", [])
    equipment = read("{stem}_equipment_bboxes.json").get("objects", [])
    line_numbers = read("{stem}_line_number_boxes.json", "{stem}_line_numbers.json").get("objects", [])
    return objects, equipment, line_numbers


def equipment_items(equipment: list[dict]) -> list[dict]:
    """Normalise Contract B equipment into {id, class_name, bbox, _ports}."""
    out = []
    for i, item in enumerate(equipment):
        bb = item.get("Bounding_box_px") or {}
        if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bb):
            continue
        tag = str(item.get("Tag") or "").strip()
        eq_id = f"equip_{tag.lower().replace('-', '_') or f'{i:03d}'}"
        ports = []
        for p in item.get("Ports") or []:
            side = str(p.get("side") or "").upper()
            pt = p.get("point_px")
            if side in DIRECTION_DELTA and isinstance(pt, (list, tuple)) and len(pt) == 2:
                ports.append({"x": int(pt[0]), "y": int(pt[1]), "direction": side})
        out.append({
            "id": eq_id,
            "class_name": str(item.get("Equipment_type") or "equipment").lower(),
            "bbox": {k: int(bb[k]) for k in ("x_min", "y_min", "x_max", "y_max")},
            "tag": tag,
            "_ports": ports,
        })
    return out


def connection_ports(image: np.ndarray, objects: list[dict], mask: np.ndarray,
                     *, min_run_px: int = 10) -> dict[str, list[dict]]:
    """Find where a real pipe meets a page/utility connection symbol.

    Scans outward from the symbol centre and accepts the first hit that has a genuine
    CONTINUING pipe run — not merely a pipe pixel. A bare nearest-pixel test lands on text or
    a stray stroke and the trace then dies after a few pixels (observed: 6 traces of 13-56 px
    starting on text). Requiring a run of `min_run_px` past the hit rejects those.
    """
    ports: dict[str, list[dict]] = {}
    h, w = mask.shape
    for obj in objects:
        cls = str(obj.get("class_name", "")).strip().lower()
        if cls not in PAGE_CONNECTION_CLASSES:
            continue
        bb = obj.get("bbox") or {}
        if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bb):
            continue
        x1, y1, x2, y2 = (int(bb[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        best = None
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            dx, dy = DIRECTION_DELTA[d]
            for step in range(1, 61):
                px, py = cx + dx * step, cy + dy * step
                if not (0 <= px < w and 0 <= py < h):
                    break
                if not _is_pipe_band(mask, px, py, d, 1):
                    continue
                # require the pipe to actually continue from here
                run = 0
                for s2 in range(1, min_run_px + 1):
                    qx, qy = px + dx * s2, py + dy * s2
                    if not (0 <= qx < w and 0 <= qy < h) or not _is_pipe_band(mask, qx, qy, d, 1):
                        break
                    run += 1
                if run >= min_run_px and (best is None or step < best[0]):
                    best = (step, px, py, d)
                break
        if best:
            ports[obj["id"]] = [{"x": int(best[1]), "y": int(best[2]), "direction": best[3]}]
    return ports


def line_number_texts(edge: dict, line_numbers: list[dict], tolerance_px: int = 30) -> list[str]:
    """Line-number labels whose box sits alongside this edge — the trace's line attachment."""
    out = []
    for seg in edge["segments"]:
        mx = (seg["x1"] + seg["x2"]) // 2
        my = (seg["y1"] + seg["y2"]) // 2
        for ln in line_numbers:
            try:
                lx, ly = int(ln["Left"]), int(ln["Top"])
                lw, lh = int(ln["Width"]), int(ln["Height"])
            except (KeyError, TypeError, ValueError):
                continue
            if (lx - tolerance_px <= mx <= lx + lw + tolerance_px
                    and ly - tolerance_px <= my <= ly + lh + tolerance_px):
                txt = str(ln.get("Text") or "").strip()
                if txt and txt not in out:
                    out.append(txt)
    return out


# ---------------------------------------------------------------------------
# overlays
# ---------------------------------------------------------------------------


def draw_trace_overlay(image, traces: dict, terminals: dict, equipment: list[dict],
                       branch: dict | None = None) -> np.ndarray:
    ov = image.copy()
    for item in equipment:
        bb = item["bbox"]
        cv2.rectangle(ov, (bb["x_min"], bb["y_min"]), (bb["x_max"], bb["y_max"]), (0, 140, 255), 3)

    for tid, res in traces.items():
        for seg in res["segments"]:
            cv2.line(ov, (seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]), (0, 200, 0), 3)
        p = res["port"]
        cv2.circle(ov, (p["x"], p["y"]), 7, (0, 255, 0), -1)
        cv2.circle(ov, (p["x"], p["y"]), 7, (255, 255, 255), 2)
        tx, ty = res["terminal_x"], res["terminal_y"]
        ttype = str(res.get("terminal_type") or "unknown")
        colour = {"equipment": (0, 140, 255), "page_connection": (0, 180, 0),
                  "utility connection": (0, 180, 0), "instrument_tag": (255, 0, 200),
                  "dead_end": (0, 0, 220), "sheet_edge": (128, 128, 128),
                  "no_pipe": (0, 0, 0)}.get(ttype, (160, 160, 160))
        cv2.circle(ov, (tx, ty), 9, colour, -1)
        cv2.circle(ov, (tx, ty), 9, (255, 255, 255), 2)
        cv2.putText(ov, f"{res['source_obj_id']}:{ttype}", (tx + 12, ty - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

    if branch:
        for bid, res in branch.items():
            for seg in res["segments"]:
                cv2.line(ov, (seg["x1"], seg["y1"]), (seg["x2"], seg["y2"]), (0, 0, 255), 3)
            p = res["port"]
            cv2.circle(ov, (p["x"], p["y"]), 6, (0, 165, 255), -1)
            cv2.putText(ov, bid, (p["x"] + 10, p["y"] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return ov


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Trace pipes from the garnet/tests/input fixtures.")
    p.add_argument("--stem", default="Test-00001")
    p.add_argument("--indir", default="garnet/tests/input")
    p.add_argument("--out", default=None, help="Default: output/trace_<stem>")
    p.add_argument("--branch-iterations", type=int, default=4)
    p.add_argument("--branch-min-run-px", type=int, default=28)
    p.add_argument("--branch-sample-px", type=int, default=16,
                   help="Stride along a traced segment when hunting for branch starts. Default: 16")
    p.add_argument("--max-branches", type=int, default=160,
                   help="Hard cap on discovered branches. Default: 160")
    p.add_argument("--no-ocr", action="store_true",
                   help="Skip ocrmac text suppression (faster, but the walker will follow "
                        "text strokes — the fixtures only cover line numbers).")
    args = p.parse_args(argv)

    indir = Path(args.indir)
    if not indir.is_absolute():
        indir = REPO_ROOT / indir
    out_dir = Path(args.out) if args.out else REPO_ROOT / "output" / f"trace_{args.stem}"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = indir / f"{args.stem}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"error: cannot read {image_path}", file=sys.stderr)
        return 1
    h, w = image.shape[:2]

    objs, eq_raw, lns = load_fixtures(indir, args.stem)
    equipment = equipment_items(eq_raw)
    print(f"{args.stem}: {w}x{h} | {len(objs)} objects, {len(equipment)} equipment, {len(lns)} line numbers")

    mask_stats: dict = {}
    text_boxes = [] if args.no_ocr else ocr_text_boxes(image)
    if text_boxes:
        print(f"OCR: {len(text_boxes)} text boxes to suppress")
    elif not args.no_ocr:
        print("OCR: unavailable (ocrmac missing?) — text suppression is fixture-only")
    mask = build_pipe_mask(image, objs, lns, equipment, extra_text_boxes=text_boxes,
                           debug_stats=mask_stats)
    cv2.imwrite(str(out_dir / "pipe_mask.png"), mask)
    print(f"pipe mask: {mask_stats['mask_px']:,} px ({mask_stats['pct_of_sheet']}% of sheet), "
          f"{mask_stats['components_removed']} specks removed, "
          f"{mask_stats['line_numbers_suppressed']} line-number boxes + "
          f"{mask_stats['ocr_boxes_suppressed']} OCR boxes suppressed, "
          f"{mask_stats['nozzle_bridges']} nozzle bridges")
    (out_dir / "pipe_mask_stats.json").write_text(json.dumps(mask_stats, indent=2), encoding="utf-8")

    # terminals: equipment bboxes + connection/instrument objects
    terminals: dict[str, dict] = {}
    for item in equipment:
        terminals[item["id"]] = {"bbox": item["bbox"], "type": "equipment",
                                 "class": item["class_name"]}
    for obj in objs:
        cls = str(obj.get("class_name", "")).strip().lower()
        bb = obj.get("bbox") or {}
        if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bb):
            continue
        box = {k: int(bb[k]) for k in ("x_min", "y_min", "x_max", "y_max")}
        if cls in PAGE_CONNECTION_CLASSES:
            terminals[obj["id"]] = {"bbox": box,
                                    "type": "utility connection" if "utility" in cls else "page_connection",
                                    "class": cls}
        elif cls in INSTRUMENT_CLASSES:
            terminals[obj["id"]] = {"bbox": box, "type": "instrument_tag", "class": cls}
    print(f"terminals: {len(terminals)}")

    inline = [{"bbox": {k: int(o["bbox"][k]) for k in ("x_min", "y_min", "x_max", "y_max")}}
              for o in objs
              if str(o.get("class_name", "")).strip().lower() in INLINE_CLASSES
              and {"x_min", "y_min", "x_max", "y_max"}.issubset(o.get("bbox") or {})]

    # starts: equipment nozzles, then page/utility connection ports
    starts: list[tuple[str, dict, int]] = []
    for item in equipment:
        for i, port in enumerate(item["_ports"], start=1):
            starts.append((item["id"], port, i))
    for oid, plist in connection_ports(image, objs, mask).items():
        for i, port in enumerate(plist, start=1):
            starts.append((oid, port, i))
    print(f"trace starts: {len(starts)}")

    visited = np.zeros_like(mask)
    walker = PipeWalker(mask, terminals=terminals, inline=inline, visited=visited)
    traces: dict[str, dict] = {}
    for oid, port, idx in starts:
        tid = oid if idx == 1 else f"{oid}:port_{idx:02d}"
        res = walker.trace(port["x"], port["y"], port["direction"], source_obj_id=oid)
        traces[tid] = {
            "source_obj_id": oid, "port_index": idx,
            "port": {"x": int(port["x"]), "y": int(port["y"]), "direction": port["direction"]},
            "terminal_type": res.terminal_type, "terminal_x": res.terminal_x,
            "terminal_y": res.terminal_y, "terminal_obj_id": res.terminal_obj_id,
            "segments": [dict(s) for s in res.segments],
            "turns": [{"x": t[0], "y": t[1], "new_dir": t[2]} for t in res.turns],
            "trace_length_px": res.trace_length_px, "status": res.status,
        }

    # Branch discovery: walk the traced polylines looking for perpendicular side runs.
    # NOTE: the `visited` mask is intentionally NOT consulted here. Port traces share one
    # `visited`, so by this point every main-line pixel is claimed and a `not visited` gate
    # suppresses all branch discovery (observed: 0 branches). What matters is whether a
    # perpendicular pipe run leaves the traced line — a tee — not whether the main line was
    # walked.
    #
    # Discovery must be BOUNDED or it never converges: with a 10px sampling stride a single
    # tee yields many near-identical starts, and newly-added branches seed more starts, so the
    # count kept growing every iteration (34, 96, 173, 229, ...). Dedupe starts by a coarse
    # grid cell + direction, and cap the total.
    branch: dict[str, dict] = {}
    seen_cells: set[tuple[int, int, str]] = set()
    for it in range(args.branch_iterations):
        if len(branch) >= args.max_branches:
            break
        found = 0
        for res in list(traces.values()) + list(branch.values()):
            for seg in res["segments"]:
                d = seg["direction"]
                steps = max(1, seg["length_px"] // args.branch_sample_px)
                for k in range(0, steps + 1):
                    t = k / steps
                    x = int(seg["x1"] + (seg["x2"] - seg["x1"]) * t)
                    y = int(seg["y1"] + (seg["y2"] - seg["y1"]) * t)
                    for turn in (TURN_LEFT[d], TURN_RIGHT[d]):
                        # coarse cell (~24px) so one tee produces one branch, not ten
                        cell = (x // 24, y // 24, turn)
                        if cell in seen_cells:
                            continue
                        # Require a SUSTAINED perpendicular run, not merely a band hit. Text
                        # strokes sit next to pipes all over the sheet and were producing
                        # branches lying on text.
                        if not walker.is_pipe_axis(x, y, turn, args.branch_min_run_px):
                            continue
                        # Reject candidates sitting on an equipment symbol. An equipment
                        # outline is drawn ink, so a "branch" off a nozzle walks down the
                        # vessel wall — observed as branches claiming 800+ px inside V-2501.
                        # Nozzles are already trace STARTS; a branch from them is spurious.
                        if any(_inside(x, y, it["bbox"], margin=8) for it in equipment):
                            continue
                        seen_cells.add(cell)
                        bid = f"branch_{len(branch) + 1:03d}"
                        r = walker.trace(x, y, turn, source_obj_id=bid)
                        if r.trace_length_px < args.branch_min_run_px:
                            continue
                        branch[bid] = {
                            "source_obj_id": bid, "from": d,
                            "port": {"x": x, "y": y, "direction": turn},
                            "terminal_type": r.terminal_type, "terminal_x": r.terminal_x,
                            "terminal_y": r.terminal_y, "terminal_obj_id": r.terminal_obj_id,
                            "segments": [dict(s) for s in r.segments],
                            "turns": [{"x": t2[0], "y": t2[1], "new_dir": t2[2]} for t2 in r.turns],
                            "trace_length_px": r.trace_length_px, "status": r.status,
                        }
                        found += 1
                        if len(branch) >= args.max_branches:
                            break
                    if len(branch) >= args.max_branches:
                        break
                if len(branch) >= args.max_branches:
                    break
            if len(branch) >= args.max_branches:
                break
        print(f"  branch iteration {it + 1}: +{found} (total {len(branch)})")
        if not found:
            break

    # line-number attachment per trace
    for res in list(traces.values()) + list(branch.values()):
        res["line_numbers"] = line_number_texts(res, lns)

    (out_dir / "trace_results.json").write_text(json.dumps(traces, indent=2), encoding="utf-8")
    (out_dir / "branch_results.json").write_text(json.dumps(branch, indent=2), encoding="utf-8")
    cv2.imwrite(str(out_dir / "trace_overlay.png"),
                draw_trace_overlay(image, traces, terminals, equipment))
    cv2.imwrite(str(out_dir / "branch_overlay.png"),
                draw_trace_overlay(image, traces, terminals, equipment, branch=branch))

    from collections import Counter
    print(f"\ntraces: {len(traces)}")
    for k, v in Counter(str(t.get("terminal_type")) for t in traces.values()).most_common():
        print(f"  {v:>4}  {k}")
    print(f"branches: {len(branch)}")
    for k, v in Counter(str(t.get("terminal_type")) for t in branch.values()).most_common():
        print(f"  {v:>4}  {k}")
    total = sum(t["trace_length_px"] for t in traces.values())
    btotal = sum(t["trace_length_px"] for t in branch.values())
    print(f"port-trace length: {total:,} px | branch length: {btotal:,} px")
    with_ln = sum(1 for t in traces.values() if t["line_numbers"])
    print(f"traces with a line-number attachment: {with_ln}/{len(traces)}")
    print(f"\nOK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
