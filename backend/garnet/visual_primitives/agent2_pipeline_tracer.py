"""Agent 2 — Pipeline Tracer (Visual Primitive, Step-by-Step).

Drives a VLM to trace pipe lines pixel-by-pixel from Stage 4 page-connection
anchors.  Each step crops a local view, the VLM decides the next direction/
distance, and the cursor advances until a terminal is reached.

Usage:
    python -m garnet.visual_primitives.agent2_pipeline_tracer \
        --image test/ppcl/Test-00001.jpg \
        --stage4 ../output/stage4_objects.json \
        --output ../output/stage_vp2_test01/
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

import cv2

from .cursor import PipelineCursor
from .prompts import (
    AGENT2_SYSTEM_PROMPT,
    AGENT2_USER_PROMPT_TEMPLATE,
    PORT_FINDER_SYSTEM,
    PORT_FINDER_USER,
)
from .schemas import TraceDirection, TraceResult, TraceSegment, TraceStep, TraceTokenType
from .trace_parser import has_terminal, last_terminal, parse_trace_response, total_trace_distance

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("agent2")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = 120
DEFAULT_CROP_SIZE = 300
DEFAULT_MAX_STEPS_PER_SEGMENT = 50

# Sheet-edge detection: if a page connection bbox edge is within this many
# pixels of the image edge, it's a boundary anchor.
SHEET_EDGE_THRESHOLD_PX = 500


# ---------------------------------------------------------------------------
# Port computation
# ---------------------------------------------------------------------------


def compute_port_from_bbox(
    bbox: dict[str, int],
    image_w: int,
    image_h: int,
) -> tuple[int, int, str]:
    """Determine the pipe exit point and direction from a page-connection bbox.

    Page connections sit at sheet edges.  The port is the midpoint of the
    inner bbox edge (the one facing into the drawing).

    Returns (x, y, direction) in original image pixel coordinates.
    """
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    center_y = (y_min + y_max) // 2
    center_x = (x_min + x_max) // 2

    # Determine which sheet edge the connection sits on
    if x_min < SHEET_EDGE_THRESHOLD_PX:
        # Left edge — pipe exits to the RIGHT
        return (x_max, center_y, "RIGHT")
    elif x_max > image_w - SHEET_EDGE_THRESHOLD_PX:
        # Right edge — pipe exits to the LEFT
        return (x_min, center_y, "LEFT")
    elif y_min < SHEET_EDGE_THRESHOLD_PX:
        # Top edge — pipe exits DOWN
        return (center_x, y_max, "DOWN")
    elif y_max > image_h - SHEET_EDGE_THRESHOLD_PX:
        # Bottom edge — pipe exits UP
        return (center_x, y_min, "UP")
    else:
        # Not at a sheet edge — guess based on aspect ratio
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        if bbox_w > bbox_h:
            # Wide box — pipe likely exits left or right
            if x_min < image_w / 2:
                return (x_max, center_y, "RIGHT")
            return (x_min, center_y, "LEFT")
        else:
            if y_min < image_h / 2:
                return (center_x, y_max, "DOWN")
            return (center_x, y_min, "UP")


def compute_port_vlm(
    image: np.ndarray,
    bbox: dict[str, int],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    crop_padding: int = 250,
) -> Optional[tuple[int, int, str]]:
    """Use VLM to determine the exact pipe port on a page connection symbol.

    Crops around the bbox, sends to VLM, parses the EDGE FRACTION response.
    Returns (x, y, direction) in image pixel coordinates, or None if VLM fails.

    The key improvement over the heuristic: VLM sees the actual symbol and
    identifies the precise attachment point, not just the bbox midpoint.
    """
    import openai

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
    )

    h, w = image.shape[:2]
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    # Generous crop around the bbox so VLM can see the pipe line
    cx1 = max(0, x_min - crop_padding)
    cy1 = max(0, y_min - crop_padding)
    cx2 = min(w, x_max + crop_padding)
    cy2 = min(h, y_max + crop_padding)

    crop = image[cy1:cy2, cx1:cx2]
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    data_uri = f"data:image/png;base64,{_encode_image(crop_pil)}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": PORT_FINDER_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PORT_FINDER_USER},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            max_tokens=50,
            temperature=0.0,
            timeout=30,
        )
        raw = response.choices[0].message.content
        if raw is None:
            log.warning("port_finder empty response")
            return None
        raw = raw.strip()
        log.info("port_finder raw: %s", raw)
    except Exception as exc:
        log.warning("port_finder VLM error: %s", exc)
        return None

    # Parse "EDGE FRACTION" e.g. "RIGHT 0.63" or just "RIGHT" → default 0.50
    m = re.match(r"(LEFT|RIGHT|TOP|BOTTOM|NONE)\s+([\d.]+)", raw, re.IGNORECASE)
    if not m:
        # Try edge-only format: "RIGHT", "LEFT", etc.
        m2 = re.match(r"(LEFT|RIGHT|TOP|BOTTOM|NONE)", raw, re.IGNORECASE)
        if not m2:
            log.warning("port_finder unparseable response: %s", raw)
            return None
        edge = m2.group(1).upper()
        if edge == "NONE":
            return None
        fraction = 0.50  # default: midpoint
    else:
        edge = m.group(1).upper()
        if edge == "NONE":
            return None
        fraction = float(m.group(2))
        fraction = max(0.0, min(1.0, fraction))

    bb_w = x_max - x_min
    bb_h = y_max - y_min

    if edge == "RIGHT":
        px = x_max
        py = y_min + int(round(fraction * bb_h))
        direction = "RIGHT"
    elif edge == "LEFT":
        px = x_min
        py = y_min + int(round(fraction * bb_h))
        direction = "LEFT"
    elif edge == "BOTTOM":
        px = x_min + int(round(fraction * bb_w))
        py = y_max
        direction = "DOWN"
    else:  # TOP
        px = x_min + int(round(fraction * bb_w))
        py = y_min
        direction = "UP"

    return (px, py, direction)


# ---------------------------------------------------------------------------
# VLM client
# ---------------------------------------------------------------------------


def _encode_image(pil_img: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_vlm(
    crop: Image.Image,
    cursor: PipelineCursor,
    crop_meta: dict,
    model: str,
    prev_direction: str = "none",
    api_key: Optional[str] = None,
) -> tuple[str, int, int]:
    """Send a cropped view to the VLM and return the text response.

    Returns (response_text, prompt_tokens, completion_tokens).
    """
    import openai

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
    )

    cx = crop_meta["cursor_x_view"]
    cy = crop_meta["cursor_y_view"]
    cw = crop_meta["crop_w"]
    ch = crop_meta["crop_h"]

    # Build visited hint
    if cursor.visited_count > 1:
        visited_hint = f"Green trail shows {cursor.visited_count} visited pixels. Do NOT retrace."
    else:
        visited_hint = "This is the first step. No previous path."

    user_prompt = AGENT2_USER_PROMPT_TEMPLATE.format(
        cursor_x=cx,
        cursor_y=cy,
        crop_w=cw,
        crop_h=ch,
        direction=cursor.direction,
        entry_direction=prev_direction,
        visited_hint=visited_hint,
    )

    data_uri = f"data:image/png;base64,{_encode_image(crop)}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": AGENT2_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        timeout=DEFAULT_TIMEOUT,
    )

    text = response.choices[0].message.content or ""
    prompt_tok = response.usage.prompt_tokens if response.usage else 0
    completion_tok = response.usage.completion_tokens if response.usage else 0

    return text, prompt_tok, completion_tok


# ---------------------------------------------------------------------------
# Agent orchestrator
# ---------------------------------------------------------------------------


class PipelineTracer:
    """Orchestrates step-by-step VLM pipeline tracing from page connections."""

    def __init__(
        self,
        image_path: str,
        stage4_path: str,
        model: str | None = None,
        crop_size: int = DEFAULT_CROP_SIZE,
    ):
        self.image = self._load_image(image_path)
        self.image_h, self.image_w = self.image.shape[:2]
        self.image_path = image_path
        self.model = model or os.environ.get("VISUAL_PRIMITIVES_MODEL", DEFAULT_MODEL)
        self.crop_size = crop_size

        # Load page connections from Stage 4
        stage4_data = json.loads(Path(stage4_path).read_text())
        self.page_connections = [
            o
            for o in stage4_data.get("objects", [])
            if o.get("class_name") == "page connection"
        ]

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        log.info("Loaded %s (%d×%d, %d channels)", path, img.shape[1], img.shape[0], img.shape[2])
        return img

    def _bbox_to_global(self, bbox: dict[str, int]) -> list[int]:
        """Convert stage4 pixel bbox to normalised [0,999]."""
        return [
            int(round(bbox["x_min"] / self.image_w * 999)),
            int(round(bbox["y_min"] / self.image_h * 999)),
            int(round(bbox["x_max"] / self.image_w * 999)),
            int(round(bbox["y_max"] / self.image_h * 999)),
        ]

    def _point_to_global(self, x: int, y: int) -> list[int]:
        """Convert pixel point to normalised [0,999]."""
        return [
            int(round(x / self.image_w * 999)),
            int(round(y / self.image_h * 999)),
        ]

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, output_dir: str | Path | None = None) -> TraceResult:
        """Run pipeline tracing on all page connections.

        Uses a work queue so tee-junction branches are explored automatically.
        Returns a TraceResult with all traced segments.
        """
        from collections import deque

        out_dir = Path(output_dir) if output_dir else Path("output/stage_vp2")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir  # store for raw response logging

        t_start = time.time()
        segments: list[TraceSegment] = []

        # Build work queue: (anchor_id, bbox_global, start_x, start_y, direction)
        WorkItem = tuple[str, list[int], int, int, str]
        queue: deque[WorkItem] = deque()

        # Seed queue with page connections
        for pc in self.page_connections:
            bbox = pc["bbox"]

            # Try VLM port detection first, fall back to heuristic
            vlm_port = compute_port_vlm(self.image, bbox, model=self.model)
            if vlm_port:
                port_x, port_y, direction = vlm_port
            else:
                port_x, port_y, direction = compute_port_from_bbox(
                    bbox, self.image_w, self.image_h
                )
            offset = min(self.crop_size // 4, 30)
            dir_dx = {"RIGHT": offset, "LEFT": -offset, "UP": 0, "DOWN": 0}
            dir_dy = {"RIGHT": 0, "LEFT": 0, "UP": -offset, "DOWN": offset}
            port_x += dir_dx.get(direction, 0)
            port_y += dir_dy.get(direction, 0)
            port_x = max(0, min(self.image_w - 1, port_x))
            port_y = max(0, min(self.image_h - 1, port_y))
            anchor_bbox_global = self._bbox_to_global(bbox)
            queue.append((pc["id"], anchor_bbox_global, port_x, port_y, direction))

        # Track visited junction points (image pixels, rounded) to avoid re-tracing
        visited_junctions: set[tuple[int, int]] = set()
        max_total_segments = 25  # hard cap to prevent explosion
        seg_count = 0

        while queue:
            if seg_count >= max_total_segments:
                log.warning("Reached max segments (%d), stopping queue", max_total_segments)
                break
            anchor_id, anchor_bbox_global, start_x, start_y, direction = queue.popleft()
            seg_count += 1
            log.info(
                "Tracing segment %d from %s (dir=%s, %d more queued)",
                seg_count, anchor_id, direction, len(queue),
            )
            seg = self._trace_from_point(
                anchor_id=anchor_id,
                anchor_bbox_global=anchor_bbox_global,
                start_x=start_x,
                start_y=start_y,
                direction=direction,
            )
            if seg:
                segments.append(seg)
                log.info(
                    "  → terminal: %s (%d steps, %d px)",
                    seg.terminal_class,
                    len(seg.steps),
                    seg.total_length_px,
                )

                # If terminal is tee_junction, queue unexplored branches
                if seg.terminal_class == "tee_junction":
                    tx = seg.terminal_point_global[0]
                    ty = seg.terminal_point_global[1]
                    # Convert junction to image pixels & snap to 20px grid for dedup
                    jx = int(tx / 999 * self.image_w)
                    jy = int(ty / 999 * self.image_h)
                    jp = (jx // 20 * 20, jy // 20 * 20)  # snap to grid
                    if jp not in visited_junctions:
                        visited_junctions.add(jp)
                        # Figure out incoming direction (last step direction)
                        incoming = seg.start_direction.value
                        for s in reversed(seg.steps):
                            if s.token_type.value == "step" and s.direction:
                                incoming = s.direction.value
                                break
                        # Queue all other directions
                        junc_id = f"tee_{tx}_{ty}"
                        junc_bbox = [tx-2, ty-2, tx+2, ty+2]
                        for new_dir in ["UP", "DOWN", "LEFT", "RIGHT"]:
                            if new_dir == incoming:
                                continue
                            doff = self.crop_size // 4
                            ndx = {"RIGHT": doff, "LEFT": -doff, "UP": 0, "DOWN": 0}
                            ndy = {"RIGHT": 0, "LEFT": 0, "UP": -doff, "DOWN": doff}
                            nx = jx + ndx.get(new_dir, 0)
                            ny = jy + ndy.get(new_dir, 0)
                            queue.append((junc_id, junc_bbox, nx, ny, new_dir))
                        log.info("  → queued %d branch directions from tee", 3)
            else:
                log.warning("  → failed to trace (no steps produced)")

        elapsed = time.time() - t_start

        result = TraceResult(
            source_image=str(self.image_path),
            model=self.model,
            source_dimensions=[self.image_w, self.image_h],
            segments=segments,
            prompt_tokens=self.total_prompt_tokens,
            completion_tokens=self.total_completion_tokens,
            elapsed_seconds=elapsed,
        )

        # Write artifacts
        self._write_artifacts(result, out_dir)
        return result

    def _trace_from_point(
        self,
        anchor_id: str,
        anchor_bbox_global: list[int],
        start_x: int,
        start_y: int,
        direction: str,
    ) -> TraceSegment | None:
        """Trace a single pipeline segment from a given starting point.

        Args:
            anchor_id: Identifier for the source (page connection id or tee_junction id)
            anchor_bbox_global: Anchor bbox in [0,999]
            start_x, start_y: Starting cursor position in image pixels
            direction: Initial trace direction
        """
        port_x, port_y = start_x, start_y

        start_point_global = self._point_to_global(port_x, port_y)

        cursor = PipelineCursor(
            image=self.image,
            x=port_x,
            y=port_y,
            direction=direction,
            crop_size=self.crop_size,
        )

        all_steps: list[TraceStep] = []
        prev_direction = "none"
        step_count = 0
        terminal_reached = False
        crop_edge_count = 0
        max_crop_edges = 10  # prevent infinite loops
        tried_directions: set[str] = set()

        while step_count < DEFAULT_MAX_STEPS_PER_SEGMENT:
            step_count += 1

            # Prepare the view
            crop, meta = cursor.crop_view()
            marked = cursor.draw_cursor_marker(crop, meta)
            marked = cursor.draw_visited_path(marked, meta)

            # Call VLM
            try:
                response_text, ptok, ctok = _call_vlm(
                    marked, cursor, meta, self.model, prev_direction
                )
                self.total_prompt_tokens += ptok
                self.total_completion_tokens += ctok
            except Exception as exc:
                log.error("VLM call failed at step %d: %s", step_count, exc)
                break

            # Save raw response for debugging
            out_dir = getattr(self, '_out_dir', None)
            if out_dir:
                raw_file = out_dir / f"trace_raw_{anchor_id}_step{step_count:03d}.txt"
                raw_file.write_text(response_text)

            # Parse
            steps = parse_trace_response(response_text)

            # If first step is "no_pipe_found", try alternate directions
            if (
                len(steps) == 1
                and steps[0].token_type == TraceTokenType.TERM
                and steps[0].symbol_class == "no_pipe_found"
                and len(tried_directions) < 3
            ):
                tried_directions.add(cursor.direction)
                alternates = [d for d in ["UP", "DOWN", "LEFT", "RIGHT"]
                              if d not in tried_directions]
                new_dir = alternates[0] if alternates else cursor.direction
                log.info("  step %d: no pipe found heading %s, trying %s",
                         step_count, cursor.direction, new_dir)
                cursor.direction = new_dir
                continue

            if not steps:
                log.warning("  step %d: no tokens parsed, stopping", step_count)
                break

            # Execute steps
            for s in steps:
                if s.token_type == TraceTokenType.STEP:
                    if s.direction and s.distance_px:
                        cursor.advance(s.direction.value, s.distance_px)
                        prev_direction = s.direction.value
                    all_steps.append(s)

                elif s.token_type == TraceTokenType.HIT:
                    all_steps.append(s)

                elif s.token_type == TraceTokenType.TERM:
                    if s.symbol_class == "crop_edge":
                        # Recenter and continue — don't stop
                        crop_edge_count += 1
                        if crop_edge_count > max_crop_edges:
                            log.warning("  step %d: too many crop edges, stopping", step_count)
                            all_steps.append(s)
                            terminal_reached = True
                            break
                        log.debug("  step %d: crop_edge — recentering cursor", step_count)
                        # Keep the last step direction to continue
                        break  # exit step loop, restart with new crop
                    else:
                        all_steps.append(s)
                        terminal_reached = True
                        break

            if terminal_reached:
                break

        if not all_steps:
            return None

        term_step = last_terminal(all_steps)
        terminal_class = term_step.symbol_class if term_step and term_step.symbol_class else "unknown"
        terminal_tag = term_step.symbol_tag if term_step else None
        terminal_point_global = self._point_to_global(cursor.x, cursor.y)
        terminal_bbox_global = None
        # Re-crop to get meta for the final position
        crop_final, meta_final = cursor.crop_view()
        if term_step and term_step.symbol_bbox_view:
            vx1, vy1, vx2, vy2 = term_step.symbol_bbox_view
            gx1, gy1 = cursor.view_to_global(vx1, vy1, meta_final)
            gx2, gy2 = cursor.view_to_global(vx2, vy2, meta_final)
            terminal_bbox_global = [
                int(round(gx1 / self.image_w * 999)),
                int(round(gy1 / self.image_h * 999)),
                int(round(gx2 / self.image_w * 999)),
                int(round(gy2 / self.image_h * 999)),
            ]

        return TraceSegment(
            anchor_id=anchor_id,
            anchor_bbox_global=anchor_bbox_global,
            start_point_global=start_point_global,
            start_direction=TraceDirection(direction),
            steps=all_steps,
            terminal_class=terminal_class,
            terminal_tag=terminal_tag,
            terminal_point_global=terminal_point_global,
            terminal_bbox_global=terminal_bbox_global,
            total_length_px=total_trace_distance(all_steps),
        )

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------

    def _write_artifacts(self, result: TraceResult, out_dir: Path) -> None:
        """Write trace result artifacts to disk."""
        # Full result JSON
        result_path = out_dir / "stage_vp2_trace_result.json"
        result_path.write_text(result.model_dump_json(indent=2))
        log.info("  Wrote %s", result_path)

        # Summary
        summary: dict[str, Any] = {
            "agent": "agent2_pipeline_tracer",
            "model": result.model,
            "source_image": result.source_image,
            "source_dimensions": result.source_dimensions,
            "segments_traced": result.total_segments,
            "total_prompt_tokens": result.prompt_tokens,
            "total_completion_tokens": result.completion_tokens,
            "elapsed_seconds": result.elapsed_seconds,
            "terminals": {},
        }
        for seg in result.segments:
            tc = seg.terminal_class
            summary["terminals"][tc] = summary["terminals"].get(tc, 0) + 1

        summary_path = out_dir / "stage_vp2_trace_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("  Wrote %s", summary_path)

        # Overlay image — draw traced paths on original P&ID
        overlay_path = out_dir / "stage_vp2_trace_overlay.png"
        self._draw_overlay(result, overlay_path)
        log.info("  Wrote %s", overlay_path)

        # Terminal summary to stdout
        print(f"\n{'='*50}")
        print(f"  Agent 2 — Pipeline Tracer")
        print(f"  Model: {result.model}")
        print(f"  Segments traced: {result.total_segments}")
        print(f"{'='*50}")
        for seg in result.segments:
            tag = seg.terminal_tag or "?"
            steps_n = len(seg.steps)
            print(
                f"  {seg.anchor_id:>16} → {seg.terminal_class:<20}"
                f" tag={tag:<12} {steps_n:>3} steps"
            )
        print(f"\n  Overlay: {overlay_path}")
        print(f"  Time: {result.elapsed_seconds:.1f}s")
        print(f"  Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")

    def _draw_overlay(self, result: TraceResult, path: Path) -> None:
        """Draw traced pipeline paths as colored overlays on the P&ID image."""
        from PIL import Image, ImageDraw, ImageFont

        rgb = self.image[..., ::-1]
        img = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        palette: dict[str, tuple[int, int, int, int]] = {
            "vessel": (0, 255, 255, 200),
            "pump": (0, 255, 0, 200),
            "tee_junction": (255, 255, 0, 200),
            "sheet_edge": (255, 80, 80, 200),
            "page_connection": (255, 160, 0, 200),
            "crop_edge": (100, 100, 255, 200),
            "no_pipe_found": (128, 128, 128, 200),
            "unknown": (255, 0, 255, 200),
        }

        for seg in result.segments:
            color = palette.get(seg.terminal_class, (200, 200, 200, 200))

            # Convert start point from [0,999] → image pixels
            ix = int(seg.start_point_global[0] / 999 * self.image_w)
            iy = int(seg.start_point_global[1] / 999 * self.image_h)
            points = [(ix, iy)]

            for step in seg.steps:
                if step.token_type.value != "step":
                    continue
                if not step.direction or not step.distance_px:
                    continue
                dx, dy = 0, 0
                d = step.direction.value
                if d == "RIGHT": dx = step.distance_px
                elif d == "LEFT": dx = -step.distance_px
                elif d == "DOWN": dy = step.distance_px
                elif d == "UP": dy = -step.distance_px
                lx, ly = points[-1]
                points.append((lx + dx, ly + dy))

            # Draw path line
            if len(points) >= 2:
                for i in range(1, len(points)):
                    draw.line([points[i-1], points[i]], fill=color, width=4)

            # Terminal dot + label
            if points:
                tx, ty = points[-1]
                r = 10
                draw.ellipse([tx-r, ty-r, tx+r, ty+r], fill=color, outline=(255,255,255,255), width=2)
                draw.text((tx+14, ty-7), seg.terminal_class[:14], fill=(255,255,255,255))

            # Start anchor dot
            draw.ellipse([ix-5, iy-5, ix+5, iy+5], fill=(0,255,0,255))

        # Composite overlay onto original
        result_img = Image.alpha_composite(img, overlay)
        result_img = result_img.convert("RGB")
        result_img.save(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-5s | %(message)s")

    parser = argparse.ArgumentParser(description="Agent 2: Pipeline Tracer")
    parser.add_argument("--image", required=True, help="Path to input P&ID image")
    parser.add_argument("--stage4", required=True, help="Path to stage4_objects.json")
    parser.add_argument("--output", default="output/stage_vp2", help="Output directory")
    parser.add_argument("--model", default=None, help="VLM model override")
    parser.add_argument("--crop-size", type=int, default=DEFAULT_CROP_SIZE,
                        help="Crop square size in pixels")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.stage4):
        print(f"ERROR: stage4 not found: {args.stage4}")
        sys.exit(1)

    tracer = PipelineTracer(
        image_path=args.image,
        stage4_path=args.stage4,
        model=args.model,
        crop_size=args.crop_size,
    )

    result = tracer.run(output_dir=args.output)
    print(f"\nDone. {result.total_segments} segments traced.")


if __name__ == "__main__":
    main()
