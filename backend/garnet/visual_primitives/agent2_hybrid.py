"""Agent 2 Hybrid — CV pipe follower + VLM classifier for pipeline tracing.

The hybrid approach: CV follows pipe-mask pixels at ~1000 pixels/ms speed.
VLM is called only for: (1) port detection on page connections, (2) equipment
classification at terminals, (3) gap-bridging across inline objects.

This replaces the pure VLM step-by-step tracer (~14 min/sheet) with a ~1 min
hybrid: CV does the walking, VLM does the interpretation.

Usage:
    python -m garnet.visual_primitives.agent2_hybrid \
        --image test/ppcl/Test-00001.jpg \
        --stage4 ../output/stage4_objects.json \
        --pipe-mask ../output/stage5_pipe_mask_raw.png \
        --output ../output/stage_vp2h_test01/
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
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

load_dotenv(Path.home() / ".env")

from .cv_pipe_follower import CVPipeFollower, CVTraceStep
from .vlm_cursor import VLMCursor, crop_around_cursor, DIR_VEC
from .vlm_trace_parser import parse_trace_response, TraceToken
from .prompts import (
    HYBRID_CLASSIFY_SYSTEM,
    HYBRID_CLASSIFY_USER,
    PORT_FINDER_SYSTEM,
    PORT_FINDER_USER,
    VLM_TRACE_SYSTEM,
    VLM_TRACE_USER,
)
from .schemas import TraceDirection, TraceResult, TraceSegment, TraceStep, TraceTokenType

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("agent2_hybrid")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_TIMEOUT = 60
SHEET_EDGE_THRESHOLD_PX = 500

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ClassifiedTerminal:
    """VLM classification result for a pipeline terminal."""

    equipment_class: str = "unknown"
    tag: str = "unknown"
    confidence: str = "medium"  # high / medium / low


@dataclass
class HybridSegment:
    """One traced pipeline segment with classification."""

    anchor_id: str = ""
    start_x: int = 0
    start_y: int = 0
    direction: str = ""
    total_length_px: int = 0
    steps: list[CVTraceStep] = field(default_factory=list)
    terminal_kind: str = "unknown"
    terminal_object: Optional[dict[str, Any]] = None
    terminal_x: int = 0
    terminal_y: int = 0
    classification: Optional[ClassifiedTerminal] = None


@dataclass
class HybridTraceResult:
    """Complete hybrid trace result for a P&ID sheet."""

    image_path: str = ""
    image_w: int = 0
    image_h: int = 0
    segments: list[HybridSegment] = field(default_factory=list)
    total_pipe_length_px: int = 0
    vlm_calls: int = 0
    total_runtime_s: float = 0.0


# ---------------------------------------------------------------------------
# VLM helpers
# ---------------------------------------------------------------------------


def _encode_image(pil_img: Image.Image) -> str:
    """Encode a PIL image as a base64 data URI."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_vlm_raw(
    system_prompt: str,
    user_text: str,
    image: np.ndarray,
    model: str,
    max_tokens: int = 128,
    timeout: int = DEFAULT_TIMEOUT,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """Send a single image + text to the VLM and return raw response text.

    Returns None on failure.
    """
    import openai

    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
    )

    crop_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_pil = Image.fromarray(crop_rgb)
    data_uri = f"data:image/png;base64,{_encode_image(crop_pil)}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
        )
        content = response.choices[0].message.content
        return content.strip() if content else None
    except Exception as exc:
        log.warning("VLM call error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Port detection (VLM)
# ---------------------------------------------------------------------------


def compute_port_vlm(
    image: np.ndarray,
    bbox: dict[str, int],
    model: str = DEFAULT_MODEL,
    crop_padding: int = 250,
    api_key: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[tuple[int, int, str]]:
    """Use VLM to determine the exact pipe port on a page connection symbol.

    Crops around the bbox, sends to VLM, parses the EDGE [FRACTION] response.
    Returns (x, y, direction) in image pixel coordinates, or None if VLM fails
    after all retries.

    Firmed process — no heuristic fallback. VLM or skip.
    """
    import time as _time

    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]

    cx1 = max(0, x_min - crop_padding)
    cy1 = max(0, y_min - crop_padding)
    cx2 = min(w, x_max + crop_padding)
    cy2 = min(h, y_max + crop_padding)
    crop = image[cy1:cy2, cx1:cx2]

    for attempt in range(max_retries):
        raw = _call_vlm_raw(PORT_FINDER_SYSTEM, PORT_FINDER_USER, crop, model, max_tokens=50)
        if raw is None:
            _time.sleep(1.0 * (attempt + 1))
            continue

        log.info("port_vlm attempt %d/%d raw: %s", attempt + 1, max_retries, raw)

        # Parse "EDGE [FRACTION]"
        m = re.match(r"(LEFT|RIGHT|TOP|BOTTOM|NONE)\s+([\d.]+)", raw, re.IGNORECASE)
        if m:
            edge = m.group(1).upper()
            if edge == "NONE":
                return None
            fraction = float(m.group(2))
            fraction = max(0.0, min(1.0, fraction))
        else:
            m2 = re.match(r"(LEFT|RIGHT|TOP|BOTTOM|NONE)", raw, re.IGNORECASE)
            if not m2:
                if attempt < max_retries - 1:
                    _time.sleep(1.0 * (attempt + 1))
                    continue
                log.warning("port_vlm unparseable after %d attempts: %s", max_retries, raw)
                return None
            edge = m2.group(1).upper()
            if edge == "NONE":
                return None
            fraction = 0.50

        bb_w = x_max - x_min
        bb_h = y_max - y_min
        edge_map = {
            "RIGHT": (x_max, y_min + int(round(fraction * bb_h)), "RIGHT"),
            "LEFT": (x_min, y_min + int(round(fraction * bb_h)), "LEFT"),
            "BOTTOM": (x_min + int(round(fraction * bb_w)), y_max, "DOWN"),
            "TOP": (x_min + int(round(fraction * bb_w)), y_min, "UP"),
        }
        return edge_map[edge]

    return None


# ---------------------------------------------------------------------------
# Port detection — VLM only (firmed process, no heuristic fallback)
# ---------------------------------------------------------------------------
# Legacy: compute_port_from_bbox removed. Use compute_port_vlm() exclusively.
# ---------------------------------------------------------------------------
# Equipment classifier (VLM)
# ---------------------------------------------------------------------------


def classify_terminal_vlm(
    image: np.ndarray,
    bbox: dict[str, int],
    model: str = DEFAULT_MODEL,
    crop_padding: int = 300,
    api_key: Optional[str] = None,
) -> ClassifiedTerminal:
    """Use VLM to classify equipment at a pipeline terminal.

    Crops generously around the object bbox and asks VLM for type + tag.
    """
    h, w = image.shape[:2]
    x_min = bbox["x_min"]
    y_min = bbox["y_min"]
    x_max = bbox["x_max"]
    y_max = bbox["y_max"]

    cx1 = max(0, x_min - crop_padding)
    cy1 = max(0, y_min - crop_padding)
    cx2 = min(w, x_max + crop_padding)
    cy2 = min(h, y_max + crop_padding)

    crop = image[cy1:cy2, cx1:cx2]

    raw = _call_vlm_raw(
        HYBRID_CLASSIFY_SYSTEM,
        HYBRID_CLASSIFY_USER,
        crop,
        model,
        max_tokens=128,
    )

    if raw is None:
        log.warning("classify_terminal: VLM returned None")
        return ClassifiedTerminal()

    log.info("classify_vlm raw: %s", raw)

    # Parse <|eq|>CLASS<|/eq|> <|tag|>TAG<|/tag|>
    # Be lenient: accept unclosed tags like "<|eq|>pump" or "<|eq|>pump<"
    m_eq = re.search(r"<\|eq\|>\s*(\S+?)\s*(?:<\|/eq\|>|<)?", raw, re.IGNORECASE)
    m_tag = re.search(r"<\|tag\|>\s*(\S+?)\s*(?:<\|/tag\|>|<)?", raw, re.IGNORECASE)

    eq_class = m_eq.group(1).strip().lower().replace(" ", "_") if m_eq else "unknown"
    tag = m_tag.group(1).strip() if m_tag else "unknown"

    # Confidence heuristic: high if both parsed, medium if one, low if none
    if m_eq and m_tag:
        confidence = "high"
    elif m_eq or m_tag:
        confidence = "medium"
    else:
        confidence = "low"

    return ClassifiedTerminal(equipment_class=eq_class, tag=tag, confidence=confidence)


# ---------------------------------------------------------------------------
# Bridge-gap logic
# ---------------------------------------------------------------------------


def find_mask_resume(
    pipe_mask: np.ndarray,
    visited_mask: np.ndarray,
    x: int,
    y: int,
    direction: str,
    obj_bbox: dict[str, int],
    max_search: int = 300,
    min_straight: int = 60,
) -> Optional[tuple[int, int]]:
    """Find where the pipe mask resumes beyond an inline object.

    P&ID Rule 4: pipes only jump gaps if the line continues straight in the
    same direction. After finding a resume point, verify at least min_straight
    pixels of straight pipe follow.
    """
    h, w = pipe_mask.shape
    dir_dx = {"RIGHT": 1, "LEFT": -1, "UP": 0, "DOWN": 0}
    dir_dy = {"RIGHT": 0, "LEFT": 0, "UP": -1, "DOWN": 1}
    dx = dir_dx.get(direction, 1)
    dy = dir_dy.get(direction, 0)

    # Start from the far side of the object bbox
    if direction == "RIGHT":
        sx = obj_bbox["x_max"] + 5
        sy = y
    elif direction == "LEFT":
        sx = obj_bbox["x_min"] - 5
        sy = y
    elif direction == "DOWN":
        sx = x
        sy = obj_bbox["y_max"] + 5
    else:  # UP
        sx = x
        sy = obj_bbox["y_min"] - 5

    # Scan forward until we find unvisited mask pixels
    for offset in range(0, max_search, 5):
        nx = sx + dx * offset
        ny = sy + dy * offset
        if not (0 <= nx < w and 0 <= ny < h):
            return None
        if pipe_mask[ny, nx] > 0 and visited_mask[ny, nx] == 0:
            # Rule 4: verify straight-line continuation for min_straight px
            straight_ok = True
            for s in range(5, min_straight, 10):
                sx2 = nx + dx * s
                sy2 = ny + dy * s
                if not (0 <= sx2 < w and 0 <= sy2 < h):
                    straight_ok = False
                    break
                if pipe_mask[sy2, sx2] == 0:
                    straight_ok = False
                    break
            if straight_ok:
                return (nx, ny)
            # Otherwise keep searching — might find a better point ahead

    return None


# ---------------------------------------------------------------------------
# Hybrid Pipeline Tracer
# ---------------------------------------------------------------------------


class HybridPipelineTracer:
    """CV pipe follower + VLM classifier for fast pipeline tracing.

    CV walks the pipe mask in milliseconds; VLM is called only for:
      - Port detection (start point on page connections)
      - Equipment classification (what's at the terminal?)
      - Gap bridging (resume pipe beyond inline objects)

    The result: ~1 min/sheet instead of ~14 min for pure VLM.
    """

    def __init__(
        self,
        image_path: str,
        stage4_path: str,
        pipe_mask_path: str,
        model: str = "",
        output_dir: str = "",
        crop_size: int = 300,
        cv_step_size: int = 5,
        cv_window_size: int = 60,
        tracer: str = "cv",
    ):
        self.tracer_mode = tracer
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        self.image_h, self.image_w = self.image.shape[:2]
        self.image_path = image_path

        stage4_data = json.loads(Path(stage4_path).read_text())
        self.stage4_objects: list[dict[str, Any]] = stage4_data.get(
            "objects", stage4_data
        )

        pipe_mask_raw = cv2.imread(pipe_mask_path, cv2.IMREAD_GRAYSCALE)
        if pipe_mask_raw is None:
            raise FileNotFoundError(f"Could not read pipe mask: {pipe_mask_path}")
        self.pipe_mask = pipe_mask_raw

        self.model = model or os.environ.get("VISUAL_PRIMITIVES_MODEL", DEFAULT_MODEL)
        self.output_dir = Path(output_dir) if output_dir else Path.home() / "Downloads" / "output" / "hybrid_trace"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_size = crop_size
        self.cv_step_size = cv_step_size
        self.cv_window_size = cv_window_size
        self.api_key = os.environ.get("OPENROUTER_API_KEY")

        # Categorize objects
        self.page_connections: list[dict] = []
        self.terminal_objects: list[dict] = []  # equipment that terminates a pipe
        self.inline_objects: list[dict] = []    # valves, instruments (inline)
        self._categorize_objects()

        # Shared CV follower — visited mask accumulates across all traces
        self._follower: Optional[CVPipeFollower] = None
        self.vlm_call_count = 0

    # ------------------------------------------------------------------
    # Object categorization
    # ------------------------------------------------------------------

    _TERMINAL_CLASSES = {
        "pump", "heat exchanger", "tank", "vessel", "column",
        "compressor", "blower", "fan", "reactor",
        "knockout drum", "filter", "strainer",
        "page connection", "connection", "utility connection",
    }

    _INLINE_CLASSES = {
        "valve", "control valve", "check valve", "ball valve",
        "gate valve", "globe valve", "butterfly valve",
        "instrument", "indicator", "transmitter", "controller",
        "solenoid", "actuator", "reducer", "strainer",
    }

    def _categorize_objects(self):
        """Sort YOLO objects into page_connections, terminals, and inline."""
        for obj in self.stage4_objects:
            cls = obj.get("class_name", "").lower()
            if cls == "page connection":
                self.page_connections.append(obj)
            elif cls in self._TERMINAL_CLASSES:
                self.terminal_objects.append(obj)
            elif cls in self._INLINE_CLASSES:
                self.inline_objects.append(obj)

    # ------------------------------------------------------------------
    # Follower factory
    # ------------------------------------------------------------------

    def _get_follower(self) -> CVPipeFollower:
        """Lazy-init the shared CV follower."""
        if self._follower is None:
            self._follower = CVPipeFollower(
                pipe_mask=self.pipe_mask,
                stage4_objects=self.stage4_objects,
                image_w=self.image_w,
                image_h=self.image_h,
                step_size=self.cv_step_size,
                window_size=self.cv_window_size,
            )
        return self._follower

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> HybridTraceResult:
        """Execute the full hybrid pipeline trace."""
        t0 = time.perf_counter()
        result = HybridTraceResult(
            image_path=self.image_path,
            image_w=self.image_w,
            image_h=self.image_h,
        )

        # Work queue: (source_id, port_x, port_y, direction, entry_direction)
        queue: deque[tuple[str, int, int, str, str]] = deque()

        # Phase 1: Seed queue from page connections (VLM port detection)
        log.info("Phase 1: Detecting ports on %d page connections", len(self.page_connections))
        for pc in self.page_connections:
            bbox = pc["bbox"]

            # VLM port detection (firmed — no heuristic fallback)
            vlm_port = compute_port_vlm(self.image, bbox, model=self.model, api_key=self.api_key)
            self.vlm_call_count += 1

            if vlm_port is None:
                log.warning("VLM port detection failed for %s — skipping", pc.get("id", "unknown"))
                continue

            port_x, port_y, direction = vlm_port

            # Nudge port slightly into the pipe
            offset = min(self.crop_size // 4, 30)
            dir_push = {
                "RIGHT": (offset, 0),
                "LEFT": (-offset, 0),
                "UP": (0, -offset),
                "DOWN": (0, offset),
            }
            px, py = dir_push.get(direction, (offset, 0))
            port_x = max(0, min(self.image_w - 1, port_x + px))
            port_y = max(0, min(self.image_h - 1, port_y + py))

            queue.append((pc["id"], port_x, port_y, direction, "none"))
            log.info("  %s: port=(%d,%d) dir=%s", pc["id"], port_x, port_y, direction)

        # Phase 2: Trace each seed
        log.info("Phase 2: Tracing %d page connection seeds (mode=%s)", len(queue), self.tracer_mode)
        for _ in range(len(queue)):
            source_id, start_x, start_y, direction, entry_dir = queue.popleft()

            if self.tracer_mode == "vlm":
                seg = self._trace_vlm_segment(source_id, start_x, start_y, direction)
            else:
                fresh = CVPipeFollower(
                    pipe_mask=self.pipe_mask,
                    stage4_objects=self.stage4_objects,
                    image_w=self.image_w,
                    image_h=self.image_h,
                    step_size=self.cv_step_size,
                    window_size=self.cv_window_size,
                )
                seg = self._trace_segment_with(
                    fresh, source_id, start_x, start_y, direction, entry_dir, set()
                )
            result.segments.append(seg)
            result.total_pipe_length_px += seg.total_length_px

        # Phase 3: VLM classification of terminals (CV tracer only — VLM handles its own)
        if self.tracer_mode == "cv":
            log.info("Phase 3: Classifying terminals with VLM")
            for seg in result.segments:
                if seg.terminal_kind in ("unknown", "max_steps", "no_pipe_found", "junction"):
                    continue
                if seg.terminal_object is None and seg.terminal_kind != "dead_end":
                    continue
                # Skip interconnects
                if seg.terminal_kind in ("page connection", "connection", "utility connection"):
                    continue

                # Only classify dead_end if segment is significant (>200px)
                if seg.terminal_kind == "dead_end" and seg.total_length_px < 200:
                    continue

                # For dead_end: use terminal coordinates as center of crop
                # For other terminals: use the object bbox
                if seg.terminal_kind == "dead_end" and seg.terminal_object is None:
                    # Crop a square around the dead-end point
                    half = 250
                    tx, ty = seg.terminal_x, seg.terminal_y
                    bbox = {
                        "x_min": max(0, tx - half),
                        "y_min": max(0, ty - half),
                        "x_max": min(self.image_w, tx + half),
                        "y_max": min(self.image_h, ty + half),
                    }
                else:
                    bbox = seg.terminal_object["bbox"]

                cls = classify_terminal_vlm(
                    self.image,
                    bbox,
                    model=self.model,
                    api_key=self.api_key,
                )
                self.vlm_call_count += 1
                seg.classification = cls
                log.info(
                    "  %s → %s (%s) tag=%s conf=%s",
                    seg.anchor_id,
                    seg.terminal_kind,
                    cls.equipment_class,
                    cls.tag,
                    cls.confidence,
                )

        result.vlm_calls = self.vlm_call_count
        result.total_runtime_s = time.perf_counter() - t0

        log.info(
            "Done: %d segments, %dpx pipe, %d VLM calls, %.1fs",
            len(result.segments),
            result.total_pipe_length_px,
            result.vlm_calls,
            result.total_runtime_s,
        )

        return result

    # ------------------------------------------------------------------
    # VLM step-by-step tracer
    # ------------------------------------------------------------------

    def _trace_vlm_segment(
        self,
        source_id: str,
        start_x: int,
        start_y: int,
        direction: str,
        crop_size: int = 350,
        max_steps: int = 25,
    ) -> HybridSegment:
        """Trace one segment using VLM-guided step-by-step walking."""
        cursor = VLMCursor(x=start_x, y=start_y, direction=direction)
        steps: list[CVTraceStep] = []

        for _ in range(max_steps):
            crop, crop_bbox = crop_around_cursor(self.image, cursor, crop_size)
            crop_b64 = self._encode_image(crop)

            response = self._call_vlm_step(crop_b64)
            tokens = parse_trace_response(response)

            for tok in tokens:
                if tok.kind == "go":
                    cursor.move(tok.direction, tok.distance)
                    steps.append(CVTraceStep(
                        kind="move", direction=tok.direction,
                        distance_px=tok.distance, x=cursor.x, y=cursor.y))
                elif tok.kind == "turn":
                    cursor.turn(tok.direction)
                    steps.append(CVTraceStep(
                        kind="move", direction=tok.direction,
                        distance_px=0, x=cursor.x, y=cursor.y))
                elif tok.kind == "hit":
                    steps.append(CVTraceStep(
                        kind="hit", x=cursor.x, y=cursor.y,
                        hit_object={"class_name": tok.class_name}))
                elif tok.kind == "jump":
                    dx, dy = DIR_VEC.get(tok.direction, (1, 0))
                    jx = cursor.x + dx * tok.distance
                    jy = cursor.y + dy * tok.distance
                    cursor.jump((jx, jy), tok.direction)
                    steps.append(CVTraceStep(
                        kind="move", direction=tok.direction,
                        distance_px=tok.distance, x=cursor.x, y=cursor.y))
                elif tok.kind == "term":
                    return HybridSegment(
                        anchor_id=source_id,
                        start_x=start_x, start_y=start_y,
                        direction=direction,
                        total_length_px=cursor.total_distance,
                        steps=steps,
                        terminal_kind=tok.class_name or "unknown",
                        terminal_object={"class_name": tok.class_name} if tok.class_name else None,
                        terminal_x=cursor.x, terminal_y=cursor.y,
                    )

            if not tokens:
                # VLM gave nothing — dead end
                return HybridSegment(
                    anchor_id=source_id,
                    start_x=start_x, start_y=start_y,
                    direction=direction,
                    total_length_px=cursor.total_distance,
                    steps=steps,
                    terminal_kind="dead_end",
                    terminal_x=cursor.x, terminal_y=cursor.y,
                )

        return HybridSegment(
            anchor_id=source_id,
            start_x=start_x, start_y=start_y,
            direction=direction,
            total_length_px=cursor.total_distance,
            steps=steps,
            terminal_kind="max_steps",
            terminal_x=cursor.x, terminal_y=cursor.y,
        )

    def _call_vlm_step(self, image_b64: str) -> Optional[str]:
        """Call VLM for step-by-step trace guidance."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": VLM_TRACE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VLM_TRACE_USER},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            self.vlm_call_count += 1
            return content
        except Exception:
            log.exception("VLM step call failed")
            return None

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 PNG."""
        import io, base64
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ------------------------------------------------------------------
    # Trace one segment (CV)
    # ------------------------------------------------------------------

    def _trace_segment_with(
        self,
        follower: CVPipeFollower,
        source_id: str,
        start_x: int,
        start_y: int,
        direction: str,
        entry_dir: str,
        visited_junctions: set,
    ) -> HybridSegment:
        """Trace one pipe segment from start to terminal using the given follower."""
        cv_result = follower.trace(
            start_x=start_x,
            start_y=start_y,
            direction=direction,
            anchor_id=source_id,
            max_steps=300,
        )

        seg = HybridSegment(
            anchor_id=source_id,
            start_x=start_x,
            start_y=start_y,
            direction=direction,
            total_length_px=cv_result.total_length_px,
            steps=cv_result.steps,
            terminal_kind=cv_result.terminal_kind,
            terminal_object=cv_result.terminal_object,
            terminal_x=cv_result.terminal_x,
            terminal_y=cv_result.terminal_y,
        )

        # Bridge gaps: if CV dead-ends and there's a nearby inline object,
        # try to resume on the far side
        if cv_result.terminal_kind == "dead_end":
            bridged = self._try_bridge_gap(seg, follower)
            if bridged:
                seg.terminal_kind = "bridged"
                # The bridge puts us past the object; queue continuation
                _ = bridged  # handled by queue logic already

        return seg

    # ------------------------------------------------------------------
    # Gap bridging
    # ------------------------------------------------------------------

    def _try_bridge_gap(self, seg: HybridSegment, follower: CVPipeFollower) -> bool:
        """Attempt to bridge a gap caused by an inline object.

        If the dead-end is near a YOLO object bbox, try to find where the
        pipe mask resumes on the far side and continue tracing.
        """
        tx, ty = seg.terminal_x, seg.terminal_y

        # Find the last move direction
        direction = seg.direction
        for step in reversed(seg.steps):
            if step.kind == "move" and step.direction:
                direction = step.direction
                break

        # Look for nearby objects in the trace direction
        nearby = self._find_nearby_object(tx, ty, direction, max_dist=200)
        if nearby is None:
            return False

        resume = find_mask_resume(
            self.pipe_mask,
            follower.visited_mask,
            tx, ty, direction,
            nearby["bbox"],
        )

        if resume is None:
            return False

        rx, ry = resume
        log.info(
            "  Bridged gap at %s object %s: (%d,%d) → (%d,%d)",
            nearby.get("class_name", "?"), nearby["id"],
            tx, ty, rx, ry,
        )

        # Trace the resumed segment and merge its steps into this one
        cv_resume = follower.trace(
            start_x=rx, start_y=ry, direction=direction,
            anchor_id=seg.anchor_id + "_bridge", max_steps=200,
        )

        if cv_resume.steps:
            seg.steps.append(CVTraceStep(
                kind="move", direction=direction,
                distance_px=0, x=rx, y=ry,
                hit_object=nearby,
            ))
            seg.steps.extend(cv_resume.steps)
            seg.total_length_px += cv_resume.total_length_px
            seg.terminal_kind = cv_resume.terminal_kind
            seg.terminal_object = cv_resume.terminal_object
            seg.terminal_x = cv_resume.terminal_x
            seg.terminal_y = cv_resume.terminal_y
            return True

        return False

    def _find_nearby_object(
        self, x: int, y: int, direction: str, max_dist: int = 200
    ) -> Optional[dict]:
        """Find a YOLO object ahead of (x,y) in the given direction."""
        dir_dx = {"RIGHT": 1, "LEFT": -1, "UP": 0, "DOWN": 0}
        dir_dy = {"RIGHT": 0, "LEFT": 0, "UP": -1, "DOWN": 1}
        dx = dir_dx.get(direction, 1)
        dy = dir_dy.get(direction, 0)

        search_x = x + dx * max_dist
        search_y = y + dy * max_dist

        best = None
        best_dist = max_dist + 1

        for obj in self.inline_objects + self.terminal_objects:
            b = obj["bbox"]
            cx = (b["x_min"] + b["x_max"]) // 2
            cy = (b["y_min"] + b["y_max"]) // 2

            # Must be in the right direction
            if dx > 0 and cx <= x:
                continue
            if dx < 0 and cx >= x:
                continue
            if dy > 0 and cy <= y:
                continue
            if dy < 0 and cy >= y:
                continue

            dist = abs(cx - x) + abs(cy - y)
            if dist < max_dist and dist < best_dist:
                best = obj
                best_dist = dist

        return best

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_result(self, result: HybridTraceResult):
        """Write trace result JSON and generate overlay image."""
        # JSON
        segments_json = []
        for seg in result.segments:
            seg_dict = {
                "anchor_id": seg.anchor_id,
                "start_x": seg.start_x,
                "start_y": seg.start_y,
                "direction": seg.direction,
                "total_length_px": seg.total_length_px,
                "terminal_kind": seg.terminal_kind,
                "terminal_x": seg.terminal_x,
                "terminal_y": seg.terminal_y,
                "terminal_object_id": seg.terminal_object.get("id", "") if seg.terminal_object else "",
                "n_steps": len(seg.steps),
            }
            if seg.classification:
                seg_dict["classification"] = {
                    "equipment_class": seg.classification.equipment_class,
                    "tag": seg.classification.tag,
                    "confidence": seg.classification.confidence,
                }
            segments_json.append(seg_dict)

        summary = {
            "image": result.image_path,
            "image_w": result.image_w,
            "image_h": result.image_h,
            "segments": segments_json,
            "total_pipe_length_px": result.total_pipe_length_px,
            "vlm_calls": result.vlm_calls,
            "total_runtime_s": result.total_runtime_s,
        }

        json_path = self.output_dir / "stage_vp2h_result.json"
        json_path.write_text(json.dumps(summary, indent=2))
        log.info("Result JSON: %s", json_path)

        # Overlay
        self._draw_overlay(result)

    def _draw_overlay(self, result: HybridTraceResult):
        """Draw traced pipe paths as connected lines with dots at step points.

        Object IDs are labeled at terminals.
        """
        from PIL import Image, ImageDraw, ImageFont

        rgb = self.image[..., ::-1]
        img = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except Exception:
            font = ImageFont.load_default()

        colors = [
            (0, 255, 255, 200),   # cyan
            (255, 0, 255, 200),   # magenta
            (0, 255, 0, 200),     # green
            (0, 180, 180, 200),   # teal (was yellow)
            (255, 128, 0, 200),   # orange
            (128, 0, 255, 200),   # purple
        ]

        for i, seg in enumerate(result.segments):
            color = colors[i % len(colors)]
            darker = tuple(max(0, c - 80) for c in color)

            # Collect movement points
            points: list[tuple[int, int]] = []
            for step in seg.steps:
                if step.kind == "move":
                    points.append((step.x, step.y))

            if not points:
                continue

            # Draw connecting lines between consecutive points
            for j in range(len(points) - 1):
                draw.line([points[j], points[j + 1]], fill=color, width=3)

            # Draw dots at each step point
            for px, py in points:
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=darker)

            # Mark terminal with red circle + object ID label
            tx, ty = seg.terminal_x, seg.terminal_y
            draw.ellipse([tx - 10, ty - 10, tx + 10, ty + 10],
                         outline=(255, 0, 0, 255), width=3)

            label = seg.terminal_kind[:20]
            if seg.terminal_object:
                obj_id = seg.terminal_object.get("id", "")
                obj_cls = seg.terminal_object.get("class_name", "")
                label = f"{obj_id} ({obj_cls[:15]})"
            if seg.classification and seg.classification.tag != "unknown":
                label += f" [{seg.classification.tag}]"

            draw.text((tx + 14, ty - 8), label, fill=(255, 255, 255, 255), font=font,
                      stroke_width=2, stroke_fill=(0, 0, 0, 200))

        combined = Image.alpha_composite(img, overlay)
        out_path = self.output_dir / "stage_vp2h_overlay.png"
        combined.save(str(out_path))
        log.info("Overlay: %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Hybrid CV+VLM Pipeline Tracer")
    parser.add_argument("--image", required=True, help="P&ID image path")
    parser.add_argument("--stage4", required=True, help="Stage 4 YOLO objects JSON")
    parser.add_argument("--pipe-mask", required=True, help="Pipe mask image (stage 5)")
    parser.add_argument("--output", default="", help="Output directory (default: ~/Downloads/output/hybrid_trace)")
    parser.add_argument("--model", default=None, help="VLM model for classification")
    parser.add_argument("--cv-step-size", type=int, default=5, help="CV follower step size")
    parser.add_argument("--tracer", choices=["cv", "vlm"], default="cv", help="Tracing engine: cv (fast) or vlm (accurate)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    tracer = HybridPipelineTracer(
        image_path=args.image,
        stage4_path=args.stage4,
        pipe_mask_path=args.pipe_mask,
        model=args.model,
        output_dir=args.output,
        cv_step_size=args.cv_step_size,
        tracer=args.tracer,
    )

    result = tracer.run()
    tracer.save_result(result)

    print(f"\n--- Summary ---")
    print(f"Segments: {len(result.segments)}")
    print(f"Total pipe length: {result.total_pipe_length_px} px")
    print(f"VLM calls: {result.vlm_calls}")
    print(f"Runtime: {result.total_runtime_s:.1f}s")
    print(f"Output: {tracer.output_dir}/")

    for seg in result.segments:
        cls = seg.classification.equipment_class if seg.classification else "-"
        tag = seg.classification.tag if seg.classification else "-"
        print(f"  {seg.anchor_id:<16} → {seg.terminal_kind:<20} {seg.total_length_px:>5}px  {cls:<16} {tag}")


if __name__ == "__main__":
    main()
