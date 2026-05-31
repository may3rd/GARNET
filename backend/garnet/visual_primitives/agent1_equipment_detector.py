"""Agent 1: Global Equipment Detector — Visual Primitives P&ID Pipeline.

Takes a P&ID image, downsamples it, sends it to a VLM (Claude via OpenRouter)
with chain-of-thought spatial grounding (<box> primitives interleaved in
reasoning), and returns a structured EquipmentRegistry.

Usage (standalone):
    python -m garnet.visual_primitives.agent1_equipment_detector \\
        --image path/to/pid.png \\
        --output output/stage_vp1/

Environment:
    OPENROUTER_API_KEY — required
    VISUAL_PRIMITIVES_MODEL — optional, defaults to anthropic/claude-sonnet-latest
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import openai

from .canvas import load_canvas, make_global_view, CanvasConfig
from .prompts import (
    AGENT1_SYSTEM_PROMPT,
    AGENT1_USER_PROMPT_TEMPLATE,
    AGENT1_DRAWING_CONTEXT_DEFAULT,
)
from .response_parser import parse_response
from .schemas import EquipmentRegistry, StageArtifactMeta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "google/gemini-2.5-pro"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 120

STAGE_PREFIX = "stage_vp1"


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class Agent1EquipmentDetector:
    """Global Equipment Detector using visual-primitives spatial grounding.

    Call .detect() to run detection on a single P&ID image.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        canvas_cfg: Optional[CanvasConfig] = None,
    ):
        self.model = model or os.getenv("VISUAL_PRIMITIVES_MODEL", DEFAULT_MODEL)
        self.canvas_cfg = canvas_cfg or CanvasConfig()
        self._client: Optional[openai.OpenAI] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        drawing_context: str = "",
    ) -> EquipmentRegistry:
        """Run Agent 1 detection on a P&ID image.

        Args:
            image_path: Path to the P&ID image (PNG, JPG, TIFF).
            output_dir: If provided, stage artifacts are written here.
            drawing_context: Optional context string (unit name, line prefix, etc.).

        Returns:
            EquipmentRegistry with all detected items in [0, 999] global coords.
        """
        t_start = time.monotonic()

        # 1. Load canvas + metadata
        img, meta = load_canvas(image_path)
        logger.info(
            "Loaded %s (%d×%d, %d channels)",
            meta.source_path,
            meta.width,
            meta.height,
            meta.channels,
        )

        # 2. Downsample for Agent 1 global view
        global_view = make_global_view(img, cfg=self.canvas_cfg)
        gv_h, gv_w = global_view.shape[:2]
        logger.info("Global view: %d×%d (max_dim=%d)", gv_w, gv_h, self.canvas_cfg.global_view_max_dim)

        # 3. Call VLM
        context = drawing_context or AGENT1_DRAWING_CONTEXT_DEFAULT
        raw_response, usage = self._call_vlm(global_view, context)

        # 4. Parse
        registry, thinking = parse_response(raw_response, gv_w, gv_h)

        elapsed = time.monotonic() - t_start

        # 5. Write stage artifacts (if output_dir provided)
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            self._write_artifacts(
                out=out,
                registry=registry,
                raw_response=raw_response,
                thinking=thinking,
                meta=meta,
                global_view=global_view,
                usage=usage,
                elapsed=elapsed,
            )

        logger.info(
            "Agent 1 complete: %d equipment detected in %.1fs",
            registry.total_count,
            elapsed,
        )
        return registry

    # ------------------------------------------------------------------
    # Internal: VLM call
    # ------------------------------------------------------------------

    def _call_vlm(self, image: "np.ndarray", drawing_context: str) -> tuple[str, dict]:
        """Send the downsampled global view to the VLM, return raw response + usage."""
        client = self._get_client()
        image_url = _encode_image_array(image)

        user_prompt = AGENT1_USER_PROMPT_TEMPLATE.replace("{drawing_context}", drawing_context)

        logger.info("Calling %s (max_tokens=%d)...", self.model, DEFAULT_MAX_TOKENS)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": AGENT1_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )

        raw = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        logger.info(
            "VLM response: %d chars, %d prompt tok, %d completion tok",
            len(raw),
            usage["prompt_tokens"],
            usage["completion_tokens"],
        )
        return raw, usage

    def _get_client(self) -> openai.OpenAI:
        if self._client is not None:
            return self._client
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. "
                "Export it or set in your env: export OPENROUTER_API_KEY=sk-or-v1-..."
            )
        self._client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            timeout=DEFAULT_TIMEOUT,
            default_headers={
                "HTTP-Referer": "https://garnet.local",
                "X-Title": "GARNET-VisualPrimitives-Agent1",
            },
        )
        return self._client

    # ------------------------------------------------------------------
    # Internal: artifact writing
    # ------------------------------------------------------------------

    def _write_artifacts(
        self,
        out: Path,
        registry: EquipmentRegistry,
        raw_response: str,
        thinking: str,
        meta,
        global_view: "np.ndarray",
        usage: dict,
        elapsed: float,
    ) -> None:
        """Write the 5 stage artifacts to the output directory."""

        # -- registry JSON --
        registry_path = out / f"{STAGE_PREFIX}_equipment_registry.json"
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info("  Wrote %s", registry_path)

        # -- summary JSON --
        summary = {
            "agent": "agent1_equipment_detector",
            "model": self.model,
            "source_image": meta.source_path,
            "source_dimensions": [meta.width, meta.height],
            "global_view_dimensions": [global_view.shape[1], global_view.shape[0]],
            "global_view_max_dim": self.canvas_cfg.global_view_max_dim,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "elapsed_seconds": round(elapsed, 1),
            "equipment_count": registry.total_count,
            "per_class": _count_per_class(registry),
            "timestamp": datetime.now().isoformat(),
        }
        summary_path = out / f"{STAGE_PREFIX}_equipment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("  Wrote %s", summary_path)

        # -- overlay PNG --
        overlay = _draw_overlay(global_view, registry, self.canvas_cfg.global_view_max_dim)
        overlay_path = out / f"{STAGE_PREFIX}_equipment_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        logger.info("  Wrote %s", overlay_path)

        # -- raw response --
        raw_path = out / f"{STAGE_PREFIX}_raw_response.txt"
        raw_path.write_text(raw_response, encoding="utf-8")
        logger.info("  Wrote %s", raw_path)

        # -- thinking chain --
        thinking_path = out / f"{STAGE_PREFIX}_thinking_chain.txt"
        thinking_path.write_text(thinking, encoding="utf-8")
        logger.info("  Wrote %s", thinking_path)


# ---------------------------------------------------------------------------
# Internal: helpers
# ---------------------------------------------------------------------------


def _encode_image_array(image: "np.ndarray") -> str:
    """Encode an OpenCV BGR image as a base64 data URL for the API."""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Failed to encode image to PNG")
    b64 = base64.b64encode(buf).decode()
    return f"data:image/png;base64,{b64}"


def _count_per_class(registry: EquipmentRegistry) -> dict:
    """Count equipment entries per class."""
    counts: dict[str, int] = {}
    for entry in registry.equipment:
        cls_name = entry.equipment_class.value
        counts[cls_name] = counts.get(cls_name, 0) + 1
    return counts


def _draw_overlay(
    image: "np.ndarray",
    registry: EquipmentRegistry,
    max_dim: int,
) -> "np.ndarray":
    """Draw bounding boxes and tags on the downsampled view for review."""
    out = image.copy()
    scale_x = image.shape[1] / 999.0
    scale_y = image.shape[0] / 999.0

    class_colors = {
        "distillation_column": (0, 165, 255),  # orange
        "pressure_vessel": (255, 0, 0),        # blue
        "heat_exchanger": (0, 255, 0),         # green
        "storage_tank": (0, 255, 255),         # yellow
        "pump": (255, 0, 255),                 # magenta
        "compressor": (255, 255, 0),           # cyan
        "reactor": (128, 0, 128),              # purple
        "other": (128, 128, 128),              # gray
    }

    for entry in registry.equipment:
        x1, y1, x2, y2 = entry.global_bbox
        px1 = int(round(x1 * scale_x))
        py1 = int(round(y1 * scale_y))
        px2 = int(round(x2 * scale_x))
        py2 = int(round(y2 * scale_y))

        color = class_colors.get(entry.equipment_class.value, (128, 128, 128))
        cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)

        label = f"{entry.tag} [{entry.equipment_class.value}] ({entry.confidence.value})"
        font_scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (px1, py1 - th - 6), (px1 + tw + 4, py1), color, -1)
        cv2.putText(
            out, label, (px1 + 2, py1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
        )

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Agent 1: Global Equipment Detector (Visual Primitives)"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to P&ID image")
    parser.add_argument("--output", "-o", default=None, help="Output directory for stage artifacts")
    parser.add_argument("--model", "-m", default=None, help="VLM model override")
    parser.add_argument("--max-dim", type=int, default=None, help="Global view max dimension")
    parser.add_argument("--context", default="", help="Drawing context string")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    cfg = CanvasConfig()
    if args.max_dim:
        cfg.global_view_max_dim = args.max_dim

    agent = Agent1EquipmentDetector(model=args.model, canvas_cfg=cfg)

    try:
        registry = agent.detect(
            image_path=args.image,
            output_dir=args.output,
            drawing_context=args.context,
        )

        # Print summary to stdout
        print(f"\n{'='*50}")
        print(f"  Agent 1 — Global Equipment Detector")
        print(f"  Model: {agent.model}")
        print(f"  Equipment detected: {registry.total_count}")
        print(f"{'='*50}")
        for entry in registry.equipment:
            print(
                f"  {entry.tag:<12} {entry.equipment_class.value:<22} "
                f"[{entry.confidence.value}]  {entry.global_bbox}"
            )
        if registry.drawing_notes:
            print(f"\n  Notes: {registry.drawing_notes}")
        print()

    except Exception as e:
        logger.error("Agent 1 failed: %s", e, exc_info=args.verbose)
        sys.exit(1)


if __name__ == "__main__":
    main()
