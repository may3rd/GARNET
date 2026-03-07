"""
Stage 1-only P&ID pipeline.

This file intentionally starts small: raw P&ID image in, normalization artifact
bundle out. Later slices can add more stages without carrying legacy pipeline
logic forward.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("pid")

DEFAULT_OUT = Path("output")
DEFAULT_OUT.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    adaptive_block_size: int = 21
    adaptive_c: int = 5
    blur_kernel: int = 5


class PIDPipeline:
    def __init__(
        self,
        image_path: str,
        out_dir: str | Path = DEFAULT_OUT,
        cfg: PipelineConfig | None = None,
        stage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **_: Any,
    ) -> None:
        self.image_path = str(image_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or PipelineConfig()
        self.stage_callback = stage_callback

        self.image_bgr: Optional[np.ndarray] = None
        self.stage_manifest: Dict[str, Any] = {}

    # ---------- Stage runner ----------
    def _stage_definitions(self) -> List[Tuple[int, str, Callable[[], None]]]:
        return [(1, "stage1_input_normalization", self.stage1_input_normalization)]

    def _manifest_path(self) -> Path:
        return self.out_dir / "stage_manifest.json"

    def _write_stage_manifest(self) -> None:
        with open(self._manifest_path(), "w") as f:
            json.dump(self.stage_manifest, f, indent=2)
        logger.info(f"saved {self._manifest_path()}")

    def _notify_stage_callback(self, payload: Dict[str, Any]) -> None:
        if self.stage_callback is not None:
            self.stage_callback(payload)

    def _reset_stage_manifest(self, stop_after: int) -> None:
        self.stage_manifest = {
            "image_path": self.image_path,
            "out_dir": str(self.out_dir),
            "stop_after": stop_after,
            "stages": [],
        }
        self._write_stage_manifest()

    def _stage_artifacts_since(self, started_at: float) -> List[str]:
        artifacts: List[str] = []
        for path in self.out_dir.iterdir():
            if not path.is_file() or path.name == "stage_manifest.json":
                continue
            if path.stat().st_mtime >= started_at:
                artifacts.append(path.name)
        return sorted(artifacts)

    def _run_stage(self, stage_num: int, stage_name: str, stage_fn: Callable[[], None]) -> None:
        started_at = time.time()
        entry = {
            "num": stage_num,
            "name": stage_name,
            "status": "started",
            "started_at": started_at,
            "artifacts": [],
        }
        self.stage_manifest["stages"].append(entry)
        self._write_stage_manifest()
        self._notify_stage_callback({"event": "stage_started", "stage": entry.copy(), "manifest": self.stage_manifest})
        try:
            stage_fn()
        except Exception as exc:
            entry["status"] = "failed"
            entry["ended_at"] = time.time()
            entry["duration_sec"] = round(entry["ended_at"] - started_at, 6)
            entry["artifacts"] = self._stage_artifacts_since(started_at)
            entry["error"] = str(exc)
            self._write_stage_manifest()
            self._notify_stage_callback({"event": "stage_failed", "stage": entry.copy(), "manifest": self.stage_manifest})
            raise
        entry["status"] = "completed"
        entry["ended_at"] = time.time()
        entry["duration_sec"] = round(entry["ended_at"] - started_at, 6)
        entry["artifacts"] = self._stage_artifacts_since(started_at)
        self._write_stage_manifest()
        self._notify_stage_callback({"event": "stage_completed", "stage": entry.copy(), "manifest": self.stage_manifest})

    def run(self, stop_after: int = 1) -> None:
        stages = self._stage_definitions()
        if stop_after < 1 or stop_after > len(stages):
            raise ValueError(f"stop_after must be between 1 and {len(stages)}, got {stop_after}")
        self._reset_stage_manifest(stop_after)
        for stage_num, stage_name, stage_fn in stages:
            if stage_num > stop_after:
                break
            self._run_stage(stage_num, stage_name, stage_fn)

    # ---------- Persistence ----------
    def _save_img(self, name: str, img: np.ndarray) -> None:
        path = self.out_dir / f"{name}.png"
        out = img
        if out.dtype == bool:
            out = out.astype(np.uint8) * 255
        elif out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        if cv2 is not None:
            cv2.imwrite(str(path), out)
        elif Image is not None:
            Image.fromarray(out).save(str(path))
        else:  # pragma: no cover
            raise RuntimeError("No image backend available")
        logger.info(f"saved {path}")

    def _save_json(self, name: str, data: Any) -> None:
        path = self.out_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"saved {path}")

    # ---------- Stage 1 ----------
    def _ensure_image_loaded(self) -> np.ndarray:
        if self.image_bgr is not None:
            return self.image_bgr
        if cv2 is not None:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {self.image_path}")
            self.image_bgr = img
            return img
        if Image is not None:
            img = Image.open(self.image_path).convert("RGB")
            self.image_bgr = np.array(img)[:, :, ::-1]
            return self.image_bgr
        raise RuntimeError("No image backend available")  # pragma: no cover

    def stage1_input_normalization(self) -> None:
        image = self._ensure_image_loaded()
        if cv2 is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (self.cfg.blur_kernel, self.cfg.blur_kernel), 0)
            adaptive = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.cfg.adaptive_block_size,
                self.cfg.adaptive_c,
            )
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            equalized = cv2.equalizeHist(gray)
        else:
            gray = np.dot(image[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
            blur = gray
            threshold = int(gray.mean())
            adaptive = (gray < threshold).astype(np.uint8) * 255
            otsu = adaptive.copy()
            equalized = gray

        self._save_img("stage1_gray", gray)
        self._save_img("stage1_gray_equalized", equalized)
        self._save_img("stage1_binary_adaptive", adaptive)
        self._save_img("stage1_binary_otsu", otsu)
        self._save_json(
            "stage1_normalization_summary",
            {
                "image_path": self.image_path,
                "dimensions": {"height": int(image.shape[0]), "width": int(image.shape[1])},
                "artifacts": [
                    "stage1_gray.png",
                    "stage1_gray_equalized.png",
                    "stage1_binary_adaptive.png",
                    "stage1_binary_otsu.png",
                ],
                "config": {
                    "adaptive_block_size": self.cfg.adaptive_block_size,
                    "adaptive_c": self.cfg.adaptive_c,
                    "blur_kernel": self.cfg.blur_kernel,
                },
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser("P&ID Stage 1 pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--stop-after", type=int, default=1, help="Run up to this stage (1-1)")
    args = parser.parse_args()
    pipe = PIDPipeline(args.image, out_dir=args.out)
    pipe.run(stop_after=args.stop_after)


if __name__ == "__main__":
    main()
