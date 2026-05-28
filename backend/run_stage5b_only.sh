#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_stage5b_only.sh [OUT_DIR]
# Example:
#   ./run_stage5b_only.sh /Users/maetee/Code/GARNET/backend/output_debug/Test-00001

OUT_DIR="${1:-/Users/maetee/Code/GARNET/backend/output_debug/Test-00001}"
PY_BIN="${PY_BIN:-/Users/maetee/Code/GARNET/.venv/bin/python}"

cd /Users/maetee/Code/GARNET/backend

"$PY_BIN" - "$OUT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from garnet.pid_extractor import PIDPipeline, PipelineConfig


def fail(msg: str) -> None:
    raise SystemExit(f"[stage5b-only] {msg}")


out_dir = Path(sys.argv[1]).resolve()
manifest_path = out_dir / "stage_manifest.json"
if not manifest_path.exists():
    fail(f"missing manifest: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
image_path = manifest.get("image_path")
if not image_path:
    fail("stage_manifest.json has no image_path")

required = [
    out_dir / "stage4_objects.json",
    out_dir / "stage4_instrument_tags.json",
    out_dir / "stage5_pipe_mask.png",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    fail("missing required artifacts:\n  - " + "\n  - ".join(missing))

pipe = PIDPipeline(
    image_path=str(image_path),
    output_dir=str(out_dir),
    cfg=PipelineConfig(use_geometric_line_detection=True),
)
pipe.stage5b_pipe_trace()
print(f"[stage5b-only] done: {out_dir / 'stage5b_trace_results.json'}")
PY

