#!/usr/bin/env python
"""Run Stage 5b (CV pipe tracing + branch tracing) and emit the two trace overlays.

Reuses the real tracer — `Stage5bPipelineMixin.stage5b_pipe_trace()` from
`backend/garnet/path_tracer/stage5b_pipeline.py` (which drives `CVPipeTracer`) — rather than
reimplementing any of it. This script only supplies the prerequisites and calls it.

Outputs (written into the job/output dir):
    stage5b_trace_overlay.png          port traces + terminals
    stage5b_branch_trace_overlay.png   the above + tee-branch traces
plus the JSON payloads:
    stage5b_trace_results.json
    stage5b_branch_candidates.json
    stage5b_branch_trace_results.json
    stage5_connection_ports.json

Prerequisites, generated on demand if absent:
    stage1_*                       (normalization)
    stage2_ocr_regions.json        (OCR; needed so the pipe mask can erase text)
    stage4_objects.json            (YOLO, or seeded from --stage4-objects)
    stage5_pipe_mask.png           (mask the tracer walks)

Usage:
    # from a raster, generating whatever prerequisites are missing
    python garnet/run_stage5b.py --image garnet/tests/input/Test-00001.jpg --out output/stage5b_test01

    # reuse an existing job that already has stage4/stage5 artifacts
    python garnet/run_stage5b.py --job-dir backend/output/pipeline_jobs/Baseline

    # seed detection from a fixture instead of re-running YOLO
    python garnet/run_stage5b.py --image ... --out ... \
        --stage4-objects garnet/tests/input/Test-00001_objects.json

Run with the repo venv: /Users/maetee/Code/GARNET/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))          # garnet.* lives under backend/
OVERLAYS = ("stage5b_trace_overlay.png", "stage5b_branch_trace_overlay.png")
PREREQS = ("stage4_objects.json", "stage5_pipe_mask.png")


def _ensure_backend_on_path() -> None:
    """`garnet.*` (the pipeline package) lives under backend/, not repo root."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (p if p.exists() else REPO_ROOT / p)


def _have(out_dir: Path, names) -> bool:
    return all((out_dir / n).exists() for n in names)


def _write_objects_fixture(src: Path, out_dir: Path) -> int:
    """Seed stage4_objects.json from a fixture, normalizing our detect_objects.py shape.

    `garnet/detect_objects.py` emits the same object keys as the pipeline except it is
    written to a file with extra run metadata, which stage 4 consumers ignore.
    """
    payload = json.loads(src.read_text())
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{src.name}: no 'objects' list")
    # Pipeline shape: {'image_id','pass_type','objects':[...]}
    normalized = {
        "image_id": payload.get("image_id", src.stem),
        "pass_type": payload.get("pass_type", "sheet"),
        "objects": objects,
    }
    (out_dir / "stage4_objects.json").write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    (out_dir / "stage4_objects_summary.json").write_text(
        json.dumps({"object_count": len(objects), "seeded_from": str(src)}, indent=2), encoding="utf-8")
    return len(objects)


def _write_equipment_seed(src: Path, out_dir: Path) -> tuple[int, int]:
    """Seed stage3_equipment_bboxes.json (and ai_equipment_ports.json) from a Contract B file.

    Stage 5b only traces from ports, and equipment ports come from these Stage 3 bboxes (or an
    external import). Without them a sheet traces only its page connections, which is why a run
    with no equipment seed yields a handful of traces instead of the full network.
    """
    payload = json.loads(src.read_text())
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{src.name}: no 'objects' list")

    # Reuse the project's own synonym table and vocabulary gate instead of duplicating them:
    # a Contract B file names equipment in prose (`static mixer`), while the loader only accepts
    # EQUIPMENT_LABELS (`mixer`). Bypassing AI_CLASS_MAP silently drops items.
    _ensure_backend_on_path()
    from garnet.ai_import import AI_CLASS_MAP
    from garnet.pid_extractor import EQUIPMENT_LABELS

    equipment, ports = [], {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        bbox = item.get("Bounding_box_px") or {}
        if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bbox):
            continue
        raw = str(item.get("Equipment_type") or "").strip().lower().replace("_", " ")
        label = raw if raw in EQUIPMENT_LABELS else AI_CLASS_MAP.get(raw, "")
        if not label:
            continue
        tag = str(item.get("Tag") or "").strip()
        eq_id = f"equip_{tag.lower().replace('-', '_').replace(' ', '_')}" if tag else f"equip_{len(equipment):03d}"
        equipment.append({
            "id": eq_id,
            "class_name": label,
            "bbox": {k: int(round(float(bbox[k]))) for k in ("x_min", "y_min", "x_max", "y_max")},
            "tag": tag,
            "review_state": "accepted",
            "source": "ai_seed",
        })
        # Nozzle points are authoritative when present; the pipeline projects them onto the box edge.
        usable = []
        for port in item.get("Ports") or []:
            if not isinstance(port, dict):
                continue
            side = str(port.get("side") or "").upper()
            pt = port.get("point_px")
            if side not in ("TOP", "BOTTOM", "LEFT", "RIGHT"):
                continue
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                continue
            usable.append({"x": int(round(float(pt[0]))), "y": int(round(float(pt[1]))), "direction": side})
        if usable:
            ports[eq_id] = usable

    (out_dir / "stage3_equipment_bboxes.json").write_text(
        json.dumps({"equipment": equipment}, indent=2), encoding="utf-8")
    if ports:
        (out_dir / "ai_equipment_ports.json").write_text(json.dumps(ports, indent=2), encoding="utf-8")
    return len(equipment), sum(len(v) for v in ports.values())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run Stage 5b pipe tracing and emit the trace overlays.")
    p.add_argument("--image", help="Sheet raster. Required unless --job-dir is used.")
    p.add_argument("--out", help="Output/job dir. Default: output/stage5b_<image-stem>")
    p.add_argument("--job-dir", help="Existing pipeline job dir; trace only using its artifacts.")
    p.add_argument("--stage4-objects", default=None,
                   help="Seed stage4_objects.json from this JSON instead of running YOLO.")
    p.add_argument("--equipment", default=None,
                   help="Seed stage3_equipment_bboxes.json + nozzle ports from a Contract B JSON. "
                        "Without this, tracing starts only from page connections and yields few traces.")
    p.add_argument("--ocr-route", default="ocrmac",
                   choices=["easyocr", "gemini", "paddleocr", "ocrmac"],
                   help="Stage 2 OCR route when OCR must be generated. Default: ocrmac")
    p.add_argument("--debug-artifacts", action="store_true",
                   help="Also emit per-trace images and branch-candidate iteration overlays.")
    p.add_argument("--force", action="store_true", help="Re-trace even if overlays already exist.")
    args = p.parse_args(argv)

    if args.job_dir:
        out_dir = _resolve(args.job_dir)
        if not out_dir.is_dir():
            print(f"error: job dir not found: {out_dir}", file=sys.stderr)
            return 1
        manifest = out_dir / "stage_manifest.json"
        if args.image:
            image_path = _resolve(args.image)
        elif manifest.exists():
            image_path = Path(json.loads(manifest.read_text()).get("image_path") or "")
        else:
            print(f"error: no --image and no image_path in {manifest}", file=sys.stderr)
            return 1
    else:
        if not args.image:
            print("error: --image is required (or use --job-dir)", file=sys.stderr)
            return 1
        image_path = _resolve(args.image)
        out_dir = _resolve(args.out) if args.out else REPO_ROOT / "output" / f"stage5b_{image_path.stem}"

    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"image   : {image_path}")
    print(f"out dir : {out_dir}")

    if _have(out_dir, OVERLAYS) and not args.force:
        print("\noverlays already present (use --force to re-trace)")
        for name in OVERLAYS:
            print(f"  {out_dir / name}")
        return 0

    from garnet.pid_extractor import PIDPipeline, PipelineConfig

    cfg = PipelineConfig(debug_artifacts=args.debug_artifacts, ocr_route=args.ocr_route)
    pipe = PIDPipeline(image_path=str(image_path), output_dir=str(out_dir), cfg=cfg)

    # ---- prerequisites -------------------------------------------------------
    stage1 = ("stage1_gray.png", "stage1_binary_adaptive.png", "stage1_binary_otsu.png")
    if not _have(out_dir, stage1):
        print("\n[stage1] normalization")
        pipe.stage1_input_normalization()

    if not (out_dir / "stage2_ocr_regions.json").exists():
        print(f"\n[stage2] OCR ({args.ocr_route})")
        pipe.stage2_ocr_discovery()

    if not (out_dir / "stage4_objects.json").exists():
        if args.stage4_objects:
            seeded = _write_objects_fixture(_resolve(args.stage4_objects), out_dir)
            print(f"\n[stage4] seeded {seeded} objects from {args.stage4_objects}")
        else:
            print("\n[stage4] object detection")
            pipe.stage4_object_detection()
            # line-number / instrument-tag fusion feed downstream stages and the overlay labels;
            # they need stage2 text, and are cheap relative to the trace itself.
            try:
                pipe.stage4_line_number_fusion()
            except Exception as exc:      # non-fatal: tracing does not depend on them
                print(f"  (line-number fusion skipped: {type(exc).__name__}: {exc})")
            try:
                pipe.stage4_instrument_tag_fusion()
            except Exception as exc:
                print(f"  (instrument-tag fusion skipped: {type(exc).__name__}: {exc})")

    # Equipment seed must land BEFORE the pipe mask / port computation: `stage5_pipe_mask`
    # applies the AI import and Stage 5 owns port detection, so seeding afterwards leaves
    # equipment out of `stage5_connection_ports.json` and the tracer never starts at nozzles.
    if args.equipment and not (out_dir / "stage3_equipment_bboxes.json").exists():
        n_eq, n_ports = _write_equipment_seed(_resolve(args.equipment), out_dir)
        print(f"\n[stage3] seeded {n_eq} equipment bboxes, {n_ports} nozzle ports "
              f"from {args.equipment}")
        stale = out_dir / "stage5_connection_ports.json"
        if stale.exists():
            stale.unlink()
            print("  (removed stale stage5_connection_ports.json so ports recompute)")

    if not (out_dir / "stage5_pipe_mask.png").exists():
        print("\n[stage5] pipe mask")
        pipe.stage5_pipe_mask()

    missing = [n for n in PREREQS if not (out_dir / n).exists()]
    if missing:
        print(f"error: prerequisites still missing after generation: {missing}", file=sys.stderr)
        return 1

    # ---- the actual work: reuse the real Stage 5b tracer ---------------------
    print("\n[stage5b] pipe trace + branch trace")
    pipe.stage5b_pipe_trace()

    # ---- report --------------------------------------------------------------
    print()
    produced = []
    for name in OVERLAYS:
        path = out_dir / name
        if path.exists():
            produced.append(path)
            print(f"  {name}")
        else:
            print(f"  {name}  MISSING")

    trace_json = out_dir / "stage5b_trace_results.json"
    branch_json = out_dir / "stage5b_branch_trace_results.json"
    if trace_json.exists():
        traces = json.loads(trace_json.read_text())
        by_terminal: dict[str, int] = {}
        for t in traces.values():
            key = str(t.get("terminal_type", "unknown"))
            by_terminal[key] = by_terminal.get(key, 0) + 1
        print(f"\ntraces: {len(traces)}")
        for k, v in sorted(by_terminal.items(), key=lambda kv: -kv[1]):
            print(f"  {v:>4}  {k}")
    if branch_json.exists():
        b = json.loads(branch_json.read_text())
        print(f"branches: {b.get('summary', {})}")

    if len(produced) != len(OVERLAYS):
        return 1
    print(f"\nOK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
