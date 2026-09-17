#!/usr/bin/env python
"""Run the copied Stage 5b pipe tracer end to end, from fixtures, with no backend/ dependency.

Pipeline:
    fixtures (objects + line numbers merged, equipment)
        -> pipe mask (fixture-suppressed, no OCR)
        -> Stage5bHost (supplies the 10 methods + config the mixin needs)
        -> Stage5bPipelineMixin.stage5b_pipe_trace()   <- the copied algorithm
        -> stage5b_trace_overlay.png + stage5b_branch_trace_overlay.png

Usage (from the repo root):
    python -m garnet.stage5b.run --stem Test-00001
    python -m garnet.stage5b.run --stem Test-00001 --out output/s5b_test01

Run with the repo venv: /Users/maetee/Code/GARNET/.venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2

if __package__ in (None, ""):                       # allow `python garnet/stage5b/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "garnet.stage5b"

from .host import (Stage5bConfig, Stage5bHost, build_pipe_mask, load_equipment_ports,
                   load_fixtures)
from .stage5b_pipeline import Stage5bPipelineMixin

REPO_ROOT = Path(__file__).resolve().parents[2]
OVERLAYS = ("stage5b_trace_overlay.png", "stage5b_branch_trace_overlay.png")


class _Runner(Stage5bPipelineMixin, Stage5bHost):
    """Combines the mixin (algorithm) with the host (I/O + config)."""


def fill_equipment_ports_from_mask(mask, equipment: list[dict],
                                   ports: dict[str, list[dict]]) -> int:
    """Give equipment that has NO port data at all a trace start, using the mask.

    Extraction ports are authoritative, so this only fires for an item the extraction left
    completely undescribed — never to top up individual empty sides. Filling empty sides of a
    described item produced bad ports: the pumps' extraction gives top+bottom (the real
    nozzles), and the LEFT/RIGHT sides I added landed on the pump symbol itself, so those
    traces terminated on the symbol after a single segment.

    For each side, scan outward along the edge normal from the edge midpoint and accept the
    first pipe pixel whose run continues for >= 6 px. Scanning perpendicular from the midpoint
    (rather than nearest-pixel-anywhere) is what keeps a nozzle on the correct side.

    Returns the number of nozzle points added.
    """
    h, w = mask.shape
    normals = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    added = 0
    for item in equipment:
        b = item["bbox"]
        have = ports.get(item["id"]) or []
        if have:
            continue                          # extraction described this item — leave it alone
        found: list[dict] = []
        for side, (dx, dy) in normals.items():
            if side in ("UP", "DOWN"):
                px, py = (b["x_min"] + b["x_max"]) // 2, (b["y_min"] if side == "UP" else b["y_max"])
            else:
                px, py = (b["x_min"] if side == "LEFT" else b["x_max"]), (b["y_min"] + b["y_max"]) // 2
            # Scan from just outside the edge outward. Step 0 (the edge itself) is often blanked
            # because symbol interiors/edges are suppressed, so start at 1 but keep going —
            # the pumps on this sheet attach ~36-54 px out.
            for step in range(1, 121):
                qx, qy = px + dx * step, py + dy * step
                if not (0 <= qx < w and 0 <= qy < h):
                    break
                if not mask[qy, qx]:
                    continue
                # accept a run along the scan direction OR perpendicular to it, because a nozzle
                # stub can meet the pipe at a T as well as continue straight out
                best_run = 0
                for odx, ody in ((dx, dy), (-dx, -dy), (dy, -dx), (-dy, dx)):
                    run = 0
                    for s2 in range(0, 10):
                        rx, ry = qx + odx * s2, qy + ody * s2
                        if 0 <= rx < w and 0 <= ry < h and mask[ry, rx]:
                            run += 1
                        else:
                            break
                    best_run = max(best_run, run)
                if best_run >= 6:
                    found.append({"x": int(qx), "y": int(qy), "direction": side})
                    added += 1
                break                          # only the first pipe per edge
        if found:
            ports[item["id"]] = found
    return added


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 5b pipe tracing from fixtures (backend-free).")
    p.add_argument("--stem", default="Test-00001")
    p.add_argument("--indir", default="garnet/tests/input")
    p.add_argument("--out", default=None, help="Default: output/stage5b_<stem>")
    p.add_argument("--debug-artifacts", action="store_true",
                   help="Also write per-trace images and branch-candidate iteration overlays.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    indir = Path(args.indir)
    if not indir.is_absolute():
        indir = REPO_ROOT / indir
    out_dir = Path(args.out) if args.out else REPO_ROOT / "output" / f"stage5b_{args.stem}"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stale artifacts from a previous run would be reused instead of recomputed: the mixin only
    # computes connection ports when stage5_connection_ports.json is absent, so a leftover file
    # silently pins the trace starts to an older equipment set.
    stale = out_dir / "stage5_connection_ports.json"
    if stale.exists():
        stale.unlink()

    image_path = indir / f"{args.stem}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"error: cannot read {image_path}", file=sys.stderr)
        return 1
    h, w = image.shape[:2]

    # ---- fixtures ---------------------------------------------------------
    objects, equipment = load_fixtures(indir, args.stem)
    n_ln = sum(1 for o in objects if str(o.get("class_name", "")).lower() == "line number")
    n_text = sum(1 for o in objects if str(o.get("text") or "").strip())
    print(f"{args.stem}: {w}x{h}")
    print(f"  objects   : {len(objects)} (of which {n_ln} line-number boxes, {n_text} carrying text)")
    print(f"  equipment : {len(equipment)}")

    # Nozzle points supplied by the fixture
    ports = load_equipment_ports(indir, args.stem)
    print(f"  ports     : {sum(len(v) for v in ports.values())} nozzle points from the fixture "
          f"on {len(ports)} of {len(equipment)} equipment items")

    # ---- mask -------------------------------------------------------------
    mask, mask_stats = build_pipe_mask(image, objects, equipment)
    # Equipment whose Ports carry a `side` but no `point_px` would never be traced from. Fill
    # those from the mask: for each bbox edge, take the nearest pipe pixel perpendicular to that
    # edge (within reach). Same information the pipeline's `_detect_equipment_ports_cv` derives,
    # but keyed to the edge midpoints so a nozzle cannot be assigned to the wrong side.
    filled = fill_equipment_ports_from_mask(mask, equipment, ports)
    if filled:
        print(f"  ports     : +{filled} nozzle(s) recovered from the pipe mask")
    total_ports = sum(len(v) for v in ports.values())
    covered = sum(1 for it in equipment if ports.get(it["id"]))
    print(f"  ports     : {total_ports} nozzle points covering {covered}/{len(equipment)} equipment items")
    if covered < len(equipment):
        missing = [it["id"] for it in equipment if not ports.get(it["id"])]
        print(f"              no trace start for: {', '.join(missing)}")
    cv2.imwrite(str(out_dir / "stage5_pipe_mask.png"), mask)
    (out_dir / "stage5_pipe_mask_summary.json").write_text(
        json.dumps(mask_stats, indent=2), encoding="utf-8")
    print(f"  pipe mask : {mask_stats['mask_px']:,} px ({mask_stats['pct_of_sheet']}% of sheet), "
          f"{mask_stats['text_boxes_suppressed']} text boxes suppressed, "
          f"{mask_stats['components_removed']} specks removed")

    # ---- stage-4 style objects (what the mixin loads) ---------------------
    (out_dir / "stage4_objects.json").write_text(
        json.dumps({"image_id": image_path.name, "pass_type": "sheet", "objects": objects},
                   indent=2), encoding="utf-8")
    # equipment in the shape _load_equipment_bboxes_for_stage5b expects
    (out_dir / "stage3_equipment_bboxes.json").write_text(
        json.dumps({"equipment": equipment}, indent=2), encoding="utf-8")
    if ports:
        (out_dir / "ai_equipment_ports.json").write_text(
            json.dumps(ports, indent=2), encoding="utf-8")

    # ---- run the copied tracer -------------------------------------------
    cfg = Stage5bConfig(debug_artifacts=args.debug_artifacts)
    runner = _Runner(image_path=str(image_path), out_dir=out_dir, cfg=cfg)
    print("\n[stage5b] pipe trace + branch trace")
    runner.stage5b_pipe_trace()

    # ---- report -----------------------------------------------------------
    print()
    produced = []
    for name in OVERLAYS:
        path = out_dir / name
        if path.exists():
            produced.append(path)
            print(f"  {name}")
        else:
            print(f"  {name}  MISSING")

    t = out_dir / "stage5b_trace_results.json"
    if t.exists():
        traces = json.loads(t.read_text())
        print(f"\ntraces: {len(traces)}")
        for k, v in Counter(str(x.get("terminal_type")) for x in traces.values()).most_common():
            print(f"  {v:>4}  {k}")
        # Port coverage: how many equipment items actually got a trace start. An equipment item
        # with no port is never traced from, so a silent drop here shrinks the network.
        cports = out_dir / "stage5_connection_ports.json"
        if cports.exists():
            got = json.loads(cports.read_text())
            eq_got = {k for k in got if str(k).startswith("equip_")}
            eq_all = {it["id"] for it in equipment}
            missing = sorted(eq_all - eq_got)
            print(f"equipment traced from: {len(eq_got)}/{len(eq_all)}"
                  + (f"  MISSING PORTS: {missing}" if missing else ""))
    b = out_dir / "stage5b_branch_trace_results.json"
    if b.exists():
        payload = json.loads(b.read_text())
        branches = payload.get("branches", {})
        print(f"branches: {len(branches)}")
        for k, v in Counter(str(x.get("terminal_type")) for x in branches.values()).most_common():
            print(f"  {v:>4}  {k}")
        print(f"summary: {payload.get('summary', {})}")

    if len(produced) != len(OVERLAYS):
        return 1
    print(f"\nOK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
