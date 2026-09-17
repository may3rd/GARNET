#!/usr/bin/env python
"""Stage 6 — attach semantic evidence (line numbers, valves, tags, arrows) to traced paths.

Vendored from `backend/garnet/trace_associations.py` + `flow_direction.py` so it runs from
repo-root `garnet/` with no backend dependency.

    python -m garnet.stage5b.run_associations --stem Test-00001

Reads the Stage 5b artifacts this package writes and produces:

    stage6_trace_associations.json          every edge with its `attachments`
    stage6_trace_association_summary.json   counts per evidence group
    stage6_line_number_review.json          line numbers accepted / needing review
    stage6_line_number_review_summary.json
    stage6_trace_association_overlay.png    edges + attachment markers   <- deliverable

**Line-number association is the point of this stage.** A line-number label is attached to the
traced path it sits along, not merely the nearest one: when the globally nearest segment runs
perpendicular to the label's long axis (a crossing pipe clipping the label corner), an
orientation-matching segment within threshold is preferred instead. Labels that find no path
within `--text-max-distance` land in `needs_review` rather than being forced onto a wrong pipe,
and traces with no label are reported in `traces_without_line_number` — never fabricated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("stage6.associations")

# Mirrors PipelineConfig defaults (backend/garnet/pid_extractor.py) so the vendored stage
# behaves like the pipeline without importing it.
DEFAULTS = {
    "equipment_port_max_distance_px": 16.0,
    "inline_object_max_distance_px": 24.0,
    "text_max_distance_px": 100.0,
    "instrument_max_distance_px": 90.0,
    "arrow_max_distance_px": 45.0,
    "flow_arrow_raster_confidence_threshold": 0.70,
    "flow_arrow_raster_asymmetry_threshold": 0.15,
}


def _load(path: Path, what: str) -> dict:
    if not path.is_file():
        print(f"error: missing {what}: {path}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", default="Test-00001")
    p.add_argument("--indir", default="garnet/tests/input",
                   help="directory holding <stem>.jpg")
    p.add_argument("--dir", default=None,
                   help="stage5b output dir (default output/stage5b_<stem>)")
    p.add_argument("--out", default=None, help="output dir (default <dir>)")
    for key, val in DEFAULTS.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=float, default=val)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    indir = Path(args.indir)
    if not indir.is_absolute():
        indir = REPO_ROOT / indir
    sdir = Path(args.dir) if args.dir else Path(f"output/stage5b_{args.stem}")
    if not sdir.is_absolute():
        sdir = REPO_ROOT / sdir
    out_dir = Path(args.out) if args.out else sdir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from .trace_associations import build_trace_associations, render_trace_association_overlay

    image_path = indir / f"{args.stem}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"error: cannot read {image_path}", file=sys.stderr)
        return 1

    object_payload = _load(sdir / "stage4_objects.json", "stage 4 objects")
    trace_payload = _load(sdir / "stage5b_trace_results.json", "stage 5b traces")
    branch_payload = json.loads((sdir / "stage5b_branch_trace_results.json").read_text()) \
        if (sdir / "stage5b_branch_trace_results.json").is_file() else {"branches": {}}
    ports_payload = json.loads((sdir / "stage5_connection_ports.json").read_text()) \
        if (sdir / "stage5_connection_ports.json").is_file() else {}

    # Line numbers live in the merged object list (no OCR stage, no separate artifact).
    objects = object_payload.get("objects", [])
    line_numbers = [o for o in objects
                    if str(o.get("class_name", "")).lower() == "line number"]
    # Instrument tags are not a separate artifact here; take them from the object list.
    instrument_tags = [o for o in objects
                       if str(o.get("class_name", "")).lower()
                       in ("instrument tag", "instrument dcs", "instrument logic")]

    print(f"{args.stem}: {len(objects)} objects, {len(line_numbers)} line numbers "
          f"({sum(1 for o in line_numbers if str(o.get('text') or '').strip())} with text), "
          f"{len(instrument_tags)} instrument tags")
    print(f"  traces {len(trace_payload)}, branches {len(branch_payload.get('branches', {}))}")

    result = build_trace_associations(
        image_id=args.stem,
        objects=objects,
        trace_payload=trace_payload,
        branch_payload=branch_payload,
        ports_payload=ports_payload,
        line_numbers=line_numbers,
        instrument_tags=instrument_tags,
        image_bgr=image,
        **{k: getattr(args, k) for k in DEFAULTS},
    )

    for name, payload in (
        ("stage6_trace_associations", result["trace_associations_payload"]),
        ("stage6_trace_association_summary", result["trace_association_summary"]),
        ("stage6_line_number_review", result["line_number_review_payload"]),
        ("stage6_line_number_review_summary", result["line_number_review_summary"]),
    ):
        (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overlay = render_trace_association_overlay(image, result["trace_edges"],
                                               result["associations"])
    cv2.imwrite(str(out_dir / "stage6_trace_association_overlay.png"), overlay)

    s = result["trace_association_summary"]
    acc, rej = s["accepted_counts"], s["rejected_counts"]
    print(f"\n  edges: {s['trace_edge_count']} "
          f"({s['port_trace_count']} port, {s['branch_trace_count']} branch), "
          f"{s['skipped_branch_count']} skipped")
    print("  accepted:", dict(sorted(acc.items())))
    print("  rejected:", dict(sorted(rej.items())))
    print(f"  traces without a line number: {s['trace_without_line_number_count']}")
    print(f"  dead-end traces: {s['dead_end_trace_count']}")

    print("\n=== line numbers (the point of this stage) ===")
    ln_ok = result["associations"]["line_numbers"]["accepted"]
    ln_no = result["associations"]["line_numbers"]["rejected"]
    print(f"  attached: {len(ln_ok)}   needs review: {len(ln_no)}")
    for item in ln_ok:
        txt = str(item.get("text") or item.get("normalized_text") or item.get("id"))
        print(f"    {item['trace_id']:26} <- {txt[:34]:34} "
              f"dist={item.get('distance_px')}px seg={item.get('segment_index')}")
    for item in ln_no[:10]:
        print(f"    UNATTACHED {str(item.get('id')):16} "
              f"{str(item.get('text') or '')[:30]:30} {item.get('reason')}")

    # traces that got no label at all — reported, never fabricated
    no_ln = result["trace_associations_payload"]["unresolved"]["traces_without_line_number"]
    print(f"\n  {len(no_ln)} traces have no line number: {no_ln[:6]}"
          f"{' ...' if len(no_ln) > 6 else ''}")

    print(f"\nOK -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
