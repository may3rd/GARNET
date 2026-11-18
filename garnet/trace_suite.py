#!/usr/bin/env python
"""Run the Stage 5b pipe trace over a set of sheets and dump a reviewable bundle.

For every sheet: seed equipment from the matching GT delivery export (so the score measures
tracing, not detection), trace, then score against the GT line paths. Writes one directory per
sheet plus a summary, so a human can open the overlays and the per-line table and decide.

    python garnet/trace_suite.py --out ~/Downloads/garnet-trace-tests \
        --sheet 0002A:~/Downloads/0002A: \
        --sheet 0002:~/Downloads/0002_0003:0002 \
        --sheet 0003:~/Downloads/0002_0003:0003
"""
from __future__ import annotations
import argparse, csv, json, os, re, shutil, subprocess, sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
INPUT = REPO / "garnet" / "tests" / "input"


def sh(cmd: list[str], *, backend: bool = False) -> str:
    env = dict(os.environ)
    if backend:
        env["PYTHONPATH"] = str(REPO / "backend")
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(f"FAILED: {' '.join(cmd)}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout + r.stderr          # the pipeline logs through logging -> stderr


def gt_model(gt_dir: Path) -> dict:
    raw = json.loads((gt_dir / "connectivity_model.json").read_text())
    return raw.get("model", raw)


def draw_overlay(image: Path, traced: Path, gt_dir: Path, sheet: str, out: Path) -> None:
    """GT paths green, traced walks red, on the sheet."""
    img = cv2.imread(str(image))
    for ln in gt_model(gt_dir).get("lines", []):
        if sheet and str(ln.get("sheet") or "") != sheet:
            continue
        pts = [(int(int(a) * .5), int(int(b) * .5))
               for a, b in re.findall(r"\((\d+),\s*(\d+)\)", ln.get("path_native_px") or "")]
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], False, (0, 200, 0), 3)
    tr = json.loads((traced / "stage5b_trace_results.json").read_text())
    raw = json.loads((traced / "stage5b_branch_trace_results.json").read_text())["branches"]
    walks = list(tr.values()) + [b for b in (raw.values() if isinstance(raw, dict) else raw)
                                 if isinstance(b, dict) and b.get("status") == "traced"]
    for w in walks:
        for s in w.get("segments") or []:
            cv2.line(img, (int(s["x1"]), int(s["y1"])), (int(s["x2"]), int(s["y2"])), (0, 0, 255), 1)
    cv2.imwrite(str(out), img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="bundle dir (e.g. ~/Downloads/garnet-trace-tests)")
    ap.add_argument("--sheet", action="append", required=True,
                    metavar="SHEET:GT_DIR[:GT_SHEET]", help="repeatable")
    ap.add_argument("--work", default="output/trace_suite", help="per-run artifacts (repo-relative)")
    a = ap.parse_args()

    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.sheet:
        parts = spec.split(":")
        sheet, gt_dir = parts[0], Path(parts[1]).expanduser()
        gt_sheet = parts[2] if len(parts) > 2 else ""
        image = INPUT / f"14780-8120-25-25-{sheet}.jpg"
        run_dir = REPO / a.work / sheet
        dest = out / sheet
        dest.mkdir(exist_ok=True)

        # A cached seed set from an earlier run would quietly mask a seeding change:
        # ports are only recomputed when this artifact is absent.
        for stale in ("stage5_connection_ports.json", "stage5b_trace_results.json",
                      "stage5b_branch_trace_results.json", "stage5b_branch_candidates.json"):
            (run_dir / stale).unlink(missing_ok=True)

        seed = Path("/tmp") / f"gt_equip_{sheet}.json"
        sh([PY, "garnet/gt_equipment_seed.py", "--gt", str(gt_dir),
            "--out", str(seed)] + (["--sheet", gt_sheet] if gt_sheet else []))
        log = sh([PY, "garnet/run_stage5b.py", "--image", str(image),
                  "--out", str(run_dir), "--equipment", str(seed), "--force"])
        (dest / "run.log").write_text(log)

        eval_json = dest / "eval.json"
        eval_txt = sh([PY, "garnet/eval_trace_gt.py", "--gt", str(gt_dir), "--traced", str(run_dir),
                       "--json-out", str(eval_json)] + (["--gt-sheet", gt_sheet] if gt_sheet else []))
        (dest / "eval.txt").write_text(eval_txt)

        # The trace writes its own endpoint verification (stage5b_trace_verification.*), so
        # just carry it into the bundle: one review list per sheet, no second pass.
        verify_path = run_dir / "stage5b_trace_verification.json"
        verification = json.loads(verify_path.read_text())["summary"] if verify_path.exists() else {}
        for name in ("stage5b_trace_verification.csv",
                     "stage5b_trace_verification_overlay.png"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, dest / name)

        metrics = json.loads(eval_json.read_text())
        rows.append({"sheet": sheet, **{k: metrics[k] for k in
                                        ("recall", "precision", "f1", "fully_traced", "gt_lines",
                                         "traces", "segments")},
                     "hold": verification.get("hold", ""),
                     "verified_item": verification.get("ok_item", ""),
                     "seeded": re.search(r"Total ports: (\d+) objects", log).group(1)
                     if re.search(r"Total ports: (\d+) objects", log) else "?"})

        with (dest / "per_line.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(metrics["per_line"][0]))
            w.writeheader()
            w.writerows(sorted(metrics["per_line"], key=lambda r: r["coverage"]))
        draw_overlay(image, run_dir, gt_dir, gt_sheet, dest / "gt_vs_traced.png")
        for name in ("stage5b_trace_overlay.png", "stage5b_branch_trace_overlay.png",
                     "stage5_pipe_mask_overlay.png"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, dest / name)
        print(f"{sheet}: R {metrics['recall']:.3f} P {metrics['precision']:.3f} "
              f"F1 {metrics['f1']:.3f} full {metrics['fully_traced']}/{metrics['gt_lines']} -> {dest}")

    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    lines = ["# Stage 5b pipe trace vs GT delivery export", "",
             "Seeds are built from the GT export's own equipment list, so the score measures the",
             "tracer, not the detector. GT px (600 dpi) is scaled x0.5 onto the 4962px raster.", "",
             "| sheet | seeded objs | recall | precision | F1 | lines >=0.9 | traces |"
             " endpoints on an item | endpoints on HOLD |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sheet']} | {r['seeded']} | {r['recall']:.3f} | {r['precision']:.3f} "
                     f"| {r['f1']:.3f} | {r['fully_traced']}/{r['gt_lines']} | {r['traces']} "
                     f"| {r['verified_item']} | {r['hold']} |")
    lines += ["", "Precision is a lower bound: the export draws paths for only part of its lines,",
              "so walks on the undrawn ones count as stray ink. Read `per_line.csv` for the",
              "worst lines and `gt_vs_traced.png` (green = GT, red = traced) for the picture."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
