#!/usr/bin/env python
"""Score Stage-5b/5c pipe traces against the DEXPI delivery export used as ground truth.

Ground truth:  ~/Downloads/0002A (or 0002_0003) — `connectivity_model.json` / `line_list.csv`
                lines carry `path_native_px` polylines in 600 dpi page pixels.
Traced output:  a `output/stage5b_<stem>` dir — `stage5b_trace_results.json`,
                `stage5b_branch_trace_results.json`, `stage5c_branch_merge.json` segments.

Both sides become ink masks; each side's points are scored by distance-transform to the other's
ink. Recall is per GT line (a line counts only if its whole drawn path is covered), precision is
per traced point. Coordinates are scaled GT -> trace space (default 0.5: 9925x7017 -> 4962x3508).

    python garnet/eval_trace_gt.py --gt ~/Downloads/0002A --traced output/stage5b_0002A_baseline
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

import cv2
import numpy as np


def gt_polylines(gt_dir: Path, sheet: str = "") -> list[dict]:
    """Lines with their drawn path, from the DEXPI export's connectivity model."""
    for name in ("connectivity_model.json", "model.json"):
        p = gt_dir / name
        if p.exists():
            raw = json.loads(p.read_text())
            model = raw.get("model", raw)
            out = []
            for ln in model.get("lines", []):
                if sheet and str(ln.get("sheet") or "") != sheet:
                    continue
                path = ln.get("path_native_px") or ""
                pts = [(int(a), int(b)) for a, b in re.findall(r"\((\d+),\s*(\d+)\)", path)]
                if len(pts) >= 2:
                    out.append({"line_id": ln.get("line_id"), "line_no": ln.get("line_no"),
                                "from": ln.get("from"), "to": ln.get("to"),
                                "status": ln.get("status"), "points": pts})
            if out:
                return out
    raise SystemExit(f"no connectivity model with line paths in {gt_dir}")


def traced_segments(traced_dir: Path) -> tuple[list[dict], int]:
    """Every traced segment from the port traces and the branch traces."""
    segs, n_traces = [], 0
    port = traced_dir / "stage5b_trace_results.json"
    if port.exists():
        for obj_id, tr in json.loads(port.read_text()).items():
            n_traces += 1
            for s in tr.get("segments") or []:
                segs.append({"x1": s["x1"], "y1": s["y1"], "x2": s["x2"], "y2": s["y2"],
                             "trace": obj_id})
    branch = traced_dir / "stage5b_branch_trace_results.json"
    if branch.exists():
        raw_branches = json.loads(branch.read_text()).get("branches", [])
        items = raw_branches.values() if isinstance(raw_branches, dict) else raw_branches
        for b in (v for v in items if isinstance(v, dict)):
            if b.get("status") != "traced":
                continue
            n_traces += 1
            for s in b.get("segments") or []:
                segs.append({"x1": s["x1"], "y1": s["y1"], "x2": s["x2"], "y2": s["y2"],
                             "trace": b.get("candidate", {}).get("id", "branch")})
    return segs, n_traces


def ink_mask(segments: list[dict], shape: tuple[int, int], thickness: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for s in segments:
        cv2.line(m, (int(s["x1"]), int(s["y1"])), (int(s["x2"]), int(s["y2"])), 255, thickness)
    return m


def polyline_mask(lines: list[dict], shape: tuple[int, int], scale: float, thickness: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for ln in lines:
        pts = np.array([[int(x * scale), int(y * scale)] for x, y in ln["points"]], np.int32)
        cv2.polylines(m, [pts], False, 255, thickness)
    return m


def distance_to_ink(mask: np.ndarray) -> np.ndarray:
    """Distance (px) from every pixel to the nearest ink pixel."""
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)


def resample(pts: list[tuple[int, int]], scale: float, step: float) -> list[tuple[int, int]]:
    """Points every `step` px along a polyline (so long lines and short stubs weigh per length)."""
    out = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale
        length = float(np.hypot(x2 - x1, y2 - y1))
        n = max(1, int(length / step))
        for i in range(n + 1):
            t = i / n
            out.append((int(x1 + (x2 - x1) * t), int(y1 + (y2 - y1) * t)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="delivery dir with connectivity_model.json")
    ap.add_argument("--traced", required=True, help="output/stage5b_<stem> dir")
    ap.add_argument("--scale", type=float, default=0.5, help="GT px -> traced px")
    ap.add_argument("--gt-sheet", default="", help="only lines of this sheet (multi-sheet GT)")
    ap.add_argument("--tol", type=float, default=5.0, help="match tolerance in traced px")
    ap.add_argument("--thickness", type=int, default=3)
    ap.add_argument("--shape", default="", help="HxW of the traced raster (default: infer from segments)")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--per-trace", action="store_true", help="rank traces by stray ink (precision side)")
    a = ap.parse_args()

    gt = gt_polylines(Path(a.gt), a.gt_sheet)
    segs, n_traces = traced_segments(Path(a.traced))
    if not segs:
        raise SystemExit(f"no traced segments in {a.traced}")
    if a.shape:
        h, w = (int(v) for v in a.shape.lower().split("x"))
    else:
        w = int(max(max(s["x1"], s["x2"]) for s in segs)) + 20
        h = int(max(max(s["y1"], s["y2"]) for s in segs)) + 20
        w, h = max(w, 4962), max(h, 3508)          # the sheet raster; segments can be short

    traced_mask = ink_mask(segs, (h, w), a.thickness)
    gt_mask = polyline_mask(gt, (h, w), a.scale, a.thickness)
    d_to_traced = distance_to_ink(traced_mask)
    d_to_gt = distance_to_ink(gt_mask)

    # recall: per GT line, the share of its drawn path that a trace got within tol
    per_line = []
    for ln in gt:
        pts = resample(ln["points"], a.scale, 3.0)
        if not pts:
            continue
        hit = 0
        for x, y in pts:
            if 0 <= y < h and 0 <= x < w and d_to_traced[y, x] <= a.tol:
                hit += 1
        cov = hit / len(pts)
        per_line.append({"line_id": ln["line_id"], "line_no": (ln["line_no"] or "")[:34],
                         "from": ln["from"], "to": ln["to"], "gt_status": ln["status"],
                         "path_px": len(pts) * 3, "coverage": round(cov, 3)})

    # precision: share of traced ink whose nearest GT path is within tol
    ys, xs = np.where(traced_mask > 0)
    near = d_to_gt[ys, xs] <= a.tol
    precision = float(near.mean()) if len(near) else 0.0
    recall = float(np.mean([p["coverage"] for p in per_line])) if per_line else 0.0
    full = sum(1 for p in per_line if p["coverage"] >= 0.9)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"GT lines {len(gt)} | traced traces {n_traces} / {len(segs)} segments | tol {a.tol}px "
          f"| scale {a.scale} | raster {w}x{h}")
    print(f"RECALL   {recall:.3f}  (mean per-line path coverage)")
    print(f"PRECISION {precision:.3f}  (traced ink within {a.tol}px of a GT path)")
    print(f"F1       {f1:.3f}")
    print(f"lines fully traced (coverage>=0.9): {full}/{len(per_line)}")
    print(f"\nworst {a.worst} GT lines by coverage:")
    for p in sorted(per_line, key=lambda p: p["coverage"])[:a.worst]:
        print(f"  {p['line_id'] or '?':6s} cov={p['coverage']:.2f} path={p['path_px']:5d}px "
              f"{(p['from'] or '?')[:18]:18s} -> {(p['to'] or '?')[:18]:18s} {p['line_no']}")
    if a.per_trace:
        # Which walks put ink where no GT line runs? Ranked by stray share x length, so a
        # long wrong walk outranks a short wrong stub.
        segs_by_trace: dict[str, list[dict]] = {}
        for s in segs:
            segs_by_trace.setdefault(s["trace"], []).append(s)
        rows = []
        for name, ss in segs_by_trace.items():
            m = ink_mask(ss, (h, w), a.thickness)
            ys_, xs_ = np.where(m > 0)
            if not len(ys_):
                continue
            off = float((d_to_gt[ys_, xs_] > a.tol).mean())
            length = sum(s_["length_px"] for s_ in ss if "length_px" in s_) or len(ys_)
            rows.append((off, length, name))
        print(f"\nstray ink per trace (worst {a.worst}):")
        for off, length, name in sorted(rows, reverse=True)[:a.worst]:
            print(f"  {name:16s} stray={off:.2f} len={length:5d}px")
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"recall": round(recall, 4), "precision": round(precision, 4),
                                                "f1": round(f1, 4), "fully_traced": full,
                                                "gt_lines": len(gt), "traces": n_traces,
                                                "segments": len(segs), "tol": a.tol, "scale": a.scale,
                                                "per_line": per_line}, indent=1))
        print("\nwrote", a.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
