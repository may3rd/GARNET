#!/usr/bin/env python
"""Stage 5b review gate — clean up the traced network, mostly in Python.

Stage 5b produces walks that are mostly real pipe but include dashed-line followers,
duplicates of an existing trace, and runs that stop in empty space. This gate applies the
three rules in `review_rules.md`.

**Most of it is deterministic.** The vision model was being asked to re-derive arithmetic that
pixels already answer:

    R1  solid line          -> ink/offset screen + "are the gaps inside a known symbol?"
    R2  one line per place  -> rasterise, intersect, count        (pure geometry)
    R3  valid terminal      -> set membership + bbox consistency  (pure lookup)

R2 and R3 are decided in Python, full stop. R1 is also decided in Python when it is clear-cut
(blank paper, or a clean centred run). Only the genuine residual reaches the model: a path
that is mostly-but-not-quite on ink with gaps no known symbol explains — the one case where a
local pixel test cannot tell an inline valve from a text label.

    python -m garnet.stage5b.review_gate --stem Test-00001 --dry-run   # inspect, no API call
    python -m garnet.stage5b.review_gate --stem Test-00001 --no-llm    # deterministic only
    python -m garnet.stage5b.review_gate --stem Test-00001             # + model on residual

The gate reports verdicts and emits the surviving network (`reviewed_walks.json`); it never
invents geometry.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("stage5b.review")

DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_BASE_URL = "https://ollama.com/v1"
ENV_PATH = Path.home() / ".hermes" / ".env"

# Terminals that are legitimate places for a pipe to end (rule R3).
# `pressure_relief_valve` is a real endpoint: a PSV terminates a run, it does not pass through.
VALID_TERMINALS = {"equipment", "page_connection", "tee_junction", "instrument_tag",
                   "pressure_relief_valve"}
# R3: a branch walk stopping on its parent line is a valid WALK but not a network endpoint.
BRANCH_TERMINALS = {"branch_connection"}
# R3: a walk that never reached anything.
DEAD_TERMINALS = {"dead_end", "no_pipe", "max_steps", "sheet_edge"}

# ---------------------------------------------------------------------------
# thresholds — see review_rules.md for reasoning and the calibration table
# ---------------------------------------------------------------------------
# Ink window: how far off the traced centreline ink may sit and still count. Calibrated on a
# known-good and a known-bad walk. The walker snaps ~1px off the drawn line and JPEG
# antialiasing leaves a fringe, so 0 is useless (both score ~0.04). At 4px the BAD walk scores
# a perfect 1.00 because it passes within 8px of UNRELATED ink — the window stops
# discriminating. 2px separated good from bad by 0.31.
INK_TOL = 2
# Perpendicular offset: ink within this many px of the centreline counts as "on the line".
ON_LINE_TOL = 1
# Beyond this, a point is "far" from the line (summary statistic only).
FAR_TOL = 3

COVERAGE_HOPELESS = 0.20   # mostly gap -> deterministic reject, no model needed
# A run of path over paper that no known symbol explains. Long ones are deterministically bad:
# there is nothing drawn to follow.
UNEXPLAINED_REJECT_PX = 100
# A gap counts as "explained" only if this fraction of it lies inside known object bboxes.
# Using any() let one known pixel anywhere in a 200px gap excuse the whole gap, which is how a
# blank-paper run scored zero unexplained pixels.
GAP_EXPLAINED_FRAC = 0.50
# Accept when every interruption is a known symbol AND there is ink under the path at all.
#
# NOTE: this deliberately does NOT gate on `on_line`. Requiring a high centred-fraction here
# was backwards — a walk following real pipe through inline valves scores a LOW on_line
# (median offset can be 0 while the fraction is 0.60) because ink inside a valve body sits off
# the centreline. Symbols both depress the score and explain the gaps, so on_line is a
# diagnostic, not a gate. Calibrated on 12 hand-labelled walks: `unexplained == 0 and
# coverage >= 0.60` accepted 5 of 6 good walks with 0 false accepts, cutting the residual from
# 24 to 7.
COVERAGE_OK = 0.60
# on_line at/above this is reported as "cleanly centred" in the reason text.
ON_LINE_CLEAN = 0.95

OVERLAP_TRIVIAL = 10       # px; a shared junction pixel is not duplication
NEAR_DUP_RATIO = 0.85      # this much of a walk already covered -> duplicate
PURE_DUP_RATIO = 0.95      # this much -> nothing new, reject rather than trim

PROMPT_PATH = Path(__file__).with_name("review_prompt.md")

# severity order for merging per-rule verdicts into one action
SEVERITY = {"accept": 0, "flag": 1, "trim": 2, "reject": 3, "review": 1, "unsure": 1}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_walks(out_dir: Path) -> dict[str, dict]:
    """Every walk from both artifacts, tagged with its kind and a stable id."""
    walks: dict[str, dict] = {}
    t = out_dir / "stage5b_trace_results.json"
    if t.is_file():
        for tid, v in json.loads(t.read_text()).items():
            walks[tid] = {**v, "_kind": "trace", "_id": tid}
    b = out_dir / "stage5b_branch_trace_results.json"
    if b.is_file():
        for bid, v in (json.loads(b.read_text()).get("branches") or {}).items():
            walks[bid] = {**v, "_kind": "branch", "_id": bid}
    return walks


def walk_pixels(walk: dict, step: int = 1) -> list[tuple[int, int]]:
    """Sample points along the walk's segments."""
    pts: list[tuple[int, int]] = []
    for s in walk.get("segments", []):
        n = max(abs(s["x2"] - s["x1"]), abs(s["y2"] - s["y1"]))
        for k in range(0, n + 1, step):
            f = k / n if n else 0
            pts.append((int(round(s["x1"] + (s["x2"] - s["x1"]) * f)),
                        int(round(s["y1"] + (s["y2"] - s["y1"]) * f))))
    return pts


def walk_pixels_dir(walk: dict, step: int = 1) -> list[tuple[int, int, str]]:
    """Same, carrying each point's travel direction (needed for the perpendicular)."""
    pts: list[tuple[int, int, str]] = []
    for s in walk.get("segments", []):
        n = max(abs(s["x2"] - s["x1"]), abs(s["y2"] - s["y1"]))
        for k in range(0, n + 1, step):
            f = k / n if n else 0
            pts.append((int(round(s["x1"] + (s["x2"] - s["x1"]) * f)),
                        int(round(s["y1"] + (s["y2"] - s["y1"]) * f)),
                        str(s.get("direction", "UP"))))
    return pts


def known_object_mask(shape: tuple[int, int], objects: list[dict],
                      equipment: list[dict]) -> np.ndarray:
    """Rasterise every known object/symbol footprint.

    Used to ask whether an interruption in the ink is *explained*: a gap inside a valve,
    instrument or equipment box is a symbol sitting on the pipe (legitimate). A gap in open
    paper is a failed walk.
    """
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    boxes = [o.get("bbox") for o in objects] + [e.get("bbox") for e in equipment]
    for b in boxes:
        if not isinstance(b, dict) or not {"x_min", "y_min", "x_max", "y_max"}.issubset(b):
            continue
        x0 = max(0, int(b["x_min"]) - 3)
        y0 = max(0, int(b["y_min"]) - 3)
        x1 = min(w, int(b["x_max"]) + 4)
        y1 = min(h, int(b["y_max"]) + 4)
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1
    return m


# ---------------------------------------------------------------------------
# R1 — solid line: ink coverage, perpendicular offset, explained gaps
# ---------------------------------------------------------------------------


def r1_profile(walk: dict, ink: np.ndarray, known: np.ndarray) -> dict:
    """Everything R1 needs, from pixels only.

    Three views of the same question:
      * coverage — is there ink under the path at all (window INK_TOL)
      * on_line  — is the ink centred on the path, or is the path beside it
      * gaps     — where ink is missing, is that explained by a known symbol
    """
    pts = walk_pixels_dir(walk, step=1)
    h, w = ink.shape
    if not pts:
        return {"n_px": 0, "coverage": 0.0, "on_line": 0.0, "far": 0.0, "median_offset": 0.0,
                "max_gap": 0, "n_gaps": 0, "n_long_gaps": 0, "n_unexplained": 0,
                "unexplained_px": 0, "worst_unexplained": 0}

    covered, offsets, expl = [], [], []
    lim = FAR_TOL + 3
    for x, y, d in pts:
        y0, y1 = max(0, y - INK_TOL), min(h, y + INK_TOL + 1)
        x0, x1 = max(0, x - INK_TOL), min(w, x + INK_TOL + 1)
        covered.append(1 if ink[y0:y1, x0:x1].any() else 0)

        # perpendicular offset: nearest ink stepping across the line of travel
        perp = (0, 1) if d in ("LEFT", "RIGHT") else (1, 0)
        best = None
        for off in range(0, lim + 1):
            hit = False
            for sgn in ((0,) if off == 0 else (-1, 1)):
                px, py = x + perp[0] * off * sgn, y + perp[1] * off * sgn
                if 0 <= px < w and 0 <= py < h and ink[py, px]:
                    hit = True
                    break
            if hit:
                best = off
                break
        offsets.append(best if best is not None else lim)

        expl.append(1 if (0 <= x < w and 0 <= y < h and known[y, x]) else 0)

    # gap runs, and whether each is explained by known symbols
    gaps: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(covered):
        if v == 0 and start is None:
            start = i
        elif v == 1 and start is not None:
            gaps.append((start, i))
            start = None
    if start is not None:
        gaps.append((start, len(covered)))

    unexpl: list[tuple[int, int]] = []
    for a, b in gaps:
        span = b - a
        if span <= 0:
            continue
        if sum(expl[a:b]) / span < GAP_EXPLAINED_FRAC:
            unexpl.append((a, b))

    arr = np.array(offsets)
    return {
        "n_px": len(pts),
        "coverage": round(sum(covered) / len(covered), 3),
        "on_line": round(float((arr <= ON_LINE_TOL).mean()), 3),
        "far": round(float((arr > FAR_TOL).mean()), 3),
        "median_offset": float(np.median(arr)),
        "max_gap": max((b - a for a, b in gaps), default=0),
        "n_gaps": len(gaps),
        "n_long_gaps": sum(1 for a, b in gaps if b - a >= 8),
        "n_unexplained": len(unexpl),
        "unexplained_px": sum(b - a for a, b in unexpl),
        "worst_unexplained": max((b - a for a, b in unexpl), default=0),
    }


def r1_verdict(prof: dict) -> dict:
    """Deterministic R1 decision, or `review` for the residual only.

    Three tiers, in order:
      reject — clear-cut: almost no ink, or a long run over paper no symbol explains.
      accept — clean: centred on the ink with every interruption explained by a symbol.
      review — the residual. Mostly on ink but with unexplained gaps small enough that a
               valve body and a text label look the same to a local pixel test.
    """
    if prof["n_px"] < 20:
        return {"verdict": "accept", "reason": "too short to judge"}

    if prof["coverage"] < COVERAGE_HOPELESS:
        return {"verdict": "reject",
                "reason": f"almost no ink under the path (coverage {prof['coverage']:.2f})"}
    if prof["unexplained_px"] >= UNEXPLAINED_REJECT_PX:
        return {"verdict": "reject",
                "reason": f"{prof['unexplained_px']}px over paper no symbol explains "
                          f"(worst run {prof['worst_unexplained']}px)"}

    # Accept only when EVERY interruption is accounted for by a known symbol. Any unexplained
    # gap at all is the residual — and note that gap SIZE does not separate the cases:
    # branch_000013 is a real pipe with a 36px unexplained gap, while branch_000023 is a bad
    # walk whose gaps are only 4px. What differs is whether the gap is a symbol or a text
    # label, which is exactly what pixels cannot tell apart and the model is asked about.
    if prof["unexplained_px"] == 0 and prof["coverage"] >= COVERAGE_OK:
        centred = ("cleanly centred" if prof["on_line"] >= ON_LINE_CLEAN
                   else f"on_line {prof['on_line']:.2f}")
        return {"verdict": "accept",
                "reason": f"ink under the path ({centred}); all {prof['n_gaps']} gap(s) "
                          f"explained by symbols"}

    return {"verdict": "review",
            "reason": f"coverage {prof['coverage']:.2f}, on_line {prof['on_line']:.2f}, "
                      f"{prof['n_unexplained']} unexplained gap(s) totalling "
                      f"{prof['unexplained_px']}px (worst {prof['worst_unexplained']}px)"}


# ---------------------------------------------------------------------------
# R2 — one walk per line (pure geometry)
# ---------------------------------------------------------------------------


def overlap_analysis(walks: dict[str, dict], shape: tuple[int, int]) -> dict[str, dict]:
    """Rasterise every walk; measure how much an earlier walk already covers, and where."""
    h, w = shape
    canvas = np.zeros((h, w), np.uint8)
    order = [k for k, v in walks.items() if v.get("segments")]
    info: dict[str, dict] = {}
    for i, wid in enumerate(order, start=1):
        m = np.zeros((h, w), np.uint8)
        for s in walks[wid].get("segments", []):
            cv2.line(m, (s["x1"], s["y1"]), (s["x2"], s["y2"]), 1, 3)
        total = int(m.sum())
        shared = int((m.astype(bool) & (canvas > 0)).sum())

        best, best_id = 0, None
        if shared:
            lab = canvas[m.astype(bool)]
            vals, counts = np.unique(lab, return_counts=True)
            j = int(np.argmax(counts))
            best = int(counts[j])
            best_id = order[int(vals[j]) - 1] if vals[j] >= 1 else None

        # Where does this walk first run on top of an earlier one? Needed for `trim`: the
        # portion before the first shared point is new linework worth keeping.
        first_shared = None
        pts = walk_pixels(walks[wid], step=1)
        for idx, (x, y) in enumerate(pts):
            if 0 <= x < w and 0 <= y < h and canvas[y, x]:
                first_shared = idx
                break

        info[wid] = {
            "mask_px": total,
            "shared_px": shared,
            "shared_ratio": round(shared / total, 3) if total else 0.0,
            "most_overlapping": best_id,
            "most_overlapping_px": best,
            "first_shared_idx": first_shared,
            "n_pts": len(pts),
        }
        canvas[m.astype(bool)] = i
    return info


def r2_verdict(ov: dict, walk: dict) -> dict:
    """Pure geometry: is this walk a duplicate of one already taken?

    Keep the earlier/longer path. A fully-covered walk has nothing new and is rejected; a
    partly-covered one is trimmed from the first shared point, keeping the new prefix.
    """
    if ov["mask_px"] <= 0:
        return {"verdict": "accept", "reason": "no linework drawn"}
    if ov["shared_ratio"] >= PURE_DUP_RATIO:
        return {"verdict": "reject",
                "reason": f"{ov['shared_ratio'] * 100:.0f}% already covered by "
                          f"{ov['most_overlapping']}"}
    if ov["shared_ratio"] >= NEAR_DUP_RATIO:
        idx = ov.get("first_shared_idx")
        trim = None
        if idx is not None:
            pts = walk_pixels(walk, step=1)
            if idx < len(pts):
                trim = [int(pts[idx][0]), int(pts[idx][1])]
        return {"verdict": "trim", "trim_from": trim,
                "reason": f"{ov['shared_ratio'] * 100:.0f}% covered by "
                          f"{ov['most_overlapping']}; keep the new prefix"}
    if ov["shared_px"] > OVERLAP_TRIVIAL:
        return {"verdict": "accept",
                "reason": f"only {ov['shared_px']}px shared (junction overlap)"}
    return {"verdict": "accept", "reason": "no overlap"}


# ---------------------------------------------------------------------------
# R3 — valid terminal (pure lookup + bbox consistency)
# ---------------------------------------------------------------------------


def r3_verdict(walk: dict, bboxes: dict[str, dict]) -> dict:
    """Pure lookup: does the walk end on a real connection?

    Also checks consistency — a walk claiming to end on an object, whose terminal point is
    nowhere near that object, has a wrong label and must not be trusted either way. On
    Test-00001, 5 of 36 bbox-naming terminals were inconsistent this way.
    """
    tt = walk.get("terminal_type")
    if tt is None:
        return {"verdict": "reject", "reason": "no terminal recorded (walk skipped)"}
    tt = str(tt)

    if tt in DEAD_TERMINALS:
        return {"verdict": "reject", "reason": f"ended as {tt} - reached no connection"}
    if tt in BRANCH_TERMINALS:
        # Legitimate as a walk, but it is not a network endpoint.
        return {"verdict": "flag", "endpoint": False,
                "reason": "stops on its parent line; valid walk, not an endpoint"}
    if tt not in VALID_TERMINALS:
        return {"verdict": "reject", "reason": f"unknown terminal {tt!r}"}

    tid = walk.get("terminal_obj_id")
    tx, ty = walk.get("terminal_x"), walk.get("terminal_y")
    if tid and tx is not None and ty is not None:
        b = bboxes.get(str(tid))
        if isinstance(b, dict):
            m = 6
            if not (b["x_min"] - m <= tx <= b["x_max"] + m
                    and b["y_min"] - m <= ty <= b["y_max"] + m):
                return {"verdict": "flag",
                        "reason": f"label says {tid} but terminal ({tx},{ty}) is "
                                  f"{_gap_px(tx, ty, b)}px outside its bbox"}
    return {"verdict": "accept", "reason": f"ended on {tt}"}


def _gap_px(x: int, y: int, b: dict) -> int:
    dx = max(b["x_min"] - x, 0, x - b["x_max"])
    dy = max(b["y_min"] - y, 0, y - b["y_max"])
    return int(round((dx * dx + dy * dy) ** 0.5))


# ---------------------------------------------------------------------------
# crop rendering for the model (residual only)
# ---------------------------------------------------------------------------


def crop_for(walk: dict, image: np.ndarray, pad: int = 90) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop around a walk with context padding; returns (crop, (x0,y0))."""
    pts = walk_pixels(walk, step=2)
    if not pts:
        return np.zeros((10, 10, 3), np.uint8), (0, 0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    h, w = image.shape[:2]
    x0 = max(0, min(xs) - pad)
    y0 = max(0, min(ys) - pad)
    x1 = min(w, max(xs) + pad)
    y1 = min(h, max(ys) + pad)
    return image[y0:y1, x0:x1].copy(), (x0, y0)


def draw_crop(walk: dict, crop: np.ndarray, origin: tuple[int, int],
              equipment: list[dict] | None = None,
              other_walks: list[dict] | None = None) -> np.ndarray:
    """Draw the walk so the model can STILL SEE the ink underneath it.

    A thick opaque overlay defeats the purpose of the review: rule R1 asks whether the path
    follows a solid line, and a solid green stroke covers the ~2px line being judged, so the
    model would be grading its own overlay. Instead: a wide semi-transparent halo (shows where
    the walk went, tinting but not hiding the ink) plus a 1px opaque centreline (pinpoints the
    exact path). The raw line stays legible through the halo, which is what makes an R1
    verdict mean anything.
    """
    ox, oy = origin
    halo = crop.copy()
    for other in (other_walks or []):
        for s in other.get("segments", []):
            cv2.line(halo, (s["x1"] - ox, s["y1"] - oy), (s["x2"] - ox, s["y2"] - oy),
                     (0, 235, 235), 2)
    for it in (equipment or []):
        b = it["bbox"]
        cv2.rectangle(halo, (b["x_min"] - ox, b["y_min"] - oy),
                      (b["x_max"] - ox, b["y_max"] - oy), (0, 140, 255), 2)
    for s in walk.get("segments", []):
        cv2.line(halo, (s["x1"] - ox, s["y1"] - oy), (s["x2"] - ox, s["y2"] - oy),
                 (0, 255, 0), 7)
    out = cv2.addWeighted(halo, 0.45, crop, 0.55, 0)
    for s in walk.get("segments", []):
        cv2.line(out, (s["x1"] - ox, s["y1"] - oy), (s["x2"] - ox, s["y2"] - oy),
                 (0, 200, 0), 1)
    pts = walk_pixels(walk, step=2)
    if pts:
        for (px, py), col in ((pts[0], (255, 0, 0)), (pts[-1], (0, 0, 255))):
            cv2.drawMarker(out, (px - ox, py - oy), col, cv2.MARKER_CROSS, 34, 4)
    return out


def segments_text(walk: dict, limit: int = 12) -> str:
    segs = walk.get("segments", [])
    parts = [f"({s['x1']},{s['y1']})->({s['x2']},{s['y2']})"
             f"[{s.get('direction', '?')},{s.get('length_px', 0)}px]" for s in segs[:limit]]
    if len(segs) > limit:
        parts.append(f"...(+{len(segs) - limit} more)")
    return "; ".join(parts) if parts else "(none)"


# ---------------------------------------------------------------------------
# vision call (residual only)
# ---------------------------------------------------------------------------


def _api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.getenv("OLLAMA_API_KEY"):
        return os.environ["OLLAMA_API_KEY"]
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("OLLAMA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def call_vision(b64: str, prompt: str, *, model: str, base_url: str, api_key: str,
                max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}],
        "max_tokens": max_tokens,
        # Without this the model burns the budget on hidden reasoning and returns empty content.
        "reasoning": {"effort": "none"},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_verdict(text: str) -> dict:
    """Parse the model's JSON, and never trust its vocabulary or geometry beyond trim_from."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    def bail(reason: str) -> dict:
        return {"verdict": "unsure", "reason": reason, "rules": [],
                "confidence": 0.0, "trim_from": None}

    try:
        v = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            return bail("unparseable model response")
        try:
            v = json.loads(m.group(0))
        except json.JSONDecodeError:
            return bail("unparseable model response")
    if not isinstance(v, dict):
        return bail("model returned non-object")

    verdict = str(v.get("verdict") or "unsure").strip().lower()
    if verdict not in {"accept", "reject", "trim", "unsure"}:
        verdict = "unsure"
    rules = [str(r).upper() for r in (v.get("rules") or [])
             if str(r).upper() in {"R1", "R2", "R3"}]
    tf = v.get("trim_from")
    if not (isinstance(tf, (list, tuple)) and len(tf) == 2
            and all(isinstance(c, (int, float)) for c in tf)):
        tf = None
    try:
        conf = float(v.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    return {"verdict": verdict, "rules": rules,
            "confidence": max(0.0, min(1.0, conf)),
            "trim_from": [int(tf[0]), int(tf[1])] if tf else None,
            "reason": str(v.get("reason") or "")[:300]}


def build_prompt(walk: dict, prof: dict, ov: dict, crop_shape: tuple[int, int],
                 origin: tuple[int, int], suspect_reason: str) -> str:
    """Fill the versioned prompt template for one residual candidate."""
    template = PROMPT_PATH.read_text(encoding="utf-8")
    head, _, body = template.partition("\n---\n")   # drop the front-matter header
    if not body:
        body = head
    pts = walk_pixels(walk, step=2)
    sx, sy = pts[0] if pts else (0, 0)
    ex, ey = pts[-1] if pts else (0, 0)
    return body.format(
        path_id=walk["_id"],
        path_kind=walk["_kind"],
        start_x=sx, start_y=sy, end_x=ex, end_y=ey,
        terminal_type=walk.get("terminal_type"),
        segment_count=len(walk.get("segments", [])),
        trace_length=walk.get("trace_length_px") or prof["n_px"],
        segments=segments_text(walk),
        crop_w=crop_shape[1], crop_h=crop_shape[0],
        crop_x0=origin[0], crop_y0=origin[1],
        coverage=prof["coverage"], max_gap=prof["max_gap"],
        n_long_gaps=prof["n_long_gaps"], shared_px=ov["shared_px"],
        on_line=prof["on_line"], unexplained_px=prof["unexplained_px"],
        suspect_reason=suspect_reason,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stem", default="Test-00001")
    p.add_argument("--indir", default="garnet/tests/input")
    p.add_argument("--dir", default=None,
                   help="stage5b output dir (default output/stage5b_<stem>)")
    p.add_argument("--out", default=None, help="review output dir (default <dir>/review)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--dry-run", action="store_true",
                   help="Write crops + prompts for the residual, but do NOT call the model.")
    p.add_argument("--no-llm", action="store_true",
                   help="Deterministic verdicts only; never call the model.")
    p.add_argument("--keep-all", action="store_true",
                   help="Retain every walk regardless of verdict. Verdicts are still computed "
                        "and recorded as `verdict_action`/`overridden`, but nothing is dropped.")
    p.add_argument("--limit", type=int, default=0, help="Cap residual calls (0 = all).")
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
    out_dir = Path(args.out) if args.out else sdir / "review"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(indir / f"{args.stem}.jpg"))
    if image is None:
        print(f"error: cannot read {indir / f'{args.stem}.jpg'}", file=sys.stderr)
        return 1
    ink = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 128).astype(np.uint8)

    walks = load_walks(sdir)
    if not walks:
        print(f"error: no stage5b artifacts in {sdir}", file=sys.stderr)
        return 1
    equipment = []
    ep = sdir / "stage3_equipment_bboxes.json"
    if ep.is_file():
        equipment = json.loads(ep.read_text()).get("equipment", [])
    objects = json.loads((sdir / "stage4_objects.json").read_text()).get("objects", [])

    # id -> bbox for the R3 consistency check
    bboxes: dict[str, dict] = {}
    for o in objects:
        if isinstance(o.get("bbox"), dict):
            bboxes[str(o.get("id"))] = o["bbox"]
    for e in equipment:
        if isinstance(e.get("bbox"), dict):
            bboxes[str(e.get("id"))] = e["bbox"]

    n_trace = sum(1 for v in walks.values() if v["_kind"] == "trace")
    print(f"{args.stem}: {len(walks)} walks ({n_trace} traces, {len(walks) - n_trace} branches)")

    # ---- measure ----
    known = known_object_mask(ink.shape, objects, equipment)
    print(f"  known-symbol footprint: {known.sum() / known.size * 100:.1f}% of the sheet")
    profs = {k: r1_profile(v, ink, known) for k, v in walks.items()}
    ovs = overlap_analysis(walks, ink.shape)

    # ---- deterministic verdicts ----
    results: dict[str, dict] = {}
    for wid, walk in walks.items():
        if walk.get("segments"):
            r1 = r1_verdict(profs[wid])
            r2 = r2_verdict(ovs[wid], walk)
        else:
            r1 = {"verdict": "reject", "reason": "no segments traced"}
            r2 = {"verdict": "accept", "reason": "no linework drawn"}
        r3 = r3_verdict(walk, bboxes)
        results[wid] = {"path_id": wid, "kind": walk["_kind"], "profile": profs[wid],
                        "R1": r1, "R2": r2, "R3": r3, "llm": None,
                        "needs_llm": r1["verdict"] == "review"}

    residual = [w for w, r in results.items() if r["needs_llm"]]
    residual.sort(key=lambda w: profs[w]["on_line"])
    if args.limit:
        residual = residual[:args.limit]
    if args.no_llm:
        residual = []

    det = Counter()
    for r in results.values():
        for rule in ("R1", "R2", "R3"):
            det[f"{rule}:{r[rule]['verdict']}"] += 1
    print("  deterministic verdicts:")
    for k in sorted(det):
        print(f"    {k:14} {det[k]:>3}")
    print(f"  residual -> model: {len(residual)}")
    for wid in residual:
        print(f"    {wid:26} {results[wid]['R1']['reason']}")

    # ---- crops + model, residual only ----
    key = "" if (args.dry_run or args.no_llm) else _api_key(args.api_key)
    if not args.dry_run and not args.no_llm and not key:
        print("error: no OLLAMA_API_KEY found (set --api-key or ~/.hermes/.env)", file=sys.stderr)
        return 1

    for i, wid in enumerate(residual, start=1):
        walk = walks[wid]
        crop, origin = crop_for(walk, image)
        others = [w for k, w in walks.items()
                  if k != wid and (ovs.get(k) or {}).get("most_overlapping") == wid]
        drawn = draw_crop(walk, crop, origin, equipment, others[:12])
        prompt = build_prompt(walk, profs[wid], ovs[wid], crop.shape[:2], origin,
                              results[wid]["R1"]["reason"])

        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", wid)
        name = f"crop_{i:03d}_{stem}.png"
        cv2.imwrite(str(out_dir / name), drawn)
        (out_dir / f"prompt_{i:03d}_{stem}.txt").write_text(prompt, encoding="utf-8")
        results[wid]["crop"] = name
        results[wid]["prompt"] = f"prompt_{i:03d}_{stem}.txt"

        if args.dry_run or args.no_llm:
            print(f"  [{i}/{len(residual)}] {wid:26} -> pending (no model call)")
            continue
        _, buf = cv2.imencode(".png", drawn)
        b64 = base64.b64encode(buf.tobytes()).decode()
        try:
            resp = call_vision(b64, prompt, model=args.model, base_url=args.base_url,
                               api_key=key, max_tokens=args.max_tokens,
                               timeout=args.timeout)
            content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            v = parse_verdict(content)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            v = {"verdict": "unsure", "reason": f"api error: {e}", "rules": [],
                 "confidence": 0.0, "trim_from": None}
        results[wid]["llm"] = v
        results[wid]["needs_llm"] = False
        print(f"  [{i}/{len(residual)}] {wid:26} -> {v['verdict']:7} "
              f"conf={v['confidence']:.2f} {v['reason'][:60]}")

    # ---- merge into one action per walk ----
    for wid, rec in results.items():
        acts = {"R1": dict(rec["R1"]), "R2": dict(rec["R2"]), "R3": dict(rec["R3"])}
        if acts["R1"]["verdict"] == "review":
            llm = rec.get("llm")
            acts["R1"] = {"verdict": llm["verdict"] if llm else "review",
                          "reason": (llm or {}).get("reason", "pending model review"),
                          "trim_from": (llm or {}).get("trim_from")}
        worst = max(acts.items(), key=lambda kv: SEVERITY.get(kv[1]["verdict"], 0))
        final = worst[1]["verdict"]
        # a `flag` must survive a lower-severity verdict, but not override reject/trim
        if final == "accept" and any(v["verdict"] == "flag" for v in acts.values()):
            final = "flag"
        rec["verdict_action"] = final          # what the rules concluded
        rec["final_action"] = "accept" if args.keep_all else final
        rec["overridden"] = bool(args.keep_all and final != "accept")
        rec["decided_by"] = "python" if (rec.get("llm") is None and not rec["needs_llm"]) \
            else "python+llm"
        rec["reasons"] = {k: v.get("reason", "") for k, v in acts.items()}
        rec["trim_from"] = next((v.get("trim_from") for v in acts.values()
                                 if v["verdict"] == "trim" and v.get("trim_from")), None)

    survivors = {w: r for w, r in results.items() if r["final_action"] != "reject"}
    (out_dir / "reviewed_walks.json").write_text(json.dumps({
        "stem": args.stem,
        "keep_all": bool(args.keep_all),
        "policy": ("keep_all: every walk is retained; `verdict_action` records what the rules "
                   "concluded (see `overridden`)"
                   if args.keep_all else
                   "reject drops the walk; trim keeps segments before trim_from; flag keeps the "
                   "walk but marks it not-an-endpoint or label-inconsistent"),
        "kept": {w: {"kind": r["kind"], "final_action": r["final_action"],
                     "verdict_action": r["verdict_action"],
                     "overridden": r["overridden"],
                     "trim_from": r["trim_from"], "reasons": r["reasons"]}
                 for w, r in survivors.items()},
        "dropped": {w: {"kind": r["kind"], "reasons": r["reasons"]}
                    for w, r in results.items() if r["final_action"] == "reject"},
    }, indent=2), encoding="utf-8")

    llm_calls = sum(1 for r in results.values() if r.get("llm"))
    payload = {
        "stem": args.stem, "model": args.model,
        "dry_run": bool(args.dry_run), "no_llm": bool(args.no_llm),
        "walks_total": len(walks), "llm_calls": llm_calls,
        "deterministic_verdicts": dict(det),
        "rules": {"R1": "solid line", "R2": "one trace per line", "R3": "valid terminal"},
        "thresholds": {"INK_TOL": INK_TOL, "COVERAGE_HOPELESS": COVERAGE_HOPELESS,
                       "UNEXPLAINED_REJECT_PX": UNEXPLAINED_REJECT_PX,
                       "GAP_EXPLAINED_FRAC": GAP_EXPLAINED_FRAC,
                       "ON_LINE_CLEAN": ON_LINE_CLEAN,
                       "NEAR_DUP_RATIO": NEAR_DUP_RATIO, "PURE_DUP_RATIO": PURE_DUP_RATIO},
        "results": results,
    }
    (out_dir / "review_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nfinal actions: {dict(Counter(r['final_action'] for r in results.values()))}")
    print(f"model calls: {llm_calls} of {len(walks)} walks")
    print(f"{'DRY RUN' if args.dry_run else 'REVIEW'}"
          f"{' (no-llm)' if args.no_llm else ''} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
