"""Stage 5b host: runs the copied pipe tracer with no dependency on backend/.

`Stage5bPipelineMixin` (stage5b_pipeline.py) expects a host object supplying 10 methods and a
`cfg` with the trace_* fields. That host was `PIDPipeline` inside backend/garnet/pid_extractor.py.
This module replaces it with a self-contained `Stage5bHost`, so the whole tracer runs from
garnet/ alone.

Inputs are the fixture JSONs already in garnet/tests/input (or any directory with the same
shapes). Line-number boxes are MERGED into the object list — see `merge_line_numbers` — so a
separate line-number artifact is not needed.

Outputs, written into --out:
    stage5_pipe_mask.png                the mask that was walked
    stage5_connection_ports.json        trace start points
    stage5b_trace_results.json          port traces
    stage5b_branch_candidates.json      tee-branch candidates
    stage5b_branch_trace_results.json   branch traces
    stage5b_trace_overlay.png           port-trace overlay      <- deliverable
    stage5b_branch_trace_overlay.png    branch-trace overlay    <- deliverable

Usage:
    python -m garnet.stage5b.run --stem Test-00001
    python -m garnet.stage5b.run --stem Test-00001 --indir garnet/tests/input --out output/s5b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .stage5b_pipeline import EQUIPMENT_LABELS

REPO_ROOT = Path(__file__).resolve().parents[2]

# `line number` is a class the YOLO weight emits; the tracer treats those as text, but they are
# also what we attach to traces, so they stay in the object list as a first-class class.
LINE_NUMBER_CLASSES = {"line number", "line_number"}

log = logging.getLogger("stage5b")


# ---------------------------------------------------------------------------
# Config: the trace_* fields Stage5bPipelineMixin reads, at the project's
# accepted baseline values (mirrors PipelineConfig defaults).
# ---------------------------------------------------------------------------


@dataclass
class Stage5bConfig:
    debug_artifacts: bool = False

    trace_max_steps: int = 5000
    trace_min_step: int = 5
    trace_straight_min_step: int = 10
    trace_turn_min_step: int = 3
    trace_lookahead_px: int = 30
    trace_raycast_start_px: int = 20
    trace_raycast_max_px: int = 50
    trace_raycast_step_px: int = 2
    trace_raycast_max_snap_shift_px: int = 4
    trace_centerline_radius_px: int = 8
    trace_side_path_inline_probe_px: int = 60
    trace_anchor_turn_max_gap_px: int = 60
    trace_turn_terminal_scan_px: int = 70
    trace_turn_terminal_scan_far_px: int = 160
    trace_axis_rewind_px: int = 12
    trace_sheet_edge_margin_px: int = 10
    trace_warmup_steps: int = 20
    trace_turn_gap_max_px: int = 60
    trace_branch_side_turn_probe_px: int = 8
    trace_branch_side_turn_terminal_px: int = 90
    trace_turn_probe_px: int = 8
    trace_tee_search_px: int = 8
    trace_terminal_current_margin_px: int = 2
    trace_terminal_current_tag_margin_px: int = 4
    trace_terminal_current_dcs_margin_px: int = 6
    trace_terminal_current_equipment_margin_px: int = 4
    trace_terminal_ahead_margin_px: int = 2
    trace_terminal_ahead_equipment_margin_px: int = 4
    trace_inline_hit_margin_px: int = 2
    trace_inline_exit_margin_px: int = 6

    trace_branch_max_iterations: int = 5
    trace_branch_min_run_px: int = 25
    trace_branch_cluster_radius_px: int = 8
    trace_branch_turn_tolerance_px: int = 8
    trace_branch_point_tolerance_px: int = 10
    trace_branch_candidate_sample_step_px: int = 5
    trace_branch_seed_from_tees: bool = True


# ---------------------------------------------------------------------------
# fixture loading
# ---------------------------------------------------------------------------


def merge_line_numbers(objects: list[dict], line_numbers: list[dict]) -> list[dict]:
    """Return `objects` with the line-number boxes folded in as `class_name: "line number"`.

    The object list already carries its own `line number` detections, so this merges rather
    than replaces:
      * an object line-number box that OVERLAPS a fixture box keeps the fixture's `Text`
        (the fixture is transcribed text; the detector only localises);
      * a fixture box with no overlapping object box is ADDED, so no transcribed label is lost;
      * object boxes already carrying `text` are left alone.
    Matching is by centre containment — IoU between the two sources runs 0.26-0.74 on this
    sheet because the detector's box is consistently a few px taller than the tight text box.
    """
    merged = [dict(o) for o in objects]
    ln_objs = [o for o in merged if str(o.get("class_name", "")).strip().lower() in LINE_NUMBER_CLASSES]
    unmatched = []

    for ln in line_numbers:
        text = str(ln.get("Text") or "").strip()
        if not text:
            continue
        try:
            x, y = int(ln["Left"]), int(ln["Top"])
            w, h = int(ln["Width"]), int(ln["Height"])
        except (KeyError, TypeError, ValueError):
            continue
        cx, cy = x + w / 2, y + h / 2
        target = None
        for o in ln_objs:
            b = o.get("bbox") or {}
            if not {"x_min", "y_min", "x_max", "y_max"}.issubset(b):
                continue
            if (b["x_min"] <= cx <= b["x_max"]) and (b["y_min"] <= cy <= b["y_max"]):
                target = o
                break
        if target is not None:
            target["text"] = text
            target["text_source"] = "line_number_boxes"
        else:
            unmatched.append({
                "id": f"line_number_{len(unmatched) + 1:04d}",
                "class_name": "line number",
                "confidence": float(ln.get("Score", 1.0)),
                "bbox": {"x_min": x, "y_min": y, "x_max": x + w, "y_max": y + h},
                "text": text,
                "text_source": "line_number_boxes",
            })

    merged.extend(unmatched)
    log.info("line numbers: %d fixture labels, %d merged into objects, %d added",
             len(line_numbers), len(line_numbers) - len(unmatched), len(unmatched))
    return merged


def load_fixtures(indir: Path, stem: str) -> tuple[list[dict], list[dict]]:
    """Return (objects_with_line_numbers_merged, equipment_as_objects)."""
    def read(name: str, *alts: str):
        for n in (name, *alts):
            p = indir / n.format(stem=stem)
            if p.is_file():
                return json.loads(p.read_text())
        return None

    obj_payload = read("{stem}_objects.json")
    if obj_payload is None:
        raise FileNotFoundError(f"no {stem}_objects.json in {indir}")
    objects = obj_payload.get("objects", [])

    ln_payload = read("{stem}_line_number_boxes.json", "{stem}_line_numbers.json")
    objects = merge_line_numbers(objects, (ln_payload or {}).get("objects", []))

    eq_payload = read("{stem}_equipment_bboxes.json", "{stem}_equipment.json")
    equipment = _equipment_as_objects((eq_payload or {}).get("objects", []))
    return objects, equipment


def _equipment_as_objects(items: list[dict]) -> list[dict]:
    """Contract B equipment -> the object shape Stage 5b's loader expects.

    Two traps, both silent:
      * the loader keeps only `class_name` values in EQUIPMENT_LABELS, so a Contract B
        `Equipment_type` of "static mixer" must map to "mixer";
      * nozzle `Ports` are read from a SEPARATE `ai_equipment_ports.json`, not from the bbox.
    """
    # synonym table, mirrored from backend/garnet/ai_import.py AI_CLASS_MAP
    synonyms = {
        "static mixer": "mixer", "inline mixer": "mixer",
        "ko drum": "knockout drum", "knockout drum": "knockout drum",
        "shell and tube exchanger": "heat exchanger", "shell & tube exchanger": "heat exchanger",
        "exchanger": "heat exchanger", "air cooler": "cooler",
        "drum": "vessel", "separator": "vessel", "accumulator": "vessel",
    }

    out = []
    for i, item in enumerate(items):
        bb = item.get("Bounding_box_px") or {}
        if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bb):
            continue
        raw = str(item.get("Equipment_type") or "").strip().lower().replace("_", " ")
        label = raw if raw in EQUIPMENT_LABELS else synonyms.get(raw, "")
        if not label:
            log.warning("equipment %r has no EQUIPMENT_LABELS equivalent; skipped",
                        item.get("Equipment_type"))
            continue
        tag = str(item.get("Tag") or "").strip()
        out.append({
            "id": f"equip_{tag.lower().replace('-', '_') or f'{i:03d}'}",
            "class_name": label,
            "bbox": {k: int(round(float(bb[k]))) for k in ("x_min", "y_min", "x_max", "y_max")},
            "tag": tag,
            "review_state": "accepted",
            "source": "fixture",
        })
    return out


# Contract B spells sides in lowercase words; the tracer wants compass directions.
# Same mapping as backend/garnet/ai_import.py:_SIDE_TO_DIRECTION — kept here so the
# copy stays backend-free.
_SIDE_TO_DIRECTION = {
    "top": "UP",
    "bottom": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
    # tolerate the compass spellings too
    "up": "UP",
    "down": "DOWN",
}


def load_equipment_ports(indir: Path, stem: str) -> dict[str, list[dict]]:
    """Nozzle points from Contract B -> the `equip_id -> [{x,y,direction}]` file the loader wants.

    Every port the extraction supplies is a usable trace start/end: the tracer walks outward
    from the bbox edge, so a nozzle marking sits on the equipment outline and is projected to
    that edge later (the mixin does this via port_projection.py). `mark`, `size` and
    `line_number` ride along as metadata — the tracer ignores them, but they let a trace be
    labelled with the nozzle and line it started from.
    """
    p = indir / f"{stem}_equipment_bboxes.json"
    if not p.is_file():
        return {}
    items = json.loads(p.read_text()).get("objects", [])
    ports: dict[str, list[dict]] = {}
    rejected: list[str] = []
    for item in items:
        tag = str(item.get("Tag") or "").strip()
        if not tag:
            continue
        eq_id = f"equip_{tag.lower().replace('-', '_')}"
        usable = []
        for index, port in enumerate(item.get("Ports") or [], start=1):
            if not isinstance(port, dict):
                continue
            side = str(port.get("side") or "").strip().lower()
            direction = _SIDE_TO_DIRECTION.get(side)
            if direction is None:
                rejected.append(f"{tag}:{side!r} (unusable side)")
                continue
            pt = port.get("point_px")
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                rejected.append(f"{tag}:{side} (no point_px)")
                continue
            usable.append({
                "x": int(pt[0]),
                "y": int(pt[1]),
                "direction": direction,
                "mark": port.get("mark") or f"port_{index:02d}",
                "size": port.get("size"),
                "line_number": port.get("line_number"),
            })
        if usable:
            ports[eq_id] = usable
    if rejected:
        log.warning("%d nozzle(s) rejected: %s", len(rejected), ", ".join(rejected[:8]))
    return ports


# ---------------------------------------------------------------------------
# the mask — built from the fixtures, no OCR stage
# ---------------------------------------------------------------------------


def build_pipe_mask(image: np.ndarray, objects: list[dict], equipment: list[dict], *,
                    ocr_padding: int = 1, object_inset: int = 1, inline_inset: int = 12,
                    min_area: int = 16, blur_kernel: int = 5,
                    adaptive_block_size: int = 21, adaptive_c: int = 5,
                    equipment_inset: int = 0) -> tuple[np.ndarray, dict]:
    """Binarize, then erase text and symbol glyphs so only pipework remains.

    Text suppression comes from the fixtures alone now (no OCR stage): the merged object list
    supplies both the `line number` boxes and the instrument-tag / note classes, which is the
    coverage we have without running an OCR pass.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, adaptive_block_size, adaptive_c)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = np.where(cv2.bitwise_or(adaptive, otsu) > 0, 255, 0).astype(np.uint8)
    h, w = mask.shape

    def blank(x1, y1, x2, y2):
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0

    keep = {"arrow", "node"}
    inline = {"gate valve", "globe valve", "check valve", "ball valve", "butterfly valve",
              "control valve", "pressure relief valve", "reducer", "spectacle blind",
              "strainer", "three way valve"}
    # Equipment symbols (pump, vessel, column, ...) belong in the same category as inline
    # symbols: pipe attaches at their boundary. Blanking their full padded bbox erased the
    # nozzle stub itself — a detected `pump` object is a few px larger than the equipment
    # bbox, so a padded blank ate ~17 px of real pipe just outside the nozzle and the trace
    # died with `no_pipe`. Insetting keeps the stub while still clearing the symbol body.
    inline |= EQUIPMENT_LABELS
    text_classes = LINE_NUMBER_CLASSES | {"instrument tag", "instrument dcs", "instrument logic",
                                          "note", "title_block", "table_text", "legend_text",
                                          "dimension", "process_label", "utility_label",
                                          "valve_tag", "equipment_tag", "unknown"}
    n_text = 0
    for obj in objects:
        cls = str(obj.get("class_name", "")).strip().lower()
        if cls in keep:
            continue
        bb = obj.get("bbox") or {}
        if not {"x_min", "y_min", "x_max", "y_max"}.issubset(bb):
            continue
        x1, y1, x2, y2 = (int(bb[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        if cls in inline:
            if inline_inset:
                blank(x1 + inline_inset, y1 + inline_inset, x2 - inline_inset, y2 - inline_inset)
        else:
            blank(x1 - ocr_padding, y1 - ocr_padding, x2 + ocr_padding, y2 + ocr_padding)
            if cls in text_classes:
                n_text += 1

    if equipment_inset > 0:
        for it in equipment:
            b = it["bbox"]
            blank(b["x_min"] + equipment_inset, b["y_min"] + equipment_inset,
                  b["x_max"] - equipment_inset, b["y_max"] - equipment_inset)

    before = int((mask > 0).sum())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    removed = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
            removed += 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    stats_out = {
        "mask_px_before_cleanup": before,
        "mask_px": int((mask > 0).sum()),
        "pct_of_sheet": round(float((mask > 0).mean()) * 100, 2),
        "components_removed": removed,
        "text_boxes_suppressed": n_text,
        "binarize": {"blur_kernel": blur_kernel, "adaptive_block_size": adaptive_block_size,
                     "adaptive_c": adaptive_c, "min_component_area": min_area},
    }
    return mask, stats_out


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------


class Stage5bHost:
    """The 10 host methods Stage5bPipelineMixin calls, backed by a plain directory."""

    def __init__(self, image_path: str, out_dir: str | Path,
                 cfg: Stage5bConfig | None = None) -> None:
        self.image_path = str(image_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or Stage5bConfig()
        self.image_bgr: np.ndarray | None = None
        self.artifacts: list[str] = []

    # ---- artifacts -------------------------------------------------------
    def _register_artifact(self, name: str) -> None:
        self.artifacts.append(name)

    def _save_json(self, name: str, data: Any) -> None:
        p = self.out_dir / f"{name}.json"
        tmp = p.with_name(f".{p.name}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(p)
        finally:
            if tmp.exists():
                tmp.unlink()
        self._register_artifact(p.name)
        log.info("saved %s", p)

    def _save_img(self, name: str, img: np.ndarray) -> None:
        p = self.out_dir / f"{name}.png"
        out = img.astype(np.uint8) * 255 if img.dtype == bool else img
        if not cv2.imwrite(str(p), out):
            raise OSError(f"failed to write {p}")
        self._register_artifact(p.name)
        log.info("saved %s", p)

    def _load_json_artifact(self, name: str) -> Any:
        p = self.out_dir / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError(f"required artifact missing: {p}")
        return json.loads(p.read_text())

    def _load_json_artifact_or_default(self, name: str, default: Any) -> Any:
        p = self.out_dir / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else default

    # ---- image -----------------------------------------------------------
    def _ensure_image_loaded(self) -> np.ndarray:
        if self.image_bgr is None:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"cannot read image: {self.image_path}")
            self.image_bgr = img
        return self.image_bgr

    # ---- equipment -------------------------------------------------------
    def _load_equipment_bboxes_for_stage5b(self) -> list[dict]:
        return self._load_json_artifact_or_default("stage3_equipment_bboxes", {}).get("equipment", [])

    # ---- mask helper -----------------------------------------------------
    @staticmethod
    def _extend_mask_to_terminals(mask: np.ndarray, terminals: list[dict],
                                  max_gap: int = 80) -> np.ndarray:
        """Fill a padded pad at the bbox edge nearest the pipe so the tracer can walk in.

        Copied in behaviour from PIDPipeline: bridges are drawn as short orthogonal stubs rather
        than long lines, which would create loops.
        """
        pipe_ys, pipe_xs = np.where(mask > 0)
        if len(pipe_xs) == 0:
            return mask
        result = mask.copy()
        h, w = result.shape
        pad = 20
        for obj in terminals:
            b = obj.get("bbox") or {}
            if not {"x_min", "y_min", "x_max", "y_max"}.issubset(b):
                continue
            bx1, by1 = max(0, int(b["x_min"])), max(0, int(b["y_min"]))
            bx2, by2 = min(w, int(b["x_max"])), min(h, int(b["y_max"]))
            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
            d = np.sqrt((pipe_xs - cx) ** 2 + (pipe_ys - cy) ** 2)
            j = int(np.argmin(d))
            if float(d[j]) > max_gap:
                continue
            px, py = int(pipe_xs[j]), int(pipe_ys[j])
            dl, dr = abs(px - bx1), abs(px - bx2)
            dt, db = abs(py - by1), abs(py - by2)
            m = min(dl, dr, dt, db)
            if m == dl:
                fx1, fx2 = bx1, min(bx1 + pad, bx2)
                fy1, fy2 = max(0, cy - pad // 2), min(h, cy + pad // 2)
            elif m == dr:
                fx1, fx2 = max(bx1, bx2 - pad), bx2
                fy1, fy2 = max(0, cy - pad // 2), min(h, cy + pad // 2)
            elif m == dt:
                fx1, fx2 = max(0, cx - pad // 2), min(w, cx + pad // 2)
                fy1, fy2 = by1, min(by1 + pad, by2)
            else:
                fx1, fx2 = max(0, cx - pad // 2), min(w, cx + pad // 2)
                fy1, fy2 = max(by1, by2 - pad), by2
            tx, ty = (fx1 + fx2) // 2, (fy1 + fy2) // 2
            cv2.line(result, (px, py), (tx, py), 255, 4)
            cv2.line(result, (tx, py), (tx, ty), 255, 4)
            result[fy1:fy2, fx1:fx2] = 255
        return result

    # ---- overlay helpers (no-ops if the artifacts are absent) -------------
    def _draw_equipment_port_markers(self, overlay: np.ndarray, ports: dict) -> None:
        for obj_id, port_list in ports.items():
            if not str(obj_id).startswith("equip_"):
                continue
            for i, (px, py, direction) in enumerate(port_list, start=1):
                cv2.circle(overlay, (int(px), int(py)), 6, (255, 200, 0), -1)
                cv2.circle(overlay, (int(px), int(py)), 6, (255, 255, 255), 1)
                cv2.putText(overlay, f"p{i:02d}", (int(px) + 8, int(py) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 120), 2)

    def _draw_equipment_ground_truth(self, overlay: np.ndarray) -> None:
        for item in self._load_equipment_bboxes_for_stage5b():
            b = item.get("bbox") or {}
            if not {"x_min", "y_min", "x_max", "y_max"}.issubset(b):
                continue
            cv2.rectangle(overlay, (b["x_min"], b["y_min"]), (b["x_max"], b["y_max"]),
                          (220, 120, 0), 2)
            cv2.putText(overlay, str(item.get("class_name", "")).title(),
                        (b["x_min"], b["y_min"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (220, 120, 0), 2)
