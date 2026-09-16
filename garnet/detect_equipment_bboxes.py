#!/usr/bin/env python
"""Equipment bounding-box extraction from a P&ID raster using a vision LLM (ollama-cloud).

Tiles the sheet, asks a vision model for equipment boxes per tile (native resolution — the
provider downscales any image to ~1024px, so tiles are sized to avoid that), maps boxes back
to sheet coordinates, dedupes across tile seams, and writes GARNET "Contract B" JSON plus a
labelled verification overlay.

Companion to `detect_objects.py` (YOLO/SAHI, symbols only). YOLO does not detect major
equipment — this script is the source for it. Feed the JSON to the pipeline with
`--ai-equipment`.

Usage:
    python garnet/detect_equipment_bboxes.py --image sheet.png
    python garnet/detect_equipment_bboxes.py --image sheet.png --tile 1600 --overlap 0.25 \
        --model deepseek-v4.1-flash --out out/equip.json

Run with the repo venv: /Users/maetee/Code/GARNET/.venv/bin/python
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "deepseek-v4.1-flash"
DEFAULT_BASE_URL = "https://ollama.com/v1"
ENV_PATH = Path.home() / ".hermes" / ".env"

# `ai_import.py` only accepts these (underscore/space variants). Anything else is skipped.
EQUIPMENT_LABELS = [
    "vessel", "column", "pump", "compressor", "blower", "heat exchanger", "tank",
    "reactor", "mixer", "pot", "knockout drum", "filter", "cooler", "heater", "injection pump",
]
AI_CLASS_MAP = {
    "static mixer": "mixer", "inline mixer": "mixer", "ko drum": "knockout drum",
    "knockout drum": "knockout drum", "shell and tube exchanger": "heat exchanger",
    "exchanger": "heat exchanger", "air cooler": "cooler", "drum": "vessel",
    "separator": "vessel", "accumulator": "vessel", "coalescer": "vessel",
    "surge tank": "tank", "column": "column", "tower": "column",
}

PROMPT = (
    "This is a crop of a P&ID (piping and instrumentation diagram), {w} pixels wide by {h} pixels tall. "
    "Identify MAJOR PROCESS EQUIPMENT only: vessels, drums, columns, towers, reactors, tanks, pumps, "
    "compressors, blowers, heat exchangers, coolers, heaters, coalescers, filters, mixers.\n"
    "DO NOT report: valves, instruments, instrument bubbles, control loops, line numbers, arrows, "
    "off-page/page connectors, utility connections, reducers, strainers, sight glasses, notes, title block.\n"
    "For each equipment item give its tag (as printed, or \"\" if unreadable) and its bounding box "
    "tightly around the equipment SYMBOL OUTLINE only (exclude surrounding piping, tag text and callouts). "
    "Coordinates must be in THIS CROP's own pixel space, origin top-left: x_min,y_min,x_max,y_max.\n"
    "Respond ONLY with JSON, no prose, no markdown fences: "
    '{{"objects":[{{"tag":"","equipment_type":"","bbox_px":[x_min,y_min,x_max,y_max]}}]}}'
)


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (p if p.exists() else REPO_ROOT / p)


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


def _parse_json(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return json.loads(t)


def _normalize_type(raw: str) -> str | None:
    t = str(raw or "").strip().lower().replace("_", " ")
    if not t:
        return None
    if t in EQUIPMENT_LABELS:
        return t
    if t in AI_CLASS_MAP:
        return AI_CLASS_MAP[t]
    for label in EQUIPMENT_LABELS:          # substring match, longest label first
        if label in t:
            return label
    return None


def _tiles(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    stride = max(int(tile * (1.0 - overlap)), 1)
    def starts(limit: int) -> list[int]:
        if limit <= tile:
            return [0]
        out, pos = [0], 0
        while pos + tile < limit:
            pos = min(pos + stride, limit - tile)
            if pos == out[-1]:
                break
            out.append(pos)
        return out
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in starts(height) for x in starts(width)]


def _iou(a: list[int], b: list[int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _containment(a: list[int], b: list[int]) -> float:
    """Fraction of the smaller box's area covered by the overlap with the larger one."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


# Instrument tags: a bubble tagged LT/PT/TT/FT/PI... is an instrument, never equipment.
# `PDIT 200` is function pair `PD` + modifiers `IT`, so up to 3 modifier letters are allowed
# directly after the function prefix, then an optional decoration letter, then the loop number.
# Equipment tags (P-2503A, D-2502, E-2501, V-101, SRT-P2503A, PRW, PWD) start with a letter
# that is not a function prefix, so they do not match.
_INSTRUMENT_TAG = re.compile(
    r"\b(LT|LG|LC|LV|LIC|LSL|LSH|LS|PT|PD|PG|PC|PV|PIC|PSV|PS|TT|TD|TG|TC|TV|TIC|TS|TE|"
    r"FT|FG|FC|FV|FIC|FI|FE|FF|FS|AT|AI|AE|AC|AV|ZS|ZT|XV|HV|MOV|SDV|BDV)"
    r"[A-Z]{0,3}[\s\-_/.]*[A-Z]?[\s\-_/.]*\d",
    re.IGNORECASE,
)
# In-line items the prompt excludes but the model still reports.
_INLINE_ITEM = re.compile(
    r"sight\s*glass|strainer|reducer|valve|blind|sampling|connection|arrow|"
    r"instrument|bubble|controller|indicator|transmitter|gauge|switch|element|"
    r"junction|note|label|callout|dimension",
    re.IGNORECASE,
)
# Tag prefixes that are in-line items by convention, whatever `equipment_type` the model picks.
# Observed: a strainer `SRT-P2503A` came back typed "filter" and would have imported as equipment.
_INLINE_TAG = re.compile(r"^(SRT|SG|BL|SP|MS)[\s\-_]", re.IGNORECASE)
# Pipe line-number-ish tags are text annotations on the pipe, not equipment — the model reads
# them as symbols when they sit alone. Observed: `NAS 41-0007` typed "mixer", and `PRW`/`PWD`
# (header text) typed "filter".
# Equipment tags are a letter prefix plus a short number (`D-2502`, `P-2503A`, `E-2501`, `V-101`),
# so the tell is a whitespace-separated second token carrying digits — `NAS 41-0007`. Matching on
# `-[0-9]{4,}` instead would wrongly reject the pumps (`P-2503A`).
_ANNOTATION_TAG = re.compile(r"^\S+\s+\S*\d")


def call_vision(b64: str, prompt: str, *, model: str, base_url: str, api_key: str,
                max_tokens: int, mime: str, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
        "max_tokens": max_tokens,
        # Essential: without this the model spends thousands of tokens on hidden reasoning
        # and returns empty `content`.
        "reasoning": {"effort": "none"},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def extract(
    image_path: Path, *, model: str, base_url: str, api_key: str, tile: int, overlap: float,
    max_tokens: int, timeout: int, min_area: int, verbose: bool,
) -> dict:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    height, width = image.shape[:2]

    boxes = _tiles(width, height, tile, overlap)
    detections: list[dict] = []
    rejected: list[dict] = []
    errors: list[str] = []

    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        crop = image[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".png", crop)
        if not ok:
            errors.append(f"tile {idx}: encode failed")
            continue
        prompt = PROMPT.format(w=crop.shape[1], h=crop.shape[0])
        try:
            resp = call_vision(base64.b64encode(buf.tobytes()).decode(), prompt,
                               model=model, base_url=base_url, api_key=api_key,
                               max_tokens=max_tokens, mime="image/png", timeout=timeout)
            content = resp["choices"][0]["message"].get("content") or ""
            payload = _parse_json(content)
        except urllib.error.HTTPError as e:
            errors.append(f"tile {idx}: HTTP {e.code} {e.read().decode()[:120]}")
            continue
        except Exception as e:
            errors.append(f"tile {idx}: {type(e).__name__}: {e}")
            continue

        for obj in payload.get("objects", []) or []:
            bb = obj.get("bbox_px")
            if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            try:
                gx1, gy1, gx2, gy2 = (int(round(float(v))) for v in bb)
            except (TypeError, ValueError):
                continue
            # clamp into the crop, then offset into sheet space
            cx1 = max(0, min(gx1, crop.shape[1])); cx2 = max(0, min(gx2, crop.shape[1]))
            cy1 = max(0, min(gy1, crop.shape[0])); cy2 = max(0, min(gy2, crop.shape[0]))
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            glob = [cx1 + x1, cy1 + y1, cx2 + x1, cy2 + y1]
            if (glob[2] - glob[0]) * (glob[3] - glob[1]) < min_area:
                continue
            detections.append({
                "tag": str(obj.get("tag") or "").strip(),
                "raw_type": str(obj.get("equipment_type") or "").strip(),
                "equipment_type": _normalize_type(obj.get("equipment_type")),
                "bbox_px": glob,
                "tile": idx,
            })
        if verbose:
            print(f"  tile {idx}/{len(boxes)} ({x1},{y1})-({x2},{y2}): {len(detections)} total",
                  file=sys.stderr)

    # Reject instrument bubbles / in-line items the model reports despite the prompt. A tag with
    # an in-line prefix goes regardless of the type the model chose (SRT-* typed "filter").
    filtered: list[dict] = []
    for det in detections:
        if _INSTRUMENT_TAG.search(det["tag"]):
            det["reject"] = "instrument tag"
        elif _INLINE_TAG.match(det["tag"]):
            det["reject"] = "in-line tag prefix"
        elif _ANNOTATION_TAG.match(det["tag"]):
            det["reject"] = "annotation text, not an equipment tag"
        elif _INLINE_ITEM.search(det["raw_type"]) or _INLINE_ITEM.search(det["tag"]):
            det["reject"] = "in-line item / non-equipment"
        if det.get("reject"):
            rejected.append(det)
        else:
            filtered.append(det)
    detections = filtered

    # Merge duplicates. Two passes:
    #   1. same non-empty tag  -> same physical item, keep the largest box
    #   2. heavy geometric overlap (tile seam or nested re-detection) -> keep the largest
    by_tag: dict[str, dict] = {}
    untagged: list[dict] = []
    merged_count = 0
    for det in detections:
        area = (det["bbox_px"][2] - det["bbox_px"][0]) * (det["bbox_px"][3] - det["bbox_px"][1])
        tag = det["tag"].upper().replace(" ", "")
        if not tag:
            untagged.append(det)
            continue
        prev = by_tag.get(tag)
        if prev is None:
            by_tag[tag] = det
            continue
        prev_area = (prev["bbox_px"][2] - prev["bbox_px"][0]) * (prev["bbox_px"][3] - prev["bbox_px"][1])
        if area > prev_area:
            det.setdefault("merged_tiles", []).append(prev["tile"])
            by_tag[tag] = det
        else:
            prev.setdefault("merged_tiles", []).append(det["tile"])
        merged_count += 1

    candidates = sorted(list(by_tag.values()) + untagged,
                        key=lambda d: (d["bbox_px"][2] - d["bbox_px"][0]) * (d["bbox_px"][3] - d["bbox_px"][1]),
                        reverse=True)
    kept: list[dict] = []
    for det in candidates:
        twin = None
        for k in kept:
            # Geometric duplicate: high IoU, or the smaller box sits largely inside the larger.
            if _iou(det["bbox_px"], k["bbox_px"]) >= 0.35:
                twin = k
            elif _containment(det["bbox_px"], k["bbox_px"]) >= 0.8:
                twin = k
            if twin is not None:
                break
        if twin is None:
            kept.append(det)
            continue
        # Prefer the labelled twin and union the geometry so a partial box cannot shrink it.
        if not twin["tag"] and det["tag"]:
            twin["tag"] = det["tag"]
        if not twin["equipment_type"] and det["equipment_type"]:
            twin["equipment_type"] = det["equipment_type"]
        union = [min(twin["bbox_px"][0], det["bbox_px"][0]), min(twin["bbox_px"][1], det["bbox_px"][1]),
                 max(twin["bbox_px"][2], det["bbox_px"][2]), max(twin["bbox_px"][3], det["bbox_px"][3])]
        twin["bbox_px"] = union
        twin.setdefault("merged_tiles", []).append(det["tile"])
        merged_count += 1

    kept.sort(key=lambda d: (d["bbox_px"][1], d["bbox_px"][0]))
    return {"image": image, "width": width, "height": height,
            "detections": kept, "rejected": rejected, "merged_count": merged_count,
            "tiles": len(boxes), "errors": errors}


def to_contract_b(result: dict, *, source_drawing: str) -> dict:
    w, h = result["width"], result["height"]
    objects, unresolved = [], []
    for i, d in enumerate(result["detections"], start=1):
        x1, y1, x2, y2 = d["bbox_px"]
        note = "model box; Equipment_type not in EQUIPMENT_LABELS" if not d["equipment_type"] else ""
        entry = {
            "Index": i,
            "Object": d["equipment_type"] or d["raw_type"],
            "Tag": d["tag"],
            "Equipment_type": d["equipment_type"] or "",
            "Service": "",
            "Size": "",
            "Evidence": "vision-LLM tile extraction" + (f" ({note})" if note else ""),
            "Left": x1, "Top": y1, "Width": x2 - x1, "Height": y2 - y1,
            "Bounding_box_px": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
            "Bounding_box_norm": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            "Score": 1.0,
            "Ports": [],          # not attempted by this script
        }
        if d["equipment_type"]:
            objects.append(entry)
        else:
            unresolved.append({"what": f"{d['tag'] or 'untagged'} @ {d['bbox_px']}",
                               "why": f"Equipment_type {d['raw_type']!r} not in EQUIPMENT_LABELS",
                               "location_hint_px": [x1, y1]})
            objects.append(entry)   # keep the geometry; importer will skip the bad type
    return {
        "source_drawing": source_drawing,
        "coordinate_frame": {"width_px": w, "height_px": h, "dpi": 0, "page_pt": [0, 0],
                             "page_rotate_deg": 0,
                             "note": f"raster frame; {result['tiles']} tiles"},
        "objects": objects,
        "unresolved": unresolved,
        "verification": "overlay rendered and visually checked: no",
    }


def draw_overlay(result: dict, *, outline_only: bool = False) -> "object":
    """Draw boxes + labels. `outline_only` draws thin outlines and no label banners, so the
    drawing underneath stays readable — use it as the actual verification image."""
    import cv2

    overlay = result["image"].copy()
    scale = max(0.6, min(1.6, result["width"] / 2500))
    thickness = 2 if outline_only else 3

    for d in result["detections"]:
        x1, y1, x2, y2 = d["bbox_px"]
        color = (36, 242, 36) if d["equipment_type"] else (0, 165, 255)   # green ok / amber bad type
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        if outline_only:
            continue
        label = f"{d['tag'] or '?'} [{d['equipment_type'] or d['raw_type'] or '?'}]"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        ly = y1 - 6 if y1 - th - 12 >= 0 else y2 + th + 6
        cv2.rectangle(overlay, (x1, ly - th - 6), (x1 + tw + 8, ly + base), color, -1)
        cv2.putText(overlay, label, (x1 + 4, ly), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0) if color == (36, 242, 36) else (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Extract major-equipment bounding boxes from a P&ID via a vision LLM.")
    p.add_argument("--image", required=True)
    p.add_argument("--out", default=None, help="Contract B JSON path. Default: output/<stem>_equipment_bboxes.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=None, help="Default: $OLLAMA_API_KEY or ~/.hermes/.env")
    p.add_argument("--tile", type=int, default=1024,
                   help="Tile size px. Provider downscales to ~1024, so 1024 = no downscale. Default: 1024")
    p.add_argument("--overlap", type=float, default=0.2, help="Tile overlap ratio. Default: 0.2")
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=300, help="Per-tile HTTP timeout seconds. Default: 300")
    p.add_argument("--min-area", type=int, default=400, help="Drop boxes smaller than this area (px^2). Default: 400")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    image_path = _resolve(args.image)
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1
    key = _api_key(args.api_key)
    if not key:
        print("error: no API key (set OLLAMA_API_KEY or pass --api-key)", file=sys.stderr)
        return 1

    result = extract(image_path, model=args.model, base_url=args.base_url, api_key=key,
                     tile=args.tile, overlap=args.overlap, max_tokens=args.max_tokens,
                     timeout=args.timeout, min_area=args.min_area, verbose=args.verbose)

    payload = to_contract_b(result, source_drawing=image_path.stem)
    out_path = Path(args.out) if args.out else REPO_ROOT / "output" / f"{image_path.stem}_equipment_bboxes.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{len(payload['objects'])} equipment boxes from {result['tiles']} tiles -> {out_path}")
    typed = Counter(o["Equipment_type"] or "(unmapped)" for o in payload["objects"])
    for name, n in typed.most_common():
        print(f"  {n:4d}  {name}")
    if payload["unresolved"]:
        print(f"  {len(payload['unresolved'])} unresolved (out-of-vocabulary Equipment_type)")
    if result["rejected"]:
        print(f"  {len(result['rejected'])} rejected (instrument/in-line item):")
        for r in result["rejected"][:6]:
            print(f"    {r['tag'] or '(no tag)':<14} {r['raw_type']:<16} {r['reject']}")
    if result["errors"]:
        print(f"  {len(result['errors'])} tile error(s):")
        for e in result["errors"][:5]:
            print(f"    {e}")

    if not args.no_overlay:
        import cv2
        overlay_path = out_path.with_suffix(".png")
        if not cv2.imwrite(str(overlay_path), draw_overlay(result)):
            print(f"error: failed to write overlay: {overlay_path}", file=sys.stderr)
            return 1
        print(f"overlay -> {overlay_path}")
        # Banner-free thin-outline render: this is the one you can actually verify from.
        verify_path = out_path.with_name(out_path.stem + "_outline.png")
        if not cv2.imwrite(str(verify_path), draw_overlay(result, outline_only=True)):
            print(f"error: failed to write outline: {verify_path}", file=sys.stderr)
            return 1
        print(f"verify  -> {verify_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
