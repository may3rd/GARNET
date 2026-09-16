"""Import externally produced (AI-generated) equipment and line-number JSON.

Two file shapes are accepted, both produced per drawing outside this pipeline:

* equipment — ``{coordinate_frame: {width_px, height_px}, objects: [{Object, Tag,
  Bounding_box_px, Ports: [{side, point_px, ...}], ...}], unresolved: [...]}``
* line numbers — ``{objects: [{Left, Top, Width, Height, Text, Score}],
  image_width, image_height}``

Everything here is pure: callers pass parsed payloads and receive artifact
payloads plus a per-item report. No file is read or written, so the CLI and the
HTTP endpoint share one implementation and one set of rules.

The hard part is the coordinate frame. The source raster is not necessarily the
raster the pipeline ran on — an export of the same sheet can be cropped
differently, which shows up as a translation with no scale change. Normalized
coordinates cannot be used to bridge that: they assume both rasters frame the
same area, and silently stretch everything when they do not. So either the
declared dimensions match exactly, or a transform is fitted from content
(`fit_transform`) and shown to a human before anything is written.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

from garnet.line_number_fusion import _normalize_line_number

__all__ = [
    "AiImportError",
    "Transform",
    "FitReport",
    "ImportResult",
    "ReportRow",
    "fit_transform",
    "convert",
    "check_frames",
]


class AiImportError(ValueError):
    """Raised when an import cannot proceed without guessing at the data."""


# AI files name equipment in prose ("Static mixer"); the pipeline matches against
# EQUIPMENT_LABELS. Only mappings onto labels that set actually contains belong
# here — anything unmapped is reported as skipped, never silently dropped.
AI_CLASS_MAP: dict[str, str] = {
    "static mixer": "mixer",
    "inline mixer": "mixer",
    "knockout drum": "knockout drum",
    "ko drum": "knockout drum",
    "shell and tube exchanger": "heat exchanger",
    "shell & tube exchanger": "heat exchanger",
    "exchanger": "heat exchanger",
    "air cooler": "cooler",
    "drum": "vessel",
    "separator": "vessel",
    "accumulator": "vessel",
}

_SIDE_TO_DIRECTION: dict[str, str] = {
    "top": "UP",
    "bottom": "DOWN",
    "left": "LEFT",
    "right": "RIGHT",
}

Status = Literal["imported", "replaced", "skipped", "info"]


@dataclass(frozen=True)
class Transform:
    """Maps a point in the source raster onto the job raster."""

    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0

    @property
    def is_identity(self) -> bool:
        return self.dx == 0.0 and self.dy == 0.0 and self.scale == 1.0

    def apply(self, x: float, y: float) -> tuple[int, int]:
        """Transform and round. Rounding happens here and nowhere else, so a
        box and the ports on its edge cannot round inconsistently."""
        return (
            int(round(x * self.scale + self.dx)),
            int(round(y * self.scale + self.dy)),
        )


@dataclass(frozen=True)
class FitReport:
    """Evidence for a fitted transform, shown to a human before it is applied."""

    pairs_matched: int
    pairs_used: int
    residual_median_x: float
    residual_median_y: float
    residual_max_x: float
    residual_max_y: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs_matched": self.pairs_matched,
            "pairs_used": self.pairs_used,
            "residual_median_x": round(self.residual_median_x, 2),
            "residual_median_y": round(self.residual_median_y, 2),
            "residual_max_x": round(self.residual_max_x, 2),
            "residual_max_y": round(self.residual_max_y, 2),
        }


@dataclass
class ReportRow:
    kind: str  # "equipment" | "line_number" | "port" | "note"
    key: str
    status: Status
    reason: str = ""
    source_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "key": self.key,
            "status": self.status,
            "reason": self.reason,
            "source_index": self.source_index,
        }


@dataclass
class ImportResult:
    """Artifact payloads ready to write, plus what happened to every input item."""

    objects: list[dict[str, Any]] = field(default_factory=list)
    equipment_ports: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    line_numbers: list[dict[str, Any]] = field(default_factory=list)
    report: list[ReportRow] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "equipment": len(self.objects),
            "ports": sum(len(v) for v in self.equipment_ports.values()),
            "line_numbers": len(self.line_numbers),
            "skipped": sum(1 for r in self.report if r.status == "skipped"),
        }


# ---------------------------------------------------------------- frame checks


def declared_frame(payload: dict[str, Any]) -> tuple[int, int] | None:
    """Source raster size declared by either file shape, if it declares one."""
    frame = payload.get("coordinate_frame")
    if isinstance(frame, dict):
        width, height = frame.get("width_px"), frame.get("height_px")
        if isinstance(width, int) and isinstance(height, int):
            return (width, height)
    width, height = payload.get("image_width"), payload.get("image_height")
    if isinstance(width, int) and isinstance(height, int):
        return (width, height)
    return None


def check_frames(payloads: Iterable[dict[str, Any]], job_size: tuple[int, int]) -> list[tuple[int, int]]:
    """Return the declared frames that disagree with the job raster."""
    mismatched = []
    for payload in payloads:
        frame = declared_frame(payload)
        if frame is not None and frame != job_size:
            mismatched.append(frame)
    return mismatched


# ------------------------------------------------------------ transform fitting


def _match_key(text: str) -> str:
    """Aggressive key for pairing AI text with OCR text.

    Deliberately harsher than `_normalize_line_number`: OCR renders the same
    line number as ``3"_ CUL-25-002008-B1A2-Nl-_``, so punctuation and the I/l/1
    family have to go before two spellings of one label can meet.
    """
    key = text.upper().replace("½", "1/2").replace("”", '"')
    return re.sub(r"[^A-Z0-9]", "", key)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _center_of_ai_line(item: dict[str, Any]) -> tuple[float, float]:
    return (
        float(item["Left"]) + float(item["Width"]) / 2,
        float(item["Top"]) + float(item["Height"]) / 2,
    )


def _center_of_bbox(bbox: dict[str, Any]) -> tuple[float, float]:
    return (
        (float(bbox["x_min"]) + float(bbox["x_max"])) / 2,
        (float(bbox["y_min"]) + float(bbox["y_max"])) / 2,
    )


def fit_transform(
    ai_line_objects: list[dict[str, Any]],
    job_line_numbers: list[dict[str, Any]],
    *,
    min_pairs: int = 4,
    max_residual_px: float = 6.0,
    fuzzy_threshold: float = 0.85,
) -> tuple[Transform, FitReport]:
    """Fit the source→job transform by matching line-number text.

    Only texts that are unique on *both* sides are used: a line number printed
    twice on a sheet gives two candidate positions and no way to tell which
    instance an OCR box belongs to, which is exactly where a naive fit picks up
    kilopixel outliers. Offsets are taken as medians with MAD rejection so a
    surviving bad pair cannot drag the result.
    """
    ai_by_key: dict[str, list[dict[str, Any]]] = {}
    for item in ai_line_objects:
        key = _match_key(str(item.get("Text") or ""))
        if len(key) > 8:
            ai_by_key.setdefault(key, []).append(item)

    job_by_key: dict[str, list[dict[str, Any]]] = {}
    for entry in job_line_numbers:
        text = str(entry.get("normalized_text") or entry.get("text") or "")
        key = _match_key(text)
        if len(key) > 8 and isinstance(entry.get("bbox"), dict):
            job_by_key.setdefault(key, []).append(entry)

    unique_job = {k: v[0] for k, v in job_by_key.items() if len(v) == 1}
    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for key, items in ai_by_key.items():
        if len(items) != 1:
            continue  # ambiguous on the AI side
        match_key, ratio = _closest_key(key, unique_job)
        if match_key is None or ratio < fuzzy_threshold:
            continue
        pairs.append(
            (_center_of_ai_line(items[0]), _center_of_bbox(unique_job[match_key]["bbox"]))
        )

    if len(pairs) < min_pairs:
        raise AiImportError(
            f"Cannot align: only {len(pairs)} line number(s) matched between the "
            f"imported file and the job (need {min_pairs}). Re-run the extraction "
            "against the job's own input image, or import with matching dimensions."
        )

    offsets_x = [job[0] - src[0] for src, job in pairs]
    offsets_y = [job[1] - src[1] for src, job in pairs]
    dx, used_x = _median_with_mad_rejection(offsets_x)
    dy, used_y = _median_with_mad_rejection(offsets_y)

    # Scale is pinned to 1.0. A differently-framed export of the same sheet at
    # the same DPI is a crop, not a resize; fitting a scale off a handful of
    # text centroids would read measurement noise as a stretch.
    transform = Transform(dx=dx, dy=dy, scale=1.0)

    residuals_x = [abs(src[0] + dx - job[0]) for src, job in pairs]
    residuals_y = [abs(src[1] + dy - job[1]) for src, job in pairs]
    inliers_x = sorted(residuals_x)[: max(used_x, 1)]
    inliers_y = sorted(residuals_y)[: max(used_y, 1)]
    report = FitReport(
        pairs_matched=len(pairs),
        pairs_used=min(used_x, used_y),
        residual_median_x=_median(inliers_x),
        residual_median_y=_median(inliers_y),
        residual_max_x=max(inliers_x),
        residual_max_y=max(inliers_y),
    )

    if report.residual_median_x > max_residual_px or report.residual_median_y > max_residual_px:
        raise AiImportError(
            "Cannot align: fitted offset "
            f"({dx:.1f}, {dy:.1f}) still leaves a median residual of "
            f"{report.residual_median_x:.1f}/{report.residual_median_y:.1f}px, above the "
            f"{max_residual_px:.0f}px limit. The two images are probably not the same sheet."
        )
    return transform, report


def _closest_key(key: str, candidates: dict[str, Any]) -> tuple[str | None, float]:
    best_key, best_ratio = None, 0.0
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, key, candidate).ratio()
        if ratio > best_ratio:
            best_key, best_ratio = candidate, ratio
    return best_key, best_ratio


def _median_with_mad_rejection(values: list[float], *, threshold: float = 3.0) -> tuple[float, int]:
    """Median after dropping points beyond `threshold` median-absolute-deviations."""
    center = _median(values)
    deviations = [abs(v - center) for v in values]
    mad = _median(deviations)
    if mad <= 0:
        return center, len(values)
    inliers = [v for v in values if abs(v - center) <= threshold * mad]
    if not inliers:
        return center, len(values)
    return _median(inliers), len(inliers)


# ------------------------------------------------------------------ conversion


def _bbox_from_ai_equipment(item: dict[str, Any], transform: Transform) -> dict[str, int] | None:
    box = item.get("Bounding_box_px")
    if isinstance(box, dict) and all(k in box for k in ("x_min", "y_min", "x_max", "y_max")):
        x_min, y_min = transform.apply(float(box["x_min"]), float(box["y_min"]))
        x_max, y_max = transform.apply(float(box["x_max"]), float(box["y_max"]))
    elif all(k in item for k in ("Left", "Top", "Width", "Height")):
        x_min, y_min = transform.apply(float(item["Left"]), float(item["Top"]))
        x_max, y_max = transform.apply(
            float(item["Left"]) + float(item["Width"]),
            float(item["Top"]) + float(item["Height"]),
        )
    else:
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def project_port_to_bbox_edge(
    x: int, y: int, direction: str, bbox: dict[str, Any]
) -> tuple[int, int]:
    """Move a supplied nozzle point onto the bounding-box edge it faces.

    An extraction marks a nozzle where it belongs engineering-wise — on the
    equipment outline — but the box usually encloses the nozzle stubs too, so
    the point lands well inside the edge (37 px on the sample column, whose box
    spans the stubs while the shell is narrower). The tracer walks *outward*
    from the box edge, so a port left inside starts its walk in equipment
    geometry rather than on the pipe, and centreline snapping cannot rescue it:
    that search radius is only 12 px.

    The coordinate on the facing axis moves to the edge; the other is clamped
    into the box so a slightly-off nozzle still starts somewhere on that side.
    """
    try:
        x_min, y_min = int(bbox["x_min"]), int(bbox["y_min"])
        x_max, y_max = int(bbox["x_max"]), int(bbox["y_max"])
    except (KeyError, TypeError, ValueError):
        return (x, y)

    facing = direction.upper()
    if facing == "UP":
        return (int(min(max(x, x_min), x_max)), y_min)
    if facing == "DOWN":
        return (int(min(max(x, x_min), x_max)), y_max)
    if facing == "LEFT":
        return (x_min, int(min(max(y, y_min), y_max)))
    if facing == "RIGHT":
        return (x_max, int(min(max(y, y_min), y_max)))
    return (x, y)


def _equipment_id(tag: str, index: int) -> str:
    """`equip_`-prefixed and stable. Stage 5b classifies a terminal as equipment
    by this prefix alone, so the tag can never be the whole id."""
    slug = re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")
    return f"equip_{slug}" if slug else f"equip_{index:03d}"


def _class_for(item: dict[str, Any], equipment_labels: set[str]) -> str | None:
    raw = str(item.get("Equipment_type") or item.get("Object") or "").strip().lower()
    if not raw:
        return None
    if raw in equipment_labels:
        return raw
    mapped = AI_CLASS_MAP.get(raw)
    if mapped and mapped in equipment_labels:
        return mapped
    return None


def convert(
    equipment_payload: dict[str, Any] | None,
    line_payload: dict[str, Any] | None,
    *,
    equipment_labels: set[str],
    transform: Transform | None = None,
    image_id: str = "",
) -> ImportResult:
    """Convert both payloads into artifact-ready structures plus a report."""
    transform = transform or Transform()
    result = ImportResult()

    if equipment_payload:
        _convert_equipment(equipment_payload, equipment_labels, transform, result)
    if line_payload:
        _convert_line_numbers(line_payload, transform, result, image_id=image_id)
    return result


def _convert_equipment(
    payload: dict[str, Any],
    equipment_labels: set[str],
    transform: Transform,
    result: ImportResult,
) -> None:
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise AiImportError("Equipment file has no 'objects' list.")

    for index, item in enumerate(objects, start=1):
        if not isinstance(item, dict):
            continue
        tag = str(item.get("Tag") or "").strip()
        label = tag or f"#{index}"
        raw_type = str(item.get("Equipment_type") or item.get("Object") or "").strip()

        class_name = _class_for(item, equipment_labels)
        if class_name is None:
            result.report.append(
                ReportRow(
                    kind="equipment",
                    key=label,
                    status="skipped",
                    reason=f"unrecognised equipment type {raw_type!r}",
                    source_index=index,
                )
            )
            continue

        bbox = _bbox_from_ai_equipment(item, transform)
        if bbox is None:
            result.report.append(
                ReportRow(
                    kind="equipment",
                    key=label,
                    status="skipped",
                    reason="missing or degenerate bounding box",
                    source_index=index,
                )
            )
            continue

        equip_id = _equipment_id(tag, index)
        result.objects.append(
            {
                "id": equip_id,
                "class_name": class_name,
                "confidence": 1.0,
                "bbox": bbox,
                "source_model": "ai_import",
                "source_weight": "",
                "text": tag,
                "ai_import": {
                    key: item[key]
                    for key in ("Service", "Size", "Evidence", "Index", "Object")
                    if key in item
                },
            }
        )
        result.report.append(
            ReportRow(kind="equipment", key=label, status="imported", source_index=index)
        )

        ports = _convert_ports(item, equip_id, label, transform, result)
        if ports:
            result.equipment_ports[equip_id] = ports

    for note in payload.get("unresolved") or []:
        if isinstance(note, dict):
            result.report.append(
                ReportRow(
                    kind="note",
                    key=str(note.get("what") or "unresolved"),
                    status="info",
                    reason=str(note.get("why") or ""),
                )
            )
    for what, why in (payload.get("excluded_by_rule") or {}).items():
        result.report.append(
            ReportRow(kind="note", key=str(what), status="info", reason=str(why))
        )


def _convert_ports(
    item: dict[str, Any],
    equip_id: str,
    label: str,
    transform: Transform,
    result: ImportResult,
) -> list[dict[str, Any]]:
    ports: list[dict[str, Any]] = []
    for port_index, port in enumerate(item.get("Ports") or [], start=1):
        if not isinstance(port, dict):
            continue
        mark = str(port.get("mark") or f"port_{port_index:02d}")
        side = str(port.get("side") or "").strip().lower()
        direction = _SIDE_TO_DIRECTION.get(side)
        point = port.get("point_px")
        if direction is None:
            result.report.append(
                ReportRow(
                    kind="port",
                    key=f"{label}:{mark}",
                    status="skipped",
                    reason=f"unusable side {side!r}",
                )
            )
            continue
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            result.report.append(
                ReportRow(
                    kind="port",
                    key=f"{label}:{mark}",
                    status="skipped",
                    reason="missing point_px",
                )
            )
            continue
        x, y = transform.apply(float(point[0]), float(point[1]))
        ports.append(
            {
                "x": x,
                "y": y,
                "direction": direction,
                "mark": mark,
                "size": port.get("size"),
                "line_number": port.get("line_number"),
            }
        )
    return ports


def _convert_line_numbers(
    payload: dict[str, Any],
    transform: Transform,
    result: ImportResult,
    *,
    image_id: str,
) -> None:
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise AiImportError("Line-number file has no 'objects' list.")

    for index, item in enumerate(objects, start=1):
        if not isinstance(item, dict):
            continue
        text = str(item.get("Text") or "").strip()
        if not text:
            result.report.append(
                ReportRow(
                    kind="line_number", key=f"#{index}", status="skipped",
                    reason="empty text", source_index=index,
                )
            )
            continue
        try:
            x_min, y_min = transform.apply(float(item["Left"]), float(item["Top"]))
            x_max, y_max = transform.apply(
                float(item["Left"]) + float(item["Width"]),
                float(item["Top"]) + float(item["Height"]),
            )
        except (KeyError, TypeError, ValueError):
            result.report.append(
                ReportRow(
                    kind="line_number", key=text, status="skipped",
                    reason="missing or malformed geometry", source_index=index,
                )
            )
            continue
        if x_max <= x_min or y_max <= y_min:
            result.report.append(
                ReportRow(
                    kind="line_number", key=text, status="skipped",
                    reason="degenerate bounding box", source_index=index,
                )
            )
            continue

        score = item.get("Score")
        confidence = float(score) if isinstance(score, (int, float)) else 1.0
        result.line_numbers.append(
            {
                "id": f"line_number_ai_{index:06d}",
                "source_object_id": "",
                "bbox": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
                "text": text,
                "normalized_text": _normalize_line_number(text),
                "ocr_region_id": None,
                "ocr_source": "ai_import",
                "score": confidence,
                "distance_px": 0.0,
                "ocr_confirmed": True,
                "detection_confidence": confidence,
                "fused_confidence": confidence,
                "semantic_class": "line_number",
                "review_state": "ocr_confirmed",
            }
        )
        result.report.append(
            ReportRow(kind="line_number", key=text, status="imported", source_index=index)
        )


def line_numbers_payload(result: ImportResult, image_id: str) -> dict[str, Any]:
    """stage4_line_numbers.json body for the imported line numbers."""
    return {
        "image_id": image_id,
        "pass_type": "sheet",
        "line_numbers": result.line_numbers,
        "rejected": [],
    }


def merge_objects(existing: list[dict[str, Any]], imported: list[dict[str, Any]],
                  equipment_labels: set[str]) -> list[dict[str, Any]]:
    """Replace the equipment bucket, keep every other detection untouched.

    Existing `obj_NNNNNN` ids are never renumbered — stage4_line_numbers entries
    reference them through `source_object_id`.
    """
    kept = [
        obj for obj in existing
        if not (
            isinstance(obj, dict)
            and str(obj.get("class_name") or "").strip().lower() in equipment_labels
        )
    ]
    return kept + list(imported)
