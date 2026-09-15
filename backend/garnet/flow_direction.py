"""Arrow evidence normalization and conservative route direction inference."""

from __future__ import annotations

import math
from typing import Any


def _point(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return [round(x, 2), round(y, 2)]


def _vector(value: Any) -> list[float] | None:
    p = _point(value)
    if p is None:
        return None
    length = math.hypot(p[0], p[1])
    if length <= 1e-9:
        return None
    return [round(p[0] / length, 6), round(p[1] / length, 6)]


def _direction_vector(value: Any) -> list[float] | None:
    if isinstance(value, str):
        vectors = {"right": [1, 0], "east": [1, 0], "left": [-1, 0], "west": [-1, 0],
                   "down": [0, 1], "south": [0, 1], "up": [0, -1], "north": [0, -1]}
        return vectors.get(value.strip().lower())
    if isinstance(value, dict):
        return _vector([value.get("dx", value.get("x", 0)), value.get("dy", value.get("y", 0))])
    return _vector(value)


def _raster_arrow(image: Any, bbox: Any, threshold: float, asymmetry: float) -> dict[str, Any] | None:
    """Infer only very clear raster arrows; ambiguous crops remain unresolved."""
    if image is None or not isinstance(bbox, dict):
        return None
    try:
        x0, y0 = int(math.floor(float(bbox["x_min"]))), int(math.floor(float(bbox["y_min"])))
        x1, y1 = int(math.ceil(float(bbox["x_max"]))), int(math.ceil(float(bbox["y_max"])))
        if x1 <= x0 or y1 <= y0:
            return None
        crop = image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        if getattr(crop, "size", 0) == 0:
            return None
        gray = crop.mean(axis=2) if getattr(crop, "ndim", 0) == 3 else crop
        dark = gray < 128
        if dark.sum() < 4:
            return None
        ys, xs = dark.nonzero()
        width, height = dark.shape[1], dark.shape[0]
        cx = float(xs.mean()) if len(xs) else 0.0
        cy = float(ys.mean()) if len(ys) else 0.0
        col_score = abs(cx - (width - 1) / 2) / max(width, 1)
        row_score = abs(cy - (height - 1) / 2) / max(height, 1)
        if max(col_score, row_score) < asymmetry:
            return None
        if col_score >= row_score:
            # The filled arrowhead biases the centroid toward its tip.
            direction = [1.0 if cx > (width - 1) / 2 else -1.0, 0.0]
        else:
            direction = [0.0, 1.0 if cy > (height - 1) / 2 else -1.0]
        confidence = min(0.99, max(0.0, (max(col_score, row_score) - asymmetry) / max(0.5 - asymmetry, 1e-9) * (1.0 - threshold) + threshold))
        return {"vector": direction, "source": "raster_crop", "confidence": round(confidence, 4)} if confidence >= threshold else None
    except (TypeError, ValueError, IndexError, AttributeError, OverflowError):
        return None


def normalize_arrow_evidence(item: dict[str, Any], *, image: Any = None,
                             raster_confidence_threshold: float = 0.70,
                             raster_asymmetry_threshold: float = 0.15) -> dict[str, Any]:
    """Return finite, normalized arrow evidence while retaining its provenance."""
    tip = _point(item.get("tip_xy", item.get("tip")))
    tail = _point(item.get("tail_xy", item.get("tail")))
    vector = _direction_vector(item.get("vector"))
    source = "explicit"
    confidence = item.get("confidence")
    reviewed = str(item.get("review_state", item.get("flow_direction_review_state", ""))).lower() in {"accepted", "reviewed", "confirmed"}
    direction = str(item.get("direction", "")).strip().lower()
    if vector is None and tip is not None and tail is not None:
        vector = _vector([tip[0] - tail[0], tip[1] - tail[1]])
    if vector is None and direction in {"right", "east", "left", "west", "up", "north", "down", "south"}:
        vector = _direction_vector(direction)
    explicit_state = direction if direction in {"forward", "reverse", "bidirectional", "conflicting"} else None
    if vector is None:
        raster = _raster_arrow(image, item.get("bbox"), raster_confidence_threshold, raster_asymmetry_threshold)
        if raster:
            vector, source, confidence = raster["vector"], raster["source"], raster["confidence"]
    try:
        confidence = float(confidence) if confidence is not None and math.isfinite(float(confidence)) else None
    except (TypeError, ValueError):
        confidence = None
    result: dict[str, Any] = {
        "tip_xy": tip, "tail_xy": tail, "vector": vector,
        "source": source if vector is not None else "unresolved",
        "confidence": round(max(0.0, min(1.0, confidence)), 4) if confidence is not None else None,
        "review_state": "accepted" if reviewed else "unresolved",
    }
    if explicit_state in {"forward", "reverse"}:
        result["explicit_state"] = explicit_state
    elif explicit_state == "bidirectional" and reviewed:
        result["explicit_state"] = "bidirectional"
    if item.get("projected_xy") is not None:
        result["projected_xy"] = _point(item.get("projected_xy"))
    if item.get("trace_distance_px") is not None:
        try:
            d = float(item["trace_distance_px"])
            result["trace_distance_px"] = round(d, 2) if math.isfinite(d) else None
        except (TypeError, ValueError):
            result["trace_distance_px"] = None
    return result


def infer_arrow_route_direction(evidence: dict[str, Any], edge: dict[str, Any], *, dot_threshold: float = 0.35) -> str:
    if evidence.get("explicit_state") == "bidirectional" and evidence.get("review_state") == "accepted":
        return "bidirectional"
    if evidence.get("explicit_state") in {"forward", "reverse"}:
        return evidence["explicit_state"]
    vector = evidence.get("vector")
    if not vector:
        return "unknown"
    try:
        index = int(evidence.get("segment_index", 0))
        segment = (edge.get("segments") or [])[index]
        dx, dy = float(segment["x2"]) - float(segment["x1"]), float(segment["y2"]) - float(segment["y1"])
        length = math.hypot(dx, dy)
        if not math.isfinite(length) or length <= 1e-9:
            return "unknown"
        dot = (float(vector[0]) * dx + float(vector[1]) * dy) / length
        if not math.isfinite(dot) or abs(dot) < dot_threshold:
            return "conflicting"
        return "forward" if dot > 0 else "reverse"
    except (TypeError, ValueError, IndexError, KeyError):
        return "unknown"


def aggregate_edge_direction(evidences: list[dict[str, Any]]) -> tuple[str, float | None]:
    states = [e.get("route_direction", e.get("explicit_state", "unknown")) for e in evidences]
    usable = [s for s in states if s in {"forward", "reverse", "bidirectional"}]
    if any(s == "bidirectional" for s in usable):
        return ("bidirectional", min((e.get("confidence") for e in evidences if e.get("route_direction") == "bidirectional" and e.get("confidence") is not None), default=None))
    if not usable:
        return ("conflicting" if states and any(s == "conflicting" for s in states) else "unknown", None)
    if len(set(usable)) > 1 or any(s == "conflicting" for s in states):
        return ("conflicting", None)
    confidence = [e.get("confidence") for e in evidences if isinstance(e.get("confidence"), (int, float))]
    return (usable[0], round(sum(confidence) / len(confidence), 4) if confidence else None)
