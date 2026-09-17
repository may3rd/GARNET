"""Port position projection — copied verbatim from backend/garnet/ai_import.py.

Stage 5b needs this one helper so the copied pipeline has no dependency on the backend
package. Kept byte-for-byte in behaviour; only the module location changed.
"""

from __future__ import annotations

from typing import Any


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
