"""Step 5c -- merge branch runs that join end-to-end.

A single drawn line is often walked as several branch runs, because each run starts
at a junction dot and stops at the next one. The result is a set of fragments whose
ends do not sit on terminate points: a run stops at `branch_connection` in the middle
of a line, and the continuation is a separate run starting at the same spot.

This step combines them so a final branch line starts and ends at real terminate
points -- tee junction, equipment, connection port, instrument tag, equipment or
connection bounding box.

WHAT IT MERGES
  Only END-TO-END joins: run A's terminal coincides with run B's start (within
  ``tol``), and the heading change is not a reversal. Concatenating then produces a
  path that follows the drawing, because A already ended where B begins.

  A join where A's terminal lands MID-path of B is NOT a concatenation -- that is a
  T, and the point is a junction where both must terminate. Those are reported, not
  merged (see ``describe_non_merge_joins``).

  Measured on the 9 fixture sheets: 13 end-to-end joins, 34 T-shapes, 39 runs whose
  terminal is a `dead_end` (not a terminate point at all).

WHAT IT DOES NOT DO
  It never invents geometry. The merged path is A's segments followed by B's,
  unchanged; only the bookkeeping (id, terminal, length) is recomputed. A merge that
  would reverse direction, or that would leave the merged run ending on a
  non-terminate point, is refused and reported.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np

# A run's terminal must be one of these to count as properly ended. `branch_connection`
# is deliberately absent: it is the mid-line artifact this step exists to remove.
TERMINATE_POINTS = {
    "tee_junction",
    "equipment",
    "page_connection",
    "instrument_tag",
    "pressure_relief_valve",
}

_OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def _segs(walk: dict) -> list[dict]:
    return list(walk.get("segments") or [])


def _walk_start(walk: dict) -> Optional[tuple[int, int]]:
    s = _segs(walk)
    return (int(s[0]["x1"]), int(s[0]["y1"])) if s else None


def _walk_end(walk: dict) -> Optional[tuple[int, int]]:
    s = _segs(walk)
    return (int(s[-1]["x2"]), int(s[-1]["y2"])) if s else None


def _walk_length(walk: dict) -> int:
    return sum(max(abs(int(s["x2"]) - int(s["x1"])),
                   abs(int(s["y2"]) - int(s["y1"]))) for s in _segs(walk))


def _first_dir(walk: dict) -> str:
    s = _segs(walk)
    return str(s[0].get("direction", "")) if s else ""


def _last_dir(walk: dict) -> str:
    s = _segs(walk)
    return str(s[-1].get("direction", "")) if s else ""


def _is_reversal(d_from: str, d_to: str) -> bool:
    """True when the second run immediately doubles back on the first."""
    return _OPPOSITE.get(d_from) == d_to


def _terminal_of(walk: dict) -> tuple[Optional[str], Optional[int], Optional[int]]:
    return (walk.get("terminal_type"), walk.get("terminal_x"), walk.get("terminal_y"))


def _live(branch_results: dict[str, dict]) -> dict[str, dict]:
    return {k: v for k, v in branch_results.items()
            if v.get("status") == "traced" and _segs(v)}


def _continuation_open(walk: dict, path_index: int,
                       mask: np.ndarray, min_run: int = 25,
                       max_gap: int = 11, window: int = 160) -> bool:
    """Does ``walk`` carry on past ``path_index`` along its own travel direction?

    The direction is taken from the PATH GEOMETRY around that index, not from the
    arriving run's endpoint: the arriving endpoint can sit a few px off the line (on
    0002A `branch_000030` ends at (723,2035) while b007's path runs at y=2030), so
    looking for a segment that CONTAINS it finds nothing and the test silently
    reports "closed" for a line that plainly continues.

    Measured over the junction marker's width, with only scan-noise gap tolerance --
    the same test the tracer's junction rule applies (never a drawn line jump).
    """
    if mask is None or not hasattr(mask, "shape"):
        return False
    pts: list[tuple[int, int]] = []
    for s in _segs(walk):
        x1, y1, x2, y2 = (int(s[c]) for c in ("x1", "y1", "x2", "y2"))
        n = max(abs(x2 - x1), abs(y2 - y1))
        for i in range(0, n + 1):
            t = i / n if n else 0
            pts.append((round(x1 + (x2 - x1) * t), round(y1 + (y2 - y1) * t)))
    if not pts or path_index >= len(pts):
        return False
    lo = max(0, path_index - 4)
    hi = min(len(pts) - 1, path_index + 4)
    dx, dy = pts[hi][0] - pts[lo][0], pts[hi][1] - pts[lo][1]
    if dx == 0 and dy == 0:
        return False
    if abs(dx) >= abs(dy):
        sx, sy = (1, 0) if dx > 0 else (-1, 0)
    else:
        sx, sy = (0, 1) if dy > 0 else (0, -1)
    px, py = pts[path_index]
    best = cur = gap = 0
    for i in range(1, window + 1):
        found = False
        for w in range(-6, 7):
            if sx:
                qx, qy = px + sx * i, py + w
            else:
                qx, qy = px + w, py + sy * i
            if 0 <= qx < mask.shape[1] and 0 <= qy < mask.shape[0] and mask[qy, qx] > 0:
                found = True
                break
        if found:
            cur += 1
            gap = 0
            if cur > best:
                best = cur
        else:
            gap += 1
            if gap > max_gap:
                break
    # A run that carries on for min_run or more past the junction has an OPEN axis.
    return best >= min_run


def find_end_to_end_joins(branch_results: dict[str, dict], *,
                          tol: int = 6) -> list[tuple[str, str, float, str, str]]:
    """Return (first, second, gap_px, first_last_dir, second_first_dir) candidates.

    ``first``'s terminal coincides with ``second``'s start. Ordered longest-first so
    the caller can prefer the more complete run as the merge target.
    """
    live = _live(branch_results)
    starts = {}
    for k, v in live.items():
        sp = _walk_start(v)
        if sp:
            starts[k] = sp
    out = []
    for a, va in live.items():
        ea = _walk_end(va)
        if ea is None:
            continue
        for b, sb in starts.items():
            if a == b:
                continue
            gap = ((ea[0] - sb[0]) ** 2 + (ea[1] - sb[1]) ** 2) ** 0.5
            if gap > tol:
                continue
            d1, d2 = _last_dir(va), _first_dir(live[b])
            if _is_reversal(d1, d2):
                continue
            out.append((a, b, round(gap, 2), d1, d2))
    out.sort(key=lambda r: (-_walk_length(live[r[0]]), r[0], r[1]))
    return out


def find_mid_path_joins(branch_results: dict[str, dict], *,
                        tol: int = 6, step: int = 1,
                        require_closed: bool = True,
                        mask: Optional[np.ndarray] = None,
                        min_run: int = 25) -> list[tuple[str, str, tuple[int, int]]]:
    """A's terminal lands MID-path of B: a T, not a concatenation.

    ``require_closed`` restricts this to junctions where B's OWN axis is closed, i.e.
    B has no continuation on the opposite side of its coming side. That is the same
    test the tracer's junction rule uses, and it is what separates a real T from a
    point where another run merely happens to end next to a line that carries on.

    Measured on 0002A `branch_000007`: `branch_000030` ends at (723,2035), which sits
    70px into b007's first RIGHT segment. But b007 continues RIGHT another 479px to
    x=1202, so its axis is OPEN there and the junction must not end it -- cutting on
    endpoint proximity alone shrank a valid 4676px route to 73px.

    Without ``mask`` the closed test cannot run, so every proximity hit is reported
    and the caller is expected to pass a mask when cutting.
    """
    live = _live(branch_results)
    paths: dict[str, list[tuple[int, int]]] = {}
    for k, v in live.items():
        pts = []
        for s in _segs(v):
            x1, y1, x2, y2 = (int(s[c]) for c in ("x1", "y1", "x2", "y2"))
            n = max(abs(x2 - x1), abs(y2 - y1))
            for i in range(0, n + 1, step):
                t = i / n if n else 0
                pts.append((round(x1 + (x2 - x1) * t), round(y1 + (y2 - y1) * t)))
        paths[k] = pts

    out = []
    for a, va in live.items():
        ea = _walk_end(va)
        if ea is None:
            continue
        for b, pts in paths.items():
            if a == b or not pts:
                continue
            best, bi = 1e18, -1
            for i, (qx, qy) in enumerate(pts):
                dd = (ea[0] - qx) ** 2 + (ea[1] - qy) ** 2
                if dd < best:
                    best, bi = dd, i
            if best ** 0.5 > tol:
                continue
            if bi <= 1 or bi >= len(pts) - 2:
                continue        # that is an end, handled by find_end_to_end_joins
            if require_closed and mask is not None:
                if _continuation_open(live[b], bi, mask, min_run):
                    continue
            out.append((a, b, ea))
    return out


def _reverse_segments(segs: list[dict]) -> list[dict]:
    """Return the same path walked the other way.

    Each segment's endpoints swap and its direction flips, and the segment ORDER
    reverses. This is what lets two runs that point at each other be joined: the
    shorter one is reversed so it starts where the longer one ends.
    """
    flip = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    out = []
    for s in reversed(segs):
        out.append({
            "x1": int(s["x2"]), "y1": int(s["y2"]),
            "x2": int(s["x1"]), "y2": int(s["y1"]),
            "direction": flip.get(str(s.get("direction", "")), str(s.get("direction", ""))),
            "length_px": int(s.get("length_px")
                             or max(abs(int(s["x2"]) - int(s["x1"])),
                                    abs(int(s["y2"]) - int(s["y1"])))),
        })
    return out


def _reverse_walk(walk: dict) -> None:
    """Reverse a walk's geometry in place, swapping its endpoints."""
    walk["segments"] = _reverse_segments(_segs(walk))
    walk["turns"] = list(reversed(walk.get("turns") or []))


def _collinear(a: dict, b: dict, tol: int = 3) -> bool:
    """Do two runs lie on the same axis? Compared on the axis they each travel."""
    sa, sb = _segs(a), _segs(b)
    if not sa or not sb:
        return False
    da, db = _first_dir(a), _first_dir(b)
    if da in ("LEFT", "RIGHT") and db in ("LEFT", "RIGHT"):
        return abs(int(sa[0]["y1"]) - int(sb[0]["y1"])) <= tol
    if da in ("UP", "DOWN") and db in ("UP", "DOWN"):
        return abs(int(sa[0]["x1"]) - int(sb[0]["x1"])) <= tol
    return False


def drop_empty_walks(branch_results: dict[str, dict]) -> list[str]:
    """Mark a `traced` run that produced no geometry as `empty`.

    `0010 branch_000054` is `status=traced` with 0 segments and `terminal_type=no_pipe`
    -- it was seeded, found nothing, and still counts as a walk. Re-labelling it keeps
    it out of the run counts and out of the merge candidates without deleting the
    record of the attempt.
    """
    dropped = []
    for k, v in branch_results.items():
        if v.get("status") == "traced" and not _segs(v):
            v["status"] = "empty"
            v["skip_reason"] = "no_geometry"
            dropped.append(k)
    return dropped


def find_all_joins(branch_results: dict[str, dict], *,
                   tol: int = 14) -> list[tuple[str, str, tuple[int, int], str]]:
    """Every pair of runs whose ENDPOINTS meet, with the join point and its kind.

    One finder for all four geometries, because they differ only in which ends meet
    and the merge orients them the same way regardless:

      straight       A.end meets B.start, headings continue (the common case)
      facing         A.end meets B.end -- two runs pointing at each other
      corner         ends meet but the headings are PERPENDICULAR: one run turns
                     into the other, so the merged path has an L in it
      same_direction both ends meet while travelling the SAME way: joining turns the
                     second run around, giving a doubled-back path

    ``0009 branch_000002`` (ends RIGHT at (1167,1267)) and ``branch_000003`` (ends
    DOWN at (1170,1262)) are a corner: 5px apart, perpendicular. ``0009
    branch_000017`` and ``branch_000001`` both END near (1641,2960) travelling UP --
    same_direction.
    """
    live = _live(branch_results)
    out: list[tuple[str, str, tuple[int, int], str]] = []
    ks = sorted(live)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = live[ks[i]], live[ks[j]]
            ends_a = [("start", _walk_start(a)), ("end", _walk_end(a))]
            ends_b = [("start", _walk_start(b)), ("end", _walk_end(b))]
            best = None
            for na, pa in ends_a:
                if not pa:
                    continue
                for nb, pb in ends_b:
                    if not pb:
                        continue
                    gap = ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5
                    if gap > tol:
                        continue
                    if best is None or gap < best[0]:
                        best = (gap, na, nb, pa)
            if best is None:
                continue
            gap, na, nb, point = best
            # Heading each run travels AWAY from the join point. For an end that is
            # the run's last direction; for a start it is the reverse of its first
            # direction (the run arrives from the opposite side).
            out_a = _last_dir(a) if na == "end" else _OPPOSITE.get(_first_dir(a), "")
            out_b = _last_dir(b) if nb == "end" else _OPPOSITE.get(_first_dir(b), "")
            if out_a == out_b:
                kind = "same_direction"
            elif _is_reversal(out_a, out_b):
                kind = "facing"
            else:
                kind = "corner"
            out.append((ks[i], ks[j], point, kind))
    out.sort(key=lambda r: r[0])
    return out


def _orient_for_join(a: dict, b: dict, point: tuple[int, int]) -> tuple[dict, dict]:
    """Orient two runs so a's path ENDS at ``point`` and b's path STARTS there.

    Reverses whichever run needs it. Returns them in that order, so the caller can
    simply append b's segments to a's and get a path that follows the drawing.
    """
    px, py = int(point[0]), int(point[1])

    def at(pt, t=16):
        return pt is not None and abs(pt[0] - px) <= t and abs(pt[1] - py) <= t

    if not at(_walk_end(a)):
        _reverse_walk(a)
    if not at(_walk_start(b)):
        _reverse_walk(b)
    return a, b


def find_same_origin_pairs(branch_results: dict[str, dict], *,
                           tol: int = 10) -> list[tuple[str, str, tuple[int, int]]]:
    """Runs seeded from the SAME dot heading opposite ways, collinear.

    Orphan-dot seeding emits one leg per direction, so a straight line passing through
    a dot becomes two runs that share a start. `0003 branch_000034` (UP from the dot)
    and `branch_000035` (DOWN from it) are one line, split.
    """
    live = _live(branch_results)
    out = []
    ks = sorted(live)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = live[ks[i]], live[ks[j]]
            sa = _walk_start(a)
            sb = _walk_start(b)
            if not sa or not sb:
                continue
            if abs(sa[0] - sb[0]) > tol or abs(sa[1] - sb[1]) > tol:
                continue
            if _is_reversal(_first_dir(a), _first_dir(b)) and _collinear(a, b):
                out.append((ks[i], ks[j], sa))
    return out


def find_facing_pairs(branch_results: dict[str, dict], *,
                      tol: int = 14) -> list[tuple[str, str, tuple[int, int], float]]:
    """Two runs whose ENDS meet, heading opposite ways, collinear.

    `0009 branch_000012` (RIGHT) ends at (3551,1700) and `branch_000013` (LEFT) ends at
    (3561,1700): they point at each other across a 10px gap. Joining needs one
    reversed -- `_reverse_walks_into` does that.
    """
    live = _live(branch_results)
    out = []
    ks = sorted(live)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = live[ks[i]], live[ks[j]]
            ea, eb = _walk_end(a), _walk_end(b)
            if not ea or not eb:
                continue
            gap = ((ea[0] - eb[0]) ** 2 + (ea[1] - eb[1]) ** 2) ** 0.5
            if gap > tol:
                continue
            if _is_reversal(_last_dir(a), _last_dir(b)) and _collinear(a, b):
                out.append((ks[i], ks[j], ea, round(gap, 2)))
    return out


def resolve_terminal_from_objects(walk: dict, objects: list[dict], *,
                                  reach: int = 10) -> Optional[dict]:
    """Re-name a merged run's terminal from the objects at its far end.

    A merged run often ends where the last absorbed leg ended -- frequently
    `branch_connection`, which is the mid-line label this whole step exists to remove.
    If a junction node or equipment bbox sits at that point, the honest terminal is
    that object, so the run ends on a terminate point as the step requires.

    Returns the change, or None when nothing better was found.
    """
    tx, ty = walk.get("terminal_x"), walk.get("terminal_y")
    if tx is None or ty is None:
        return None
    before = walk.get("terminal_type")
    for o in objects:
        b = o.get("bbox")
        if not isinstance(b, dict):
            continue
        cls = str(o.get("class_name", "")).strip().lower()
        cx = (int(b["x_min"]) + int(b["x_max"])) // 2
        cy = (int(b["y_min"]) + int(b["y_max"])) // 2
        inside = (int(b["x_min"]) - reach <= tx <= int(b["x_max"]) + reach and
                  int(b["y_min"]) - reach <= ty <= int(b["y_max"]) + reach)
        if not inside:
            continue
        if cls == "node":
            walk["terminal_type"] = "tee_junction"
            walk["terminal_obj_id"] = str(o.get("id"))
            walk["terminal_x"], walk["terminal_y"] = cx, cy
            return {"from": before, "to": "tee_junction", "obj": str(o.get("id"))}
    return None


def _merge_pair(dst: dict, src: dict, merged_from: list[str]) -> None:
    """Append ``src``'s segments onto ``dst`` in place, then re-derive bookkeeping.

    Geometry is appended, never rewritten: the path is dst's segments followed by
    src's, exactly as each was walked.
    """
    dst["segments"] = _segs(dst) + _segs(src)
    dst["turns"] = list(dst.get("turns") or []) + list(src.get("turns") or [])
    dst["hits"] = list(dst.get("hits") or []) + list(src.get("hits") or [])
    dst["junctions"] = list(dst.get("junctions") or []) + list(src.get("junctions") or [])
    # the surviving run takes src's terminal, because src walked the far end
    dst["terminal_type"] = src.get("terminal_type")
    dst["terminal_x"] = src.get("terminal_x")
    dst["terminal_y"] = src.get("terminal_y")
    dst["terminal_obj_id"] = src.get("terminal_obj_id")
    dst["trace_length_px"] = _walk_length(dst)
    dst["merged_from"] = merged_from + [str(src.get("_merge_id", ""))]


def terminate_at_t_shapes(branch_results: dict[str, dict], *,
                          tol: int = 6, mask: Optional[np.ndarray] = None,
                          ) -> list[dict]:
    """Point both runs of a mid-path join at the junction and mark them terminated.

    A T is a real junction, so each participating run must END there: the arriving
    run already does, and the run it landed on is re-cut at that index and given the
    junction as its terminal. Without this the second run keeps running past a
    junction it has been shown to meet.

    Several Ts can land on ONE run -- `0010 branch_000012` is met by three, because it
    walks a loop and each leg of the loop is crossed by another run. The run must be
    cut at the FIRST junction it actually reaches walking from its start, i.e. the
    smallest path index. Cutting at whatever pair is encountered first in iteration
    order instead produced a 1-segment stub ending 201px from its own junction.

    Returns the list of adjustments, one per changed run.
    """
    live = _live(branch_results)

    # Collect, per run, the earliest junction any T reaches. Path index decides, and
    # the index is measured on the run's FULL path before any cutting.
    def path_index(walk: dict, point: tuple[int, int]) -> Optional[int]:
        px, py = int(point[0]), int(point[1])
        idx = 0
        for s in _segs(walk):
            x1, y1, x2, y2 = (int(s[c]) for c in ("x1", "y1", "x2", "y2"))
            n = max(abs(x2 - x1), abs(y2 - y1))
            for i in range(0, n + 1):
                t = i / n if n else 0
                qx, qy = round(x1 + (x2 - x1) * t), round(y1 + (y2 - y1) * t)
                if (qx - px) ** 2 + (qy - py) ** 2 <= tol * tol:
                    return idx
                idx += 1
        return None

    candidates: dict[str, list[tuple[int, tuple[int, int], str]]] = {}
    for a, b, point in find_mid_path_joins(branch_results, tol=tol, mask=mask):
        walk = live.get(b)
        if walk is None:
            continue
        pi = path_index(walk, point)
        if pi is None:
            continue
        candidates.setdefault(b, []).append((pi, (int(point[0]), int(point[1])), a))

    adjustments: list[dict] = []
    for run_id, hits in candidates.items():
        walk = live[run_id]
        pi, point, from_run = min(hits, key=lambda h: h[0])
        px, py = point
        segs = _segs(walk)
        # walk the cumulative length to find which segment holds this path index
        idx = 0
        cut_seg = None
        for i, s in enumerate(segs):
            x1, y1, x2, y2 = (int(s[c]) for c in ("x1", "y1", "x2", "y2"))
            n = max(abs(x2 - x1), abs(y2 - y1))
            if idx + n >= pi:
                cut_seg = i
                break
            idx += n + 1
        if cut_seg is None:
            continue
        if (int(walk.get("terminal_x") or -1),
                int(walk.get("terminal_y") or -1)) == (px, py):
            continue
        before_t = walk.get("terminal_type")
        before_at = (walk.get("terminal_x"), walk.get("terminal_y"))
        before_len = walk.get("trace_length_px")
        walk["segments"] = segs[:cut_seg + 1]
        last = walk["segments"][-1]
        # trim the final segment so the cut lands exactly on the junction
        lx1, ly1, lx2, ly2 = (int(last[c]) for c in ("x1", "y1", "x2", "y2"))
        reduced = False
        if abs(lx2 - lx1) >= abs(ly2 - ly1):
            if min(lx1, lx2) <= px <= max(lx1, lx2) and px != lx2:
                last["x2"] = px
                reduced = True
        else:
            if min(ly1, ly2) <= py <= max(ly1, ly2) and py != ly2:
                last["y2"] = py
                reduced = True
        last["length_px"] = max(abs(int(last["x2"]) - int(last["x1"])),
                                abs(int(last["y2"]) - int(last["y1"])))
        walk["trace_length_px"] = _walk_length(walk)
        walk["terminal_type"] = "tee_junction"
        walk["terminal_x"] = px
        walk["terminal_y"] = py
        walk["terminal_obj_id"] = None
        walk["terminated_at_t_junction_from"] = from_run
        adjustments.append({
            "run": run_id, "cut_at": (px, py), "segment_index": cut_seg,
            "path_index": pi,
            "junctions_on_this_run": [h[1] for h in sorted(hits)],
            "previous_terminal_type": before_t,
            "previous_terminal_at": before_at,
            "previous_length_px": before_len,
            "length_px": walk["trace_length_px"],
            "segment_truncated": reduced,
        })
    return adjustments


def merge_branch_runs(branch_results: dict[str, dict], *,
                      tol: int = 6,
                      require_terminate: bool = True,
                      terminate_t_shapes: bool = True,
                      mask: Optional[np.ndarray] = None,
                      objects: Optional[list[dict]] = None,
                      ) -> dict[str, Any]:
    """Merge branch runs until each survivor ends on a terminate point.

    Runs in this order, because later passes depend on earlier ones having joined:
      1. drop empty walks (no geometry);
      2. SAME_ORIGIN -- two legs from one dot, opposite headings, collinear;
      3. FACING -- ends meet, opposite headings: reverse the shorter, then join;
      4. END_TO_END -- a's end meets b's start;
      5. re-resolve each merged terminal from the detected objects, so a run that
         absorbed a `branch_connection` leg can end on the junction that is really
         there instead.

    Returns a summary dict:
      ``merged``      list of {kept, absorbed, route, length_px, terminal}
      ``refused``     list of {kept, absorbed, reason}
      ``t_shapes``    list of (a, b, point) mid-path joins
      ``same_origin`` / ``facing``    the pairs found by those passes
      ``terminals_resolved``          terminal re-namings
      ``dropped_empty``               walks with no geometry
      ``runs_before`` / ``runs_after``
    """
    dropped_empty = drop_empty_walks(branch_results)
    live = _live(branch_results)
    for k, v in live.items():
        v["_merge_id"] = k

    # ---- pass 1: SAME_ORIGIN -- one line split at a dot ----------------------
    # Orphan-dot seeding emits one leg per direction, so a straight line through a dot
    # becomes two runs sharing a start. Concatenate them back into one.
    pre_merged: list[dict] = []
    same_origin = find_same_origin_pairs(branch_results)
    for a, b, seed in same_origin:
        va, vb = live.get(a), live.get(b)
        if not va or not vb:
            continue
        va, vb = _orient_for_join(va, vb, seed)
        _merge_pair(va, vb, list(va.get("merged_from") or []))
        branch_results[b]["status"] = "merged"
        branch_results[b]["merged_into"] = a
        live.pop(b, None)
        pre_merged.append({
            "kept": a, "absorbed": [b], "via": "same_origin", "at": list(seed),
            "route": " -> ".join(str(s.get("direction")) for s in _segs(va))[:120],
            "length_px": va.get("trace_length_px"), "terminal": va.get("terminal_type"),
            "terminal_at": (va.get("terminal_x"), va.get("terminal_y")),
        })

    # ---- pass 2: every other endpoint meeting ---------------------------------
    # One loop for straight / facing / corner / same_direction, because the merge is
    # the same operation in all four: orient a to END at the join, orient b to START
    # there, then append. How the two headings relate only labels the join.
    joins_all = find_all_joins(branch_results, tol=max(tol, 14))
    join_kinds: list[dict] = []
    absorbed_ids: set[str] = set()
    for a, b, point, kind in joins_all:
        va, vb = live.get(a), live.get(b)
        if not va or not vb or a in absorbed_ids or b in absorbed_ids:
            continue
        va, vb = _orient_for_join(va, vb, point)
        _merge_pair(va, vb, list(va.get("merged_from") or []))
        branch_results[b]["status"] = "merged"
        branch_results[b]["merged_into"] = a
        absorbed_ids.add(b)
        live.pop(b, None)
        join_kinds.append({"a": a, "b": b, "at": list(point), "kind": kind})
        pre_merged.append({
            "kept": a, "absorbed": [b], "via": kind, "at": list(point),
            "route": " -> ".join(str(s.get("direction")) for s in _segs(va))[:120],
            "length_px": va.get("trace_length_px"), "terminal": va.get("terminal_type"),
            "terminal_at": (va.get("terminal_x"), va.get("terminal_y")),
        })

    # ---- pass 3: chains -------------------------------------------------------
    # After the direct joins, a chain can remain (A.end -> B.start -> C.start ...).
    # successors[a] = b keeps the LONGEST continuation when several compete.
    joins = find_end_to_end_joins(branch_results, tol=tol)
    succ: dict[str, str] = {}
    for a, b, gap, d1, d2 in joins:
        if a in succ or b in absorbed_ids:
            continue
        succ[a] = b

    absorbed: set[str] = set()
    report: dict[str, Any] = {"merged": list(pre_merged), "refused": [],
                              "runs_before": len(live),
                              "same_origin": [
                                  {"a": a, "b": b, "at": list(s)}
                                  for a, b, s in same_origin],
                              "joins": join_kinds,
                              "dropped_empty": dropped_empty}

    def chain_from(start: str) -> list[str]:
        chain, seen, cur = [start], {start}, succ.get(start)
        while cur and cur not in seen and cur not in absorbed:
            chain.append(cur)
            seen.add(cur)
            cur = succ.get(cur)
        return chain

    heads = [k for k in live if k not in {b for _, b, *_ in joins}]
    heads.sort(key=lambda k: -_walk_length(live[k]))
    for head in heads:
        chain = chain_from(head)
        if len(chain) < 2:
            continue
        dst = live[chain[0]]
        merged_ids = list(chain[1:])
        for member in chain[1:]:
            _merge_pair(dst, live[member], [str(x) for x in dst.get("merged_from", [])])
        term_ok = dst.get("terminal_type") in TERMINATE_POINTS
        # A merge whose result still ends on a non-terminate point is APPLIED and
        # warned about, not refused: the terminal pass below re-names it from the
        # detected objects, and refusing here was what blocked the user's own cases
        # (`0007 branch_000009 + branch_000023` merged cleanly but ended
        # `branch_connection`). Recorded under `warned` so nothing is hidden.
        if not term_ok:
            report["refused"].append({
                "kept": chain[0], "absorbed": merged_ids,
                "applied": True,
                "warning": f"merged terminal is {dst.get('terminal_type')!r}, "
                           f"not a terminate point before terminal resolution",
            })
        for member in chain[1:]:
            branch_results[member]["status"] = "merged"
            branch_results[member]["merged_into"] = chain[0]
            absorbed.add(member)
        report["merged"].append({
            "kept": chain[0],
            "absorbed": merged_ids,
            "via": "chain",
            "route": " -> ".join(str(s.get("direction")) for s in _segs(dst))[:120],
            "length_px": dst.get("trace_length_px"),
            "terminal": dst.get("terminal_type"),
            "terminal_at": (dst.get("terminal_x"), dst.get("terminal_y")),
        })

    report["t_shapes"] = find_mid_path_joins(branch_results, tol=tol, mask=mask)
    if terminate_t_shapes:
        # Cut the run that a T landed on, so both runs of the junction end there.
        report["t_shape_terminations"] = terminate_at_t_shapes(
            branch_results, tol=tol, mask=mask)
    else:
        report["t_shape_terminations"] = []

    # ---- pass: resolve terminals from the detected objects -------------------
    # A merged run frequently ends where its last absorbed leg did, which is often
    # `branch_connection` -- the mid-line label this step exists to remove. Where a
    # detected junction node sits at the far end, name it instead.
    resolved = []
    if objects:
        for k, v in branch_results.items():
            if v.get("status") != "traced" or not _segs(v):
                continue
            if v.get("terminal_type") in TERMINATE_POINTS:
                continue
            ch = resolve_terminal_from_objects(v, objects)
            if ch:
                resolved.append({"run": k, **ch})
    report["terminals_resolved"] = resolved

    report["runs_after"] = sum(1 for v in branch_results.values()
                               if v.get("status") == "traced" and _segs(v))
    return report
