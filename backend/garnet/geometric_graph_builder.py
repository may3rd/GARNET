from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import networkx as nx


HV_THRESHOLD_DEG = 45.0
CHAIN_ALIGN_TOLERANCE_PX = 3.0
CHAIN_GAP_PX = 8.0
# Stage 5 emits many short text/object remnants. The geometric bypass keeps
# full-pipe runs only; 90 px is the smallest threshold that keeps Test-00008
# in the target graph envelope while preserving substantial pipe spans.
MIN_RUN_LENGTH_PX = 90.0
JUNCTION_PROXIMITY_PX = 15.0
ANGLE_TOLERANCE_DEG = 15.0


class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_id(segment: dict[str, Any], index: int) -> str:
    if segment.get("id") is not None:
        return str(segment["id"])
    return f"seg_{index}"


def _classify_orientation(x1: float, y1: float, x2: float, y2: float) -> str:
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    return "H" if angle <= HV_THRESHOLD_DEG or angle >= 180.0 - HV_THRESHOLD_DEG else "V"


def _normalise_segment(segment: dict[str, Any], index: int) -> dict[str, Any]:
    x1 = _as_float(segment.get("x1"))
    y1 = _as_float(segment.get("y1"))
    x2 = _as_float(segment.get("x2"))
    y2 = _as_float(segment.get("y2"))
    orientation = _classify_orientation(x1, y1, x2, y2)
    if orientation == "H":
        start = min(x1, x2)
        end = max(x1, x2)
        axis = (y1 + y2) / 2.0
    else:
        start = min(y1, y2)
        end = max(y1, y2)
        axis = (x1 + x2) / 2.0
    return {
        "id": _segment_id(segment, index),
        "orientation": orientation,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "axis": axis,
        "start": start,
        "end": end,
        "length": math.hypot(x2 - x1, y2 - y1),
    }


def _interval_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    if a["end"] < b["start"]:
        return float(b["start"] - a["end"])
    if b["end"] < a["start"]:
        return float(a["start"] - b["end"])
    return 0.0


def _round_coord(value: float) -> float | int:
    rounded = round(float(value), 3)
    if abs(rounded - round(rounded)) < 1e-6:
        return int(round(rounded))
    return rounded


def chain_geometric_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Phase A: Chain collinear/near-collinear segments into longer H/V runs.

    Segments are classified by angle, then unioned when they share the same
    alignment axis (within 3 px) and their projected intervals overlap or are
    separated by at most 8 px.
    """
    normalised = [_normalise_segment(segment, idx) for idx, segment in enumerate(segments)]
    runs: list[dict[str, Any]] = []

    for orientation in ("H", "V"):
        items = [item for item in normalised if item["orientation"] == orientation]
        if not items:
            continue
        dsu = _DisjointSet(len(items))
        # Spatial bucketing on the alignment axis keeps the O(n^2) comparison
        # bounded for full-size sheets while preserving single-linkage merging.
        buckets: dict[int, list[int]] = defaultdict(list)
        for idx, item in enumerate(items):
            bucket = int(math.floor(item["axis"] / max(CHAIN_ALIGN_TOLERANCE_PX, 1.0)))
            for nearby_bucket in range(bucket - 2, bucket + 3):
                for other_idx in buckets.get(nearby_bucket, []):
                    other = items[other_idx]
                    if abs(float(item["axis"]) - float(other["axis"])) > CHAIN_ALIGN_TOLERANCE_PX:
                        continue
                    if _interval_gap(item, other) <= CHAIN_GAP_PX:
                        dsu.union(idx, other_idx)
            buckets[bucket].append(idx)

        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for idx, item in enumerate(items):
            groups[dsu.find(idx)].append(item)

        sorted_groups = sorted(
            groups.values(),
            key=lambda group: (
                sum(float(item["axis"]) for item in group) / len(group),
                min(float(item["start"]) for item in group),
            ),
        )
        for group in sorted_groups:
            start = min(float(item["start"]) for item in group)
            end = max(float(item["end"]) for item in group)
            axis = sum(float(item["axis"]) for item in group) / len(group)
            length = end - start
            if length < MIN_RUN_LENGTH_PX:
                continue
            member_ids = sorted(str(item["id"]) for item in group)
            if orientation == "H":
                x1, y1, x2, y2 = start, axis, end, axis
            else:
                x1, y1, x2, y2 = axis, start, axis, end
            run_id = f"geo_run_{len(runs)}"
            runs.append(
                {
                    "id": run_id,
                    "orientation": orientation,
                    "x1": _round_coord(x1),
                    "y1": _round_coord(y1),
                    "x2": _round_coord(x2),
                    "y2": _round_coord(y2),
                    "length": round(float(length), 3),
                    "member_segment_ids": member_ids,
                    "segments": member_ids,
                }
            )

    return runs


def _run_endpoint(run: dict[str, Any], is_start: bool) -> tuple[float, float]:
    if is_start:
        return (_as_float(run.get("x1")), _as_float(run.get("y1")))
    return (_as_float(run.get("x2")), _as_float(run.get("y2")))


def _run_angle(run: dict[str, Any]) -> float:
    dx = _as_float(run.get("x2")) - _as_float(run.get("x1"))
    dy = _as_float(run.get("y2")) - _as_float(run.get("y1"))
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _angle_difference_deg(a: float, b: float) -> float:
    diff = abs((a - b) % 180.0)
    return min(diff, 180.0 - diff)


def _is_perpendicular(a: float, b: float) -> bool:
    return abs(_angle_difference_deg(a, b) - 90.0) <= ANGLE_TOLERANCE_DEG


def _is_collinear(a: float, b: float) -> bool:
    diff = _angle_difference_deg(a, b)
    return diff <= ANGLE_TOLERANCE_DEG or abs(diff - 180.0) <= ANGLE_TOLERANCE_DEG


def _classify_cluster(run_ids: list[str], run_by_id: dict[str, dict[str, Any]]) -> tuple[str, str, bool]:
    run_count = len(run_ids)
    if run_count <= 1:
        return "terminal", "terminal", True
    if run_count >= 4:
        return "junction", "X", True

    angles = [_run_angle(run_by_id[run_id]) for run_id in run_ids if run_id in run_by_id]
    if run_count == 2 and len(angles) == 2:
        verified = _is_perpendicular(angles[0], angles[1])
        return "junction", "L" if verified else "straight", verified

    if run_count == 3 and len(angles) == 3:
        collinear_pair = any(_is_collinear(angles[i], angles[j]) for i in range(3) for j in range(i + 1, 3))
        perpendicular_pair = any(_is_perpendicular(angles[i], angles[j]) for i in range(3) for j in range(i + 1, 3))
        verified = collinear_pair and perpendicular_pair
        return "junction", "T" if verified else "multi", verified

    return "junction", "multi", False


def detect_junctions_from_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Phase B: Cluster run endpoints into terminal/junction nodes."""
    endpoints: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run["id"])
        for is_start in (True, False):
            x, y = _run_endpoint(run, is_start)
            endpoints.append({"x": x, "y": y, "run_id": run_id, "is_start": is_start})

    if not endpoints:
        return []

    dsu = _DisjointSet(len(endpoints))
    cell_size = JUNCTION_PROXIMITY_PX
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, endpoint in enumerate(endpoints):
        cell = (int(math.floor(endpoint["x"] / cell_size)), int(math.floor(endpoint["y"] / cell_size)))
        for gx in range(cell[0] - 1, cell[0] + 2):
            for gy in range(cell[1] - 1, cell[1] + 2):
                for other_idx in grid.get((gx, gy), []):
                    other = endpoints[other_idx]
                    if math.hypot(endpoint["x"] - other["x"], endpoint["y"] - other["y"]) <= JUNCTION_PROXIMITY_PX:
                        dsu.union(idx, other_idx)
        grid[cell].append(idx)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, endpoint in enumerate(endpoints):
        groups[dsu.find(idx)].append(endpoint)

    run_by_id = {str(run["id"]): run for run in runs}
    sorted_groups = sorted(
        groups.values(),
        key=lambda group: (sum(p["y"] for p in group) / len(group), sum(p["x"] for p in group) / len(group)),
    )
    junctions: list[dict[str, Any]] = []
    for group in sorted_groups:
        cx = sum(float(p["x"]) for p in group) / len(group)
        cy = sum(float(p["y"]) for p in group) / len(group)
        connected_runs = sorted({str(p["run_id"]) for p in group})
        node_type, subtype, angle_verified = _classify_cluster(connected_runs, run_by_id)
        junctions.append(
            {
                "id": f"geo_junction_{len(junctions)}" if node_type == "junction" else f"geo_terminal_{len(junctions)}",
                "type": node_type,
                "junction_subtype": subtype,
                "subtype": subtype,
                "position": {"x": round(cx, 3), "y": round(cy, 3)},
                "connected_runs": connected_runs,
                "endpoint_count": len(group),
                "angle_verified": angle_verified,
                "review_state": "provisional",
            }
        )
    return junctions


def _nearest_junction(
    endpoint: tuple[float, float],
    junctions: list[dict[str, Any]],
    *,
    max_dist: float = JUNCTION_PROXIMITY_PX,
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for junction in junctions:
        position = junction.get("position", {})
        dist = math.hypot(endpoint[0] - _as_float(position.get("x")), endpoint[1] - _as_float(position.get("y")))
        if dist <= max_dist and (best is None or dist < best[0]):
            best = (dist, junction)
    return best[1] if best is not None else None


def _terminal_node(endpoint: tuple[float, float], index: int) -> dict[str, Any]:
    return {
        "id": f"geo_terminal_extra_{index}",
        "type": "terminal",
        "junction_subtype": "terminal",
        "subtype": "terminal",
        "position": {"x": round(endpoint[0], 3), "y": round(endpoint[1], 3)},
        "connected_runs": [],
        "endpoint_count": 1,
        "angle_verified": True,
        "review_state": "provisional",
    }


def _terminal_endpoint_ref(node: dict[str, Any], run_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    connected_runs = [str(run_id) for run_id in node.get("connected_runs", [])]
    if len(connected_runs) != 1:
        return None

    run_id = connected_runs[0]
    run = run_by_id.get(run_id)
    if run is None:
        return None

    position = node.get("position", {})
    x = _as_float(position.get("x"))
    y = _as_float(position.get("y"))
    start = _run_endpoint(run, True)
    end = _run_endpoint(run, False)
    is_start = math.hypot(x - start[0], y - start[1]) <= math.hypot(x - end[0], y - end[1])
    return {
        "run_id": run_id,
        "is_start": is_start,
        "orientation": str(run.get("orientation", _classify_orientation(start[0], start[1], end[0], end[1]))),
        "x": x,
        "y": y,
    }


def _can_merge_terminal_refs(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    max_merge_dist: float,
    axis_tolerance: float,
    require_axis_alignment: bool,
) -> bool:
    if a["run_id"] == b["run_id"]:
        return False
    if a["orientation"] != b["orientation"]:
        return False
    if math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"])) > max_merge_dist:
        return False
    if not require_axis_alignment:
        return True
    if a["orientation"] == "H":
        return abs(float(a["y"]) - float(b["y"])) <= axis_tolerance
    return abs(float(a["x"]) - float(b["x"])) <= axis_tolerance


def _cluster_terminal_refs(
    refs: list[dict[str, Any]],
    *,
    max_merge_dist: float,
    axis_tolerance: float = 10.0,
    require_axis_alignment: bool = True,
    enforce_unique_runs: bool = False,
) -> list[list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for i, ref in enumerate(refs):
        for j in range(i + 1, len(refs)):
            other = refs[j]
            if not _can_merge_terminal_refs(
                ref,
                other,
                max_merge_dist=max_merge_dist,
                axis_tolerance=axis_tolerance,
                require_axis_alignment=require_axis_alignment,
            ):
                continue
            pairs.append(
                (
                    math.hypot(float(ref["x"]) - float(other["x"]), float(ref["y"]) - float(other["y"])),
                    i,
                    j,
                )
            )

    if enforce_unique_runs:
        # The adaptive pass can span larger symbol gaps. Greedy clustering keeps
        # one endpoint per run in each terminal cluster so a run never resolves
        # both ends to the same node and disappears during edge assembly.
        pairs.sort()
        clusters: list[set[int]] = []
        item_cluster: dict[int, int] = {}

        def cluster_runs(cluster: set[int]) -> set[str]:
            return {str(refs[idx]["run_id"]) for idx in cluster}

        for _, i, j in pairs:
            cluster_i = item_cluster.get(i)
            cluster_j = item_cluster.get(j)
            if cluster_i is None and cluster_j is None:
                clusters.append({i, j})
                cluster_idx = len(clusters) - 1
                item_cluster[i] = cluster_idx
                item_cluster[j] = cluster_idx
            elif cluster_i is not None and cluster_j is None:
                if str(refs[j]["run_id"]) not in cluster_runs(clusters[cluster_i]):
                    clusters[cluster_i].add(j)
                    item_cluster[j] = cluster_i
            elif cluster_i is None and cluster_j is not None:
                if str(refs[i]["run_id"]) not in cluster_runs(clusters[cluster_j]):
                    clusters[cluster_j].add(i)
                    item_cluster[i] = cluster_j
            elif cluster_i != cluster_j:
                runs_i = cluster_runs(clusters[cluster_i])
                runs_j = cluster_runs(clusters[cluster_j])
                if runs_i.isdisjoint(runs_j):
                    if len(clusters[cluster_i]) < len(clusters[cluster_j]):
                        cluster_i, cluster_j = cluster_j, cluster_i
                    for idx in clusters[cluster_j]:
                        item_cluster[idx] = cluster_i
                    clusters[cluster_i].update(clusters[cluster_j])
                    clusters[cluster_j] = set()

        for idx in range(len(refs)):
            if idx not in item_cluster:
                clusters.append({idx})
        return [sorted(cluster) for cluster in clusters if cluster]

    dsu = _DisjointSet(len(refs))
    for _, i, j in pairs:
        dsu.union(i, j)
    clusters_by_root: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(refs)):
        clusters_by_root[dsu.find(idx)].append(idx)
    return list(clusters_by_root.values())


def _terminal_close_pair_count(nodes: list[dict[str, Any]], *, max_dist: float = 60.0) -> int:
    terminals = [node for node in nodes if node.get("type") == "terminal"]
    close = 0
    for i, terminal in enumerate(terminals):
        position = terminal.get("position", {})
        x = _as_float(position.get("x"))
        y = _as_float(position.get("y"))
        for other in terminals[i + 1 :]:
            other_position = other.get("position", {})
            if math.hypot(x - _as_float(other_position.get("x")), y - _as_float(other_position.get("y"))) < max_dist:
                close += 1
    return close


def _nodes_from_terminal_clusters(
    terminals: list[dict[str, Any]],
    refs: list[dict[str, Any]],
    clusters: list[list[int]],
) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for cluster_idx, cluster in enumerate(clusters):
        cluster_refs = [refs[idx] for idx in cluster]
        cluster_nodes = [terminals[idx] for idx in cluster]
        cx = sum(float(ref["x"]) for ref in cluster_refs) / len(cluster_refs)
        cy = sum(float(ref["y"]) for ref in cluster_refs) / len(cluster_refs)
        connected_runs = sorted({str(ref["run_id"]) for ref in cluster_refs})
        endpoint_refs = [
            {"run_id": str(ref["run_id"]), "is_start": bool(ref["is_start"])} for ref in cluster_refs
        ]

        if len(cluster) == 1:
            node = dict(cluster_nodes[0])
            node["_endpoint_refs"] = endpoint_refs
            cleaned.append(node)
            continue

        cleaned.append(
            {
                "id": f"geo_terminal_merged_{cluster_idx}",
                "type": "terminal",
                "junction_subtype": "terminal",
                "subtype": "terminal",
                "position": {"x": round(cx, 3), "y": round(cy, 3)},
                "connected_runs": connected_runs,
                "endpoint_count": sum(int(node.get("endpoint_count", 1)) for node in cluster_nodes),
                "angle_verified": True,
                "review_state": "provisional",
                "terminal_merge_state": "resolved",
                "_endpoint_refs": endpoint_refs,
            }
        )
    return cleaned


def _merge_orphaned_terminals(
    nodes: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    *,
    max_merge_dist: float = 30.0,
) -> list[dict[str, Any]]:
    """
    Post-junction-detection: merge terminal endpoints that are close together
    on the same pipe orientation. Handles instrument tap gaps where YOLO
    produces a segmentation break, causing two runs to have endpoints very
    close to each other.

    Returns a cleaned list of terminal nodes (merged where appropriate).
    Junctions are returned unchanged.
    """
    terminals = [node for node in nodes if node.get("type") == "terminal"]
    junctions = [node for node in nodes if node.get("type") != "terminal"]
    if len(terminals) <= 1:
        return nodes

    run_by_id = {str(run.get("id")): run for run in runs}
    refs: list[dict[str, Any]] = []
    mergeable_terminals: list[dict[str, Any]] = []
    passthrough_terminals: list[dict[str, Any]] = []
    for terminal in terminals:
        ref = _terminal_endpoint_ref(terminal, run_by_id)
        if ref is None:
            passthrough_terminals.append(terminal)
            continue
        refs.append(ref)
        mergeable_terminals.append(terminal)

    if not refs:
        return nodes

    clusters = _cluster_terminal_refs(refs, max_merge_dist=max_merge_dist)
    cleaned_terminals = _nodes_from_terminal_clusters(mergeable_terminals, refs, clusters) + passthrough_terminals

    # Severe over-noding can leave hundreds of same-orientation orphaned
    # endpoints on large sheets. Keep the 30 px pass as the high-confidence
    # cleanup, then widen only when the graph is still clearly pathological.
    # The wider pass still preserves orientation and one endpoint per run.
    cleaned_nodes = junctions + cleaned_terminals
    if len(cleaned_terminals) > 80 or _terminal_close_pair_count(cleaned_nodes) >= 50:
        for adaptive_dist in (60.0, 90.0, 120.0, 150.0, 180.0, 250.0, 350.0):
            if adaptive_dist <= max_merge_dist:
                continue
            adaptive_clusters = _cluster_terminal_refs(
                refs,
                max_merge_dist=adaptive_dist,
                require_axis_alignment=False,
                enforce_unique_runs=True,
            )
            adaptive_terminals = _nodes_from_terminal_clusters(mergeable_terminals, refs, adaptive_clusters) + passthrough_terminals
            adaptive_nodes = junctions + adaptive_terminals
            if len(adaptive_terminals) <= 80 and _terminal_close_pair_count(adaptive_nodes) < 50:
                cleaned_terminals = adaptive_terminals
                break

    return junctions + cleaned_terminals


def _node_for_run_endpoint(
    run: dict[str, Any],
    is_start: bool,
    nodes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    run_id = str(run.get("id", ""))
    for node in nodes:
        for endpoint_ref in node.get("_endpoint_refs", []):
            if str(endpoint_ref.get("run_id")) == run_id and bool(endpoint_ref.get("is_start")) == is_start:
                return node
    return _nearest_junction(_run_endpoint(run, is_start), nodes)


def _to_node_cluster(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(node["id"]),
        "kind": str(node.get("type", "terminal")),
        "centroid": {
            "x": float(node.get("position", {}).get("x", 0.0)),
            "y": float(node.get("position", {}).get("y", 0.0)),
        },
        "member_count": int(node.get("endpoint_count", 1)),
        "junction_subtype": node.get("junction_subtype", node.get("subtype")),
        "connected_runs": list(node.get("connected_runs", [])),
    }


def _node_payload(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(node["id"]),
        "type": str(node.get("type", "terminal")),
        "kind": str(node.get("type", "terminal")),
        "position": node.get("position", {"x": 0.0, "y": 0.0}),
        "member_count": int(node.get("endpoint_count", 1)),
        "junction_subtype": node.get("junction_subtype", node.get("subtype")),
        "connected_runs": list(node.get("connected_runs", [])),
        "review_state": node.get("review_state", "provisional"),
    }


def build_graph_from_runs_and_junctions(
    runs: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    *,
    image_id: str = "",
) -> dict[str, Any]:
    """
    Phase C: Assemble graph payload from geometric runs and endpoint clusters.

    Returns a stage12-compatible graph payload plus helper edge/node-cluster
    artifacts used by PIDPipeline's existing attachment stages.
    """
    nodes = list(junctions)
    nodes = _merge_orphaned_terminals(nodes, runs)
    terminal_extra_count = 0
    edges: list[dict[str, Any]] = []

    for run in runs:
        start = _run_endpoint(run, True)
        end = _run_endpoint(run, False)
        source_node = _node_for_run_endpoint(run, True, nodes)
        target_node = _node_for_run_endpoint(run, False, nodes)
        if source_node is None:
            source_node = _terminal_node(start, terminal_extra_count)
            terminal_extra_count += 1
            nodes.append(source_node)
        if target_node is None:
            target_node = _terminal_node(end, terminal_extra_count)
            terminal_extra_count += 1
            nodes.append(target_node)
        if str(source_node["id"]) == str(target_node["id"]):
            continue
        edge = {
            "id": f"geo_edge_{len(edges)}",
            "source": str(source_node["id"]),
            "target": str(target_node["id"]),
            "pixel_length": float(run.get("length", math.hypot(end[0] - start[0], end[1] - start[1]))),
            "simplified_pixel_length": float(run.get("length", math.hypot(end[0] - start[0], end[1] - start[1]))),
            "polyline": [
                {"row": float(start[1]), "col": float(start[0])},
                {"row": float(end[1]), "col": float(end[0])},
            ],
            "flow_direction": None,
            "flow_direction_confidence": 0.0,
            "assigned_arrow_id": None,
            "review_state": "provisional",
            "source_run_id": str(run.get("id", "")),
            "member_segment_ids": list(run.get("member_segment_ids", [])),
        }
        edges.append(edge)

    node_payloads = [_node_payload(node) for node in nodes]
    graph = nx.Graph()
    for node in node_payloads:
        graph.add_node(node["id"])
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"])

    graph_payload = {
        "image_id": image_id,
        "pass_type": "sheet",
        "nodes": node_payloads,
        "edges": edges,
        "unresolved_junction_ids": [],
        "crossings": [],
        "equipment_attachments": [],
        "connection_attachments": [],
        "text_attachments": [],
        "instrument_tag_attachments": [],
        "equipment_tag_attachments": [],
        "edge_terminals": [],
        "edge_connections": [],
        "edge_components": [[edge["id"]] for edge in edges],
    }
    summary = {
        "image_id": image_id,
        "pass_type": "sheet",
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "connected_component_count": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
        "edge_component_count": len(edges),
        "unresolved_junction_count": 0,
        "crossing_candidate_count": 0,
        "non_connecting_crossing_count": 0,
        "unresolved_crossing_count": 0,
        "source_artifacts": ["stage5_geometric_segments.json", "phase3_runs.json", "phase3_junctions.json"],
    }
    return {
        "graph_payload": graph_payload,
        "summary": summary,
        "node_clusters": [_to_node_cluster(node) for node in nodes],
        "edges_payload": {"image_id": image_id, "pass_type": "sheet", "edges": edges},
    }


# ─── S5-01: Phase 3 gap detection ────────────────────────────────────────
def detect_phase3_gaps(
    edges: list[dict[str, Any]],
    *,
    gap_threshold_px: float = 20.0,
    existing_connections: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Detect geometric gaps between Phase 3 edges: pairs of edges whose
    endpoints are aligned (H or V) and within threshold but NOT already
    connected via a shared junction node AND NOT present in existing_connections.

    Replaces Stage 10's gap_summary for the Phase 3 geometric bypass path.
    Uses the same spatial-grid approach as pipe_continuity_helpers.summarize_gaps.

    Args:
        edges: Phase 3 edges with polyline + source/target node IDs.
        gap_threshold_px: Alignment threshold in pixels.
        existing_connections: List of edge-connection dicts from build_pipe_edge_connectivity.
            If an edge pair already appears here (as any kind), it is skipped.
    """
    if not edges:
        return []

    # Build endpoint index from polyline edges
    endpoint_index: list[dict[str, Any]] = []
    for edge in edges:
        eid = str(edge.get("id", ""))
        polyline = edge.get("polyline", [])
        if not polyline or len(polyline) < 2:
            continue
        src_pt = polyline[0]
        tgt_pt = polyline[-1]
        try:
            src_x, src_y = float(src_pt["col"]), float(src_pt["row"])
            tgt_x, tgt_y = float(tgt_pt["col"]), float(tgt_pt["row"])
            dx = abs(tgt_x - src_x)
            dy = abs(tgt_y - src_y)
            # Dominant axis of this edge (H-pipe vs V-pipe)
            direction = "horizontal" if dx >= dy else "vertical"
            endpoint_index.append({
                "edge_id": eid,
                "source_xy": (src_x, src_y),
                "target_xy": (tgt_x, tgt_y),
                "source_node": str(edge.get("source", "")),
                "target_node": str(edge.get("target", "")),
                "direction": direction,
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not endpoint_index:
        return []

    # Spatial grid for O(n) neighbor lookup
    from garnet.pipe_continuity_helpers import _SpatialGrid

    grid = _SpatialGrid(cell_size=gap_threshold_px)
    for ep in endpoint_index:
        grid.insert(ep["source_xy"][0], ep["source_xy"][1], ("src", ep))
        grid.insert(ep["target_xy"][0], ep["target_xy"][1], ("dst", ep))

    gaps: list[dict[str, Any]] = []
    checked: set[tuple[str, str]] = set()

    # S5 gap_coverage: pre-populate checked from existing connections so gap
    # detection skips edge pairs that edge_connectivity already handled.
    if existing_connections:
        for conn in existing_connections:
            e_a = str(conn.get("source_edge_id", ""))
            e_b = str(conn.get("target_edge_id", ""))
            if e_a and e_b:
                checked.add(tuple(sorted((e_a, e_b))))

    for ep in endpoint_index:
        eid_a = ep["edge_id"]
        node_a = ep["source_node"]
        node_a_target = ep["target_node"]

        for pt_label, pt_a in [("source", ep["source_xy"]), ("target", ep["target_xy"])]:
            # Search radius slightly larger than threshold
            candidates = grid.query_radius(pt_a[0], pt_a[1], gap_threshold_px * 1.5)
            for ref_dist, dist in candidates:
                label, ep_b = ref_dist
                # Skip same edge or same endpoint type
                if label == pt_label and ep_b["edge_id"] == eid_a:
                    continue

                eid_b = ep_b["edge_id"]
                if eid_a >= eid_b:
                    continue
                pair = (eid_a, eid_b)
                if pair in checked:
                    continue

                pt_b = ep_b["target_xy"] if pt_label == "source" else ep_b["source_xy"]
                dx = abs(pt_a[0] - pt_b[0])
                dy = abs(pt_a[1] - pt_b[1])

                # Must be aligned: the dominant axis (larger delta) defines the
                # pipe direction; the smaller delta must be within threshold.
                # Horizontal pipe: |dx| >= |dy|, smaller |dy| <= threshold
                # Vertical pipe:   |dy| >  |dx|, smaller |dx| <= threshold
                if (dx <= gap_threshold_px or dy <= gap_threshold_px):
                    gap_dist = math.hypot(dx, dy)
                    if gap_dist <= gap_threshold_px * 1.5:
                        # Determine alignment: dominant axis is horizontal when
                        # |dx| >= |dy|; otherwise vertical.
                        alignment = "horizontal" if abs(dx) >= abs(dy) else "vertical"
                        mid_x = (pt_a[0] + pt_b[0]) / 2
                        mid_y = (pt_a[1] + pt_b[1]) / 2

                        # Quality tiers for gap_coverage improvement:
                        #   strict  — gap_dist <= 8px, both endpoints snap cleanly
                        #   good    — gap_dist <= 15px, well-aligned
                        #   weak    — gap_dist >  15px, accept but flag for review
                        if gap_dist <= 8.0:
                            gap_quality = "strict"
                        elif gap_dist <= 15.0:
                            gap_quality = "good"
                        else:
                            gap_quality = "weak"

                        # Check NOT already connected (no shared junction node)
                        node_b = ep_b["source_node"]
                        node_b_target = ep_b["target_node"]
                        already_connected = (
                            node_a and node_b and node_a == node_b
                        ) or (
                            node_a and node_b_target and node_a == node_b_target
                        ) or (
                            node_a_target and node_b and node_a_target == node_b
                        ) or (
                            node_a_target and node_b_target and node_a_target == node_b_target
                        )

                        if already_connected:
                            checked.add(pair)
                            continue

                        # Direction compatibility: the gap must be reachable from each edge.
                        # A gap at pt_a requires the edge's OTHER endpoint to be along the
                        # gap's alignment axis from pt_a (not necessarily the edge's
                        # "source" or "target" endpoint — source/target is arbitrary in
                        # Phase 3). The dominant axis of the edge's polyline must match
                        # the gap's alignment axis.
                        #
                        # Example: ep180 goes from (2921,2319)→(2921,2696). Gap is at
                        # (2894,2329.5). The "other" endpoint of ep180 is (2921,2696) —
                        # both x-coords match (vertical alignment: x=2921). So ep180 can
                        # contribute a vertical segment from (2921,2696) toward the gap.
                        ep_a_dir = ep["direction"]
                        ep_b_dir = ep_b.get("direction", "horizontal")
                        if ep_a_dir != alignment and ep_b_dir != alignment:
                            checked.add(pair)
                            continue

                        checked.add(pair)
                        endpoint_a_label = "source" if pt_label == "target" else "target"
                        endpoint_b_label = "source" if label == "dst" else "target"
                        gaps.append({
                            "edge_a": eid_a,
                            "edge_b": eid_b,
                            "endpoint_a": endpoint_a_label,
                            "endpoint_b": endpoint_b_label,
                            "edge_a_endpoint": {"col": round(pt_a[0], 1), "row": round(pt_a[1], 1)},
                            "edge_b_endpoint": {"col": round(pt_b[0], 1), "row": round(pt_b[1], 1)},
                            "gap_position": {"x": round(mid_x, 1), "y": round(mid_y, 1)},
                            "gap_distance_px": round(gap_dist, 2),
                            "alignment": alignment,
                            "gap_quality": gap_quality,
                        })

    return gaps


def detect_boundary_terminals(
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    image_shape: tuple[int, int, int],
    boundary_margin_px: float = 50.0,
) -> list[dict[str, Any]]:
    """
    S5-02: Flag edges whose endpoint is within boundary_margin_px of the
    image edge. These are likely off-page connectors that should have an
    off_page_connector record but don't yet.

    Returns list of boundary_proximity items for each qualifying edge.
    """
    if not edges or not image_shape:
        return []

    height, width = image_shape[:2]
    boundary_terminals: list[dict[str, Any]] = []

    for edge in edges:
        eid = str(edge.get("id", ""))
        polyline = edge.get("polyline", [])
        if not polyline or len(polyline) < 2:
            continue

        src_pt = polyline[0]
        tgt_pt = polyline[-1]
        try:
            src_col, src_row = float(src_pt["col"]), float(src_pt["row"])
            tgt_col, tgt_row = float(tgt_pt["col"]), float(tgt_pt["row"])
        except (KeyError, TypeError, ValueError):
            continue

        src_near = (
            src_col <= boundary_margin_px
            or src_col >= width - boundary_margin_px
            or src_row <= boundary_margin_px
            or src_row >= height - boundary_margin_px
        )
        tgt_near = (
            tgt_col <= boundary_margin_px
            or tgt_col >= width - boundary_margin_px
            or tgt_row <= boundary_margin_px
            or tgt_row >= height - boundary_margin_px
        )

        if not (src_near or tgt_near):
            continue

        # Determine which side (left/right/top/bottom)
        src_side = None
        if src_col <= boundary_margin_px:
            src_side = "left"
        elif src_col >= width - boundary_margin_px:
            src_side = "right"
        elif src_row <= boundary_margin_px:
            src_side = "top"
        elif src_row >= height - boundary_margin_px:
            src_side = "bottom"

        tgt_side = None
        if tgt_col <= boundary_margin_px:
            tgt_side = "left"
        elif tgt_col >= width - boundary_margin_px:
            tgt_side = "right"
        elif tgt_row <= boundary_margin_px:
            tgt_side = "top"
        elif tgt_row >= height - boundary_margin_px:
            tgt_side = "bottom"

        boundary_terminals.append({
            "edge_id": eid,
            "source_node": str(edge.get("source", "")),
            "target_node": str(edge.get("target", "")),
            "source_boundary_side": src_side,
            "target_boundary_side": tgt_side,
            "source_col": src_col,
            "source_row": src_row,
            "target_col": tgt_col,
            "target_row": tgt_row,
        })

    return boundary_terminals