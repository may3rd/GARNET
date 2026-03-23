from __future__ import annotations

from typing import Any

import cv2
import networkx as nx


_COMPONENT_COLORS = [
    (0, 0, 255),
    (0, 165, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
]


def _draw_component_overlay(
    image_bgr: Any,
    *,
    components: list[list[str]],
    edges: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> Any:
    overlay = image_bgr.copy()
    edge_by_component: dict[str, int] = {}
    node_by_component: dict[str, int] = {}
    edge_lookup = {str(edge.get("id", "")): edge for edge in edges}
    for idx, component in enumerate(components):
        for item_id in component:
            edge = edge_lookup.get(str(item_id))
            if edge is not None:
                edge_by_component[str(item_id)] = idx
                node_by_component[str(edge.get("source", ""))] = idx
                node_by_component[str(edge.get("target", ""))] = idx
                continue
            node_by_component[str(item_id)] = idx

    for edge in edges:
        component_idx = edge_by_component.get(str(edge.get("id", "")))
        if component_idx is None:
            continue
        color = _COMPONENT_COLORS[component_idx % len(_COMPONENT_COLORS)]
        polyline = edge.get("polyline", [])
        for start, end in zip(polyline, polyline[1:]):
            cv2.line(
                overlay,
                (int(start["col"]), int(start["row"])),
                (int(end["col"]), int(end["row"])),
                color,
                2,
            )

    for node in nodes:
        position = node.get("position") or {}
        x = int(round(float(position.get("x", 0.0))))
        y = int(round(float(position.get("y", 0.0))))
        component_idx = node_by_component.get(str(node.get("id", "")))
        if component_idx is None:
            color = (255, 255, 255)
        else:
            color = _COMPONENT_COLORS[component_idx % len(_COMPONENT_COLORS)]
        cv2.circle(overlay, (x, y), 3, color, -1)
    return overlay


def run_pipe_graph_qa_stage(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
    image_bgr: Any,
) -> dict[str, Any]:
    graph = nx.Graph()
    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    crossings = graph_payload.get("crossings", [])
    edge_terminals = graph_payload.get("edge_terminals", [])
    edge_components = graph_payload.get("edge_components", [])
    edge_connections = graph_payload.get("edge_connections", [])
    edge_terminals = graph_payload.get("edge_terminals", [])
    node_component_index: dict[str, int] = {}
    edge_lookup = {str(edge.get("id", "")): edge for edge in edges}
    for idx, component in enumerate(edge_components):
        for item_id in component:
            edge = edge_lookup.get(str(item_id))
            if edge is None:
                node_component_index[str(item_id)] = idx
                continue
            node_component_index[str(edge.get("source", ""))] = idx
            node_component_index[str(edge.get("target", ""))] = idx
    edge_component_index: dict[str, int] = {}
    for idx, component in enumerate(edge_components):
        for edge_id in component:
            edge_component_index[str(edge_id)] = idx

    for node in nodes:
        graph.add_node(node["id"], **node)
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], **edge)

    edge_graph = nx.Graph()
    for edge in edges:
        edge_id = str(edge.get("id", ""))
        if edge_id.startswith("attach_edge::"):
            continue
        edge_graph.add_node(edge_id, **edge)
    for item in edge_connections:
        source_edge_id = str(item.get("source_edge_id", ""))
        target_edge_id = str(item.get("target_edge_id", ""))
        if source_edge_id in edge_graph and target_edge_id in edge_graph and source_edge_id != target_edge_id:
            edge_graph.add_edge(source_edge_id, target_edge_id, **item)

    raw_components = [sorted(component) for component in nx.connected_components(graph)] if graph.number_of_nodes() else []
    components = [list(map(str, component)) for component in edge_components] if edge_components else raw_components
    if edge_connections and edge_graph.number_of_nodes():
        articulation_points = sorted(nx.articulation_points(edge_graph))
    else:
        articulation_points = sorted(nx.articulation_points(graph)) if graph.number_of_nodes() else []

    low_degree_nodes: list[dict[str, Any]] = []
    for node_id, attrs in graph.nodes(data=True):
        degree = graph.degree(node_id)
        node_type = attrs.get("type", "unknown")
        if degree == 0 or (node_type == "endpoint" and degree == 0):
            low_degree_nodes.append(
                {
                    "node_id": node_id,
                    "type": node_type,
                    "degree": degree,
                    "reason": "isolated_node",
                }
            )

    terminal_exposed_components: set[int] = set()
    for item in edge_terminals:
        edge_id = str(item.get("edge_id", ""))
        component_idx = edge_component_index.get(edge_id)
        if component_idx is None:
            continue
        roles = [
            str((item.get("source_terminal") or {}).get("terminal_role", "")),
            str((item.get("destination_terminal") or {}).get("terminal_role", "")),
        ]
        if any(role in {"equipment_terminal", "connection_terminal"} for role in roles):
            terminal_exposed_components.add(component_idx)

    review_queue: list[dict[str, Any]] = []
    articulation_groups: dict[str, dict[str, Any]] = {}
    for node_id in articulation_points[:]:
        component_idx = edge_component_index.get(str(node_id))
        if component_idx is not None and component_idx not in terminal_exposed_components:
            continue
        group_key = (
            f"edge_component::{component_idx}"
            if component_idx is not None
            else (
                f"edge_component::{node_component_index[node_id]}"
                if node_id in node_component_index
                else f"node::{node_id}"
            )
        )
        group = articulation_groups.setdefault(
            group_key,
            {
                "category": "articulation_point",
                "group_key": group_key,
                "priority": "medium",
                "node_ids": [],
            },
        )
        group["node_ids"].append(node_id)
    review_queue.extend(articulation_groups.values())
    isolated_groups: dict[str, dict[str, Any]] = {}
    for item in low_degree_nodes:
        node_id = str(item["node_id"])
        group_key = (
            f"edge_component::{node_component_index[node_id]}"
            if node_id in node_component_index
            else f"type::{item['type']}"
        )
        group = isolated_groups.setdefault(
            group_key,
            {
                "category": "isolated_node",
                "group_key": group_key,
                "priority": "high",
                "node_ids": [],
            },
        )
        group["node_ids"].append(node_id)
    review_queue.extend(isolated_groups.values())

    unresolved_crossings: list[dict[str, Any]] = []
    for item in crossings:
        if str(item.get("classification", "")) != "unresolved":
            continue
        crossing_id = str(item.get("id", ""))
        component_idx = node_component_index.get(crossing_id)
        if component_idx is not None and component_idx not in terminal_exposed_components:
            continue
        unresolved_crossings.append(
            {
                "crossing_id": crossing_id,
                "branch_count": int(item.get("branch_count", 0)),
                "reasons": list(item.get("unresolved_reasons", [])),
            }
        )
        review_queue.append(
            {
                "category": "unresolved_crossing",
                "crossing_id": crossing_id,
                "priority": "high",
            }
        )

    unresolved_terminal_edges: list[dict[str, Any]] = []
    unresolved_terminal_groups: dict[str, dict[str, Any]] = {}
    for item in edge_terminals:
        if not bool(item.get("provisional_due_to_unresolved_terminal")):
            continue
        edge_id = str(item.get("edge_id", ""))
        component_idx = edge_component_index.get(edge_id)
        if component_idx is not None and component_idx not in terminal_exposed_components:
            continue
        payload = {
            "edge_id": edge_id,
            "source_node_id": str(item.get("source_node_id", "")),
            "destination_node_id": str(item.get("destination_node_id", "")),
            "source_terminal_role": str((item.get("source_terminal") or {}).get("terminal_role", "")),
            "destination_terminal_role": str((item.get("destination_terminal") or {}).get("terminal_role", "")),
            "edge_component_index": component_idx,
        }
        unresolved_terminal_edges.append(payload)
        group_key = (
            f"edge_component::{component_idx}"
            if component_idx is not None
            else f"edge::{edge_id}"
        )
        group = unresolved_terminal_groups.setdefault(
            group_key,
            {
                "category": "unresolved_terminal_edge",
                "group_key": group_key,
                "priority": "high",
                "edge_ids": [],
            },
        )
        group["edge_ids"].append(edge_id)

    review_queue.extend(unresolved_terminal_groups.values())
    grouped_unresolved_terminal_edges = [
        {
            "group_key": item["group_key"],
            "edge_ids": sorted(item["edge_ids"]),
        }
        for item in unresolved_terminal_groups.values()
    ]

    anomaly_report = {
        "image_id": image_id,
        "pass_type": "sheet",
        "connected_component_count": len(components),
        "largest_component_size": max((len(component) for component in components), default=0),
        "articulation_point_count": len(articulation_points),
        "isolated_node_count": len(isolated_groups),
        "unresolved_crossing_count": len(unresolved_crossings),
        "unresolved_terminal_edge_count": len(grouped_unresolved_terminal_edges),
        "articulation_points": articulation_points,
        "isolated_nodes": low_degree_nodes,
        "unresolved_crossings": unresolved_crossings,
        "unresolved_terminal_edges": grouped_unresolved_terminal_edges,
    }

    return {
        "anomaly_report": anomaly_report,
        "component_overlay_image": _draw_component_overlay(
            image_bgr,
            components=components,
            edges=edges,
            nodes=nodes,
        ),
        "review_queue": {
            "image_id": image_id,
            "pass_type": "sheet",
            "items": review_queue,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "connected_component_count": len(components),
            "articulation_point_count": len(articulation_points),
            "isolated_node_count": len(isolated_groups),
            "unresolved_crossing_count": len(unresolved_crossings),
            "unresolved_terminal_edge_count": len(grouped_unresolved_terminal_edges),
            "review_queue_count": len(review_queue),
            "source_artifacts": [
                "stage12_graph.json",
            ],
        },
    }
