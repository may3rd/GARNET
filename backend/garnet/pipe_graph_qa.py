from __future__ import annotations

from typing import Any

import networkx as nx


def run_pipe_graph_qa_stage(
    *,
    image_id: str,
    graph_payload: dict[str, Any],
) -> dict[str, Any]:
    graph = nx.Graph()
    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    crossings = graph_payload.get("crossings", [])
    edge_terminals = graph_payload.get("edge_terminals", [])

    for node in nodes:
        graph.add_node(node["id"], **node)
    for edge in edges:
        graph.add_edge(edge["source"], edge["target"], **edge)

    components = [sorted(component) for component in nx.connected_components(graph)] if graph.number_of_nodes() else []
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

    review_queue: list[dict[str, Any]] = []
    for node_id in articulation_points[:]:
        review_queue.append(
            {
                "category": "articulation_point",
                "node_id": node_id,
                "priority": "medium",
            }
        )
    for item in low_degree_nodes:
        review_queue.append(
            {
                "category": "isolated_node",
                "node_id": item["node_id"],
                "priority": "high",
            }
        )

    unresolved_crossings: list[dict[str, Any]] = []
    for item in crossings:
        if str(item.get("classification", "")) != "unresolved":
            continue
        unresolved_crossings.append(
            {
                "crossing_id": str(item.get("id", "")),
                "branch_count": int(item.get("branch_count", 0)),
                "reasons": list(item.get("unresolved_reasons", [])),
            }
        )
        review_queue.append(
            {
                "category": "unresolved_crossing",
                "crossing_id": str(item.get("id", "")),
                "priority": "high",
            }
        )

    unresolved_terminal_edges: list[dict[str, Any]] = []
    for item in edge_terminals:
        if not bool(item.get("provisional_due_to_unresolved_terminal")):
            continue
        unresolved_terminal_edges.append(
            {
                "edge_id": str(item.get("edge_id", "")),
                "source_node_id": str(item.get("source_node_id", "")),
                "destination_node_id": str(item.get("destination_node_id", "")),
                "source_terminal_role": str((item.get("source_terminal") or {}).get("terminal_role", "")),
                "destination_terminal_role": str((item.get("destination_terminal") or {}).get("terminal_role", "")),
            }
        )
        review_queue.append(
            {
                "category": "unresolved_terminal_edge",
                "edge_id": str(item.get("edge_id", "")),
                "priority": "high",
            }
        )

    anomaly_report = {
        "image_id": image_id,
        "pass_type": "sheet",
        "connected_component_count": len(components),
        "largest_component_size": max((len(component) for component in components), default=0),
        "articulation_point_count": len(articulation_points),
        "isolated_node_count": len(low_degree_nodes),
        "unresolved_crossing_count": len(unresolved_crossings),
        "unresolved_terminal_edge_count": len(unresolved_terminal_edges),
        "articulation_points": articulation_points,
        "isolated_nodes": low_degree_nodes,
        "unresolved_crossings": unresolved_crossings,
        "unresolved_terminal_edges": unresolved_terminal_edges,
    }

    return {
        "anomaly_report": anomaly_report,
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
            "isolated_node_count": len(low_degree_nodes),
            "unresolved_crossing_count": len(unresolved_crossings),
            "unresolved_terminal_edge_count": len(unresolved_terminal_edges),
            "review_queue_count": len(review_queue),
            "source_artifacts": [
                "stage12_graph.json",
            ],
        },
    }
