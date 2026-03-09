from __future__ import annotations

from typing import Any

import networkx as nx


def _node_payload(cluster: dict[str, Any], *, reviewed_junction_ids: set[str], unresolved_junction_ids: set[str]) -> dict[str, Any]:
    node_type = cluster["kind"]
    review_state = "accepted"
    if cluster["kind"] == "junction":
        if cluster["id"] in unresolved_junction_ids:
            review_state = "unresolved"
        elif cluster["id"] in reviewed_junction_ids:
            review_state = "accepted"
        else:
            review_state = "provisional"
    return {
        "id": cluster["id"],
        "type": node_type,
        "position": {
            "x": cluster["centroid"]["x"],
            "y": cluster["centroid"]["y"],
        },
        "member_count": cluster.get("member_count", 0),
        "review_state": review_state,
    }


def run_pipe_graph_stage(
    *,
    image_id: str,
    node_clusters: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    confirmed_junctions: list[dict[str, Any]],
    unresolved_junctions: list[dict[str, Any]],
) -> dict[str, Any]:
    confirmed_ids = {str(item["id"]) for item in confirmed_junctions}
    unresolved_ids = {str(item["id"]) for item in unresolved_junctions}

    nodes = [
        _node_payload(
            cluster,
            reviewed_junction_ids=confirmed_ids,
            unresolved_junction_ids=unresolved_ids,
        )
        for cluster in node_clusters
    ]

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["id"], **node)

    graph_edges: list[dict[str, Any]] = []
    for edge in edges:
        src = edge["source"]
        dst = edge["target"]
        if src == dst:
            continue
        graph.add_edge(src, dst, id=edge["id"], pixel_length=edge.get("pixel_length", 0))
        graph_edges.append(
            {
                "id": edge["id"],
                "source": src,
                "target": dst,
                "pixel_length": edge.get("pixel_length", 0),
                "polyline": edge.get("polyline", []),
                "review_state": "provisional",
            }
        )

    return {
        "graph_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "nodes": nodes,
            "edges": graph_edges,
            "unresolved_junction_ids": sorted(unresolved_ids),
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "connected_component_count": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
            "unresolved_junction_count": len(unresolved_ids),
            "source_artifacts": [
                "stage9_node_clusters.json",
                "stage10_pipe_edges.json",
                "stage11_junctions.json",
            ],
        },
    }
