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
    crossing_candidates: list[dict[str, Any]] | None = None,
    equipment_attachments: list[dict[str, Any]] | None = None,
    text_attachments: list[dict[str, Any]] | None = None,
    instrument_tag_attachments: list[dict[str, Any]] | None = None,
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

    accepted_text_attachments = text_attachments or []
    accepted_instrument_tag_attachments = instrument_tag_attachments or []
    edge_texts: dict[str, list[dict[str, Any]]] = {}
    for attachment in accepted_text_attachments:
        edge_id = attachment.get("edge_id")
        if edge_id is None:
            continue
        edge_texts.setdefault(str(edge_id), []).append(
            {
                "region_id": attachment["region_id"],
                "text": attachment["text"],
                "normalized_text": attachment.get("normalized_text", ""),
            }
        )
    edge_instrument_tags: dict[str, list[dict[str, Any]]] = {}
    for attachment in accepted_instrument_tag_attachments:
        edge_id = attachment.get("edge_id")
        if edge_id is None:
            continue
        edge_instrument_tags.setdefault(str(edge_id), []).append(
            {
                "region_id": attachment["region_id"],
                "text": attachment["text"],
                "normalized_text": attachment.get("normalized_text", ""),
            }
        )

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
                "line_texts": edge_texts.get(str(edge["id"]), []),
                "instrument_tags": edge_instrument_tags.get(str(edge["id"]), []),
            }
        )

    accepted_attachments = equipment_attachments or []
    for attachment in accepted_attachments:
        equipment_node_id = f"equipment::{attachment['det_id']}"
        attachment_node_id = f"attach::{attachment['det_id']}"
        nearest_point = attachment.get("nearest_point_xy") or (None, None)
        graph.add_node(
            equipment_node_id,
            id=equipment_node_id,
            type=attachment["class_name"],
            position={
                "x": float((attachment["bbox"][0] + attachment["bbox"][2]) / 2),
                "y": float((attachment["bbox"][1] + attachment["bbox"][3]) / 2),
            },
            member_count=1,
            review_state="provisional",
        )
        graph.add_node(
            attachment_node_id,
            id=attachment_node_id,
            type="equipment_attachment",
            position={
                "x": float(nearest_point[0]) if nearest_point and nearest_point[0] is not None else 0.0,
                "y": float(nearest_point[1]) if nearest_point and nearest_point[1] is not None else 0.0,
            },
            member_count=1,
            review_state="provisional",
        )
        graph.add_edge(equipment_node_id, attachment_node_id, id=f"attach_edge::{attachment['det_id']}", pixel_length=0)
        graph_edges.append(
            {
                "id": f"attach_edge::{attachment['det_id']}",
                "source": equipment_node_id,
                "target": attachment_node_id,
                "pixel_length": 0,
                "polyline": [],
                "review_state": "provisional",
            }
        )

    serialized_nodes = [
        {
            "id": node_id,
            "type": attrs.get("type", "unknown"),
            "position": attrs.get("position"),
            "member_count": attrs.get("member_count", 0),
            "review_state": attrs.get("review_state", "provisional"),
        }
        for node_id, attrs in graph.nodes(data=True)
    ]

    return {
        "graph_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "nodes": serialized_nodes,
            "edges": graph_edges,
            "unresolved_junction_ids": sorted(unresolved_ids),
            "crossings": crossing_candidates or [],
            "equipment_attachments": accepted_attachments,
            "text_attachments": accepted_text_attachments,
            "instrument_tag_attachments": accepted_instrument_tag_attachments,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "connected_component_count": nx.number_connected_components(graph) if graph.number_of_nodes() else 0,
            "unresolved_junction_count": len(unresolved_ids),
            "crossing_candidate_count": len(crossing_candidates or []),
            "accepted_attachment_count": len(accepted_attachments),
            "accepted_text_attachment_count": len(accepted_text_attachments),
            "accepted_instrument_tag_attachment_count": len(accepted_instrument_tag_attachments),
            "source_artifacts": [
                "stage9_node_clusters.json",
                "stage10_pipe_edges.json",
                "stage10_crossing_resolution.json",
                "stage11_junctions.json",
            ],
        },
    }
