from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def _extract_points(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask > 0)


def _cluster_points(points: np.ndarray, eps: float, min_samples: int, kind: str) -> list[dict[str, Any]]:
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points.astype(float))
    clusters: dict[int, list[tuple[int, int]]] = {}
    for point, label in zip(points, clustering.labels_):
        clusters.setdefault(int(label), []).append((int(point[0]), int(point[1])))

    results: list[dict[str, Any]] = []
    for cluster_idx, members in sorted(clusters.items()):
        coords = np.array(members, dtype=float)
        centroid_row, centroid_col = coords.mean(axis=0)
        results.append(
            {
                "id": f"{kind}_{cluster_idx}",
                "kind": kind,
                "centroid": {
                    "x": float(centroid_col),
                    "y": float(centroid_row),
                },
                "member_count": len(members),
                "members": [{"row": row, "col": col} for row, col in members],
            }
        )
    return results


def _draw_cluster_points(shape: tuple[int, int], clusters: list[dict[str, Any]]) -> np.ndarray:
    image = np.zeros(shape, dtype=np.uint8)
    for cluster in clusters:
        x = int(round(cluster["centroid"]["x"]))
        y = int(round(cluster["centroid"]["y"]))
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            cv2.circle(image, (x, y), 2, 255, -1)
    return image


def _draw_overlay(image_bgr: np.ndarray, endpoint_clusters: list[dict[str, Any]], junction_clusters: list[dict[str, Any]]) -> np.ndarray:
    overlay = image_bgr.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for cluster in endpoint_clusters:
        x = int(round(cluster["centroid"]["x"]))
        y = int(round(cluster["centroid"]["y"]))
        cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)
    for cluster in junction_clusters:
        x = int(round(cluster["centroid"]["x"]))
        y = int(round(cluster["centroid"]["y"]))
        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
    return overlay


def run_pipe_node_cluster_stage(
    *,
    image_bgr: np.ndarray,
    endpoint_mask: np.ndarray,
    junction_mask: np.ndarray,
    image_id: str,
    cluster_eps: float = 6.0,
    cluster_min_samples: int = 1,
) -> dict[str, Any]:
    endpoint_points = _extract_points(endpoint_mask)
    junction_points = _extract_points(junction_mask)

    endpoint_clusters = _cluster_points(endpoint_points, cluster_eps, cluster_min_samples, "endpoint")
    junction_clusters = _cluster_points(junction_points, cluster_eps, cluster_min_samples, "junction")

    return {
        "endpoint_cluster_image": _draw_cluster_points(endpoint_mask.shape, endpoint_clusters),
        "junction_cluster_image": _draw_cluster_points(junction_mask.shape, junction_clusters),
        "overlay_image": _draw_overlay(image_bgr, endpoint_clusters, junction_clusters),
        "clusters_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "clusters": endpoint_clusters + junction_clusters,
        },
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "endpoint_cluster_count": len(endpoint_clusters),
            "junction_cluster_count": len(junction_clusters),
            "raw_endpoint_count": int(len(endpoint_points)),
            "raw_junction_count": int(len(junction_points)),
            "cluster_eps": cluster_eps,
            "cluster_min_samples": cluster_min_samples,
            "source_artifacts": [
                "stage8_endpoints.png",
                "stage8_junctions.png",
            ],
        },
    }
