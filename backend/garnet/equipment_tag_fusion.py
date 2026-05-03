from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from garnet.text_classify import classify_text_region, normalize_text_token


def run_equipment_tag_fusion_stage(
    *,
    image_id: str,
    object_regions: list[dict[str, Any]],
    image_bgr: np.ndarray | None = None,
    max_distance_px: float = 60.0,
) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, region in enumerate(object_regions, start=1):
        text_class = classify_text_region(region)
        text = str(region.get("text", "")).strip()
        payload = {
            "id": f"equipment_tag_{idx:06d}",
            "source_region_id": region.get("id"),
            "text": text,
            "normalized_text": normalize_text_token(text),
            "semantic_class": "equipment_tag",
            "bbox": region.get("bbox"),
            "confidence": float(region.get("confidence", 0.0)),
        }
        if text_class == "equipment_tag" and text and region.get("bbox") is not None:
            accepted.append(payload)
        else:
            rejected.append({**payload, "semantic_class": text_class})

    if image_bgr is None:
        overlay = np.zeros((1, 1, 3), dtype=np.uint8)
    else:
        overlay = image_bgr.copy()
        for item in accepted:
            bbox = item["bbox"]
            cv2.rectangle(
                overlay,
                (int(bbox["x_min"]), int(bbox["y_min"])),
                (int(bbox["x_max"]), int(bbox["y_max"])),
                (0, 200, 0),
                2,
            )

    return {
        "equipment_tags_payload": {
            "image_id": image_id,
            "pass_type": "sheet",
            "equipment_tags": accepted,
            "rejected": rejected,
        },
        "overlay_image": overlay,
        "summary": {
            "image_id": image_id,
            "pass_type": "sheet",
            "candidate_text_region_count": len(object_regions),
            "matched_equipment_tag_count": len(accepted),
            "rejected_text_region_count": len(rejected),
            "max_distance_px": max_distance_px,
        },
    }

