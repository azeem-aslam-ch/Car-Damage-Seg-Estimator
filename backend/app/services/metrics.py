"""Damage metrics computation from YOLOv8-seg detection results."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


def compute_metrics(
    results,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """
    Extract per-detection metrics from ultralytics Results.

    Returns a list of dicts with:
      - class_name
      - confidence
      - mask_area_px
      - image_area_px
      - damage_area_percent
      - polygon (list of [x, y] points, may be empty)
      - mask_bitmap (H×W bool array)
    """
    metrics: List[Dict[str, Any]] = []

    if results is None or len(results) == 0:
        return metrics

    result = results[0]  # single image inference
    img_h, img_w = result.orig_shape[:2]
    image_area_px = img_h * img_w

    # No detections
    if result.boxes is None or len(result.boxes) == 0:
        return metrics

    boxes = result.boxes
    masks = result.masks  # may be None if no seg data

    for idx, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        # Map class id → name (use model names or our config)
        if hasattr(result, "names") and result.names:
            cls_name = result.names.get(cls_id, class_names[cls_id] if cls_id < len(class_names) else "unknown")
        elif cls_id < len(class_names):
            cls_name = class_names[cls_id]
        else:
            cls_name = "unknown"

        conf = float(box.conf.item())

        # Extract mask
        mask_bitmap: Optional[np.ndarray] = None
        polygon: List[List[float]] = []
        mask_area_px = 0

        if masks is not None and idx < len(masks):
            mask_data = masks[idx]
            # masks.data → (N, H, W) float tensor
            bitmask = mask_data.data.cpu().numpy().squeeze().astype(bool)
            mask_bitmap = bitmask
            mask_area_px = int(bitmask.sum())

            # Polygon from xy attribute (list of (N,2) arrays)
            if masks.xy is not None and idx < len(masks.xy):
                polygon = masks.xy[idx].tolist()

        damage_area_percent = (mask_area_px / image_area_px) * 100 if image_area_px > 0 else 0.0

        metrics.append(
            {
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "mask_area_px": mask_area_px,
                "image_area_px": image_area_px,
                "damage_area_percent": round(damage_area_percent, 4),
                "polygon": polygon,
                "mask_bitmap": mask_bitmap,
            }
        )

    logger.info(f"Computed metrics for {len(metrics)} detections.")
    return metrics
