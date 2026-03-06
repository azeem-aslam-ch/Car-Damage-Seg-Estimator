"""Render segmentation overlays on car images."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Distinct colours per damage class (BGR for OpenCV)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "dent": (255, 160, 0),       # orange
    "scratch": (0, 200, 255),    # cyan
    "peel_paint": (200, 0, 255), # purple
    "broken": (0, 60, 255),      # red-blue
}
DEFAULT_COLOR = (0, 255, 120)  # lime green


def _get_color(class_name: str) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def draw_overlay(
    image_path: str,
    metrics: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Draw segmentation masks + labels on image.

    Returns:
        (overlay_image_path, mask_preview_path)
    """
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_h, img_w = img.shape[:2]
    overlay = img.copy()
    mask_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    for det in metrics:
        cls_name = det["class_name"]
        color = _get_color(cls_name)
        bitmask = det.get("mask_bitmap")

        if bitmask is not None and bitmask.shape == (img_h, img_w):
            # Semi-transparent fill
            overlay[bitmask] = (
                overlay[bitmask] * 0.45 + np.array(color) * 0.55
            ).astype(np.uint8)
            mask_canvas[bitmask] = color

            # Contour outline
            contours, _ = cv2.findContours(
                bitmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Label near centroid
            ys, xs = np.where(bitmask)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                label = f"{cls_name} {det['confidence']:.2f}"
                severity = det.get("severity", "")
                if severity:
                    label += f" [{severity}]"
                _put_label(overlay, label, cx, cy, color)

        # Fallback: bounding box from polygon
        elif det.get("polygon"):
            pts = np.array(det["polygon"], dtype=np.int32)
            cv2.polylines(overlay, [pts], True, color, 2)

    uid = uuid.uuid4().hex[:8]
    overlay_path = os.path.join(settings.OUTPUT_DIR, f"overlay_{uid}.jpg")
    mask_path = os.path.join(settings.OUTPUT_DIR, f"mask_{uid}.jpg")

    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, mask_canvas)

    logger.info(f"Saved overlay: {overlay_path}, mask: {mask_path}")
    return overlay_path, mask_path


def _put_label(
    img: np.ndarray,
    text: str,
    cx: int,
    cy: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a background-filled label at a centroid."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1 = max(cx - tw // 2, 0)
    y1 = max(cy - th - baseline - 4, 0)
    x2 = min(x1 + tw + 4, img.shape[1])
    y2 = y1 + th + baseline + 4
    cv2.rectangle(img, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.putText(
        img, text, (x1 + 2, y2 - baseline - 2),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
