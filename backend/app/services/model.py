"""YOLOv8-seg model service — singleton pattern with fallback loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelService:
    """Singleton wrapper around a YOLOv8-seg model."""

    _model = None
    _loaded: bool = False

    def load(self) -> None:
        """Load or download the YOLOv8-seg model."""
        if self._loaded:
            return

        # Try custom weights first
        weights_path = Path(settings.MODEL_WEIGHTS_PATH)
        if weights_path.exists():
            logger.info(f"Loading custom weights from {weights_path}")
            model_path = str(weights_path)
        else:
            logger.warning(
                f"Custom weights not found at {weights_path}. "
                f"Falling back to pretrained {settings.PRETRAINED_WEIGHTS}."
            )
            model_path = settings.PRETRAINED_WEIGHTS

        try:
            from ultralytics import YOLO  # deferred import to avoid heavy load at module level
            self._model = YOLO(model_path)
            self._loaded = True
            logger.info(f"Model loaded: {model_path}")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, image_path: str) -> list:
        """
        Run YOLOv8-seg inference on an image file.

        Returns a list of ultralytics Results objects.
        """
        if not self._loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        results = self._model.predict(
            source=image_path,
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            task="segment",
            verbose=False,
        )
        return results


# Module-level singleton
model_service = ModelService()
