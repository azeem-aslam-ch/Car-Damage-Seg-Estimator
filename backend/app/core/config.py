"""Application configuration loaded from environment variables and YAML files."""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[3]  # car_damage_seg_estimator/


class Settings:
    # Model
    MODEL_WEIGHTS_PATH: str = os.getenv("MODEL_WEIGHTS_PATH", "data/weights/best.pt")
    PRETRAINED_WEIGHTS: str = os.getenv("PRETRAINED_WEIGHTS", "yolov8s-seg.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # Storage
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "data/outputs")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "data/reports")
    WEIGHTS_DIR: str = os.getenv("WEIGHTS_DIR", "data/weights")

    # Config YAMLs (resolved at app root / config/)
    SEVERITY_RULES_PATH: str = os.getenv(
        "SEVERITY_RULES_PATH", str(ROOT_DIR / "config" / "severity_rules.yaml")
    )
    PRICING_TABLE_PATH: str = os.getenv(
        "PRICING_TABLE_PATH", str(ROOT_DIR / "config" / "pricing_table.yaml")
    )

    # API
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Class names matching YOLO training labels
    CLASS_NAMES: List[str] = ["dent", "scratch", "peel_paint", "broken"]


settings = Settings()
