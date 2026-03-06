"""Rule-based severity estimation driven by severity_rules.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Severity priority for overall aggregation
_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "none": -1}


def _load_rules() -> Dict:
    path = Path(settings.SEVERITY_RULES_PATH)
    if not path.exists():
        logger.warning(f"Severity rules not found at {path}. Using defaults.")
        return _default_rules()
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _default_rules() -> Dict:
    return {
        "scratch": {"low": {"max": 0.3}, "medium": {"min": 0.3, "max": 1.0}, "high": {"min": 1.0}},
        "dent": {"low": {"max": 0.5}, "medium": {"min": 0.5, "max": 1.5}, "high": {"min": 1.5}},
        "peel_paint": {"low": {"max": 0.4}, "medium": {"min": 0.4, "max": 1.2}, "high": {"min": 1.2}},
        "broken": {"low": {"max": 0.2}, "medium": {"min": 0.2, "max": 0.8}, "high": {"min": 0.8}},
    }


def classify_severity(class_name: str, damage_area_percent: float) -> str:
    """
    Classify severity for a single detection.

    Returns 'low', 'medium', or 'high'.
    Unknown classes default to area-based generic thresholds.
    """
    rules = _load_rules()
    class_rules = rules.get(class_name, None)

    if class_rules is None:
        # Generic fallback
        if damage_area_percent < 0.3:
            return "low"
        elif damage_area_percent < 1.0:
            return "medium"
        else:
            return "high"

    for level in ["high", "medium", "low"]:
        level_rule = class_rules.get(level, {})
        min_val = level_rule.get("min", float("-inf"))
        max_val = level_rule.get("max", float("inf"))
        if min_val <= damage_area_percent < max_val or (
            level == "high" and damage_area_percent >= min_val
        ):
            return level

    return "low"


def overall_severity(severities: List[str]) -> str:
    """Return the worst severity level from a list of individual severities."""
    if not severities:
        return "none"
    return max(severities, key=lambda s: _SEVERITY_RANK.get(s, 0))
