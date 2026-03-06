"""Cost estimation service driven by pricing_table.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _load_pricing() -> Dict:
    path = Path(settings.PRICING_TABLE_PATH)
    if not path.exists():
        logger.warning(f"Pricing table not found at {path}. Using defaults.")
        return _default_pricing()
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _default_pricing() -> Dict:
    return {
        "base_costs": {"dent": 7000, "scratch": 4000, "peel_paint": 6000, "broken": 12000},
        "severity_multipliers": {"low": 1.0, "medium": 1.7, "high": 2.7},
        "location_multipliers": {
            "bumper": 1.2, "door": 1.1, "fender": 1.15,
            "hood": 1.25, "roof": 1.3, "unknown": 1.0,
        },
        "range_factor": 0.15,
    }


def estimate_cost(
    class_name: str,
    severity: str,
    location: str = "unknown",
) -> Tuple[float, float]:
    """
    Compute min/max PKR repair cost for a single detection.

    Returns (min_cost, max_cost).
    """
    pricing = _load_pricing()

    base = pricing["base_costs"].get(class_name, 5000)
    sev_mult = pricing["severity_multipliers"].get(severity, 1.0)
    loc_mult = pricing["location_multipliers"].get(location, 1.0)
    range_factor = pricing.get("range_factor", 0.15)

    mid_cost = base * sev_mult * loc_mult
    min_cost = round(mid_cost * (1 - range_factor))
    max_cost = round(mid_cost * (1 + range_factor))

    return float(min_cost), float(max_cost)


def aggregate_costs(cost_pairs: list) -> Tuple[float, float]:
    """Sum all (min, max) cost pairs into total (min, max)."""
    total_min = sum(pair[0] for pair in cost_pairs)
    total_max = sum(pair[1] for pair in cost_pairs)
    return round(total_min), round(total_max)
