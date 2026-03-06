"""Pydantic schemas for API request/response validation."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class CostRange(BaseModel):
    min: float = Field(..., description="Minimum repair cost in PKR")
    max: float = Field(..., description="Maximum repair cost in PKR")


class DetectionResult(BaseModel):
    class_name: str = Field(..., alias="class", description="Damage class label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str = Field(..., description="low | medium | high")
    mask_area_px: int = Field(..., description="Number of pixels in damage mask")
    image_area_px: int = Field(..., description="Total image pixels")
    damage_area_percent: float = Field(..., description="Mask area as % of image")
    cost_pkr: CostRange

    model_config = {"populate_by_name": True}


class Summary(BaseModel):
    total_instances: int
    total_damage_percent: float
    estimated_cost_pkr: CostRange


class Artifacts(BaseModel):
    overlay_image_path: Optional[str] = None
    mask_preview_path: Optional[str] = None


class PredictResponse(BaseModel):
    damage_detected: bool
    overall_severity: str = Field(..., description="low | medium | high | none")
    summary: Summary
    detections: List[DetectionResult]
    artifacts: Artifacts


class ReportRequest(BaseModel):
    prediction: PredictResponse
    car_model: Optional[str] = None
    panel_location: Optional[str] = "unknown"
    notes: Optional[str] = None
    currency: str = "PKR"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"
