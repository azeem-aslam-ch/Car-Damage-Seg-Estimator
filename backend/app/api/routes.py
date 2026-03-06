"""FastAPI API routes for prediction and report generation."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas import (
    Artifacts,
    CostRange,
    DetectionResult,
    HealthResponse,
    PredictResponse,
    ReportRequest,
    Summary,
)
from app.services.costing import aggregate_costs, estimate_cost
from app.services.metrics import compute_metrics
from app.services.model import model_service
from app.services.pdf_report import generate_pdf
from app.services.render import draw_overlay
from app.services.severity import classify_severity, overall_severity

logger = get_logger(__name__)

router = APIRouter()


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
    )


# ─── Predict ──────────────────────────────────────────────────────────────────

@router.post("/api/v1/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(
    image: UploadFile = File(..., description="Car image file (jpg/png)"),
    panel_location: Optional[str] = Form("unknown", description="bumper/door/fender/hood/roof/unknown"),
):
    """
    Run YOLOv8-seg on an uploaded car image.

    Returns detections with masks, severity, cost estimates, and overlay artifact paths.
    """
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Save upload to temp file
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Inference
        results = model_service.predict(tmp_path)

        # Metrics
        raw_metrics = compute_metrics(results, settings.CLASS_NAMES)

        if not raw_metrics:
            # No detections
            return PredictResponse(
                damage_detected=False,
                overall_severity="none",
                summary=Summary(
                    total_instances=0,
                    total_damage_percent=0.0,
                    estimated_cost_pkr=CostRange(min=0, max=0),
                ),
                detections=[],
                artifacts=Artifacts(),
            )

        # Severity + cost per detection
        detection_results = []
        cost_pairs = []
        severity_list = []
        total_damage_pct = 0.0

        for m in raw_metrics:
            sev = classify_severity(m["class_name"], m["damage_area_percent"])
            min_c, max_c = estimate_cost(m["class_name"], sev, panel_location or "unknown")
            cost_pairs.append((min_c, max_c))
            severity_list.append(sev)
            total_damage_pct += m["damage_area_percent"]

            # Attach severity to metrics dict for the overlay renderer
            m["severity"] = sev

            detection_results.append(
                DetectionResult(
                    **{
                        "class": m["class_name"],
                        "confidence": m["confidence"],
                        "severity": sev,
                        "mask_area_px": m["mask_area_px"],
                        "image_area_px": m["image_area_px"],
                        "damage_area_percent": m["damage_area_percent"],
                        "cost_pkr": CostRange(min=min_c, max=max_c),
                    }
                )
            )

        overall_sev = overall_severity(severity_list)
        total_min, total_max = aggregate_costs(cost_pairs)

        # Render overlay
        overlay_path, mask_path = draw_overlay(tmp_path, raw_metrics)

        return PredictResponse(
            damage_detected=True,
            overall_severity=overall_sev,
            summary=Summary(
                total_instances=len(detection_results),
                total_damage_percent=round(total_damage_pct, 4),
                estimated_cost_pkr=CostRange(min=total_min, max=total_max),
            ),
            detections=detection_results,
            artifacts=Artifacts(
                overlay_image_path=overlay_path,
                mask_preview_path=mask_path,
            ),
        )

    except Exception as exc:
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── Report ───────────────────────────────────────────────────────────────────

@router.post("/api/v1/report", tags=["Report"])
async def generate_report(payload: ReportRequest):
    """
    Generate a PDF damage assessment report.

    Returns a downloadable PDF file.
    """
    prediction_dict = payload.prediction.model_dump(by_alias=True)

    overlay_path = prediction_dict.get("artifacts", {}).get("overlay_image_path")

    try:
        pdf_bytes = generate_pdf(
            prediction=prediction_dict,
            car_model=payload.car_model,
            panel_location=payload.panel_location,
            notes=payload.notes,
            overlay_image_path=overlay_path,
        )
    except Exception as exc:
        logger.error(f"PDF generation failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    # Save to disk and return as file
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    report_path = os.path.join(settings.REPORTS_DIR, f"report_{uid}.pdf")
    with open(report_path, "wb") as f:
        f.write(pdf_bytes)

    return FileResponse(
        report_path,
        media_type="application/pdf",
        filename=f"car_damage_report_{uid}.pdf",
        headers={"Content-Disposition": f"attachment; filename=car_damage_report_{uid}.pdf"},
    )
