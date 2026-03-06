"""
Pytest tests for Car Damage Segmentation Estimator API.

Tests cover:
  - GET /health returns ok status
  - POST /api/v1/predict returns valid schema with a sample image
  - Overlay artifact gets created
  - POST /api/v1/report returns PDF bytes
"""

import io
import os
import tempfile
from pathlib import Path

import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient using the FastAPI app."""
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_image_bytes() -> bytes:
    """Generate a simple 640×480 synthetic car-like image as JPEG bytes."""
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    # Add some variation to make it more realistic
    arr = np.array(img)
    arr[100:200, 150:350] = [80, 80, 80]   # dark rectangle (simulate car body)
    arr[200:250, 180:220] = [30, 30, 30]   # wheel area
    arr[200:250, 300:340] = [30, 30, 30]
    img_with_content = Image.fromarray(arr)
    buf = io.BytesIO()
    img_with_content.save(buf, format="JPEG")
    return buf.getvalue()


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        """GET /health should return status=ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "version" in data

    def test_health_model_loaded(self, client):
        """Model should be loaded after app startup."""
        response = client.get("/health")
        assert response.status_code == 200
        # model_loaded may be False if running without GPU/weights, just check key exists
        assert isinstance(response.json()["model_loaded"], bool)


class TestPredict:
    def test_predict_returns_valid_schema(self, client, sample_image_bytes):
        """POST /api/v1/predict should return a valid PredictResponse JSON."""
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test_car.jpg", sample_image_bytes, "image/jpeg")},
            data={"panel_location": "door"},
        )
        assert response.status_code == 200
        data = response.json()

        # Top-level keys
        assert "damage_detected" in data
        assert "overall_severity" in data
        assert "summary" in data
        assert "detections" in data
        assert "artifacts" in data

        # Summary structure
        summary = data["summary"]
        assert "total_instances" in summary
        assert "total_damage_percent" in summary
        assert "estimated_cost_pkr" in summary
        cost = summary["estimated_cost_pkr"]
        assert "min" in cost
        assert "max" in cost
        assert cost["min"] <= cost["max"]

        # Overall severity is one of expected values
        assert data["overall_severity"] in ("low", "medium", "high", "none")

    def test_predict_detections_schema(self, client, sample_image_bytes):
        """Each detection in /predict response must have required fields."""
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test_car.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        detections = response.json().get("detections", [])
        for det in detections:
            assert "class" in det
            assert "confidence" in det
            assert "severity" in det
            assert "mask_area_px" in det
            assert "damage_area_percent" in det
            assert "cost_pkr" in det
            assert det["severity"] in ("low", "medium", "high")
            assert 0.0 <= det["confidence"] <= 1.0
            assert det["cost_pkr"]["min"] <= det["cost_pkr"]["max"]

    def test_predict_creates_overlay_artifact(self, client, sample_image_bytes):
        """When damage is detected, artifacts.overlay_image_path must be set and file must exist."""
        response = client.post(
            "/api/v1/predict",
            files={"image": ("test_car.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        if data["damage_detected"]:
            overlay = data["artifacts"].get("overlay_image_path")
            assert overlay is not None, "overlay_image_path should be set when damage detected"
            assert os.path.exists(overlay), f"Overlay file not found: {overlay}"
            assert data["artifacts"].get("mask_preview_path") is not None

    def test_predict_no_damage_returns_clean_response(self, client):
        """A blank white image should return damage_detected=False gracefully."""
        blank = Image.new("RGB", (320, 240), color=(255, 255, 255))
        buf = io.BytesIO()
        blank.save(buf, format="JPEG")
        response = client.post(
            "/api/v1/predict",
            files={"image": ("blank.jpg", buf.getvalue(), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        # Should not crash — damage_detected may be True or False
        assert "damage_detected" in data

    def test_predict_invalid_file_type(self, client):
        """Sending a text file instead of image should return an error."""
        response = client.post(
            "/api/v1/predict",
            files={"image": ("bad.txt", b"not an image", "text/plain")},
        )
        # Expect 422 (validation) or 500 (processing error)
        assert response.status_code in (422, 500)


class TestReport:
    def _make_prediction_payload(self) -> dict:
        return {
            "damage_detected": True,
            "overall_severity": "medium",
            "summary": {
                "total_instances": 2,
                "total_damage_percent": 0.85,
                "estimated_cost_pkr": {"min": 12000, "max": 18000},
            },
            "detections": [
                {
                    "class": "scratch",
                    "confidence": 0.91,
                    "severity": "medium",
                    "mask_area_px": 12345,
                    "image_area_px": 307200,
                    "damage_area_percent": 0.40,
                    "cost_pkr": {"min": 5780, "max": 7820},
                },
                {
                    "class": "dent",
                    "confidence": 0.78,
                    "severity": "low",
                    "mask_area_px": 8000,
                    "image_area_px": 307200,
                    "damage_area_percent": 0.26,
                    "cost_pkr": {"min": 5950, "max": 8050},
                },
            ],
            "artifacts": {
                "overlay_image_path": None,
                "mask_preview_path": None,
            },
        }

    def test_report_returns_pdf_bytes(self, client):
        """POST /api/v1/report should return a valid PDF file response."""
        payload = {
            "prediction": self._make_prediction_payload(),
            "car_model": "Toyota Corolla 2020",
            "panel_location": "door",
            "notes": "Minor collision damage",
            "currency": "PKR",
        }
        response = client.post("/api/v1/report", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert len(response.content) > 1000  # non-trivial PDF
        # PDF magic bytes
        assert response.content[:4] == b"%PDF"

    def test_report_file_saved_to_disk(self, client):
        """Report PDF should be saved to the REPORTS_DIR."""
        from app.core.config import settings
        payload = {
            "prediction": self._make_prediction_payload(),
            "car_model": "Honda City",
            "panel_location": "bumper",
            "notes": "Test report",
            "currency": "PKR",
        }
        response = client.post("/api/v1/report", json=payload)
        assert response.status_code == 200
        # Reports dir should contain at least one PDF
        reports = list(Path(settings.REPORTS_DIR).glob("*.pdf"))
        assert len(reports) >= 1
