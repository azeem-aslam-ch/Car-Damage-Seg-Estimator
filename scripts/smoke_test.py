"""
smoke_test.py — End-to-end smoke test for the Car Damage Estimator API.

Usage:
    python scripts/smoke_test.py [--url http://localhost:8000] [--image data/sample_images/sample_car.jpg]

This script:
1. Checks /health
2. Sends an image to /predict
3. Verifies overlay + mask artifact files exist
4. Requests a PDF report
5. Verifies PDF is valid
"""

import argparse
import io
import os
import sys
from pathlib import Path

import requests
from PIL import Image


def create_test_image() -> bytes:
    """Create a synthetic car image for testing if no real image is available."""
    import numpy as np
    arr = np.ones((480, 640, 3), dtype=np.uint8) * 150
    arr[100:200, 150:350] = [80, 80, 80]
    arr[200:250, 180:220] = [30, 30, 30]
    arr[200:250, 300:340] = [30, 30, 30]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def run_smoke_test(base_url: str, image_path: str):
    print(f"\n🧪 Smoke Test — API: {base_url}")
    print("=" * 60)
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))
            failed += 1

    # ── 1. Health check ──────────────────────────────────────────────────────
    print("\n[1/4] Health Check")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        check("Status 200", r.status_code == 200, f"Got {r.status_code}")
        data = r.json()
        check("status == ok", data.get("status") == "ok", str(data))
        check("model_loaded key present", "model_loaded" in data)
    except Exception as exc:
        check("Health endpoint reachable", False, str(exc))

    # ── 2. Predict ───────────────────────────────────────────────────────────
    print("\n[2/4] Predict Endpoint")
    prediction = None
    try:
        img_path = Path(image_path)
        if img_path.exists():
            image_bytes = img_path.read_bytes()
            fname = img_path.name
        else:
            print(f"   ⚠️  Sample not found at {image_path}. Using synthetic image.")
            image_bytes = create_test_image()
            fname = "synthetic_car.jpg"

        r = requests.post(
            f"{base_url}/api/v1/predict",
            files={"image": (fname, image_bytes, "image/jpeg")},
            data={"panel_location": "door"},
            timeout=120,
        )
        check("Status 200", r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}")
        if r.status_code == 200:
            prediction = r.json()
            check("damage_detected key", "damage_detected" in prediction)
            check("overall_severity key", "overall_severity" in prediction)
            check("summary key", "summary" in prediction)
            check("detections key", "detections" in prediction)
            check("artifacts key", "artifacts" in prediction)

            # Verify artifact files
            overlay_path = prediction.get("artifacts", {}).get("overlay_image_path")
            mask_path = prediction.get("artifacts", {}).get("mask_preview_path")
            if prediction.get("damage_detected"):
                check("Overlay file exists", overlay_path and os.path.exists(overlay_path), str(overlay_path))
                check("Mask file exists", mask_path and os.path.exists(mask_path), str(mask_path))
            else:
                print("   ℹ️  No damage detected — skipping artifact file check.")
    except Exception as exc:
        check("Predict request succeeded", False, str(exc))

    # ── 3. Report ────────────────────────────────────────────────────────────
    print("\n[3/4] Report Endpoint")
    if prediction is not None:
        try:
            payload = {
                "prediction": prediction,
                "car_model": "Smoke Test Vehicle",
                "panel_location": "door",
                "notes": "Automated smoke test",
                "currency": "PKR",
            }
            r = requests.post(f"{base_url}/api/v1/report", json=payload, timeout=60)
            check("Status 200", r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}")
            check("Content-Type PDF", "application/pdf" in r.headers.get("content-type", ""))
            check("PDF size > 1KB", len(r.content) > 1024, f"{len(r.content)} bytes")
            check("PDF magic bytes", r.content[:4] == b"%PDF", f"Got {r.content[:4]}")

            # Save to disk
            out_path = Path("data/reports/smoke_test_report.pdf")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(r.content)
            check("PDF saved to disk", out_path.exists())
        except Exception as exc:
            check("Report request succeeded", False, str(exc))
    else:
        print("   ⚠️  Skipping report test (predict failed).")

    # ── 4. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed  |  {failed} failed")
    if failed == 0:
        print("🎉 All smoke tests PASSED!")
    else:
        print("⚠️  Some tests failed. Check the API logs for details.")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Car Damage Estimator API Smoke Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend API base URL")
    parser.add_argument(
        "--image", default="data/sample_images/sample_car.jpg", help="Sample image path"
    )
    args = parser.parse_args()

    ok = run_smoke_test(args.url, args.image)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
