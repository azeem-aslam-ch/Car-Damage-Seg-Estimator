# 🚗 Segmentation-Based Car Damage Estimator

> Detect, segment, and cost-estimate car damage — powered by YOLOv8-seg + FastAPI + Streamlit.

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/model-YOLOv8--seg-green)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/frontend-Streamlit-FF4B4B)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/deploy-Docker--Compose-2496ED)](https://docs.docker.com/)

---

## 🎯 Features

| Feature | Detail |
|---|---|
| **Damage Classes** | dent · scratch · peel_paint · broken |
| **Output** | Segmentation masks, severity (low/medium/high), cost range (PKR) |
| **PDF Report** | Branded ReportLab report with overlay image |
| **Config-Driven** | Edit `config/severity_rules.yaml` and `config/pricing_table.yaml` without code changes |
| **Docker** | Single command deployment |

---

## 🚀 Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- 8 GB RAM minimum (16 GB recommended for model loading)

### 1. Clone / Download

```bash
cd car_damage_seg_estimator
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults work out-of-the-box)
```

### 3. One-command launch 🎉

```bash
docker-compose up --build
```

- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## 📁 Repository Structure

```
car_damage_seg_estimator/
├── frontend/
│   ├── app.py                    # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Pydantic models
│   │   ├── api/routes.py         # API endpoints
│   │   ├── core/config.py        # Settings
│   │   ├── core/logging.py       # Structured logging
│   │   └── services/
│   │       ├── model.py          # YOLOv8-seg singleton
│   │       ├── metrics.py        # Mask metrics
│   │       ├── severity.py       # Rule-based severity
│   │       ├── costing.py        # PKR cost estimator
│   │       ├── render.py         # OpenCV overlay renderer
│   │       └── pdf_report.py     # ReportLab PDF generator
│   ├── tests/test_api.py         # Pytest tests
│   ├── conftest.py
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   ├── prepare_dataset.py        # Dataset setup + synthetic demo
│   ├── train_seg.py              # YOLOv8-seg training
│   ├── evaluate.py               # mAP evaluation
│   ├── infer.py                  # CLI single-image inference
│   └── smoke_test.py             # End-to-end smoke test
├── config/
│   ├── severity_rules.yaml       # Severity thresholds (editable)
│   └── pricing_table.yaml        # PKR pricing (editable)
├── data/
│   ├── sample_images/
│   ├── outputs/                  # Generated overlays (volume)
│   ├── reports/                  # Generated PDFs (volume)
│   ├── weights/                  # Model weights (volume)
│   └── dataset/                  # Training data
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🔌 API Reference

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "version": "1.0.0"}
```

### `POST /api/v1/predict`
**Form data:** `image` (file), `panel_location` (str, optional)

**Response:**
```json
{
  "damage_detected": true,
  "overall_severity": "medium",
  "summary": {
    "total_instances": 2,
    "total_damage_percent": 0.84,
    "estimated_cost_pkr": {"min": 14000, "max": 21000}
  },
  "detections": [
    {
      "class": "scratch",
      "confidence": 0.91,
      "severity": "medium",
      "mask_area_px": 12345,
      "image_area_px": 307200,
      "damage_area_percent": 0.42,
      "cost_pkr": {"min": 5780, "max": 7820}
    }
  ],
  "artifacts": {
    "overlay_image_path": "data/outputs/overlay_abc123.jpg",
    "mask_preview_path": "data/outputs/mask_abc123.jpg"
  }
}
```

### `POST /api/v1/report`
**Body:** `ReportRequest` JSON  
**Returns:** `application/pdf` file download

---

## ⚙️ Configuration

### Severity Rules (`config/severity_rules.yaml`)
```yaml
scratch:
  low:   {max: 0.3}
  medium: {min: 0.3, max: 1.0}
  high:  {min: 1.0}
```

### Pricing Table (`config/pricing_table.yaml`)
```yaml
base_costs:
  dent: 7000
  scratch: 4000
  ...
severity_multipliers:
  low: 1.0
  medium: 1.7
  high: 2.7
```

---

## 🧠 Training Your Own Model

### Step 1: Prepare dataset
```bash
# Generate synthetic demo dataset
python scripts/prepare_dataset.py --synthetic-count 20

# Or bring your own YOLO-seg format dataset to data/dataset/
```

### Step 2: Train
```bash
python scripts/train_seg.py \
  --data data/dataset/dataset.yaml \
  --epochs 100 \
  --imgsz 1024 \
  --model yolov8s-seg.pt
```

### Step 3: Evaluate
```bash
python scripts/evaluate.py \
  --model data/weights/best.pt \
  --data data/dataset/dataset.yaml
```

### Step 4: Use custom weights
Place `best.pt` in `data/weights/` — the API auto-loads it on startup.

---

## 🧪 Testing

### Run pytest
```bash
cd backend
pip install -r requirements.txt
pytest tests/test_api.py -v
```

### Run smoke test (requires running API)
```bash
# Start API first:  docker-compose up
python scripts/smoke_test.py --url http://localhost:8000
```

---

## 🐳 Docker Services

| Service | Port | Description |
|---|---|---|
| `backend` | 8000 | FastAPI inference API |
| `frontend` | 8501 | Streamlit web UI |

### Volume mounts
| Volume | Description |
|---|---|
| `./data/outputs` | Overlay images |
| `./data/reports` | PDF reports |
| `./data/weights` | Model weights |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `Model not loaded` at startup | First-run downloads `yolov8s-seg.pt` (~24 MB). Wait for health check to pass. |
| `503 Service Unavailable` | Backend initialising. Retry after 30–60s. |
| `Overlay not visible` in UI | Check backend logs: `docker-compose logs backend` |
| PDF generation fails | Ensure `data/reports/` has write permission. |
| Out of memory | Reduce batch size or use CPU by setting `device=cpu` in training. |
| `Connection refused` on `localhost:8000` | Run `docker-compose ps` to check service status. |

---

## 📊 10 Example Use Cases

1. **Single dent — door** Upload door photo → `dent`, low severity → ~PKR 7,000–9,000
2. **Multi-scratch — hood** Upload scratched hood → 3× scratch instances → total ~PKR 15,000–22,000
3. **Paint peeling — roof** Upload flaking roof → `peel_paint` high severity → ~PKR 18,000–26,000
4. **Broken bumper** Upload cracked bumper → `broken` high severity → ~PKR 32,000–48,000
5. **Complex damage** Combination of dent + scratch + peel_paint → overall high → PDF report download
6. **Clean car** No damage detected → cost = 0, PDF shows clean bill
7. **Insurance assessment** User fills car model + notes → PDF for insurer
8. **Workshop quote comparison** Use `/predict` JSON for automated quote comparison
9. **Fleet monitoring** Batch inference via script: `python scripts/infer.py`
10. **Training custom model** Run `prepare_dataset.py` → `train_seg.py` → swap `best.pt`

---

## 📄 License

MIT License — built for educational and commercial use.

---

> ⚠️ **Disclaimer**: Cost estimates are AI-generated approximations based on configurable PKR pricing tables. Always confirm with a qualified auto repair assessor.
