FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Install PyTorch CPU-only FIRST ──────────────────────────────────────────
# This prevents ultralytics from pulling 1.5 GB of NVIDIA CUDA wheels.
# The system will use CPU inference; to enable GPU, remove this step and
# let ultralytics resolve torch from its own dependencies.
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create output directories
RUN mkdir -p data/outputs data/reports data/weights data/sample_images

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
