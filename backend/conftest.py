"""
pytest conftest.py — shared fixtures and configuration.
"""

import os
import sys
from pathlib import Path

# Ensure the backend app package is importable during tests
backend_root = Path(__file__).resolve().parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# Set environment variables before importing app
os.environ.setdefault("OUTPUT_DIR", "data/outputs")
os.environ.setdefault("REPORTS_DIR", "data/reports")
os.environ.setdefault("WEIGHTS_DIR", "data/weights")
os.environ.setdefault("LOG_LEVEL", "WARNING")  # reduce noise during tests

# Create required dirs
for d in ["data/outputs", "data/reports", "data/weights"]:
    Path(d).mkdir(parents=True, exist_ok=True)
