"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.core.config import settings
from app.core.logging import get_logger
from app.api.routes import router
from app.services.model import model_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Starting Car Damage Segmentation Estimator API...")
    model_service.load()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Car Damage Segmentation Estimator",
    description="Detect, segment, and estimate repair costs for car damage using YOLOv8-seg.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for outputs
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.REPORTS_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

# API routes
app.include_router(router)
