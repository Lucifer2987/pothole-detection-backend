"""
FastAPI application entry point.

Startup lifecycle:
    1. Pre-load the YOLOv8 model (avoids cold start on first request)
    2. All routers are registered under /api/v1

Run locally:
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import cameras, incidents, potholes, reports, scan, upload
from src.database.db import ping_db
from src.detection.infer import get_detector


# ---------------------------------------------------------------------------
# Lifespan  (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load the YOLO model on startup so the first request isn't slow.
    """
    print("[startup] Loading YOLOv8 model...")
    try:
        get_detector()
        print("[startup] Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: {e}")
        print("[startup] Model will be loaded on first inference request.")

    db_ok = ping_db()
    print(f"[startup] Database reachable: {db_ok}")

    yield
    # Shutdown cleanup (nothing needed for now)
    print("[shutdown] Pothole Detection API shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pothole Detection API",
    description=(
        "AI-powered pothole and road-defect detection backend. "
        "Detects potholes/cracks from video/image footage using YOLOv8, "
        "tracks defects over time, and exposes REST endpoints for "
        "querying results and managing maintenance incidents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS  (open for prototype — restrict in production)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(cameras.router,  prefix=API_PREFIX)
app.include_router(potholes.router, prefix=API_PREFIX)
app.include_router(incidents.router, prefix=API_PREFIX)
app.include_router(scan.router,     prefix=API_PREFIX)
app.include_router(upload.router,   prefix=API_PREFIX)
app.include_router(reports.router,  prefix=API_PREFIX)


# ---------------------------------------------------------------------------
# Root / health endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    """API root — returns basic info."""
    return {
        "name": "Pothole Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["health"])
def health():
    """Health check — verifies DB connectivity and model status."""
    db_ok = ping_db()
    try:
        det = get_detector()
        model_loaded = det._model is not None
    except Exception:
        model_loaded = False

    return {
        "status": "ok" if db_ok else "degraded",
        "db_reachable": db_ok,
        "model_loaded": model_loaded,
    }
