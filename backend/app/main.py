"""FastAPI Entrypoint for Nepal Multi-Hazard Early Warning System (Himalayan-MHEWS)."""

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.api.routes import router as mhews_router
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("nepal_mhews")
DASHBOARD_HTML_PATH = Path(__file__).resolve().parent / "static" / "dashboard.html"

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Tier A Static Susceptibility Mapping (30m) + Tier B Dynamic Precursor & "
        "Cascade Detection for Trishuli/Langtang and Budhi Gandaki watersheds."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mhews_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_interactive_dashboard():
    """Serve the interactive 3D Himalayan-MHEWS Dashboard at the root URL."""
    if DASHBOARD_HTML_PATH.exists():
        return HTMLResponse(content=DASHBOARD_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Nepal-MHEWS API Running. Visit /docs</h1>")


@app.get("/health")
def health_check():
    """Service liveness and readiness probe."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
    }
