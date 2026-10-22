"""FastAPI Entrypoint for Nepal Multi-Hazard Early Warning System (Himalayan-MHEWS)."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as mhews_router
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("nepal_mhews")

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


@app.get("/health")
def health_check():
    """Service liveness and readiness probe."""
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
    }
