from datetime import datetime, timezone
from fastapi import APIRouter
from claims.api.dependencies import registry
from claims.config import settings
from claims.monitoring.metrics import get_all

router = APIRouter(tags=["health"])

@router.get("/health")
def get_health() -> dict:
    """Check API health and model loading status without raising 503."""
    status_str = "ok" if registry.is_loaded else "degraded"
    return {
        "status": status_str,
        "models_loaded": registry.is_loaded,
        "version": settings.API_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@router.get("/metrics")
def get_metrics() -> dict:
    """Return basic Prometheus-style in-memory counters."""
    return get_all()