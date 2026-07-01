from claims.config import settings
from claims.api.dependencies import load_all_models
from claims.config import ensure_directories
from claims.monitoring.metrics import record_startup
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from claims.api.middleware import RequestLoggingMiddleware
from claims.api.routes import claims, analytics, health

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info("Starting API server...")
    ensure_directories()
    load_all_models()
    record_startup()
    try:
        yield
    except Exception as e:
        logger.error(f"Error starting API server: {e}", exc_info=True)
        raise
    # Shutdown phase
    logger.info("Shutting down API server...")

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan
)

app.add_middleware(RequestLoggingMiddleware)
# CORS middleware added to allow all origins (needed for the HTML dashboard to call the API)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Include routers both at root and with /api prefix for maximum client compatibility
app.include_router(claims.router)
app.include_router(claims.router, prefix="/api")

app.include_router(analytics.router)
app.include_router(analytics.router, prefix="/api")

app.include_router(health.router)
app.include_router(health.router, prefix="/api")

@app.get("/")
def read_root():
    """Return a welcome message and a link to /docs."""
    return {"message": "Welcome to the Medical Claims Intelligence API", "docs": "/docs"}
