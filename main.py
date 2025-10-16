"""Main FastAPI application for Homelab AI Services."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.api.routers import crawl

# Create FastAPI app
app = FastAPI(
    title="Homelab AI Services API",
    description="REST API wrapping common AI capabilities for homelab developers",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Include routers
app.include_router(crawl.router)


@app.get("/api")
async def root():
    """Root API endpoint with service information."""
    return {
        "name": "Homelab AI Services API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "crawl": "/api/crawl",
            "docs": "/api/docs",
            "health": "/api/health",
            "ready": "/api/ready",
        },
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/ready")
async def ready():
    """Readiness check endpoint."""
    return {"status": "ready", "services": {"crawl": "available"}}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
