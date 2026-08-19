import logging
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.gzip import GZipMiddleware

from infrastructure.di_container import get_container
from presentation.api.routes import inference
from presentation.api.exception_handlers import add_exception_handlers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Eventsnap Inference API",
    description="Local testing server for HF Endpoint Handler",
)

# Initialize and attach DI container
container = get_container()
setattr(app, "container", container)

app.add_middleware(GZipMiddleware, minimum_size=1000)

add_exception_handlers(app)

app.include_router(inference.router)

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = 5000
    logger.info("=" * 50)
    logger.info("🚀 Local Eventsnap Inference Server (FastAPI) is Running!")
    logger.info(f"Send HTTP POST requests to: http://0.0.0.0:{port}/")
    logger.info(
        f"Auto-generated interactive docs available at: http://0.0.0.0:{port}/docs"
    )
    logger.info("=" * 50)
    uvicorn.run(app="presentation.api.main:app", host="0.0.0.0", port=port)
