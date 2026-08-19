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

import os
import httpx2
from infrastructure.config import settings
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    main_api_url = settings.MAIN_API_URL
    webhook_secret = settings.WEBHOOK_SECRET
    
    import asyncio
    
    async def fire_online_webhook_with_retry():
        while True:
            try:
                logger.info(f"Firing online webhook to {main_api_url}...")
                async with httpx2.AsyncClient() as client:
                    response = await client.post(
                        f"{main_api_url.rstrip('/')}/api/webhooks/inference-status",
                        json={"status": "online"},
                        headers={"Authorization": f"Bearer {webhook_secret}"},
                        timeout=5.0
                    )
                    response.raise_for_status()
                logger.info("Successfully sent online webhook! Celery is unpaused.")
                break  # Exit loop on success
            except Exception as e:
                logger.error(f"Failed to send online webhook: {e}. Retrying in 5 seconds...")
                await asyncio.sleep(5)

    task = asyncio.create_task(fire_online_webhook_with_retry())
        
    yield

    task.cancel()

app = FastAPI(
    title="Eventsnap Inference API",
    description="Local testing server for HF Endpoint Handler",
    lifespan=lifespan,
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
