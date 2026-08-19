from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from presentation.api.exception_handlers import add_exception_handlers
from presentation.api.schemas import HealthResponse

from presentation.api.routers import events, attendees, webhooks
from infrastructure.di_container import get_container
from sqlalchemy import text
from infrastructure.queue.celery_app import celery_app

import asyncio
import httpx
import logging
from config import settings

container = get_container()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Any necessary init happens via Dependency Injector singletons if needed
    
    async def poll_inference_api():
        consecutive_failures = 0
        current_state = "online" 
        
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{settings.INFERENCE_API_URL.rstrip('/')}/health", timeout=5.0)
                    response.raise_for_status()
                    
                consecutive_failures = 0
                if current_state == "offline":
                    logger.info("Poller: Inference API is back online! Resuming Celery queue...")
                    await asyncio.to_thread(celery_app.control.add_consumer, 'celery', reply=True)
                    current_state = "online"
                    
            except Exception as e:
                consecutive_failures += 1
                if consecutive_failures >= 3 and current_state == "online":
                    logger.warning(f"Poller: Inference API health check failed {consecutive_failures} times! Pausing Celery queue...")
                    await asyncio.to_thread(celery_app.control.cancel_consumer, 'celery', reply=True)
                    current_state = "offline"
                    
            # Wait 10 seconds before next ping
            await asyncio.sleep(10)
            
    # Start the background task
    task = asyncio.create_task(poll_inference_api())
    
    yield
    
    # Clean up on shutdown
    task.cancel()

app = FastAPI(
    title="Eventsnap Main API (Orchestrator) - Clean",
    description="Handles Event Encodings, Background Celery Tasks, and Attendee Sorting using pgvector.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

add_exception_handlers(app)

app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(attendees.router, prefix="/api/attendees", tags=["Attendees"])
app.include_router(webhooks.router, prefix="/api")


@app.get("/api/tasks/{task_id}", tags=["Tasks"])
def get_task_status(task_id: str):
    """Checks the status of any Celery task via Use Case."""
    use_case = container.check_encoding_status_use_case()
    return use_case.execute(task_id)


@app.get("/", tags=["Health"], response_model=HealthResponse)
async def health_check():
    health_response = HealthResponse(
        status="ok", service="Eventsnap Main API", checks={}
    )

    # Check Postgres
    try:
        engine = container.db_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health_response.checks["postgres"] = "ok"
    except Exception as e:
        health_response.status = "degraded"
        health_response.checks["postgres"] = f"error: {str(e)}"

    # Check RabbitMQ (Celery Broker)
    try:
        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=1, timeout=2)
        health_response.checks["rabbitmq"] = "ok"
    except Exception as e:
        health_response.status = "degraded"
        health_response.checks["rabbitmq"] = f"error: {str(e)}"

    return health_response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app="presentation.api.main:app", host="0.0.0.0", port=8000, reload=True)
