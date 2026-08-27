from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from presentation.api.exception_handlers import add_exception_handlers
from presentation.api.schemas import HealthResponse, TaskStatusResponse
from presentation.api.routers import events, attendees, images
from infrastructure.di_container import get_container
from infrastructure.queue.celery_app import celery_app
from sqlalchemy import text
from config.app import app_settings
from contextlib import asynccontextmanager
import logging

container = get_container()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Eventsnap Main API (Orchestrator) - Clean",
    description="Handles Event Encodings, Background Celery Tasks, and Attendee Sorting using pgvector.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

cors_origins = [
    origin.strip() for origin in app_settings.CORS_ORIGINS.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

add_exception_handlers(app)

app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(attendees.router, prefix="/api/attendees", tags=["Attendees"])
app.include_router(images.router, prefix="/api/images", tags=["Images"])


@app.get("/api/tasks/stream", tags=["Tasks"])
async def stream_task_status(request: Request, taskId: str):
    """Streams the status of a Celery task using SSE."""
    from sse_starlette.sse import EventSourceResponse
    import asyncio

    use_case = container.check_encoding_status_use_case()

    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            status = await asyncio.to_thread(use_case.execute, taskId)

            import json
            import dataclasses

            yield {"event": "message", "data": json.dumps(dataclasses.asdict(status))}

            if status.state in ["SUCCESS", "FAILURE", "REVOKED"]:
                yield {"event": "done", "data": "done"}
                break

            await asyncio.sleep(1)

    return EventSourceResponse(event_generator())


@app.get("/api/tasks/{task_id}", tags=["Tasks"], response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """Checks the status of any Celery task via Use Case."""
    use_case = container.check_encoding_status_use_case()
    dto = use_case.execute(task_id)
    return TaskStatusResponse(state=dto.state, info=dto.info, result=dto.result)


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
