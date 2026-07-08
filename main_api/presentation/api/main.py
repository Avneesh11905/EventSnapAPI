from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from config import settings
from presentation.api.routers import events, attendees
from infrastructure.di_container import get_container
from sqlalchemy import text
from infrastructure.queue.celery_app import celery_app

# Create the container at startup to wire everything
container = get_container()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Any necessary init happens via Dependency Injector singletons if needed
    yield

app = FastAPI(
    title="Eventsnap Main API (Orchestrator) - Clean",
    description="Handles Event Encodings, Background Celery Tasks, and Attendee Sorting using pgvector.",
    version="2.0.0",
    lifespan=lifespan
)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key == settings.API_KEY:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid API Key"
        )

app.include_router(events.router, prefix="/api/events", tags=["Events"], dependencies=[Depends(get_api_key)])
app.include_router(attendees.router, prefix="/api/attendees", tags=["Attendees"], dependencies=[Depends(get_api_key)])

@app.get("/api/tasks/{task_id}", tags=["Tasks"], dependencies=[Depends(get_api_key)])
def get_task_status(task_id: str):
    """Checks the status of any Celery task via Use Case."""
    use_case = container.check_encoding_status_use_case()
    return use_case.execute(task_id)

@app.get("/", tags=["Health"])
async def health_check():
    health_status = {
        "status": "ok",
        "service": "Eventsnap Main API",
        "checks": {}
    }
    
    # Check Postgres
    try:
        engine = container.db_engine()
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health_status["checks"]["postgres"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["postgres"] = f"error: {str(e)}"
        
    # Check RabbitMQ (Celery Broker)
    try:
        with celery_app.connection() as conn:
            conn.ensure_connection(max_retries=1, timeout=2)
        health_status["checks"]["rabbitmq"] = "ok"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["checks"]["rabbitmq"] = f"error: {str(e)}"
        
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app='presentation.api.main:app', host="0.0.0.0", port=8000, reload=True)
