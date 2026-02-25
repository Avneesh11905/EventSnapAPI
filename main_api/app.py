from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

from config import settings
from routers import events, attendees
from database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database (ensure pgvector extension)
    await init_db()
    yield

app = FastAPI(
    title="Eventsnap Main API (Orchestrator)",
    description="Handles Event Encodings, Background Celery Tasks, and Attendee Sorting using pgvector.",
    version="2.0.0",
    lifespan=lifespan
)

# API Key Security
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


# Include Routers
app.include_router(events.router, prefix="/api", tags=["Events"])#, dependencies=[Depends(get_api_key)]
app.include_router(attendees.router, prefix="/api", tags=["Attendees"])#, dependencies=[Depends(get_api_key)]

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "Eventsnap Main API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app='app:app', host="0.0.0.0", port=8000, reload=True)
