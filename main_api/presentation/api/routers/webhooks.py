from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel
from infrastructure.queue.celery_app import celery_app
from config import settings
import logging

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])
logger = logging.getLogger(__name__)

class InferenceStatusPayload(BaseModel):
    status: str  # "online" or "offline"

@router.post("/inference-status")
async def update_inference_status(
    payload: InferenceStatusPayload,
    x_webhook_secret: str = Header(...)
):
    # Security check: ensure the secret matches
    if x_webhook_secret != settings.WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    
    if payload.status == "offline":
        logger.warning("Received offline webhook! Pausing Celery queue consumption...")
        # Cancel the consumer to stop taking new tasks
        response = celery_app.control.cancel_consumer('celery', reply=True)
        return {"message": "Celery queue consumption paused", "details": response}
        
    elif payload.status == "online":
        logger.info("Received online webhook! Resuming Celery queue consumption...")
        # Add the consumer back to start processing tasks again
        response = celery_app.control.add_consumer('celery', reply=True)
        return {"message": "Celery queue consumption resumed", "details": response}
        
    else:
        raise HTTPException(status_code=400, detail="Invalid status payload")
