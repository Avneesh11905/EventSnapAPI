from fastapi import APIRouter, Depends
from pydantic import BaseModel
from celery.result import AsyncResult

# We import the celery app and our specific background task
from celery_app import celery_app
from tasks import encode_event_task

router = APIRouter()

class EncodeEventRequest(BaseModel):
    minio_folder_path: str
    max_faces: int = 0
    detection_conf: float = 0.5
    nms_threshold: float = 0.4

@router.post("/encode-event/")
async def start_event_encoding(request: EncodeEventRequest):
    """
    Kicks off a background Celery worker to ingest all images in a MinIO folder.
    Returns the RabbitMQ Task ID immediately so the frontend won't timeout.
    """
    # Enqueue task to RabbitMQ with custom detection parameters
    task = encode_event_task.delay(
        request.minio_folder_path,
        request.max_faces,
        request.detection_conf,
        request.nms_threshold
    )
    
    return {
        "message": "Event encoding task has been enqueued to RabbitMQ Worker.",
        "task_id": task.id,
        "folder": request.minio_folder_path
    }

def fetch_task_status_sync(task_id: str):
    """Synchronous background function to poll RabbitMQ without blocking FastAPI."""
    task_result = AsyncResult(task_id, app=celery_app)
    return task_result.state, task_result.info, task_result.result

@router.get("/encode-status/{task_id}")
async def get_encoding_status(task_id: str):
    """
    Poll this endpoint with the Task ID to get real-time processing updates
    from the Celery Worker via the RabbitMQ RPC backend.
    """
    import asyncio
    
    try:
        # Offload RabbitMQ TCP requests to a threadpool to prevent freezing the event loop
        state, info, result = await asyncio.to_thread(fetch_task_status_sync, task_id)
        
        response = {
            "task_id": task_id,
            "status": state
        }
        
        # If the task has custom meta info (like progress percentage) available
        if state == 'PROCESSING' and info:
            response.update({
                "progress": f"{info.get('progress', 0)}%",
                "images_processed": info.get('processed', 0),
                "total_images": info.get('total', 0),
            })
        elif state == 'SUCCESS':
             response["message"] = result

        return response
        
    except Exception as e:
         return {"task_id": task_id, "status": "ERROR", "message": f"Failed to connect to backend: {str(e)}"}
