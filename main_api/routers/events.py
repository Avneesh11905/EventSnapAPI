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

@router.get("/encode-status/{task_id}")
async def get_encoding_status(task_id: str):
    """
    Poll this endpoint with the Task ID to get real-time processing updates
    from the Celery Worker via the RabbitMQ RPC backend.
    """
    # Ask Celery for the task state
    task_result = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task_result.status
    }
    
    # If the task has custom meta info (like progress percentage) available
    if task_result.state == 'PROCESSING' and task_result.info:
        response.update({
            "progress": f"{task_result.info.get('progress', 0)}%",
            "images_processed": task_result.info.get('processed', 0),
            "total_images": task_result.info.get('total', 0),
        })
    elif task_result.state == 'SUCCESS':
         response["message"] = task_result.result

    return response
