from fastapi import APIRouter, Depends
from pydantic import BaseModel
from dependency_injector.wiring import inject, Provide
from infrastructure.di_container import Container
from application.use_cases.events import (
    StartEventEncodingUseCase, 
    CheckEncodingStatusUseCase, 
    GetEncodedCountUseCase, 
    DeleteEventTableUseCase
)
import asyncio

router = APIRouter()

class EncodeEventRequest(BaseModel):
    minio_folder_path: str
    max_faces: int = 0
    detection_conf: float = 0.5
    nms_threshold: float = 0.4

@router.post("/encode-event/")
@inject
async def start_event_encoding(
    request: EncodeEventRequest,
    use_case: StartEventEncodingUseCase = Depends(Provide[Container.start_event_encoding_use_case])
):
    task_id = use_case.execute(
        request.minio_folder_path,
        request.max_faces,
        request.detection_conf,
        request.nms_threshold
    )
    
    return {
        "message": "Event encoding task has been enqueued to RabbitMQ Worker.",
        "task_id": task_id,
        "folder": request.minio_folder_path
    }

@router.get("/encode-status/{task_id}")
@inject
async def get_encoding_status(
    task_id: str,
    use_case: CheckEncodingStatusUseCase = Depends(Provide[Container.check_encoding_status_use_case])
):
    try:
        response = await asyncio.to_thread(use_case.execute, task_id)
        
        state = response.get("status")
        formatted = {
            "task_id": task_id,
            "status": state
        }
        
        if state == 'PROCESSING' and 'progress' in response:
            formatted.update({
                "progress": f"{response.get('progress', 0)}%",
                "images_processed": response.get('processed', 0),
                "total_images": response.get('total', 0),
            })
        elif state == 'SUCCESS' and 'result' in response:
             formatted["message"] = response["result"]

        return formatted
    except Exception as e:
         return {"task_id": task_id, "status": "ERROR", "message": f"Failed to connect to backend: {str(e)}"}

@router.get("/encode-count/{folder}")
@inject
async def get_encoded_image_count(
    folder: str,
    use_case: GetEncodedCountUseCase = Depends(Provide[Container.get_encoded_count_use_case])
):
    try:
        return await use_case.execute(folder)
    except Exception as e:
        return {"encoded_count": 0, "error": str(e)}
    
@router.delete("/delete-event-table/{folder}")
@inject
async def delete_event_table(
    folder: str,
    use_case: DeleteEventTableUseCase = Depends(Provide[Container.delete_event_table_use_case])
):
    try:
        return await use_case.execute(folder)
    except Exception as e:
        return {
            "success": False, 
            "error": str(e),
            "table_name": folder
        }
