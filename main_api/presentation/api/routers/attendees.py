from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from dependency_injector.wiring import inject, Provide
from infrastructure.di_container import Container
from application.use_cases.attendees import (
    EncodeAttendeeUseCase, 
    SortAttendeeUseCase, 
    GenerateZipUseCase, 
    CheckZipExistsUseCase
)

router = APIRouter()

class EncodeAttendeeRequest(BaseModel):
    attendee_images_base64: list[str]

class SortAttendeeRequest(BaseModel):
    minio_folder_path: str
    attendee_encodings: list[list[float]]

@router.post("/encode-attendee/")
@inject
async def encode_attendee(
    request: EncodeAttendeeRequest,
    use_case: EncodeAttendeeUseCase = Depends(Provide[Container.encode_attendee_use_case])
):
    try:
        embeddings_list = await use_case.execute(request.attendee_images_base64)
        return {
            "message": f"Successfully generated {len(embeddings_list)} encodings from 3 reference images.",
            "encodings": embeddings_list
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference API Failed: {e}")

@router.post("/sort-attendee/")
@inject
async def sort_event_attendee(
    request: SortAttendeeRequest,
    use_case: SortAttendeeUseCase = Depends(Provide[Container.sort_attendee_use_case])
):
    try:
        return await use_case.execute(request.minio_folder_path, request.attendee_encodings)
    except ValueError as e:
        if "No encoded data" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))

class GenerateZipRequest(BaseModel):
    event_id: str
    user_id: str
    image_paths: list[dict]

@router.post("/generate-zip/")
@inject
async def generate_zip(
    request: GenerateZipRequest,
    use_case: GenerateZipUseCase = Depends(Provide[Container.generate_zip_use_case])
):
    task_id = use_case.execute(
        request.event_id,
        request.user_id,
        request.image_paths
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": "ZIP generation started in background"
    }

@router.get("/check-zip/{event_id}/{user_id}")
@inject
async def check_zip(
    event_id: str, 
    user_id: str,
    use_case: CheckZipExistsUseCase = Depends(Provide[Container.check_zip_exists_use_case])
):
    return await use_case.execute(event_id, user_id)
