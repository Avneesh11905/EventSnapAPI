from fastapi import APIRouter, Depends
from pydantic import BaseModel
from dependency_injector.wiring import inject, Provide
from infrastructure.di_container import Container
from typing import List, Optional
from infrastructure.queue.celery_workers import delete_image_batch_task

router = APIRouter()


class DeleteBulkRequest(BaseModel):
    event_code: str
    keys: List[str]
    cancel_task_id: Optional[str] = None


class DeleteBulkResponse(BaseModel):
    task_id: str


@router.post("/delete-bulk", response_model=DeleteBulkResponse)
async def delete_bulk_images(request: DeleteBulkRequest):
    task = delete_image_batch_task.delay(
        event_code=request.event_code,
        keys=request.keys,
        cancel_task_id=request.cancel_task_id,
    )
    return DeleteBulkResponse(task_id=task.id)


@router.get("/status/{event_code}")
@inject
async def get_image_status(event_code: str, uow=Depends(Provide[Container.uow])):
    async with uow as u:
        return await u.event_repo.get_image_status(event_code)
