from fastapi import APIRouter, Depends
from dependency_injector.wiring import inject, Provide
from infrastructure.di_container import Container
from application.use_cases.events import (
    StartEventEncodingUseCase,
    CheckEncodingStatusUseCase,
    GetEncodedCountUseCase,
    DeleteEventDataUseCase,
)
from presentation.api.schemas import (
    EncodeEventRequest,
    EnqueueTaskResponse,
    EncodingStatusResponse,
    EncodedCountResponse,
    DeleteDataResponse,
)
import asyncio
import dataclasses
from typing import Any

router = APIRouter()


@router.post("/encode-event/", response_model=EnqueueTaskResponse)
@inject
async def start_event_encoding(
    request: EncodeEventRequest,
    use_case: StartEventEncodingUseCase = Depends(
        Provide[Container.start_event_encoding_use_case]
    ),
):
    task_id = use_case.execute(
        request.event_code,
        request.max_faces,
        request.detection_conf,
        request.nms_threshold,
    )

    return EnqueueTaskResponse(
        message="Event encoding task has been enqueued to RabbitMQ Worker.",
        task_id=task_id,
        event_code=request.event_code,
    )


@router.get("/encode-status/{task_id}", response_model=EncodingStatusResponse)
@inject
async def get_encoding_status(
    task_id: str,
    use_case: CheckEncodingStatusUseCase = Depends(
        Provide[Container.check_encoding_status_use_case]
    ),
):
    response = await asyncio.to_thread(use_case.execute, task_id)

    state = response.get("status")
    formatted: dict[str, Any] = {"task_id": task_id, "status": state}

    if state == "PROCESSING" and "progress" in response:
        formatted.update(
            {
                "progress": f"{response.get('progress', 0)}%",
                "images_processed": response.get("processed", 0),
                "total_images": response.get("total", 0),
            }
        )
    elif state == "SUCCESS" and "result" in response:
        res_data = response["result"]
        if isinstance(res_data, dict):
            formatted["message"] = res_data.get("result", str(res_data))
        else:
            formatted["message"] = str(res_data)

    return EncodingStatusResponse(**formatted)


@router.get("/encode-count/{event_code}", response_model=EncodedCountResponse)
@inject
async def get_encoded_image_count(
    event_code: str,
    use_case: GetEncodedCountUseCase = Depends(
        Provide[Container.get_encoded_count_use_case]
    ),
):
    dto = await use_case.execute(event_code)
    return EncodedCountResponse(**dataclasses.asdict(dto))


@router.delete("/delete-event-data/{event_code}", response_model=DeleteDataResponse)
@inject
async def delete_event_data(
    event_code: str,
    event_id: str | None = None,
    use_case: DeleteEventDataUseCase = Depends(
        Provide[Container.delete_event_data_use_case]
    ),
):
    dto = await use_case.execute(event_code, event_id)
    return DeleteDataResponse(**dataclasses.asdict(dto))
