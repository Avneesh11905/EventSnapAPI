from fastapi import APIRouter, Depends
from dependency_injector.wiring import inject, Provide
from infrastructure.di_container import Container
from application.use_cases.attendees import (
    EncodeAttendeeUseCase,
    SortAttendeeUseCase,
    GenerateZipUseCase,
    CheckZipExistsUseCase,
)
from presentation.api.schemas import (
    EncodeAttendeeRequest,
    SortAttendeeRequest,
    GenerateZipRequest,
    EncodeAttendeeResponse,
    GenerateZipResponse,
    AttendeeSortResponse,
    ZipCheckResponse,
)
import dataclasses

router = APIRouter()


@router.post("/encode-attendee/", response_model=EncodeAttendeeResponse)
@inject
async def encode_attendee(
    request: EncodeAttendeeRequest,
    use_case: EncodeAttendeeUseCase = Depends(
        Provide[Container.encode_attendee_use_case]
    ),
):
    encoding = await use_case.execute(request.attendee_images_base64)
    return EncodeAttendeeResponse(
        message="Successfully generated 1 averaged encoding from 3 reference images.",
        encoding=encoding,
    )


@router.post("/sort-attendee/", response_model=AttendeeSortResponse)
@inject
async def sort_event_attendee(
    request: SortAttendeeRequest,
    use_case: SortAttendeeUseCase = Depends(Provide[Container.sort_attendee_use_case]),
):
    dto = await use_case.execute(request.event_code, request.attendee_encoding)
    return AttendeeSortResponse(**dataclasses.asdict(dto))


@router.post("/generate-zip/", response_model=GenerateZipResponse)
@inject
async def generate_zip(
    request: GenerateZipRequest,
    use_case: GenerateZipUseCase = Depends(Provide[Container.generate_zip_use_case]),
):
    task_id = use_case.execute(request.event_id, request.user_id, request.image_paths)

    return GenerateZipResponse(
        success=True, task_id=task_id, message="ZIP generation started in background"
    )


@router.get("/check-zip/{event_id}/{user_id}", response_model=ZipCheckResponse)
@inject
async def check_zip(
    event_id: str,
    user_id: str,
    use_case: CheckZipExistsUseCase = Depends(
        Provide[Container.check_zip_exists_use_case]
    ),
):
    dto = await use_case.execute(event_id, user_id)
    return ZipCheckResponse(**dataclasses.asdict(dto))
