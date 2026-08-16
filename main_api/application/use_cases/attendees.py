from typing import List
import asyncio
from application.ports.inference import IInferenceService
from application.ports.queue import ITaskQueueService
from application.ports.uow import IUnitOfWork
from application.ports.storage import IStorageService
from application.ports.image_services import IImageAugmenter
from application.dtos import AttendeeSortDTO, ZipCheckDTO
from domain.exceptions import (
    InvalidReferenceImagesError,
    EventNotFoundError,
    NoMatchesFoundError,
)


class EncodeAttendeeUseCase:
    def __init__(
        self, inference_service: IInferenceService, augmenter: IImageAugmenter
    ):
        self.inference_service = inference_service
        self.augmenter = augmenter

    async def execute(self, attendee_images_base64: List[str]) -> List[List[float]]:
        if len(attendee_images_base64) != 3:
            raise InvalidReferenceImagesError(
                "Must provide exactly 3 attendee images (front, left, right)."
            )

        augmented_b64_images = await asyncio.to_thread(
            self.augmenter.augment, attendee_images_base64
        )

        results = await self.inference_service.get_face_encodings(augmented_b64_images)

        embeddings_list = []
        for image_faces in results:
            if len(image_faces) == 1:
                embeddings_list.append(image_faces[0]["embedding"])

        if not embeddings_list:
            raise InvalidReferenceImagesError(
                "Could not detect clear faces in the provided and augmented reference images."
            )

        return embeddings_list


class SortAttendeeUseCase:
    def __init__(self, uow: IUnitOfWork):
        self.uow = uow

    async def execute(
        self, event_code: str, attendee_encodings: List[List[float]]
    ) -> AttendeeSortDTO:
        if len(attendee_encodings) == 0:
            raise InvalidReferenceImagesError(
                "Must provide at least one attendee encoding."
            )

        async with self.uow as uow:
            has_data = await uow.event_repo.check_event_has_data(event_code)
            if not has_data:
                raise EventNotFoundError(
                    f"No encoded data found for event {event_code}."
                )

            SIMILARITY_THRESHOLD = 0.55
            MIN_MATCHES = 2

            matched_paths = await uow.event_repo.find_matches(
                event_code, attendee_encodings, SIMILARITY_THRESHOLD, MIN_MATCHES
            )

        if not matched_paths:
            raise NoMatchesFoundError(
                f"Could not find any matches for the attendee in event {event_code}."
            )

        return AttendeeSortDTO(
            event_code=event_code,
            matches_found=len(matched_paths),
            photos=matched_paths,
        )


class GenerateZipUseCase:
    def __init__(self, queue_service: ITaskQueueService):
        self.queue_service = queue_service

    def execute(self, event_id: str, user_id: str, image_paths: list[dict]) -> str:
        return self.queue_service.enqueue_create_zip(event_id, user_id, image_paths)


class CheckZipExistsUseCase:
    def __init__(self, storage_service: IStorageService):
        self.storage_service = storage_service

    async def execute(self, event_id: str, user_id: str) -> ZipCheckDTO:
        zip_key = f"zip/{event_id}/{user_id}.zip"
        exists = await self.storage_service.check_zip_exists(zip_key)
        if exists:
            return ZipCheckDTO(exists=True, zip_path=zip_key, filename=f"{user_id}.zip")
        return ZipCheckDTO(exists=False)
