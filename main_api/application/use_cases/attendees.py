import numpy as np
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
    FaceValidationError,
)
from config import settings


from typing import Callable


class EncodeAttendeeUseCase[T]:
    def __init__(
        self,
        inference_service: IInferenceService[T],
        augmenter: IImageAugmenter,
        decode_fn: Callable[[str], T],
    ):
        self.inference_service = inference_service
        self.augmenter = augmenter
        self.decode_fn = decode_fn

    async def execute(self, attendee_images_base64: list[str]) -> list[float]:
        if len(attendee_images_base64) != 3:
            raise InvalidReferenceImagesError(
                "Must provide exactly 3 attendee images (front, left, right)."
            )

        converted_images = [self.decode_fn(img) for img in attendee_images_base64]

        # 1. Validation Step: Check original images for multiple faces or no faces
        original_results = await self.inference_service.get_face_encodings(
            converted_images
        )

        validation_errors = []
        no_faces_count = 0
        multi_faces_count = 0

        for i, image_faces in enumerate(original_results):
            if len(image_faces) > 1:
                bboxes = [face["bbox"] for face in image_faces]
                validation_errors.append(
                    {"image_index": i, "bboxes": bboxes, "issue": "multiple"}
                )
                multi_faces_count += 1
            elif len(image_faces) == 0:
                validation_errors.append(
                    {"image_index": i, "bboxes": [], "issue": "none"}
                )
                no_faces_count += 1

        if validation_errors:
            msg_parts = []
            if no_faces_count:
                msg_parts.append(
                    f"no face in {no_faces_count} {'photo' if no_faces_count == 1 else 'photos'}"
                )
            if multi_faces_count:
                msg_parts.append(
                    f"multiple faces in {multi_faces_count} {'photo' if multi_faces_count == 1 else 'photos'}"
                )

            error_msg = (
                "Issues detected: "
                + " and ".join(msg_parts)
                + ". Please retake the highlighted ones."
            )

            raise FaceValidationError(
                error_msg,
                details=validation_errors,
            )

        # 2. Execution Step: Augment and process
        augmented_b64_images = await asyncio.to_thread(
            self.augmenter.augment, attendee_images_base64
        )
        converted_augmented_images = [
            self.decode_fn(img) for img in augmented_b64_images
        ]

        results = await self.inference_service.get_face_encodings(
            converted_augmented_images
        )

        embeddings_list = []
        for image_faces in results:
            if len(image_faces) == 1:
                embeddings_list.append(image_faces[0]["embedding"])

        if not embeddings_list:
            raise InvalidReferenceImagesError(
                "Could not detect clear faces in the provided and augmented reference images."
            )

        avg_embedding = np.mean(embeddings_list, axis=0).tolist()
        return avg_embedding


class SortAttendeeUseCase:
    def __init__(self, uow: IUnitOfWork):
        self.uow = uow

    async def execute(
        self, event_code: str, attendee_encoding: List[float]
    ) -> AttendeeSortDTO:
        if not attendee_encoding:
            raise InvalidReferenceImagesError("Must provide a valid attendee encoding.")

        async with self.uow as uow:
            has_data = await uow.event_repo.check_event_has_data(event_code)
            if not has_data:
                raise EventNotFoundError(
                    f"No encoded data found for event {event_code}."
                )

            matched_paths = await uow.event_repo.find_matches(
                event_code,
                attendee_encoding,
                settings.SIMILARITY_THRESHOLD,
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
