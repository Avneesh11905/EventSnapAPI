from config import settings
from application.ports.storage import IStorageService
from application.ports.inference import IInferenceService
from application.ports.uow import IUnitOfWork
from application.ports.queue import ITaskQueueService
from typing import Callable
import uuid
from domain.exceptions import ZipGenerationError, InferenceError
from application.dtos import (
    BackgroundEncodingResult,
    BackgroundZipResult,
    EventEncodingDTO,
)


class EncodeImageBatchUseCase:
    def __init__(
        self,
        storage_service: IStorageService,
        inference_service: IInferenceService,
        uow: IUnitOfWork,
    ):
        self.storage_service = storage_service
        self.inference_service = inference_service
        self.uow = uow

    async def execute(
        self,
        event_code: str,
        keys: list[str],
        det_conf: float,
        nms_thresh: float,
    ) -> dict:

        import time
        import logging

        logger = logging.getLogger(__name__)

        start_time = time.time()
        batch_thumb_keys = [k.replace("/raw/", "/thumbs/", 1) for k in keys]
        b64_images = await self.storage_service.download_images_b64(batch_thumb_keys)

        dl_time = time.time()
        logger.info(
            f"download_images_b64 took {dl_time - start_time:.2f}s for {len(keys)} images"
        )

        valid_keys: list[str] = []
        valid_b64: list[str] = []
        failed_keys: list[str] = []
        for key, b64 in zip(keys, b64_images):
            if isinstance(b64, Exception):
                logger.warning(
                    f"Failed to download {key} after retries: {b64}. Skipping this image."
                )
                failed_keys.append(key)
                continue
            elif b64 is not None:
                valid_keys.append(key)
                valid_b64.append(str(b64))

        if not valid_keys:
            return {
                "encoded": 0,
                "no_encodings_found": 0,
                "total": len(keys),
                "failures": failed_keys,
            }

        try:
            inf_start = time.time()
            results = await self.inference_service.get_face_encodings(
                valid_b64, det_conf, nms_thresh
            )
            inf_time = time.time()
            logger.info(f"inference API call took {inf_time - inf_start:.2f}s")

            insert_data = []
            for key, image_faces in zip(valid_keys, results):
                for face in image_faces:
                    emb = face.get("embedding")
                    conf = face.get("confidence")
                    if emb and conf:
                        insert_data.append(
                            EventEncodingDTO(
                                id=uuid.uuid7(),
                                event_code=event_code,
                                image_path=key,
                                embedding=emb,
                                confidence=conf,
                            )
                        )

            db_start = time.time()
            async with self.uow as uow:
                if insert_data:
                    await uow.event_repo.save_encodings(insert_data)

                if valid_keys:
                    processed_data = [
                        {"event_code": event_code, "image_path": key}
                        for key in valid_keys
                    ]
                    await uow.event_repo.save_processed_images(processed_data)

                await uow.commit()
            db_time = time.time()
            logger.info(
                f"Database insert for {len(insert_data)} encodings and {len(valid_keys)} processed images took {db_time - db_start:.2f}s"
            )

            encoded = sum(
                1
                for image_faces in results
                if any(f.get("embedding") for f in image_faces)
            )
            no_encodings_found = len(valid_keys) - encoded
            return {
                "encoded": encoded,
                "no_encodings_found": no_encodings_found,
                "total": len(keys),
                "failures": failed_keys,
            }
        except Exception as e:
            raise InferenceError(f"Failed to infer batch: {e}") from e


class ProcessEventEncodingUseCase:
    def __init__(
        self,
        storage_service: IStorageService,
        uow: IUnitOfWork,
        queue_service: ITaskQueueService,
        cache_service,
    ):
        self.storage_service = storage_service
        self.uow = uow
        self.queue_service = queue_service
        self.cache_service = cache_service

    async def execute(
        self,
        event_code: str,
        det_conf: float,
        nms_thresh: float,
        update_state_cb: Callable[[str, dict], None],
    ) -> BackgroundEncodingResult:
        lock_name = f"lock:encode:{event_code}"

        # Try to acquire the lock. If it exists, return 409 Conflict via Exception handler.
        lock_acquired = await self.cache_service.acquire_lock(
            lock_name, 3600
        )  # 1 hour expiration
        if not lock_acquired:
            from domain.exceptions import TaskAlreadyInProgressError

            raise TaskAlreadyInProgressError(
                "Encoding task already in progress for this event"
            )

        try:
            update_state_cb(
                "INITIALIZING", {"progress": 0, "status": "Listing Storage files..."}
            )

            base_folder = f"event/{event_code}"
            thumbs_folder = f"{base_folder}/thumbs"

            all_thumb_keys = await self.storage_service.list_images(thumbs_folder)

            if len(all_thumb_keys) == 0:
                return BackgroundEncodingResult(total=0)

            all_raw_keys = [k.replace("/thumbs/", "/raw/", 1) for k in all_thumb_keys]

            async with self.uow as uow:
                already_encoded = await uow.event_repo.get_already_encoded_images(
                    event_code
                )

            new_raw_keys = [k for k in all_raw_keys if k not in already_encoded]
            skipped = len(all_thumb_keys) - len(new_raw_keys)
            total_images = len(new_raw_keys)

            if total_images == 0:
                return BackgroundEncodingResult(
                    total=len(all_thumb_keys), skipped=skipped
                )

            update_state_cb(
                "PROCESSING",
                {
                    "progress": 0,
                    "processed": 0,
                    "total": total_images,
                    "skipped": skipped,
                    "status_msg": f"Skipped {skipped} already-encoded images. Processing {total_images} new images...",
                },
            )

            batch_size = settings.INFERENCE_BATCH_SIZE
            chunks = []
            for i in range(0, total_images, batch_size):
                chunks.append(new_raw_keys[i : i + batch_size])

            group_id = self.queue_service.enqueue_encode_group(
                event_code=event_code,
                chunks=chunks,
                detection_conf=det_conf,
                nms_threshold=nms_thresh,
            )

            return BackgroundEncodingResult(
                total=total_images,
                skipped=skipped,
                group_id=group_id,
            )
        finally:
            await self.cache_service.release_lock(lock_name)


class CreateEventZipUseCase:
    def __init__(self, storage_service: IStorageService):
        self.storage_service = storage_service

    async def execute(
        self,
        event_id: str,
        user_id: str,
        image_paths: list[dict],
        update_state_cb: Callable[[str, dict], None],
    ) -> BackgroundZipResult:
        total = len(image_paths)
        if total == 0:
            raise ZipGenerationError("No images to zip")

        update_state_cb(
            "INITIALIZING",
            {"progress": 0, "status_msg": f"Starting ZIP for {total} images..."},
        )

        zip_filename = f"{user_id}.zip"
        storage_path = f"zip/{event_id}/{zip_filename}"

        def progress_callback(current, state_name, status_msg):
            progress_pct = int((current / total) * 90)
            update_state_cb(
                state_name, {"progress": progress_pct, "status_msg": status_msg}
            )

        try:
            await self.storage_service.create_zip_from_images(
                storage_path, image_paths, progress_callback
            )

            return BackgroundZipResult(
                zip_path=storage_path,
                filename=zip_filename,
                images_zipped=total,
            )
        except Exception as e:
            update_state_cb("FAILED", {"error": str(e)})
            raise e


class DeleteImageBatchUseCase:
    def __init__(
        self,
        storage_service: IStorageService,
        uow: IUnitOfWork,
    ):
        self.storage_service = storage_service
        self.uow = uow

    async def execute(
        self,
        event_code: str,
        keys: list[str],
        cancel_task_id: str | None = None,
        update_state_cb: Callable[[str, dict], None] = lambda *args: None,
    ) -> dict:
        import logging
        from celery import current_app

        logger = logging.getLogger(__name__)

        if cancel_task_id:
            logger.info(f"Revoking Celery task {cancel_task_id}")
            current_app.control.revoke(cancel_task_id, terminate=True)

        update_state_cb(
            "PROCESSING",
            {"status": f"Deleting {len(keys)} images from S3 and Database..."},
        )

        # S3 Deletion (both raw and thumbs)
        all_s3_keys = keys.copy()
        for key in keys:
            raw_key = key.replace("/thumbs/", "/raw/", 1)
            if raw_key != key:
                all_s3_keys.append(raw_key)

        CHUNK_SIZE = 1000
        for i in range(0, len(all_s3_keys), CHUNK_SIZE):
            chunk = all_s3_keys[i : i + CHUNK_SIZE]
            await self.storage_service.delete_objects(chunk)
            logger.info(f"Deleted S3 chunk of {len(chunk)} objects")

        # DB Deletion
        async with self.uow as uow:
            await uow.event_repo.delete_keys(event_code, keys)
            await uow.commit()

        update_state_cb(
            "SUCCESS", {"status": f"Successfully deleted {len(keys)} images."}
        )
        return {"success": True, "deleted": len(keys)}
