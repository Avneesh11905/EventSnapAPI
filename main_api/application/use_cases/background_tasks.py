from config import settings
from application.ports.storage import IStorageService
from application.ports.inference import IInferenceService
from application.ports.uow import IUnitOfWork
from application.ports.queue import ITaskQueueService
import asyncio
from typing import Callable, Any
import uuid
from domain.exceptions import ZipGenerationError, StorageDownloadError, InferenceError
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
    ) -> None:

        batch_thumb_keys = [k.replace("/raw/", "/thumbs/", 1) for k in keys]
        download_tasks = [
            self.storage_service.download_image_b64(key) for key in batch_thumb_keys
        ]
        b64_images = await asyncio.gather(*download_tasks, return_exceptions=True)

        valid_keys: list[str] = []
        valid_b64: list[str] = []
        for key, b64 in zip(keys, b64_images):
            if isinstance(b64, Exception):
                raise StorageDownloadError(
                    f"Failed to download {key} after retries: {b64}"
                )
            elif b64 is not None:
                valid_keys.append(key)
                valid_b64.append(str(b64))

        if not valid_keys:
            return

        try:
            results = await self.inference_service.get_face_encodings(
                valid_b64, det_conf, nms_thresh
            )

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

            if insert_data:
                async with self.uow as uow:
                    await uow.event_repo.save_encodings(insert_data)
                    await uow.commit()
        except Exception as e:
            raise InferenceError(f"Failed to infer batch: {e}") from e


class ProcessEventEncodingUseCase:
    def __init__(
        self,
        storage_service: IStorageService,
        uow: IUnitOfWork,
        queue_service: ITaskQueueService,
    ):
        self.storage_service = storage_service
        self.uow = uow
        self.queue_service = queue_service

    async def execute(
        self,
        event_code: str,
        det_conf: float,
        nms_thresh: float,
        update_state_cb: Callable[[str, dict], Any],
    ) -> BackgroundEncodingResult:
        update_state_cb(
            "INITIALIZING", {"progress": 0, "status": "Listing Storage files..."}
        )

        base_folder = f"event/{event_code}"
        thumbs_folder = f"{base_folder}/thumbs"

        all_thumb_keys = await self.storage_service.list_images(thumbs_folder)

        if len(all_thumb_keys) == 0:
            return BackgroundEncodingResult(
                result="No images found in folder.", total=0
            )

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
                result="All images already encoded.",
                total=len(all_thumb_keys),
                skipped=skipped,
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
        task_ids = []
        for i in range(0, total_images, batch_size):
            chunk = new_raw_keys[i : i + batch_size]
            tid = self.queue_service.enqueue_encode_batch(
                event_code=event_code,
                keys=chunk,
                detection_conf=det_conf,
                nms_threshold=nms_thresh,
            )
            task_ids.append(tid)

        import asyncio

        completed_tasks: set[str] = set()
        while len(completed_tasks) < len(task_ids):
            for tid in task_ids:
                if tid not in completed_tasks:
                    status = self.queue_service.get_task_status(tid)
                    if status.get("status") in ("SUCCESS", "FAILURE"):
                        completed_tasks.add(tid)

            processed_batches = len(completed_tasks)
            processed_images = min(processed_batches * batch_size, total_images)
            pct = (
                int((processed_images / total_images) * 100)
                if total_images > 0
                else 100
            )

            update_state_cb(
                "PROCESSING",
                {
                    "progress": pct,
                    "processed": processed_images,
                    "total": total_images,
                    "skipped": skipped,
                    "status_msg": f"Processing {processed_images}/{total_images} images...",
                },
            )
            await asyncio.sleep(2)

        update_state_cb(
            "PROCESSING",
            {
                "progress": 100,
                "processed": total_images,
                "total": total_images,
                "skipped": skipped,
                "status_msg": f"Finished processing {total_images} images.",
            },
        )

        return BackgroundEncodingResult(
            result=f"Successfully processed {total_images} images.",
            total=len(all_thumb_keys),
            skipped=skipped,
        )


class CreateEventZipUseCase:
    def __init__(self, storage_service: IStorageService):
        self.storage_service = storage_service

    async def execute(
        self,
        event_id: str,
        user_id: str,
        image_paths: list[dict],
        update_state_cb: Callable[[str, dict], Any],
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
                status="COMPLETED",
                progress=100,
                zip_path=storage_path,
                filename=zip_filename,
            )
        except Exception as e:
            update_state_cb("FAILED", {"error": str(e)})
            raise e
