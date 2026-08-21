from celery import shared_task
import asyncio
from infrastructure.di_container import get_container
import dataclasses


@shared_task(
    bind=True,
    name="encode_event_task",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 5},
    retry_backoff=True,
)
def encode_event_task(
    self,
    event_code: str,
    det_conf: float = 0.5,
    nms_thresh: float = 0.4,
):
    container = get_container()
    use_case = container.process_event_encoding_use_case()

    def update_state_cb(state_name, meta_dict):
        self.update_state(state=state_name, meta=meta_dict)

    result = asyncio.run(
        use_case.execute(
            event_code=event_code,
            det_conf=det_conf,
            nms_thresh=nms_thresh,
            update_state_cb=update_state_cb,
        )
    )
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    return result


@shared_task(
    bind=True,
    name="encode_image_batch_task",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 5},
    retry_backoff=True,
)
def encode_image_batch_task(
    self,
    event_code: str,
    keys: list[str],
    det_conf: float = 0.5,
    nms_thresh: float = 0.4,
):
    container = get_container()
    use_case = container.encode_image_batch_use_case()

    return asyncio.run(
        use_case.execute(
            event_code=event_code,
            keys=keys,
            det_conf=det_conf,
            nms_thresh=nms_thresh,
        )
    )


@shared_task(bind=True, name="create_event_zip_task", acks_late=True)
def create_event_zip_task(self, event_id: str, user_id: str, image_paths: list[dict]):
    container = get_container()
    use_case = container.create_event_zip_use_case()

    def update_state_cb(state_name, meta_dict):
        self.update_state(state=state_name, meta=meta_dict)

    result = asyncio.run(
        use_case.execute(
            event_id=event_id,
            user_id=user_id,
            image_paths=image_paths,
            update_state_cb=update_state_cb,
        )
    )
    if dataclasses.is_dataclass(result):
        return dataclasses.asdict(result)
    return result


@shared_task(bind=True, name="delete_event_data_task", acks_late=True)
def delete_event_data_task(self, event_code: str, event_id: str | None = None):
    container = get_container()
    uow = container.uow()
    storage = container.storage_service()

    async def _delete():
        async with uow:
            await uow.event_repo.delete_event_data(event_code)
            await uow.commit()
        await storage.delete_folder(f"event/{event_code}/")
        if event_id:
            await storage.delete_folder(f"zip/{event_id}/")

    asyncio.run(_delete())
    return {"success": True, "message": f"Deleted event {event_code}"}
