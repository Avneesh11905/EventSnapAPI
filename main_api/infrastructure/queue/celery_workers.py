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
) -> dict:
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
)
def encode_image_batch_task(
    self,
    event_code: str,
    keys: list[str],
    det_conf: float = 0.5,
    nms_thresh: float = 0.4,
) -> dict:
    container = get_container()
    use_case = container.encode_image_batch_use_case()

    from domain.exceptions import TaskCanceledError
    from celery.exceptions import Ignore
    import asyncio

    try:
        return asyncio.run(
            use_case.execute(
                event_code=event_code,
                keys=keys,
                det_conf=det_conf,
                nms_thresh=nms_thresh,
            )
        )
    except TaskCanceledError as e:
        self.update_state(state="REVOKED", meta={"reason": str(e)})
        raise Ignore()
    except Exception as e:
        # Manually retry with backoff to avoid catching Ignore via autoretry_for=(Exception,)
        raise self.retry(exc=e, max_retries=5, countdown=2**self.request.retries)


@shared_task(bind=True, name="create_event_zip_task", acks_late=True)
def create_event_zip_task(
    self, event_id: str, user_id: str, image_paths: list[dict]
) -> dict:
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


@shared_task(
    bind=True,
    name="delete_event_data_task",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 5},
    retry_backoff=True,
)
def delete_event_data_task(self, event_code: str, event_id: str | None = None) -> dict:
    container = get_container()
    uow = container.uow()
    storage = container.storage_service()

    async def _delete():
        try:
            cache = container.cache_service()
            await cache.set_flag(f"cancel_encode:{event_code}", expiration=3600)
            async with uow:
                await uow.event_repo.delete_event_data(event_code)
                await uow.commit()

            await storage.delete_folder(f"event/{event_code}/")
            msg = f"Successfully deleted event {event_code} from database and removed 'event/{event_code}/' from storage."
            if event_id:
                await storage.delete_folder(f"zip/{event_id}/")
                msg = f"Successfully deleted event {event_code} from database and removed 'event/{event_code}/' and 'zip/{event_id}/' from storage."

            return {"success": True, "message": msg}
        except Exception as e:
            self.update_state(
                state="FAILURE",
                meta={
                    "error": f"Failed to delete event data for event_code={event_code}. Reason: {str(e)}"
                },
            )
            raise

    return asyncio.run(_delete())


@shared_task(
    bind=True,
    name="delete_image_batch_task",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 5},
    retry_backoff=True,
)
def delete_image_batch_task(
    self, event_code: str, keys: list[str], cancel_task_id: str | None = None
) -> dict:
    container = get_container()
    use_case = container.delete_image_batch_use_case()

    def update_state_cb(state_name, meta_dict):
        self.update_state(state=state_name, meta=meta_dict)

    result = asyncio.run(
        use_case.execute(
            event_code=event_code,
            keys=keys,
            cancel_task_id=cancel_task_id,
            update_state_cb=update_state_cb,
        )
    )
    return result
