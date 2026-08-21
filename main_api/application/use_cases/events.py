from application.ports.queue import ITaskQueueService
from application.ports.uow import IUnitOfWork
from application.dtos import EncodedCountDTO, DeleteDataDTO, TaskStatusDTO


class StartEventEncodingUseCase:
    def __init__(self, queue_service: ITaskQueueService):
        self.queue_service = queue_service

    def execute(
        self,
        event_code: str,
        detection_conf: float,
        nms_threshold: float,
    ) -> str:
        return self.queue_service.enqueue_encode_event(
            event_code, detection_conf, nms_threshold
        )


class CheckEncodingStatusUseCase:
    def __init__(self, queue_service: ITaskQueueService):
        self.queue_service = queue_service

    def execute(self, task_id: str) -> TaskStatusDTO:
        return self.queue_service.get_task_status(task_id)


class GetEncodedCountUseCase:
    def __init__(self, uow: IUnitOfWork):
        self.uow = uow

    async def execute(self, event_code: str) -> EncodedCountDTO:
        async with self.uow as uow:
            exists = await uow.event_repo.check_event_has_data(event_code)
            if not exists:
                return EncodedCountDTO(encoded_count=0, table_exists=False)
            count = await uow.event_repo.get_encoded_count(event_code)
            return EncodedCountDTO(encoded_count=count, table_exists=True)


class DeleteEventDataUseCase:
    def __init__(self, queue_service: ITaskQueueService):
        self.queue_service = queue_service

    async def execute(
        self, event_code: str, event_id: str | None = None
    ) -> DeleteDataDTO:
        self.queue_service.enqueue_delete_event(event_code, event_id)
        return DeleteDataDTO(
            success=True,
            message=f"Enqueued deletion task for event '{event_code}'.",
        )
