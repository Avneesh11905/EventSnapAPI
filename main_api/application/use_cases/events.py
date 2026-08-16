from application.ports.queue import ITaskQueueService
from application.ports.repository import IEventRepository
from application.dtos import EncodedCountDTO, DeleteTableDTO


class StartEventEncodingUseCase:
    def __init__(self, queue_service: ITaskQueueService):
        self.queue_service = queue_service

    def execute(
        self,
        event_code: str,
        max_faces: int,
        detection_conf: float,
        nms_threshold: float,
    ) -> str:
        return self.queue_service.enqueue_encode_event(
            event_code, max_faces, detection_conf, nms_threshold
        )


class CheckEncodingStatusUseCase:
    def __init__(self, queue_service: ITaskQueueService):
        self.queue_service = queue_service

    def execute(self, task_id: str) -> dict:
        return self.queue_service.get_task_status(task_id)


class GetEncodedCountUseCase:
    def __init__(self, repository: IEventRepository):
        self.repository = repository

    async def execute(self, event_code: str) -> EncodedCountDTO:
        exists = await self.repository.check_event_has_data(event_code)
        if not exists:
            return EncodedCountDTO(encoded_count=0, table_exists=False)
        count = await self.repository.get_encoded_count(event_code)
        return EncodedCountDTO(encoded_count=count, table_exists=True)


class DeleteEventTableUseCase:
    def __init__(self, repository: IEventRepository):
        self.repository = repository

    async def execute(self, event_code: str) -> DeleteTableDTO:
        await self.repository.delete_event_data(event_code)
        return DeleteTableDTO(
            success=True,
            message=f"Data for event '{event_code}' deleted successfully if it existed.",
        )
