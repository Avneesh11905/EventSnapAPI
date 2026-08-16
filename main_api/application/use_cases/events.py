from application.ports.queue import TaskQueueService
from application.ports.repository import EventRepository
from application.dtos import EncodedCountDTO, DeleteTableDTO


class StartEventEncodingUseCase:
    def __init__(self, queue_service: TaskQueueService):
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
    def __init__(self, queue_service: TaskQueueService):
        self.queue_service = queue_service

    def execute(self, task_id: str) -> dict:
        return self.queue_service.get_task_status(task_id)


class GetEncodedCountUseCase:
    def __init__(self, repository: EventRepository):
        self.repository = repository

    async def execute(self, event_code: str) -> EncodedCountDTO:
        folder_path = f"event/{event_code}"
        exists = await self.repository.check_table_exists(folder_path)
        if not exists:
            return EncodedCountDTO(encoded_count=0, table_exists=False)
        count = await self.repository.get_encoded_count(folder_path)
        return EncodedCountDTO(encoded_count=count, table_exists=True)


class DeleteEventTableUseCase:
    def __init__(self, repository: EventRepository):
        self.repository = repository

    async def execute(self, event_code: str) -> DeleteTableDTO:
        folder_path = f"event/{event_code}"
        await self.repository.delete_event_table(folder_path)
        return DeleteTableDTO(
            success=True,
            message=f"Table for event '{folder_path}' deleted successfully if it existed.",
        )
