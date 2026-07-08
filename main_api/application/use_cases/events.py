from application.ports.queue import TaskQueueService
from application.ports.repository import EventRepository

class StartEventEncodingUseCase:
    def __init__(self, queue_service: TaskQueueService):
        self.queue_service = queue_service
        
    def execute(self, folder_path: str, max_faces: int, detection_conf: float, nms_threshold: float) -> str:
        return self.queue_service.enqueue_encode_event(
            folder_path, max_faces, detection_conf, nms_threshold
        )

class CheckEncodingStatusUseCase:
    def __init__(self, queue_service: TaskQueueService):
        self.queue_service = queue_service
        
    def execute(self, task_id: str) -> dict:
        return self.queue_service.get_task_status(task_id)

class GetEncodedCountUseCase:
    def __init__(self, repository: EventRepository):
        self.repository = repository
        
    async def execute(self, folder_path: str) -> dict:
        exists = await self.repository.check_table_exists(folder_path)
        if not exists:
            return {"encoded_count": 0, "table_exists": False}
        count = await self.repository.get_encoded_count(folder_path)
        return {"encoded_count": count, "table_exists": True}

class DeleteEventTableUseCase:
    def __init__(self, repository: EventRepository):
        self.repository = repository
        
    async def execute(self, folder_path: str) -> dict:
        await self.repository.delete_event_table(folder_path)
        return {
            "success": True, 
            "message": f"Table for event '{folder_path}' deleted successfully if it existed.",
        }
