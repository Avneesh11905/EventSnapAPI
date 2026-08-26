from typing import Protocol
from application.dtos import TaskStatusDTO


class ITaskQueueService(Protocol):
    def enqueue_encode_event(
        self,
        folder_path: str,
        detection_conf: float,
        nms_threshold: float,
    ) -> str:
        """Returns the task ID"""
        pass

    def enqueue_encode_group(
        self,
        event_code: str,
        chunks: list[list[str]],
        detection_conf: float,
        nms_threshold: float,
    ) -> str:
        """Returns the group ID"""
        pass

    def enqueue_create_zip(
        self, event_id: str, user_id: str, image_paths: list[dict]
    ) -> str:
        """Returns the task ID"""
        pass

    def get_task_status(self, task_id: str) -> TaskStatusDTO:
        """Returns TaskStatusDTO with state, info, result"""
        pass

    def enqueue_delete_event(self, event_code: str, event_id: str | None = None) -> str:
        """Enqueues a task to delete all event data from the database and storage."""
        pass

    async def cancel_event_tasks(self, event_code: str) -> None:
        """Cancels all active and queued tasks related to the given event_code."""
        pass
