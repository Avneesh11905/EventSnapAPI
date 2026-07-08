from typing import Any, Dict, Protocol

class TaskQueueService(Protocol):
    def enqueue_encode_event(self, folder_path: str, max_faces: int, detection_conf: float, nms_threshold: float) -> str:
        """Returns the task ID"""
        pass

    def enqueue_create_zip(self, event_id: str, user_id: str, image_paths: list[dict]) -> str:
        """Returns the task ID"""
        pass

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Returns dict with state, info, result"""
        pass
