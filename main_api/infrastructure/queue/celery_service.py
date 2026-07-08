from application.ports.queue import TaskQueueService
from celery.result import AsyncResult
from typing import Any, Dict
from infrastructure.queue.celery_app import celery_app

class CeleryTaskQueueService(TaskQueueService):
    def enqueue_encode_event(self, folder_path: str, max_faces: int, detection_conf: float, nms_threshold: float) -> str:
        from infrastructure.queue.celery_workers import encode_event_task
        task = encode_event_task.delay(folder_path, max_faces, detection_conf, nms_threshold)
        return task.id

    def enqueue_create_zip(self, event_id: str, user_id: str, image_paths: list[dict]) -> str:
        from infrastructure.queue.celery_workers import create_event_zip_task
        task = create_event_zip_task.delay(event_id, user_id, image_paths)
        return task.id

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        res = AsyncResult(task_id, app=celery_app)
        
        response = {
            "task_id": task_id,
            "status": res.state,
        }
        
        if res.ready():
            if res.successful():
                response["result"] = res.result
            else:
                try:
                    response["error"] = str(res.result)
                except Exception as e:
                    response["error"] = "Task failed, but the result/exception could not be parsed."
        else:
            if isinstance(res.info, dict):
                response.update(res.info)
                
        return response
