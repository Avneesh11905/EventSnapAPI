from application.ports.queue import ITaskQueueService
from celery.result import AsyncResult
from application.dtos import TaskStatusDTO
from infrastructure.queue.celery_app import celery_app


class CeleryTaskQueueService(ITaskQueueService):
    def enqueue_encode_event(
        self,
        folder_path: str,
        detection_conf: float,
        nms_threshold: float,
    ) -> str:
        from infrastructure.queue.celery_workers import encode_event_task

        task = encode_event_task.delay(folder_path, detection_conf, nms_threshold)
        return task.id

    def enqueue_encode_group(
        self,
        event_code: str,
        chunks: list[list[str]],
        detection_conf: float,
        nms_threshold: float,
    ) -> str:
        from celery import group
        from infrastructure.queue.celery_workers import encode_image_batch_task

        job = group(
            encode_image_batch_task.s(event_code, chunk, detection_conf, nms_threshold)
            for chunk in chunks
        )
        group_res = job.apply_async()
        group_res.save()
        return group_res.id

    def enqueue_create_zip(
        self, event_id: str, user_id: str, image_paths: list[dict]
    ) -> str:
        from infrastructure.queue.celery_workers import create_event_zip_task

        task = create_event_zip_task.delay(event_id, user_id, image_paths)
        return task.id

    def enqueue_delete_event(self, event_code: str, event_id: str | None = None) -> str:
        from infrastructure.queue.celery_workers import delete_event_data_task

        task = delete_event_data_task.delay(event_code, event_id)
        return task.id

    def get_task_status(self, task_id: str) -> TaskStatusDTO:
        res = AsyncResult(task_id, app=celery_app)

        status_info: dict[str, str | int] = {}
        status_result = None
        state = res.state

        if res.ready():
            if res.successful():
                status_result = (
                    res.result
                    if isinstance(res.result, dict)
                    else {"result": res.result}
                )
                # Check if this task delegated to a group
                if isinstance(res.result, dict) and "group_id" in res.result:
                    group_id = res.result["group_id"]
                    if group_id:
                        from celery.result import GroupResult

                        group = GroupResult.restore(group_id, app=celery_app)
                        if group:
                            completed = group.completed_count()
                            total = len(group)
                            if not group.ready():
                                state = "PROCESSING"
                                status_info["progress"] = (
                                    int((completed / total) * 100) if total else 0
                                )
                                # Multiply batches by batch size to get image count, capped at total
                                from config import settings

                                total_images = res.result.get("total", total)
                                status_info["processed"] = min(
                                    completed * settings.INFERENCE_BATCH_SIZE,
                                    total_images,
                                )
                                status_info["total"] = total_images
            else:
                try:
                    status_info["error"] = str(res.result)
                except Exception:
                    status_info["error"] = (
                        "Task failed, but the result/exception could not be parsed."
                    )
        else:
            info = res.info
            if isinstance(info, dict):
                status_info.update(info)

        return TaskStatusDTO(
            state=state, info=status_info if status_info else None, result=status_result
        )
