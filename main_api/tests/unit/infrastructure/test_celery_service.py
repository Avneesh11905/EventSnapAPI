from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.queue.celery_service import CeleryTaskQueueService


@pytest.fixture
def cache_service():
    return AsyncMock()


@pytest.fixture
def celery_service(cache_service):
    return CeleryTaskQueueService(cache_service=cache_service)


def test_enqueue_encode_event(celery_service):
    with patch("infrastructure.queue.celery_workers.encode_event_task.delay") as mock_delay:
        mock_delay.return_value.id = "task-123"
        task_id = celery_service.enqueue_encode_event("folder", 0.5, 0.4)
        assert task_id == "task-123"
        mock_delay.assert_called_once_with("folder", 0.5, 0.4)


def test_enqueue_encode_group(celery_service):
    with patch("celery.group") as mock_group:
        mock_apply = MagicMock()
        mock_apply.id = "group-123"
        mock_group.return_value.apply_async.return_value = mock_apply

        task_id = celery_service.enqueue_encode_group("event1", [["img1"], ["img2"]], 0.5, 0.4)
        assert task_id == "group-123"
        mock_group.return_value.apply_async.assert_called_once()
        mock_apply.save.assert_called_once()


def test_enqueue_create_zip(celery_service):
    with patch("infrastructure.queue.celery_workers.create_event_zip_task.delay") as mock_delay:
        mock_delay.return_value.id = "task-123"
        task_id = celery_service.enqueue_create_zip("event", "user", [])
        assert task_id == "task-123"
        mock_delay.assert_called_once_with("event", "user", [])


@pytest.mark.asyncio
async def test_cancel_event_tasks(celery_service, cache_service):
    with (
        patch("infrastructure.queue.celery_service.celery_app.control.revoke") as mock_revoke,
        patch("infrastructure.queue.celery_service.celery_app.control.inspect") as mock_inspect,
    ):
        mock_inspect.return_value.active.return_value = {
            "worker1": [
                {"id": "t1", "name": "encode_event_task", "args": ["event1"]},
                {"id": "t2", "name": "other_task", "args": ["event2"]},
                {"id": "t3", "name": "encode_image_batch_task", "args": ["event1"]},
            ]
        }

        await celery_service.cancel_event_tasks("event1")

        cache_service.set_flag.assert_called_once_with("cancel_encode:event1", expiration=3600)
        mock_revoke.assert_any_call("t1", terminate=True)
        mock_revoke.assert_any_call("t3", terminate=True)


def test_get_task_status(celery_service):
    from unittest.mock import MagicMock, patch

    with patch("infrastructure.queue.celery_service.AsyncResult") as mock_result_cls:
        mock_res = MagicMock()
        mock_res.state = "SUCCESS"
        mock_res.ready.return_value = True
        mock_res.successful.return_value = True
        mock_res.result = {"key": "val"}
        mock_result_cls.return_value = mock_res

        dto = celery_service.get_task_status("t1")
        assert dto.state == "SUCCESS"
        assert dto.result == {"key": "val"}


def test_get_task_status_pending(celery_service):
    from unittest.mock import MagicMock, patch

    with patch("infrastructure.queue.celery_service.AsyncResult") as mock_result_cls:
        mock_res = MagicMock()
        mock_res.state = "PENDING"
        mock_res.ready.return_value = False
        mock_res.info = {"step": 1}
        mock_result_cls.return_value = mock_res

        dto = celery_service.get_task_status("t1")
        assert dto.state == "PENDING"
        assert dto.info == {"step": 1}


def test_get_task_status_group(celery_service):
    from unittest.mock import MagicMock, patch

    with (
        patch("infrastructure.queue.celery_service.AsyncResult") as mock_result_cls,
        patch("celery.result.GroupResult.restore") as mock_restore,
    ):
        mock_res = MagicMock()
        mock_res.state = "SUCCESS"
        mock_res.ready.return_value = True
        mock_res.successful.return_value = True
        mock_res.result = {"group_id": "g1"}
        mock_result_cls.return_value = mock_res

        mock_group = MagicMock()
        mock_group.completed_count.return_value = 5
        mock_group.ready.return_value = False
        mock_group.__len__.return_value = 10
        mock_sub1 = MagicMock(state="SUCCESS", result={"faces": 2})
        mock_sub2 = MagicMock(state="FAILURE", result=Exception("err"))
        mock_group.results = [mock_sub1, mock_sub2]
        mock_restore.return_value = mock_group

        dto = celery_service.get_task_status("t1")
        assert dto.state == "PROCESSING"
        assert dto.info["progress"] == 50
