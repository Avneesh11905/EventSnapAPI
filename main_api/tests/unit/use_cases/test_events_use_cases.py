from unittest.mock import AsyncMock, MagicMock

import pytest

from application.dtos import TaskStatusDTO
from application.use_cases.events import (
    CheckEncodingStatusUseCase,
    DeleteEventDataUseCase,
)


def test_check_encoding_status():
    mock_queue = MagicMock()
    mock_queue.get_task_status.return_value = TaskStatusDTO(state="SUCCESS")

    use_case = CheckEncodingStatusUseCase(queue_service=mock_queue)
    result = use_case.execute("task-123")

    assert result.state == "SUCCESS"
    mock_queue.get_task_status.assert_called_once_with("task-123")


@pytest.mark.asyncio
async def test_delete_event_data():
    mock_queue = MagicMock()
    mock_queue.cancel_event_tasks = AsyncMock()

    use_case = DeleteEventDataUseCase(queue_service=mock_queue)
    result = await use_case.execute("TEST_EVENT", "event-id-123")

    assert result.success is True
    mock_queue.cancel_event_tasks.assert_called_once_with("TEST_EVENT")
    mock_queue.enqueue_delete_event.assert_called_once_with("TEST_EVENT", "event-id-123")
