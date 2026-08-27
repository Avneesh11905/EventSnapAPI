from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_delete_bulk_api(test_client, app_container):
    from unittest.mock import patch

    with patch("presentation.api.routers.images.delete_image_batch_task.delay") as mock_delay:
        mock_task = MagicMock()
        mock_task.id = "delete-task-123"
        mock_delay.return_value = mock_task

        response = await test_client.post(
            "/api/images/delete-bulk", json={"event_code": "EVT", "keys": ["img1.jpg", "img2.jpg"]}
        )

        assert response.status_code == 200
        assert response.json()["task_id"] == "delete-task-123"
        mock_delay.assert_called_once_with(
            event_code="EVT", keys=["img1.jpg", "img2.jpg"], cancel_task_id=None
        )


@pytest.mark.asyncio
async def test_image_status_api(test_client, app_container):
    mock_repo = AsyncMock()
    mock_repo.get_image_status.return_value = {"has_faces": ["img1.jpg"], "no_faces": []}

    class MockUOW:
        def __init__(self):
            self.event_repo = mock_repo

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with app_container.uow.override(MockUOW()):
        response = await test_client.get("/api/images/status/EVT")

        assert response.status_code == 200
        data = response.json()
        assert "img1.jpg" in data["has_faces"]
        mock_repo.get_image_status.assert_called_once_with("EVT")
