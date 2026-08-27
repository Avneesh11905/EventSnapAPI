import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_get_event_count_mocked(test_client, app_container):
    # Arrange
    mock_repo = AsyncMock()
    mock_repo.check_event_has_data.return_value = True
    mock_repo.get_encoded_count.return_value = 2

    # Override the UOW in the container
    class MockUOW:
        def __init__(self):
            self.event_repo = mock_repo
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with app_container.uow.override(MockUOW()):
        # Act
        response = await test_client.get("/api/events/encode-count/TEST_EVENT")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["encoded_count"] == 2

@pytest.mark.asyncio
async def test_encode_event_starts_pipeline(test_client, app_container):
    # Arrange
    mock_repo = AsyncMock()
    mock_repo.get_already_encoded_images.return_value = {"img1.jpg"}
    
    mock_s3 = AsyncMock()
    mock_s3.list_images.return_value = ["img1.jpg", "img2.jpg", "img3.jpg"]

    mock_queue = MagicMock()
    mock_queue.enqueue_encode_event.return_value = "dummy-task-id"
    
    class MockUOW:
        def __init__(self):
            self.event_repo = mock_repo
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with app_container.uow.override(MockUOW()), \
         app_container.storage_service.override(mock_s3), \
         app_container.queue_service.override(mock_queue):
         
        # Act
        response = await test_client.post("/api/events/encode-event/", json={"event_code": "TEST_EVENT", "detection_conf": 0.5, "nms_threshold": 0.4})

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "dummy-task-id"
        
        # Verify queue was called
        mock_queue.enqueue_encode_event.assert_called_once()
