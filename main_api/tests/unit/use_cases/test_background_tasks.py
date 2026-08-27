import pytest
from unittest.mock import MagicMock, AsyncMock
from application.use_cases.background_tasks import EncodeImageBatchUseCase
from domain.exceptions import TaskCanceledError

class MockUOW:
    def __init__(self, mock_repo=None):
        if mock_repo is None:
            self.event_repo = AsyncMock()
        else:
            self.event_repo = mock_repo
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    async def commit(self):
        pass

@pytest.mark.asyncio
async def test_encode_image_batch_success():
    # Arrange
    mock_repo = AsyncMock()
    mock_storage = AsyncMock()
    # 2 images return normally, 1 returns an exception to test failure skipping
    mock_storage.download_images.return_value = ["img1_data", Exception("Network error"), "img3_data"]
    
    mock_inference = AsyncMock()
    mock_inference.get_face_encodings.return_value = [
        [{"embedding": [0.1]*512, "confidence": 0.99}, {"embedding": [0.2]*512, "confidence": 0.85}], # img1 has 2 faces
        [] # img3 has 0 faces
    ]
    
    mock_cache = AsyncMock()
    mock_cache.get_flag.return_value = False # Not cancelled
    
    use_case = EncodeImageBatchUseCase(
        storage_service=mock_storage,
        inference_service=mock_inference,
        uow=MockUOW(mock_repo),
        cache_service=mock_cache
    )
    
    # Act
    keys = ["img1.jpg", "img2.jpg", "img3.jpg"]
    result = await use_case.execute("TEST_EVENT", keys, 0.5, 0.4)
    
    # Assert
    assert result["total"] == 3
    assert result["encoded"] == 1 # 1 image had faces
    assert result["no_encodings_found"] == 1 # 1 image had 0 faces
    assert "img2.jpg" in result["failures"]
    
    # Check what was saved
    mock_repo.save_encodings.assert_called_once()
    saved_encodings = mock_repo.save_encodings.call_args[0][0]
    assert len(saved_encodings) == 2
    assert saved_encodings[0].image_path == "img1.jpg"
    
    mock_repo.save_processed_images.assert_called_once()
    saved_processed = mock_repo.save_processed_images.call_args[0][0]
    assert len(saved_processed) == 2 # Only valid images are logged (img1, img3)

@pytest.mark.asyncio
async def test_encode_image_batch_canceled():
    # Arrange
    mock_cache = AsyncMock()
    mock_cache.get_flag.return_value = True # Task was cancelled before start!
    
    use_case = EncodeImageBatchUseCase(
        storage_service=AsyncMock(),
        inference_service=AsyncMock(),
        uow=AsyncMock(), # Irrelevant, shouldn't reach here
        cache_service=mock_cache
    )
    
    # Act / Assert
    with pytest.raises(TaskCanceledError, match="deleted"):
        await use_case.execute("TEST_EVENT", ["img1.jpg"], 0.5, 0.4)

@pytest.mark.asyncio
async def test_process_event_encoding_use_case():
    from application.use_cases.background_tasks import ProcessEventEncodingUseCase
    mock_storage = AsyncMock()
    mock_storage.list_images.return_value = ['event/TEST/thumbs/img1.jpg', 'event/TEST/thumbs/img2.jpg']
    mock_uow = MockUOW()
    mock_queue = MagicMock()
    mock_cache = AsyncMock()
    mock_cache.acquire_lock.return_value = True
    use_case = ProcessEventEncodingUseCase(mock_storage, mock_uow, mock_queue, mock_cache)
    def cb(state, meta): pass
    res = await use_case.execute('TEST', 0.5, 0.4, cb)
    assert res.total == 2


@pytest.mark.asyncio
async def test_create_event_zip_use_case():
    from application.use_cases.background_tasks import CreateEventZipUseCase
    mock_storage = AsyncMock()
    use_case = CreateEventZipUseCase(mock_storage)
    def cb(state, meta): pass
    await use_case.execute('ev1', 'usr1', [{'path':'a','type':'b'}], cb)


@pytest.mark.asyncio
async def test_delete_image_batch_use_case():
    from application.use_cases.background_tasks import DeleteImageBatchUseCase
    mock_storage = AsyncMock()
    mock_uow = MockUOW()
    use_case = DeleteImageBatchUseCase(mock_storage, mock_uow)
    res = await use_case.execute('TEST', ['img1.jpg'])
    assert res == {'success': True, 'deleted': 1}
