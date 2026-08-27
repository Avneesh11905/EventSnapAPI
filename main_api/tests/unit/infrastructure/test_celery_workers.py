from unittest.mock import patch, MagicMock, AsyncMock
from infrastructure.queue.celery_workers import encode_event_task, encode_image_batch_task, create_event_zip_task, delete_event_data_task, delete_image_batch_task

@patch("infrastructure.queue.celery_workers.get_container")
def test_encode_event_task(mock_get_container):
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock()
    mock_get_container.return_value.process_event_encoding_use_case.return_value = mock_use_case
    
    # Mock update_state on the task object itself
    with patch.object(encode_event_task, 'update_state'):
        encode_event_task("folder", 0.5, 0.4)
        mock_use_case.execute.assert_called_once()

@patch("infrastructure.queue.celery_workers.get_container")
def test_encode_image_batch_task(mock_get_container):
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock()
    mock_get_container.return_value.encode_image_batch_use_case.return_value = mock_use_case
    
    encode_image_batch_task("event1", ["img1"], 0.5, 0.4)
    mock_use_case.execute.assert_called_once()

@patch("infrastructure.queue.celery_workers.get_container")
def test_create_event_zip_task(mock_get_container):
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock()
    mock_get_container.return_value.create_event_zip_use_case.return_value = mock_use_case
    
    with patch.object(create_event_zip_task, 'update_state'):
        create_event_zip_task("event", "user", [])
        mock_use_case.execute.assert_called_once()
@patch('infrastructure.queue.celery_workers.get_container')
def test_delete_event_data_task(mock_get_container):
    mock_container = MagicMock()
    mock_get_container.return_value = mock_container
    
    mock_uow = AsyncMock()
    mock_uow.__aenter__.return_value = mock_uow
    mock_uow.__aexit__.return_value = None
    mock_container.uow.return_value = mock_uow
    
    mock_storage = AsyncMock()
    mock_container.storage_service.return_value = mock_storage
    
    mock_cache = AsyncMock()
    mock_container.cache_service.return_value = mock_cache
    
    res = delete_event_data_task('ev1', 'id1')
    assert res['success']

@patch('infrastructure.queue.celery_workers.get_container')
def test_delete_image_batch_task(mock_get_container):
    mock_container = MagicMock()
    mock_get_container.return_value = mock_container
    
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value={'status': 'ok'})
    mock_container.delete_image_batch_use_case.return_value = mock_use_case
    
    res = delete_image_batch_task('ev1', ['k1'])
    assert res['status'] == 'ok'
