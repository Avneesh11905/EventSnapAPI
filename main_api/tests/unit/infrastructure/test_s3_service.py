import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from infrastructure.storage.s3_service import S3StorageServiceB64

@pytest.fixture
def s3_service():
    return S3StorageServiceB64('http://mock', 'test-bucket', 'key', 'secret')

@pytest.mark.asyncio
async def test_list_images(s3_service):
    with patch('infrastructure.storage.s3_service.Session') as mock_session_cls:
        mock_client = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_client
        mock_ctx.__aexit__.return_value = False
        mock_session_cls.return_value.client.return_value = mock_ctx
        
        async def mock_paginate(*args, **kwargs):
            yield {"Contents": [{"Key": "folder/img1.jpg"}, {"Key": "folder/img2.jpg"}]}
            
        mock_paginator = MagicMock()
        mock_paginator.paginate = mock_paginate
        mock_client.get_paginator.return_value = mock_paginator
        
        keys = await s3_service.list_images("folder")
        assert keys == ["folder/img1.jpg", "folder/img2.jpg"]
        mock_client.get_paginator.assert_called_once_with("list_objects_v2")

@pytest.mark.asyncio
async def test_download_image_success(s3_service):
    with patch("infrastructure.storage.s3_service.Session") as mock_session_cls:
        mock_client = AsyncMock()
        
        async def mock_read():
            return b"image_data"
            
        async def mock_get_object(*args, **kwargs):
            mock_body = MagicMock()
            mock_body.read = mock_read
            return {"Body": mock_body}
            
        mock_client.get_object = mock_get_object
        
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_client
        mock_ctx.__aexit__.return_value = False
        mock_session_cls.return_value.client.return_value = mock_ctx
        
        data = await s3_service.download_images(["img1.jpg"])
        print("data is:", data)
        import base64
        expected_b64 = base64.b64encode(b"image_data").decode("utf-8")
        assert data[0] == expected_b64

@pytest.mark.asyncio
async def test_delete_images(s3_service):
    with patch("infrastructure.storage.s3_service.Session") as mock_session_cls:
        mock_client = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_client
        mock_ctx.__aexit__.return_value = False
        mock_session_cls.return_value.client.return_value = mock_ctx
        
        async def mock_delete_objects(*args, **kwargs):
            pass
        mock_client.delete_objects = MagicMock(side_effect=mock_delete_objects)
        
        await s3_service.delete_objects(["img1.jpg", "img2.jpg"])


