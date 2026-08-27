from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.storage.s3_service import S3StorageServiceBytes


@pytest.fixture
def s3_service():
    return S3StorageServiceBytes("http://test", "test_bucket", "key", "secret")


@pytest.mark.asyncio
async def test_delete_folder(s3_service):
    with patch("infrastructure.storage.s3_service.Session") as mock_session_cls:
        mock_client = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_client
        mock_ctx.__aexit__.return_value = False
        mock_session_cls.return_value.client.return_value = mock_ctx

        mock_paginator = MagicMock()

        async def mock_paginate(*args, **kwargs):
            yield {"Contents": [{"Key": "f1"}, {"Key": "f2"}]}

        mock_paginator.paginate = mock_paginate
        mock_client.get_paginator = MagicMock(return_value=mock_paginator)

        await s3_service.delete_folder("folder/")
        mock_client.delete_objects.assert_called()


@pytest.mark.asyncio
async def test_create_zip_from_images(s3_service):
    with patch("infrastructure.storage.s3_service.Session") as mock_session_cls, patch("os.remove"):
        mock_client = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__.return_value = mock_client
        mock_ctx.__aexit__.return_value = False
        mock_session_cls.return_value.client.return_value = mock_ctx

        mock_client.get_object = AsyncMock(
            return_value={"Body": MagicMock(read=AsyncMock(return_value=b"data"))}
        )

        await s3_service.create_zip_from_images("zip_path", [{"filename": "f1.jpg", "path": "k1"}])
        mock_client.put_object.assert_called()
