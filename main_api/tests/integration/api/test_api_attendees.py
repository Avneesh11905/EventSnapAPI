from unittest.mock import AsyncMock, MagicMock

import pytest

from application.dtos import AttendeeSortDTO, ZipCheckDTO


@pytest.mark.asyncio
async def test_encode_attendee_api(test_client, app_container):
    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = [0.1, 0.2, 0.3]

    with app_container.encode_attendee_use_case.override(mock_use_case):
        response = await test_client.post(
            "/api/attendees/encode-attendee/",
            json={"attendee_images_base64": ["base64_img1", "base64_img2", "base64_img3"]},
        )

        assert response.status_code == 200
        assert response.json()["encoding"] == [0.1, 0.2, 0.3]
        mock_use_case.execute.assert_called_once_with(["base64_img1", "base64_img2", "base64_img3"])


@pytest.mark.asyncio
async def test_sort_attendee_api(test_client, app_container):
    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = AttendeeSortDTO(
        event_code="EVT", matches_found=2, photos=["img1.jpg", "img2.jpg"]
    )

    with app_container.sort_attendee_use_case.override(mock_use_case):
        response = await test_client.post(
            "/api/attendees/sort-attendee/",
            json={"event_code": "EVT", "attendee_encoding": [0.1] * 512},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["matches_found"] == 2
        assert data["photos"] == ["img1.jpg", "img2.jpg"]
        mock_use_case.execute.assert_called_once()


@pytest.mark.asyncio
async def test_generate_zip_api(test_client, app_container):
    mock_use_case = MagicMock()
    mock_use_case.execute.return_value = "task-zip-123"

    with app_container.generate_zip_use_case.override(mock_use_case):
        response = await test_client.post(
            "/api/attendees/generate-zip/",
            json={
                "event_id": "EVT",
                "user_id": "USER1",
                "image_paths": [{"url": "http", "key": "img1.jpg"}],
            },
        )

        assert response.status_code == 200
        assert response.json()["task_id"] == "task-zip-123"
        mock_use_case.execute.assert_called_once()


@pytest.mark.asyncio
async def test_check_zip_api(test_client, app_container):
    mock_use_case = AsyncMock()
    mock_use_case.execute.return_value = ZipCheckDTO(
        exists=True, zip_path="path/to.zip", filename="to.zip"
    )

    with app_container.check_zip_exists_use_case.override(mock_use_case):
        response = await test_client.get("/api/attendees/check-zip/EVT/USER1")

        assert response.status_code == 200
        data = response.json()
        assert data["exists"] is True
        assert data["zip_path"] == "path/to.zip"
