import uuid

import pytest

from application.dtos import EventEncodingDTO


@pytest.mark.asyncio
async def test_postgres_event_repository_lifecycle(app_container, db_session):
    # The UOW provides the repository instance using Dependency Injector
    uow = app_container.uow()

    event_code = "TEST_EVENT_123"

    async with uow:
        # Act: save processed images
        await uow.event_repo.save_processed_images(
            [
                {"event_code": event_code, "image_path": "img1.jpg"},
                {"event_code": event_code, "image_path": "img2.jpg"},
            ]
        )

        # Act: save encodings
        dummy_embedding = [0.1] * 512
        encodings = [
            EventEncodingDTO(
                id=uuid.uuid4(),
                event_code=event_code,
                image_path="img1.jpg",
                embedding=dummy_embedding,
                confidence=0.99,
            )
        ]
        await uow.event_repo.save_encodings(encodings)
        await uow.commit()

    # Act: fetch status
    async with uow:
        status = await uow.event_repo.get_image_status(event_code)

        # Assert
        assert "img1.jpg" in status["has_faces"]
        assert "img2.jpg" in status["no_faces"]

        # Act: find matches
        matches = await uow.event_repo.find_matches(event_code, dummy_embedding, 0.5)

        # Assert
        assert matches == ["img1.jpg"]

        # Act: delete event data
        await uow.event_repo.delete_event_data(event_code)
        await uow.commit()

    # Act: Verify deletion
    async with uow:
        processed = await uow.event_repo.get_already_encoded_images(event_code)
        assert len(processed) == 0
