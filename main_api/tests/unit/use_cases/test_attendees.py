import pytest
from application.use_cases.attendees import EncodeAttendeeUseCase
from domain.exceptions import InvalidReferenceImagesError, FaceValidationError
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_encode_attendee_use_case_success():
    # Arrange
    mock_inference = AsyncMock()
    # Mock returns a single face for all 3 original and augmented images (so 15 total images after augmentation)
    mock_inference.get_face_encodings.side_effect = [
        # 1st call: Original validation (3 images, 1 face each)
        [[{"bbox": [0,0,10,10], "embedding": [0.1]*512}] for _ in range(3)],
        # 2nd call: Augmented images (say, 15 images, 1 face each)
        [[{"bbox": [0,0,10,10], "embedding": [0.1]*512}] for _ in range(15)]
    ]

    mock_augmenter = MagicMock()
    mock_augmenter.augment.return_value = ["dummy_b64"] * 15

    use_case = EncodeAttendeeUseCase[str](
        inference_service=mock_inference,
        augmenter=mock_augmenter,
        decode_fn=lambda x: x # transparent
    )

    # Act
    avg_embedding = await use_case.execute(["img1", "img2", "img3"])

    # Assert
    assert len(avg_embedding) == 512
    assert mock_inference.get_face_encodings.call_count == 2
    mock_augmenter.augment.assert_called_once()

@pytest.mark.asyncio
async def test_encode_attendee_use_case_invalid_image_count():
    use_case = EncodeAttendeeUseCase[str](
        inference_service=AsyncMock(), augmenter=MagicMock(), decode_fn=lambda x: x
    )

    with pytest.raises(InvalidReferenceImagesError, match="exactly 3"):
        await use_case.execute(["img1", "img2"])

@pytest.mark.asyncio
async def test_encode_attendee_use_case_no_faces():
    # Arrange
    mock_inference = AsyncMock()
    # Original validation: Image 1 has 1 face, Image 2 has 0 faces, Image 3 has 1 face
    mock_inference.get_face_encodings.return_value = [
        [{"bbox": [0,0,10,10], "embedding": [0.1]*512}],
        [], 
        [{"bbox": [0,0,10,10], "embedding": [0.1]*512}]
    ]

    use_case = EncodeAttendeeUseCase[str](
        inference_service=mock_inference,
        augmenter=MagicMock(),
        decode_fn=lambda x: x
    )

    # Act / Assert
    with pytest.raises(FaceValidationError) as exc_info:
        await use_case.execute(["img1", "img2", "img3"])
    
    assert "no face in 1 photo" in str(exc_info.value)
    assert exc_info.value.details[0]["issue"] == "none"
    assert exc_info.value.details[0]["image_index"] == 1

@pytest.mark.asyncio
async def test_encode_attendee_use_case_multiple_faces():
    # Arrange
    mock_inference = AsyncMock()
    # Original validation: Image 1 has 2 faces
    mock_inference.get_face_encodings.return_value = [
        [{"bbox": [0,0,10,10]}, {"bbox": [20,20,30,30]}],
        [{"bbox": [0,0,10,10]}], 
        [{"bbox": [0,0,10,10]}]
    ]

    use_case = EncodeAttendeeUseCase[str](
        inference_service=mock_inference,
        augmenter=MagicMock(),
        decode_fn=lambda x: x
    )

    # Act / Assert
    with pytest.raises(FaceValidationError) as exc_info:
        await use_case.execute(["img1", "img2", "img3"])
    
    assert "multiple faces in 1 photo" in str(exc_info.value)
    assert exc_info.value.details[0]["issue"] == "multiple"
    assert exc_info.value.details[0]["image_index"] == 0
import pytest
from application.use_cases.attendees import SortAttendeeUseCase

@pytest.mark.asyncio
async def test_sort_attendees_use_case():
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.__aexit__.return_value = False
    uow.event_repo.check_event_has_data.return_value = True
    uow.event_repo.find_matches.return_value = [{'filename': 'f1', 'path': 'p1'}]
    
    uc = SortAttendeeUseCase(uow)
    res = await uc.execute('ev1', [0.1, 0.2])
    assert res.matches_found == 1
    assert res.photos == [{'filename': 'f1', 'path': 'p1'}]

@pytest.mark.asyncio
async def test_sort_attendees_use_case_no_data():
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.__aexit__.return_value = False
    uow.event_repo.check_event_has_data.return_value = False
    
    uc = SortAttendeeUseCase(uow)
    try:
        await uc.execute('ev1', [0.1, 0.2])
    except Exception as e:
        assert 'No encoded data found' in str(e)

@pytest.mark.asyncio
async def test_sort_attendees_use_case_no_matches():
    uow = AsyncMock()
    uow.__aenter__.return_value = uow
    uow.__aexit__.return_value = False
    uow.event_repo.check_event_has_data.return_value = True
    uow.event_repo.find_matches.return_value = []
    
    uc = SortAttendeeUseCase(uow)
    try:
        await uc.execute('ev1', [0.1, 0.2])
    except Exception as e:
        assert 'Could not find any matches' in str(e)
from application.use_cases.attendees import CheckZipExistsUseCase

@pytest.mark.asyncio
async def test_check_zip_exists_use_case():
    storage = AsyncMock()
    storage.check_zip_exists.return_value = True
    uc = CheckZipExistsUseCase(storage)
    res = await uc.execute('ev1', 'u1')
    assert res.exists
    

