from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)


def test_inference_http_route_success(mock_container, dummy_image_base64):
    """
    Test the HTTP POST route using base64 encoded images.
    Relies on `mock_container` to swap the heavy ONNX models for instant Mocks.
    """
    payload = {
        "inputs": [dummy_image_base64, dummy_image_base64],
        "parameters": {"max_faces": "0", "detection_conf": 0.5, "nms_threshold": 0.4},
    }

    response = client.post("/", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()
    assert "batch_faces" in data

    # We sent 2 images, so we expect 2 batches of faces
    assert len(data["batch_faces"]) == 2

    # Each image gets 1 fake face from our MockFaceDetector
    for faces in data["batch_faces"]:
        assert len(faces) == 1

        face = faces[0]
        assert "bbox" in face
        assert "confidence" in face
        assert "embedding" in face
        assert len(face["embedding"]) == 512


def test_inference_http_route_invalid_base64(mock_container):
    payload = {
        "inputs": ["not-a-valid-base64-string"],
        "parameters": {"max_faces": "0"},
    }
    response = client.post("/", json=payload)
    assert response.status_code == 422  # InvalidInputFormatError returns 422
