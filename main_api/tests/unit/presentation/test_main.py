from fastapi.testclient import TestClient
from presentation.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json()['status'] in ['ok', 'degraded']

def test_get_task_status():
    from unittest.mock import patch, MagicMock
    from application.dtos import TaskStatusDTO
    with patch('presentation.api.main.container') as mock_container:
        mock_use_case = MagicMock()
        mock_use_case.execute.return_value = TaskStatusDTO(state='SUCCESS', info={}, result={})
        mock_container.check_encoding_status_use_case.return_value = mock_use_case
        response = client.get('/api/tasks/t1')
        assert response.status_code == 200
