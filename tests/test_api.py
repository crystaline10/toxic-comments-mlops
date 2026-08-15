from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

mock_model = MagicMock()
mock_model.predict.return_value = [[1, 0, 0, 0, 1, 0]]


with patch(
    "api.model_loader.load_production_model",
    return_value=mock_model,
):
    from api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.main.log_prediction")
def test_predict_endpoint(mock_log_prediction):
    response = client.post(
        "/predict",
        json={"text": "You are rude and insulting."},
    )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert "prediction" in data
    assert "latency_ms" in data

    assert data["prediction"]["toxic"] == 1
    assert data["prediction"]["insult"] == 1

    mock_log_prediction.assert_called_once()


@patch("api.main.update_feedback")
def test_feedback_endpoint(mock_update_feedback):
    response = client.post(
        "/feedback",
        json={
            "request_id": "test-request-123",
            "is_correct": True,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "feedback recorded"
    assert data["is_correct"] is True

    mock_update_feedback.assert_called_once_with(
        request_id="test-request-123",
        is_correct=True,
    )