import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.data.mock_generator import generate_mock_payload

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_teardown():
    # FastAPI test client automatically runs startup events so DB is seeded 
    yield

def test_health_check():
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_version"] is not None

def test_ingest_student():
    payload = generate_mock_payload()
    response = client.post("/v1/students/ingest", json=payload.model_dump(mode='json'))
    assert response.status_code == 200
    assert response.json()["student_id"] == payload.student_id

def test_ingest_invalid_schema():
    payload = generate_mock_payload().model_dump(mode='json')
    payload["academic"]["cgpa"] = 15.0 # Invalid
    response = client.post("/v1/students/ingest", json=payload)
    assert response.status_code == 422

def test_prediction_score():
    # Assuming testclient seeded data
    from src.core.dependencies import get_feature_store
    store = get_feature_store()
    student = store.get_all_students()[0]
    
    response = client.post("/v1/predictions/score", json={"student_id": student.student_id})
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == student.student_id
    assert len(data["placement_scores"]) == 3
    assert len(data["shap_drivers"]) > 0

def test_get_exception_alerts():
    response = client.get("/v1/alerts/exceptions?risk_tier=HIGH&horizon=12")
    assert response.status_code == 200

def test_revoke_consent():
    from src.core.dependencies import get_feature_store
    store = get_feature_store()
    student = store.get_all_students()[0]
    result = client.post("/v1/consent/revoke", json={"student_id": student.student_id})
    assert result.status_code == 200
    
    # Assert signals deleted in store
    updated_student = store.get_student(student.student_id)
    assert updated_student.psych is None
    assert updated_student.behavioral is None
    assert updated_student.consent.psycho_signals_consent is False
