import pytest
from src.jobs.scheduler import RefreshJob
from src.stores.sqlite_store import SQLiteFeatureStore
from src.data.mock_generator import generate_mock_payload

def test_idempotent_refresh():
    store = SQLiteFeatureStore()
    for _ in range(50):
        store.save_student(generate_mock_payload())
        
    job = RefreshJob(store, min_interval_seconds=1000)
    # First run
    success_1 = job.run()
    assert success_1 is True
    
    # Second run immediately
    success_2 = job.run()
    assert success_2 is False

def test_shaps_consistency():
    store = SQLiteFeatureStore()
    for _ in range(50):
        store.save_student(generate_mock_payload())
        
    job = RefreshJob(store)
    job.run()
    
    predictor = job.current_predictor
    payload = store.get_all_students()[0]
    res1 = predictor.predict(payload)
    res2 = predictor.predict(payload)
    
    # SHAP outputs mapped to human labels should be strictly deterministic for same payload
    assert [x.feature_label for x in res1.shap_drivers] == [x.feature_label for x in res2.shap_drivers]
