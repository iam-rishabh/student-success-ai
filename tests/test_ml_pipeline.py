import pytest
import numpy as np
import pandas as pd
from src.data.mock_generator import generate_mock_payload
from src.ml.trainer import ModelTrainer
from src.ml.predictor import Predictor
from src.ml.drift import compute_psi, DriftMonitor

@pytest.fixture
def mock_payloads():
    return [generate_mock_payload() for _ in range(50)]

@pytest.fixture
def trained_artifacts(mock_payloads):
    trainer = ModelTrainer()
    artifacts = trainer.train(mock_payloads)
    return artifacts

def test_feature_assembly(mock_payloads):
    from src.ml.feature_assembler import FeatureAssembler
    assembler = FeatureAssembler()
    df = assembler.assemble(mock_payloads)
    
    assert len(df) == 50
    assert "cgpa_normalized" in df.columns
    assert "institute_tier" in df.columns
    # Check nullable features imputation worked (no NaNs)
    assert df["grit_score"].isna().sum() == 0

def test_ensemble_pipeline_training(trained_artifacts):
    models = trained_artifacts["models"]
    assert "3" in models
    assert "6" in models
    assert "12" in models
    
def test_predictor_outputs_schema(trained_artifacts, mock_payloads):
    predictor = Predictor(
        feature_assembler=trained_artifacts["assembler"],
        models=trained_artifacts["models"],
        salary_model=trained_artifacts["salary_model"],
        explainer=trained_artifacts["explainer"]
    )
    
    single_payload = mock_payloads[0]
    output = predictor.predict(single_payload)
    
    assert output.student_id == single_payload.student_id
    assert len(output.placement_scores) == 3
    
    # SHAP outputs
    assert len(output.shap_drivers) <= 6
    for driver in output.shap_drivers:
        assert driver.direction in ["positive", "negative"]
        assert driver.feature_label != ""
        
    # Salary estimate
    assert output.salary_estimate.lower_lpa <= output.salary_estimate.median_lpa
    assert output.salary_estimate.median_lpa <= output.salary_estimate.upper_lpa

def test_drift_computation():
    base = np.random.normal(0, 1, 1000)
    current = np.random.normal(0.01, 1, 1000)
    
    psi = compute_psi(base, current)
    assert psi < 0.1  # Minimal drift

    shifted = np.random.normal(2.0, 1, 1000)
    psi_shifted = compute_psi(base, shifted)
    assert psi_shifted > 0.2  # Massive drift
