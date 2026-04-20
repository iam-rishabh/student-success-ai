import pytest
from pydantic import ValidationError
from datetime import datetime, timedelta
from src.models.schemas import (
    StudentIngestionPayload, ConsentLayer, AcademicProfile, 
    CourseTypeEnum, SalaryEstimate
)
from src.data.mock_generator import generate_mock_payload

def test_valid_payload_ingestion():
    payload = generate_mock_payload()
    assert payload.student_id.startswith("STU-")
    assert payload.consent.academic_data_consent is True

def test_academic_profile_boundaries():
    valid = generate_mock_payload().academic.model_dump()
    
    # Negative CGPA should fail
    valid_copy = valid.copy()
    valid_copy["cgpa"] = -0.1
    with pytest.raises(ValidationError) as exc_info:
        AcademicProfile(**valid_copy)
    assert "cgpa" in str(exc_info.value)
    
    # CGPA > 10 should fail
    valid_copy["cgpa"] = 10.1
    with pytest.raises(ValidationError) as exc_info:
        AcademicProfile(**valid_copy)
    assert "cgpa" in str(exc_info.value)

def test_mandatory_consent_false_raises():
    valid = generate_mock_payload().consent.model_dump()
    
    # Test academic_data_consent
    valid_copy = valid.copy()
    valid_copy["academic_data_consent"] = False
    with pytest.raises(ValidationError) as exc_info:
        ConsentLayer(**valid_copy)
    assert "Core consents are mandatory" in str(exc_info.value)

    # Test lender_sharing_consent
    valid_copy = valid.copy()
    valid_copy["lender_sharing_consent"] = False
    with pytest.raises(ValidationError) as exc_info:
        ConsentLayer(**valid_copy)
    assert "Core consents are mandatory" in str(exc_info.value)

def test_student_id_format():
    payload = generate_mock_payload()
    valid = payload.model_dump()
    valid["student_id"] = "INVALID-ID-FORMAT"
    with pytest.raises(ValidationError) as exc_info:
        StudentIngestionPayload(**valid)
    assert "student_id" in str(exc_info.value)

def test_semester_cgpas_empty_raises():
    valid = generate_mock_payload().academic.model_dump()
    valid["semester_cgpas"] = []
    with pytest.raises(ValidationError) as exc_info:
        AcademicProfile(**valid)
    assert "semester_cgpas" in str(exc_info.value)

def test_consent_revoked_at_validation():
    valid = generate_mock_payload().consent.model_dump()
    valid["consented_at"] = datetime.utcnow()
    valid["revoked_at"] = datetime.utcnow() - timedelta(days=1)
    
    with pytest.raises(ValidationError) as exc_info:
        ConsentLayer(**valid)
    assert "cannot be before" in str(exc_info.value)

def test_psych_data_without_consent_raises():
    mock = generate_mock_payload()
    valid = mock.model_dump()
    valid["consent"]["psycho_signals_consent"] = False
    # Since psych payload is present, it should fail
    with pytest.raises(ValidationError) as exc_info:
        StudentIngestionPayload(**valid)
    assert "Psych signals provided without consent" in str(exc_info.value)

def test_salary_estimate_bounds():
    # Valid
    SalaryEstimate(lower_lpa=5.0, median_lpa=6.0, upper_lpa=8.0)
    
    # Invalid
    with pytest.raises(ValidationError) as exc_info:
        SalaryEstimate(lower_lpa=8.0, median_lpa=6.0, upper_lpa=5.0)
    assert "lower_lpa <= median_lpa <= upper_lpa" in str(exc_info.value)
