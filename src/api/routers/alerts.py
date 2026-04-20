from fastapi import APIRouter, HTTPException
from typing import Optional, List
from src.core.dependencies import get_feature_store, get_predictor
from src.models.schemas import SuccessScoreOutput

router = APIRouter(prefix="/v1", tags=["Alerts and Consent"])

@router.get("/alerts/exceptions", response_model=List[SuccessScoreOutput])
async def get_exception_alerts(risk_tier: str = "HIGH", horizon: str = "12"):
    store = get_feature_store()
    try:
        predictor = get_predictor()
    except Exception:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
        
    all_students = store.get_all_students()
    alerts = []
    
    # In production, this would be computed offline. Here we compute real-time for MVP:
    for student in all_students:
        score_output = predictor.predict(student)
        
        # Check against requested filters
        if score_output.is_exception_alert:
            # We specifically check the horizon
            for p in score_output.placement_scores:
                if str(p.horizon_months) == horizon and p.risk_tier == risk_tier:
                    alerts.append(score_output)
                    break
                    
    return alerts

@router.post("/consent/revoke")
async def revoke_consent(payload: dict):
    student_id = payload.get("student_id")
    if not student_id:
        raise HTTPException(status_code=400, detail="Missing student_id")
        
    store = get_feature_store()
    if not store.get_student(student_id):
        raise HTTPException(status_code=404, detail="Student not found")
        
    store.revoke_student_signals(student_id)
    return {"status": "success", "message": "Optional signals purged."}
