from fastapi import APIRouter, Depends, HTTPException
import asyncio
from typing import Dict, Any
from src.core.dependencies import get_predictor, get_feature_store
from src.models.schemas import SuccessScoreOutput
from concurrent.futures import ThreadPoolExecutor

router = APIRouter(prefix="/v1/predictions", tags=["Predictions"])
executor = ThreadPoolExecutor(max_workers=4)

@router.post("/score", response_model=SuccessScoreOutput)
async def get_success_score(payload: Dict[str, str]):
    student_id = payload.get("student_id")
    if not student_id:
        raise HTTPException(status_code=400, detail="Missing student_id")
        
    store = get_feature_store()
    student_payload = store.get_student(student_id)
    
    if not student_payload:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")

    try:
        predictor = get_predictor()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
        
    loop = asyncio.get_event_loop()
    try:
        # Run synchronous CPU-bound XGBoost/SHAP prediction in thread pool
        result = await loop.run_in_executor(
            executor,
            predictor.predict,
            student_payload
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
