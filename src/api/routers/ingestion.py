from fastapi import APIRouter, Depends, HTTPException
from src.models.schemas import StudentIngestionPayload
from src.core.dependencies import get_feature_store
from datetime import datetime
import logging

router = APIRouter(prefix="/v1/students", tags=["Ingestion"])
logger = logging.getLogger(__name__)

@router.post("/ingest")
async def ingest_student(payload: StudentIngestionPayload):
    store = get_feature_store()
    try:
        store.save_student(payload)
        return {
            "student_id": payload.student_id,
            "status": "success",
            "ingested_at": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Ingestion failed for {payload.student_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal ingestion error")
