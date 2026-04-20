from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.core.dependencies import init_dependencies, get_scheduler
from src.core.logging import logger
from src.api.routers.ingestion import router as ingestion_router
from src.api.routers.predictions import router as predictions_router
from src.api.routers.alerts import router as alerts_router
from typing import Dict, Any

def create_app() -> FastAPI:
    app = FastAPI(title="Student-Success Guidance API")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup_event():
        init_dependencies()
        # Seed the DB with some initial mock data and trigger initial training to warm up the models
        from src.data.mock_generator import generate_mock_payload
        from src.core.dependencies import get_feature_store
        store = get_feature_store()
        logger.info("Warming up store with mock data...")
        for _ in range(100):
            store.save_student(generate_mock_payload())
        
        # Initial run
        sched = get_scheduler()
        sched.run(force=True)

    app.include_router(ingestion_router)
    app.include_router(predictions_router)
    app.include_router(alerts_router)

    @app.get("/v1/health")
    async def health_check() -> Dict[str, Any]:
        sched = get_scheduler()
        model_version = sched.current_predictor.model_version if sched and sched.current_predictor else None
        return {
            "status": "ok",
            "model_version": model_version,
            "last_refresh_at": sched.last_run.isoformat() if sched and sched.last_run else None
        }

    return app

app = create_app()
