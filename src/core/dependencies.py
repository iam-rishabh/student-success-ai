from src.stores.sqlite_store import SQLiteFeatureStore
from src.jobs.scheduler import RefreshJob
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Global singletons for runtime
feature_store = None
scheduler = None

def init_dependencies():
    global feature_store, scheduler
    logger.info(f"Initializing dependencies for ENV: {settings.app_env}")
    
    if settings.app_env == "simulation":
        feature_store = SQLiteFeatureStore()
        scheduler = RefreshJob(feature_store, min_interval_seconds=settings.refresh_interval_seconds)
    else:
        # In production this would return PostgresFeatureStore/Redis etc.
        # Fallback to simulation for this MVP
        logger.warning(f"Production dependencies not fully implemented for {settings.app_env}. Using simulation.")
        feature_store = SQLiteFeatureStore()
        scheduler = RefreshJob(feature_store, min_interval_seconds=settings.refresh_interval_seconds)

def get_feature_store():
    return feature_store

def get_scheduler():
    return scheduler

def get_predictor():
    if not scheduler or not scheduler.current_predictor:
        raise RuntimeError("Predictor not ready. Scheduler must run first.")
    return scheduler.current_predictor
