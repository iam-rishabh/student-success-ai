import logging
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Optional
from src.ml.trainer import ModelTrainer
from src.ml.predictor import Predictor
from src.stores.base import FeatureStore
from src.ml.drift import DriftMonitor, compute_psi
from src.ml.feature_assembler import FeatureAssembler
import traceback

logger = logging.getLogger(__name__)

class RefreshJob:
    def __init__(self, feature_store: FeatureStore, min_interval_seconds: int = 120):
        self.job_id = uuid4()
        self.last_run: Optional[datetime] = None
        self.feature_store = feature_store
        self.min_interval_seconds = min_interval_seconds
        self.trainer = ModelTrainer()
        self.drift_monitor = DriftMonitor()
        
        # State
        self.current_predictor: Optional[Predictor] = None

    def run(self, force=False):
        now = datetime.utcnow()
        if not force and self.last_run and (now - self.last_run).seconds < self.min_interval_seconds:
            logger.warning(f"Refresh attempted before {self.min_interval_seconds}s interval. Skipping.")
            return False

        r_students = self.feature_store.get_all_students()
        if len(r_students) < 10:
            logger.info("Not enough data to train. Skipping refresh.")
            return False

        try:
            logger.info(f"Starting refresh job. training on {len(r_students)} students.")
            artifacts = self.trainer.train(r_students)
            
            # Instantiate new predictor to make it active
            self.current_predictor = Predictor(
                feature_assembler=artifacts["assembler"],
                models=artifacts["models"],
                salary_model=artifacts["salary_model"],
                explainer=artifacts["explainer"],
                model_version=f"v1.0.0-run-{now.strftime('%Y%m%d%H%M%S')}"
            )
            
            self.last_run = now
            logger.info(f"Refresh complete. New model version: {self.current_predictor.model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
            logger.debug(traceback.format_exc())
            return False
