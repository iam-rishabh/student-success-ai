import json
from datetime import datetime
import pandas as pd
from src.models.schemas import SuccessScoreOutput, PlacementHorizonScore
from src.ml.feature_assembler import FeatureAssembler
from config.settings import settings

class Predictor:
    """Unified interface wrapping the models, assembler, and config for standard API interactions."""
    def __init__(self, 
                 feature_assembler: FeatureAssembler, 
                 models: dict, 
                 salary_model, 
                 explainer,
                 thresholds_path="config/thresholds.json",
                 model_version="v1.0.0-simulation"):
        
        self.assembler = feature_assembler
        # models holds the {"3": model_3mo, "6": model_6mo, "12": model_12mo}
        self.models = models
        self.salary_model = salary_model
        self.explainer = explainer
        self.model_version = model_version
        
        with open(thresholds_path, 'r') as f:
            self.thresholds = json.load(f)

    def _determine_confidence(self, payload) -> str:
        if payload.institute.data_vintage_months > 24:
            return "LOW"
        if payload.psych is None or payload.behavioral is None:
            return "MEDIUM"
        return "HIGH"

    def _get_risk_tier(self, prob: float, horizon: str) -> str:
        thresh = self.thresholds.get(horizon, {"HIGH": 0.40, "LOW": 0.70})
        if prob < thresh["HIGH"]:
            return "HIGH"
        if prob > thresh["LOW"]:
            return "LOW"
        return "MEDIUM"

    def _generate_actions(self, payload) -> list:
        actions = []
        if payload.academic.cgpa < 6.5:
            actions.append("Focus on improving academic consistency and semester CGPAs.")
        if not payload.internships:
            actions.append("Acquire practical experience through an internship or live project.")
        if len(actions) == 0:
            actions.append("Maintain current trajectory; begin interview preparation.")
        return actions

    def predict(self, payload) -> SuccessScoreOutput:
        # Assemble feature vector
        df = self.assembler.assemble([payload])
        feature_vector = df.iloc[0].to_numpy().reshape(1, -1)
        
        # Placement Horizons
        scores = []
        is_exception = False
        for horizon in ["3", "6", "12"]:
            if horizon in self.models:
                prob = float(self.models[horizon].predict_proba(df)[0][1])  # Class 1 probability
                tier = self._get_risk_tier(prob, horizon)
                if tier == "HIGH":
                    is_exception = True
                
                scores.append(PlacementHorizonScore(
                    horizon_months=int(horizon),
                    probability=prob,
                    risk_tier=tier
                ))

        # Salary Estimate
        salary = self.salary_model.predict(df)[0]
        
        # Explainer
        explanation = self.explainer.explain(feature_vector)
        
        confidence = self._determine_confidence(payload)
        actions = self._generate_actions(payload)
        
        return SuccessScoreOutput(
            student_id=payload.student_id,
            placement_scores=scores,
            salary_estimate=salary,
            shap_drivers=explanation["shap_drivers"],
            confidence_level=confidence,
            is_exception_alert=is_exception,
            next_best_actions=actions,
            model_version=self.model_version,
            predicted_at=datetime.utcnow()
        )
