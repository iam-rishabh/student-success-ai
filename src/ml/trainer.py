import numpy as np
import pandas as pd
from typing import List
from src.ml.feature_assembler import FeatureAssembler
from src.ml.ensemble import build_placement_pipeline
from src.ml.salary_estimator import QuantileSalaryEstimator
from src.ml.explainer import PlacementExplainer

class ModelTrainer:
    """Coordinates the training of the multi-horizon classifiers and quantile regressor."""
    def __init__(self):
        # We manually list categoricals and numericals based on FeatureAssembler output
        self.numerical_features = [
            "cgpa_normalized", "cgpa_trend", "cgpa_consistency", "year_of_study",
            "certifications_count", "internship_duration_weeks", "internship_count",
            "employer_tier_best", "role_relevance_avg", "institute_tier",
            "placement_rate_3mo", "placement_rate_6mo", "placement_rate_12mo",
            "placement_rate_yoy_delta", "median_salary_lpa", "recruiter_participation_normalized",
            "sector_hiring_index", "region_job_density", "macro_gdp_growth_pct",
            "sector_demand_trend", "grit_score", "growth_mindset_score",
            "self_efficacy_score", "job_portal_activity_index", "resume_update_count"
        ]
        self.categorical_features = [
            "course_type_encoded", "has_verified_internship"
        ]
        self.feature_names = self.numerical_features + self.categorical_features
        
    def generate_labels_for_simulation(self, df: pd.DataFrame):
        """Generates synthetic targets using heuristic logic to avoid pure random noise."""
        np.random.seed(42)
        n = len(df)
        
        # Base probability derived from features
        base_prob = (
            df["cgpa_normalized"] * 0.3 + 
            df["placement_rate_12mo"] * 0.4 +
            df["sector_hiring_index"] * 0.3
        )
        
        y_12mo = (base_prob + np.random.normal(0, 0.1, n) > 0.4).astype(int)
        y_6mo = (base_prob + np.random.normal(0, 0.1, n) > 0.6).astype(int)
        y_3mo = (base_prob + np.random.normal(0, 0.1, n) > 0.75).astype(int)
        
        # Salary roughly tied to institute tier and current median
        y_salary = df["median_salary_lpa"] * (1 + (base_prob - 0.5)) + np.random.normal(0, 1.0, n)
        y_salary = np.clip(y_salary, 2.5, 30.0)
        
        return y_3mo, y_6mo, y_12mo, y_salary
        
    def train(self, payloads: List):
        assembler = FeatureAssembler()
        df = assembler.assemble(payloads)
        
        y_3, y_6, y_12, y_sal = self.generate_labels_for_simulation(df)
        
        models = {}
        for horizon, y in zip(["3", "6", "12"], [y_3, y_6, y_12]):
            pipeline = build_placement_pipeline(self.categorical_features, self.numerical_features)
            # Ensure proper class mapping, as synthetic subsets might randomly only have 1 class
            if len(np.unique(y)) > 1:
                pipeline.fit(df, y)
            else:
                # Mock fit with a dummy negative case to prevent pipeline crash on very small batches
                y_mock = y.copy()
                y_mock[0] = 0
                y_mock[1] = 1
                pipeline.fit(df, y_mock)
                
            models[horizon] = pipeline

        salary_model = QuantileSalaryEstimator()
        salary_model.fit(df, y_sal)
        
        # Explainer initialized using 12mo XGBoost component as the primary driver
        # pipeline component 'classifier' is CalibratedClassifierCV -> estimator -> SoftVotingEnsemble -> xgb
        xgb_base = models["12"].named_steps['classifier'].estimator.xgb
        explainer = PlacementExplainer(xgb_base, df.columns.tolist())
        
        return {
            "assembler": assembler,
            "models": models,
            "salary_model": salary_model,
            "explainer": explainer
        }
