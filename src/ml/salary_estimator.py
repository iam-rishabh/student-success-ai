import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from src.models.schemas import SalaryEstimate

class QuantileSalaryEstimator:
    """Uses XGBRegressor for point estimate and quantile regression for bounds."""
    def __init__(self):
        self.median_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
        self.lower_model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.15, n_estimators=100)
        self.upper_model = XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.85, n_estimators=100)
        
    def fit(self, X, y):
        """Fit models on X and salary labels y"""
        self.median_model.fit(X, y)
        self.lower_model.fit(X, y)
        self.upper_model.fit(X, y)
        return self
        
    def predict(self, X) -> list[SalaryEstimate]:
        medians = self.median_model.predict(X)
        lowers = self.lower_model.predict(X)
        uppers = self.upper_model.predict(X)
        
        results = []
        for l, m, u in zip(lowers, medians, uppers):
            # Enforce strictly valid constraints in case of cross-over during inference
            lo = max(0.0, float(min(l, m - 0.5)))
            hi = float(max(u, m + 0.5))
            med = float(m)
            results.append(SalaryEstimate(lower_lpa=lo, median_lpa=med, upper_lpa=hi))
        return results
