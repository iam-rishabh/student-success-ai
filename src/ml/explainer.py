import shap
import json
import os
import numpy as np
from src.models.schemas import ShapDriver

class PlacementExplainer:
    def __init__(self, xgb_model, feature_names: list, label_map_path="src/data/feature_label_map.json"):
        # SHAP requires the underlying raw model. Since we use an ensemble,
        # we extract the XGBoost component to run TreeExplainer for high performance.
        self.explainer = shap.TreeExplainer(xgb_model)
        self.feature_names = feature_names
        
        try:
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
        except Exception:
            self.label_map = {feat: feat for feat in feature_names}

    def explain(self, feature_vector: np.ndarray) -> dict:
        """Returns the top 3 positive and top 3 negative drivers"""
        shap_values = self.explainer.shap_values(feature_vector)
        
        # Binary classification SHAP values array extraction
        if isinstance(shap_values, list): # For multi-class or some xgb configurations
            shap_values = shap_values[1] 
            
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0] # Single row explanation

        drivers = []
        for val, name in zip(shap_values, self.feature_names):
            label = self.label_map.get(name, name)
            drivers.append((label, val))
            
        # Sort by impact
        sorted_drivers = sorted(drivers, key=lambda x: x[1])
        
        top_negative = sorted_drivers[:3]
        top_positive = sorted_drivers[-3:]
        
        shap_driver_objects = []
        for label, val in top_negative:
            if val < 0:
                shap_driver_objects.append(ShapDriver(feature_label=label, direction="negative", magnitude=abs(float(val))))
        
        for label, val in reversed(top_positive):
            if val > 0:
                shap_driver_objects.append(ShapDriver(feature_label=label, direction="positive", magnitude=float(val)))

        # Ensure we pad empty if less than 6 drivers exist
        return {
            "shap_drivers": shap_driver_objects,
            "raw_shap": dict(zip(self.feature_names, [float(x) for x in shap_values]))
        }
