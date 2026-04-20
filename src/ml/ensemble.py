import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble model aggregating XGB, RF and LR per specifications."""
    def __init__(self, xgb_weight=0.5, rf_weight=0.3, lr_weight=0.2):
        self.xgb_weight = xgb_weight
        self.rf_weight = rf_weight
        self.lr_weight = lr_weight
        self.xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        self.rf = RandomForestClassifier(n_estimators=100)
        self.lr = LogisticRegression(max_iter=1000)
        
    def fit(self, X, y):
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        self.lr.fit(X, y)
        self.classes_ = self.xgb.classes_
        return self
        
    def predict_proba(self, X):
        p_xgb = self.xgb.predict_proba(X)
        p_rf = self.rf.predict_proba(X)
        p_lr = self.lr.predict_proba(X)
        
        return (p_xgb * self.xgb_weight) + (p_rf * self.rf_weight) + (p_lr * self.lr_weight)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]

def build_placement_pipeline(categorical_features, numerical_features):
    """Builds the end-to-end preprocessing + calibrated ensemble pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    ensemble = SoftVotingEnsemble()
    calibrated = CalibratedClassifierCV(estimator=ensemble, method='isotonic', cv=5)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', calibrated)
    ])
    return pipeline
