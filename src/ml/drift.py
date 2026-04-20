import numpy as np

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index. PSI > 0.2 means significant drift."""
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input
    
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    expected_percents = np.percentile(expected, breakpoints)
    
    expected_fractions = np.histogram(expected, expected_percents)[0] / len(expected)
    actual_fractions = np.histogram(actual, expected_percents)[0] / len(actual)
    
    # Avoid zero division
    expected_fractions = np.where(expected_fractions == 0, 0.0001, expected_fractions)
    actual_fractions = np.where(actual_fractions == 0, 0.0001, actual_fractions)
    
    psi_values = (expected_fractions - actual_fractions) * np.log(expected_fractions / actual_fractions)
    return float(np.sum(psi_values))

class DriftMonitor:
    def check_feature_drift(self, reference_df, current_df) -> dict:
        psi_dict = {}
        for col in reference_df.columns:
            psi_dict[col] = compute_psi(reference_df[col].values, current_df[col].values)
        overall_drift = any(v > 0.20 for v in psi_dict.values())
        return {"overall_drift": overall_drift, "feature_psi": psi_dict}
        
    def check_prediction_drift(self, reference_probs, current_probs) -> dict:
        psi = compute_psi(reference_probs, current_probs)
        return {"prediction_drift": psi > 0.20, "psi": psi}
