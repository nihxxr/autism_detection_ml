"""Optional model loader.

The app works without a model: the AQ-10 score is the result. If you run
scripts/train_model.py, it saves a classifier to models/aq10_model.joblib and
the result page also shows that model's probability as a cross-check.
"""

from pathlib import Path

FEATURES = [f"A{i}_Score" for i in range(1, 11)]


class AQ10Model:
    def __init__(self, estimator):
        self.estimator = estimator

    def probability(self, points) -> float:
        import pandas as pd

        row = pd.DataFrame([list(points)], columns=FEATURES)
        return float(self.estimator.predict_proba(row)[0][1])


def load_model(path: Path):
    if not path.exists():
        return None
    import joblib

    estimator = joblib.load(path)
    if list(getattr(estimator, "feature_names_in_", [])) != FEATURES:
        return None
    return AQ10Model(estimator)
