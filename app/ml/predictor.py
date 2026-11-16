import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.config import ARTIFACTS_DIR
from app.schemas import (
    ClassificationModel,
    RegressionModel,
    RiskLevel,
    RiskPredictionRequest,
    RiskPredictionResponse,
)


class RiskPredictor:
    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self._regressor: Any = None
        self._classifier: Any = None
        self._metrics: dict = {}

    def load(self) -> None:
        reg_path = self.artifacts_dir / "contingency_regressor.joblib"
        clf_path = self.artifacts_dir / "risk_classifier.joblib"
        metrics_path = self.artifacts_dir / "metrics.json"

        self._regressor = joblib.load(reg_path)
        self._classifier = joblib.load(clf_path)
        self._metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    @property
    def is_loaded(self) -> bool:
        return self._regressor is not None and self._classifier is not None

    def get_metrics(self) -> dict:
        if not self.is_loaded:
            self.load()
        return self._metrics

    def predict(self, request: RiskPredictionRequest) -> RiskPredictionResponse:
        if not self.is_loaded:
            self.load()

        features = pd.DataFrame(
            [
                {
                    "project_type": request.project_type.value,
                    "base_effort_points": request.base_effort_points,
                    "total_effort_hours": request.total_effort_hours,
                    "stories_total": request.stories_total,
                    "tasks_total": request.tasks_total,
                    "total_sprints": request.total_sprints,
                    "planned_days": request.planned_days,
                    "junior_ratio": request.junior_ratio,
                    "senior_ratio": request.senior_ratio,
                }
            ]
        )

        margin = float(self._regressor.predict(features)[0])
        predicted_label = str(self._classifier.predict(features)[0])
        probs = self._classifier.predict_proba(features)[0]
        prob_map = {
            str(label): round(float(prob), 4)
            for label, prob in zip(self._classifier.classes_, probs)
        }

        return RiskPredictionResponse(
            recommended_contingency_margin=round(max(0.0, margin), 2),
            predicted_risk_level=RiskLevel(predicted_label),
            risk_probabilities=prob_map,
            regression_model_used=RegressionModel(self._metrics["selected_regression_model"]),
            classification_model_used=ClassificationModel(
                self._metrics["selected_classification_model"]
            ),
        )

