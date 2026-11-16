from pydantic import BaseModel
from app.schemas.enums.classification_model import ClassificationModel
from app.schemas.enums.regression_model import RegressionModel
from app.schemas.enums.risk_level import RiskLevel


class RiskPredictionResponse(BaseModel):
    recommended_contingency_margin: float
    predicted_risk_level: RiskLevel
    risk_probabilities: dict[str, float]
    regression_model_used: RegressionModel
    classification_model_used: ClassificationModel
