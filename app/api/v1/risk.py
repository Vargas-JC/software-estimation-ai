from fastapi import APIRouter
from app.schemas import RiskPredictionRequest, RiskPredictionResponse
from app.services import risk_service

router = APIRouter(tags=["Risk & Models"])


@router.post("/risk/predict", response_model=RiskPredictionResponse)
def predict_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
    return risk_service.predict(request)


@router.get("/models/metrics")
def get_model_metrics() -> dict:
    return risk_service.get_metrics()
