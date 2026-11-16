from fastapi import APIRouter
from app.services import risk_service

router = APIRouter(tags=["Health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": risk_service.is_ready()}
