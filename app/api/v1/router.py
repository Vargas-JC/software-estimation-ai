from fastapi import APIRouter
from app.api.v1 import assignment, risk

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(risk.router)
api_v1_router.include_router(assignment.router)
