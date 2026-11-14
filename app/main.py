from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn

from app.config import HOST, PORT
from app.ml.predictor import RiskPredictor
from app.schemas import (
    AllocationOptimizeRequest,
    AllocationOptimizeResponse,
    RiskPredictionRequest,
    RiskPredictionResponse,
)
from app.search.algorithms import run_search
from app.search.problem import AllocationProblem

predictor = RiskPredictor()


@asynccontextmanager
async def lifespan(_: FastAPI):
    predictor.load()
    yield


app = FastAPI(title="InScope AI Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": predictor.is_loaded}


@app.post("/api/v1/risk/predict", response_model=RiskPredictionResponse)
def predict_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
    return predictor.predict(request)


@app.post("/api/v1/allocation/optimize", response_model=AllocationOptimizeResponse)
def optimize_allocation(request: AllocationOptimizeRequest) -> AllocationOptimizeResponse:
    problem = AllocationProblem(
        tasks=request.tasks,
        profiles=request.profiles,
        max_hours_per_profile=request.max_hours_per_profile,
    )
    try:
        result = run_search(problem, request.algorithm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AllocationOptimizeResponse(
        algorithm=request.algorithm,
        assignments=result.assignments,
        total_cost=result.total_cost,
        nodes_explored=result.nodes_explored,
        execution_time_ms=result.execution_time_ms,
        optimal_guaranteed=result.optimal_guaranteed,
    )


@app.get("/api/v1/models/metrics")
def get_model_metrics() -> dict:
    return predictor.get_metrics()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT)

