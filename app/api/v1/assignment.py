from fastapi import APIRouter
from app.schemas import AssignmentOptimizeRequest, AssignmentOptimizeResponse
from app.services import assignment_service

router = APIRouter(prefix="/assignments", tags=["Assignments"])


@router.post("/optimize", response_model=AssignmentOptimizeResponse)
def optimize_assignments(request: AssignmentOptimizeRequest) -> AssignmentOptimizeResponse:
    return assignment_service.optimize(request)
