from pydantic import BaseModel
from app.schemas.enums.search_algorithm import SearchAlgorithm
from app.schemas.response.task_assignment_response import TaskAssignmentResponse


class AssignmentOptimizeResponse(BaseModel):
    algorithm: SearchAlgorithm
    assignments: list[TaskAssignmentResponse]
    total_cost: float
    nodes_explored: int
    execution_time_ms: float
    optimal_guaranteed: bool
