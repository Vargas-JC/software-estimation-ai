from collections.abc import Callable
from app.schemas import AssignmentOptimizeRequest, AssignmentOptimizeResponse, SearchAlgorithm
from app.search.algorithms import SearchResult, run_search
from app.search.problem import AssignmentProblem


class AssignmentService:
    def __init__(
        self,
        search_runner: Callable[[AssignmentProblem, SearchAlgorithm], SearchResult] = run_search,
    ):
        self._search_runner = search_runner

    def optimize(self, request: AssignmentOptimizeRequest) -> AssignmentOptimizeResponse:
        problem = AssignmentProblem(
            tasks=request.tasks,
            profiles=request.profiles,
            max_hours_per_profile=request.max_hours_per_profile,
        )
        result = self._search_runner(problem, request.algorithm)
        return AssignmentOptimizeResponse(
            algorithm=request.algorithm,
            assignments=result.assignments,
            total_cost=result.total_cost,
            nodes_explored=result.nodes_explored,
            execution_time_ms=result.execution_time_ms,
            optimal_guaranteed=result.optimal_guaranteed,
        )
