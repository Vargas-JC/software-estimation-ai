from collections import deque
from dataclasses import dataclass
import heapq
import random
import time

from app.core.exceptions import UnprocessableEntityException
from app.schemas import SearchAlgorithm, TaskAssignmentResponse
from app.search.problem import AssignmentProblem, AssignmentState


@dataclass
class SearchResult:
    assignments: list[TaskAssignmentResponse]
    total_cost: float
    nodes_explored: int
    execution_time_ms: float
    optimal_guaranteed: bool


def _build_assignments(
    problem: AssignmentProblem, assignment_indices: tuple[int, ...] | list[int]
) -> list[TaskAssignmentResponse]:
    result: list[TaskAssignmentResponse] = []
    for task, profile_idx in zip(problem.tasks, assignment_indices):
        profile = problem.profiles[profile_idx]
        cost = round(task.estimate_hours * profile.hourly_rate, 2)
        result.append(
            TaskAssignmentResponse(
                task_id=task.id,
                task_title=task.title,
                profile_id=profile.id,
                profile_name=profile.name,
                estimate_hours=task.estimate_hours,
                hourly_rate=profile.hourly_rate,
                cost=cost,
            )
        )
    return result


def astar_search(problem: AssignmentProblem) -> SearchResult:
    start_time = time.perf_counter()
    start_state = problem.initial_state()
    counter = 0
    frontier: list[tuple[float, int, AssignmentState]] = [
        (problem.heuristic(start_state), counter, start_state)
    ]
    nodes_explored = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        nodes_explored += 1

        if problem.is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                assignments=_build_assignments(problem, current.assignments),
                total_cost=round(current.accumulated_cost, 2),
                nodes_explored=nodes_explored,
                execution_time_ms=round(elapsed, 4),
                optimal_guaranteed=True,
            )

        for action in problem.actions(current):
            nxt = problem.transition(current, action)
            counter += 1
            f_score = nxt.accumulated_cost + problem.heuristic(nxt)
            heapq.heappush(frontier, (f_score, counter, nxt))

    raise UnprocessableEntityException(
        "No se encontro una asignacion factible con las restricciones dadas."
    )


def bfs_search(problem: AssignmentProblem) -> SearchResult:
    start_time = time.perf_counter()
    queue = deque([problem.initial_state()])
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if problem.is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                assignments=_build_assignments(problem, current.assignments),
                total_cost=round(current.accumulated_cost, 2),
                nodes_explored=nodes_explored,
                execution_time_ms=round(elapsed, 4),
                optimal_guaranteed=False,
            )

        for action in problem.actions(current):
            queue.append(problem.transition(current, action))

    raise UnprocessableEntityException(
        "No se encontro una asignacion factible con las restricciones dadas."
    )


def dfs_search(problem: AssignmentProblem) -> SearchResult:
    start_time = time.perf_counter()
    stack = [problem.initial_state()]
    nodes_explored = 0

    while stack:
        current = stack.pop()
        nodes_explored += 1

        if problem.is_goal(current):
            elapsed = (time.perf_counter() - start_time) * 1000
            return SearchResult(
                assignments=_build_assignments(problem, current.assignments),
                total_cost=round(current.accumulated_cost, 2),
                nodes_explored=nodes_explored,
                execution_time_ms=round(elapsed, 4),
                optimal_guaranteed=False,
            )

        for action in reversed(problem.actions(current)):
            stack.append(problem.transition(current, action))

    raise UnprocessableEntityException(
        "No se encontro una asignacion factible con las restricciones dadas."
    )


def _evaluate_cost(problem: AssignmentProblem, assignment: list[int]) -> float | None:
    profile_hours = [0.0] * len(problem.profiles)
    total_cost = 0.0
    for task, p_idx in zip(problem.tasks, assignment):
        profile_hours[p_idx] += task.estimate_hours
        if problem.max_hours is not None and profile_hours[p_idx] > problem.max_hours:
            return None
        total_cost += task.estimate_hours * problem.profiles[p_idx].hourly_rate
    return total_cost


def hill_climbing_search(problem: AssignmentProblem, restarts: int = 5) -> SearchResult:
    # Busqueda local con reinicios aleatorios para mitigar optimos locales.
    start_time = time.perf_counter()
    rng = random.Random(42)
    best_assignment: list[int] | None = None
    best_cost = float("inf")
    nodes_explored = 0

    for attempt in range(restarts):
        if attempt == 0:
            current = [0] * len(problem.tasks)
            if _evaluate_cost(problem, current) is None:
                continue
        else:
            current = [rng.randrange(len(problem.profiles)) for _ in problem.tasks]

        current_cost = _evaluate_cost(problem, current)
        if current_cost is None:
            continue

        while True:
            nodes_explored += 1
            best_neighbor: list[int] | None = None
            best_neighbor_cost = current_cost

            for t_idx in range(len(problem.tasks)):
                for p_idx in range(len(problem.profiles)):
                    if current[t_idx] == p_idx:
                        continue
                    candidate = current.copy()
                    candidate[t_idx] = p_idx
                    candidate_cost = _evaluate_cost(problem, candidate)
                    if candidate_cost is not None and candidate_cost < best_neighbor_cost:
                        best_neighbor = candidate
                        best_neighbor_cost = candidate_cost

            if best_neighbor is None:
                break
            current = best_neighbor
            current_cost = best_neighbor_cost

        if current_cost < best_cost:
            best_assignment = current
            best_cost = current_cost

    if best_assignment is None:
        return astar_search(problem)

    elapsed = (time.perf_counter() - start_time) * 1000
    return SearchResult(
        assignments=_build_assignments(problem, best_assignment),
        total_cost=round(best_cost, 2),
        nodes_explored=nodes_explored,
        execution_time_ms=round(elapsed, 4),
        optimal_guaranteed=False,
    )


def run_search(problem: AssignmentProblem, algorithm: SearchAlgorithm) -> SearchResult:
    dispatch = {
        SearchAlgorithm.ASTAR: astar_search,
        SearchAlgorithm.HILL_CLIMBING: hill_climbing_search,
        SearchAlgorithm.BFS: bfs_search,
        SearchAlgorithm.DFS: dfs_search,
    }
    return dispatch[algorithm](problem)

