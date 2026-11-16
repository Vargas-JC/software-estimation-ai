from app.schemas import ExperienceLevel, ProfileRequest, SearchAlgorithm, TaskRequest, TechnicalRole
from app.search.algorithms import run_search
from app.search.problem import AssignmentProblem


def _sample_problem(max_hours: float | None = 12.0) -> AssignmentProblem:
    tasks = [
        TaskRequest(id="t1", title="API Auth", estimate_hours=8),
        TaskRequest(id="t2", title="Dashboard UI", estimate_hours=6),
        TaskRequest(id="t3", title="Unit Tests", estimate_hours=4),
    ]
    profiles = [
        ProfileRequest(
            id="p1",
            name="Junior Dev",
            role=TechnicalRole.BACKEND,
            experience_level=ExperienceLevel.JUNIOR,
            hourly_rate=20.0,
        ),
        ProfileRequest(
            id="p2",
            name="Senior Dev",
            role=TechnicalRole.FULLSTACK,
            experience_level=ExperienceLevel.SENIOR,
            hourly_rate=50.0,
        ),
    ]
    return AssignmentProblem(tasks=tasks, profiles=profiles, max_hours_per_profile=max_hours)


def test_astar_finds_minimum_cost():
    problem = _sample_problem(max_hours=12.0)
    res_astar = run_search(problem, SearchAlgorithm.ASTAR)
    res_bfs = run_search(problem, SearchAlgorithm.BFS)

    assert len(res_astar.assignments) == 3
    assert res_astar.optimal_guaranteed is True
    assert res_astar.total_cost <= res_bfs.total_cost
    assert res_astar.total_cost == 540.0


def test_all_algorithms_produce_valid_assignments():
    problem = _sample_problem(max_hours=14.0)
    for algo in SearchAlgorithm:
        res = run_search(problem, algo)
        assert len(res.assignments) == 3
        assert res.nodes_explored >= 1
        assert res.total_cost > 0

