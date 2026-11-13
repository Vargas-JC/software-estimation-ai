from app.main import get_model_metrics, health, optimize_allocation, predict_risk, predictor
from app.schemas import (
    AllocationOptimizeRequest,
    ExperienceLevel,
    ProfileInput,
    RiskPredictionRequest,
    SearchAlgorithm,
    TaskInput,
    TechnicalRole,
    ProjectType,
)


def test_health_and_metrics_and_predict():
    predictor.load()
    h = health()
    assert h["models_loaded"] is True

    metrics = get_model_metrics()
    assert "regression_comparison" in metrics
    assert "classification_logistic_regression" in metrics

    pred = predict_risk(
        RiskPredictionRequest(
            project_type=ProjectType.AUTOMATION,
            base_effort_points=45,
            total_effort_hours=180.0,
            stories_total=12,
            tasks_total=30,
            total_sprints=3,
            planned_days=30,
            junior_ratio=0.4,
            senior_ratio=0.3,
        )
    )
    assert pred.recommended_contingency_margin > 0
    assert pred.predicted_risk_level.value in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert abs(sum(pred.risk_probabilities.values()) - 1.0) < 0.01


def test_optimize_endpoint():
    res = optimize_allocation(
        AllocationOptimizeRequest(
            tasks=[
                TaskInput(id="t1", title="Task 1", estimate_hours=6),
                TaskInput(id="t2", title="Task 2", estimate_hours=4),
            ],
            profiles=[
                ProfileInput(
                    id="p1",
                    name="Dev A",
                    role=TechnicalRole.BACKEND,
                    experience_level=ExperienceLevel.MID,
                    hourly_rate=30.0,
                ),
                ProfileInput(
                    id="p2",
                    name="Dev B",
                    role=TechnicalRole.FULLSTACK,
                    experience_level=ExperienceLevel.SENIOR,
                    hourly_rate=50.0,
                ),
            ],
            max_hours_per_profile=8.0,
            algorithm=SearchAlgorithm.ASTAR,
        )
    )
    assert res.total_cost == 380.0
