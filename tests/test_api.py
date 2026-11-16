from app.api.v1.assignment import optimize_assignments
from app.api.v1.health import health
from app.api.v1.risk import get_model_metrics, predict_risk
from app.schemas import (
    AssignmentOptimizeRequest,
    ClassificationModel,
    ExperienceLevel,
    ProfileRequest,
    ProjectType,
    RegressionModel,
    RiskPredictionRequest,
    SearchAlgorithm,
    TaskRequest,
    TechnicalRole,
)
from app.services import risk_service


def test_health_and_metrics_and_predict():
    risk_service.load_models()
    h = health()
    assert h["models_loaded"] is True

    metrics = get_model_metrics()
    assert "regression_comparison" in metrics
    assert "classification_comparison" in metrics
    assert "selected_classification_model" in metrics

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
    assert isinstance(pred.regression_model_used, RegressionModel)
    assert isinstance(pred.classification_model_used, ClassificationModel)


def test_optimize_endpoint():
    res = optimize_assignments(
        AssignmentOptimizeRequest(
            tasks=[
                TaskRequest(id="t1", title="Task 1", estimate_hours=6),
                TaskRequest(id="t2", title="Task 2", estimate_hours=4),
            ],
            profiles=[
                ProfileRequest(
                    id="p1",
                    name="Dev A",
                    role=TechnicalRole.BACKEND,
                    experience_level=ExperienceLevel.MID,
                    hourly_rate=30.0,
                ),
                ProfileRequest(
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
