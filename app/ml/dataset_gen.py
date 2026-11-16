import numpy as np
import pandas as pd

from app.schemas import ProjectType, RiskLevel

PROJECT_TYPE_COMPLEXITY = {
    ProjectType.MONITORING.value: 0.9,
    ProjectType.INTEGRATION.value: 1.0,
    ProjectType.AUTOMATION.value: 1.1,
    ProjectType.TELEMETRY.value: 1.15,
    ProjectType.CONTROL_SUPERVISION.value: 1.25,
    ProjectType.PREDICTIVE_MAINTENANCE.value: 1.35,
}


def _map_risk_level(margin: float) -> str:
    if margin < 12.0:
        return RiskLevel.LOW.value
    if margin < 19.0:
        return RiskLevel.MEDIUM.value
    if margin < 27.0:
        return RiskLevel.HIGH.value
    return RiskLevel.CRITICAL.value


def generate_training_dataset(n_samples: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Genera datos iniciales de proyectos calibrados a margenes realistas
    (media ~15%, consistente con DEFAULT_MARGIN de InScope).
    """
    rng = np.random.default_rng(seed)
    project_types = list(PROJECT_TYPE_COMPLEXITY.keys())

    rows: list[dict] = []
    for _ in range(n_samples):
        p_type = rng.choice(project_types)
        complexity = PROJECT_TYPE_COMPLEXITY[p_type]

        stories_total = int(rng.integers(5, 50))
        tasks_total = int(stories_total * rng.uniform(2.0, 4.5))
        base_effort_points = int(stories_total * rng.uniform(3.0, 7.0))
        total_effort_hours = round(float(tasks_total * rng.uniform(4.0, 11.0)), 2)
        total_sprints = max(1, int(np.ceil(base_effort_points / rng.uniform(12.0, 26.0))))
        planned_days = int(total_sprints * rng.integers(8, 15) * rng.uniform(0.65, 1.45))
        planned_days = max(7, planned_days)

        junior_ratio = round(float(rng.uniform(0.0, 0.75)), 3)
        senior_ratio = round(float(rng.uniform(0.0, 1.0 - junior_ratio)), 3)

        schedule_pressure = total_effort_hours / (planned_days * 11.0)
        margin = (
            4.5
            + 3.5 * complexity
            + 6.2 * (schedule_pressure ** 1.45)
            + 8.5 * junior_ratio
            - 4.5 * senior_ratio
            + 0.04 * stories_total
            + float(rng.normal(0.0, 1.8))
        )
        contingency_margin = round(float(np.clip(margin, 5.0, 38.0)), 2)

        rows.append(
            {
                "project_type": p_type,
                "base_effort_points": base_effort_points,
                "total_effort_hours": total_effort_hours,
                "stories_total": stories_total,
                "tasks_total": tasks_total,
                "total_sprints": total_sprints,
                "planned_days": planned_days,
                "junior_ratio": junior_ratio,
                "senior_ratio": senior_ratio,
                "contingency_margin": contingency_margin,
                "risk_level": _map_risk_level(contingency_margin),
            }
        )

    return pd.DataFrame(rows)
