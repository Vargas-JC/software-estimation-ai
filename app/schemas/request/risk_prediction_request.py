from pydantic import BaseModel, Field
from app.schemas.enums.project_type import ProjectType


class RiskPredictionRequest(BaseModel):
    project_type: ProjectType
    base_effort_points: int = Field(ge=0)
    total_effort_hours: float = Field(ge=0)
    stories_total: int = Field(ge=0)
    tasks_total: int = Field(ge=0)
    total_sprints: int = Field(ge=0)
    planned_days: int = Field(ge=1)
    junior_ratio: float = Field(ge=0.0, le=1.0)
    senior_ratio: float = Field(ge=0.0, le=1.0)
