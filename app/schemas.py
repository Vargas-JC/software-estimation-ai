from enum import Enum
from pydantic import BaseModel, Field


class ProjectType(str, Enum):
    MONITORING = "MONITORING"
    AUTOMATION = "AUTOMATION"
    TELEMETRY = "TELEMETRY"
    CONTROL_SUPERVISION = "CONTROL_SUPERVISION"
    PREDICTIVE_MAINTENANCE = "PREDICTIVE_MAINTENANCE"
    INTEGRATION = "INTEGRATION"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TechnicalRole(str, Enum):
    FRONTEND = "FRONTEND"
    BACKEND = "BACKEND"
    FULLSTACK = "FULLSTACK"
    QA = "QA"
    DEVOPS = "DEVOPS"
    UI_UX_DESIGNER = "UI_UX_DESIGNER"
    PRODUCT_MANAGER = "PRODUCT_MANAGER"
    TECH_LEAD = "TECH_LEAD"
    FUNCTIONAL_ANALYST = "FUNCTIONAL_ANALYST"
    OTHER = "OTHER"


class ExperienceLevel(str, Enum):
    JUNIOR = "JUNIOR"
    MID = "MID"
    SENIOR = "SENIOR"
    LEAD = "LEAD"


class SearchAlgorithm(str, Enum):
    ASTAR = "astar"
    HILL_CLIMBING = "hill_climbing"
    BFS = "bfs"
    DFS = "dfs"


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


class RiskPredictionResponse(BaseModel):
    recommended_contingency_margin: float
    predicted_risk_level: RiskLevel
    risk_probabilities: dict[str, float]
    regression_model_used: str


class TaskInput(BaseModel):
    id: str
    title: str
    estimate_hours: int = Field(gt=0)


class ProfileInput(BaseModel):
    id: str
    name: str
    role: TechnicalRole
    experience_level: ExperienceLevel
    hourly_rate: float = Field(gt=0)


class AllocationOptimizeRequest(BaseModel):
    tasks: list[TaskInput] = Field(min_length=1)
    profiles: list[ProfileInput] = Field(min_length=1)
    max_hours_per_profile: float | None = Field(default=None, gt=0)
    algorithm: SearchAlgorithm = SearchAlgorithm.ASTAR


class TaskAssignment(BaseModel):
    task_id: str
    task_title: str
    profile_id: str
    profile_name: str
    estimate_hours: int
    hourly_rate: float
    cost: float


class AllocationOptimizeResponse(BaseModel):
    algorithm: SearchAlgorithm
    assignments: list[TaskAssignment]
    total_cost: float
    nodes_explored: int
    execution_time_ms: float
    optimal_guaranteed: bool

