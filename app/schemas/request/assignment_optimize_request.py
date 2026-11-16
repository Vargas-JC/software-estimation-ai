from pydantic import BaseModel, Field
from app.schemas.enums.search_algorithm import SearchAlgorithm
from app.schemas.request.profile_request import ProfileRequest
from app.schemas.request.task_request import TaskRequest


class AssignmentOptimizeRequest(BaseModel):
    tasks: list[TaskRequest] = Field(min_length=1)
    profiles: list[ProfileRequest] = Field(min_length=1)
    max_hours_per_profile: float | None = Field(default=None, gt=0)
    algorithm: SearchAlgorithm = SearchAlgorithm.ASTAR
