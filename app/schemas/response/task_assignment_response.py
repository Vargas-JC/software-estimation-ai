from pydantic import BaseModel


class TaskAssignmentResponse(BaseModel):
    task_id: str
    task_title: str
    profile_id: str
    profile_name: str
    estimate_hours: int
    hourly_rate: float
    cost: float
