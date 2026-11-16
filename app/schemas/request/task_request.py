from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    id: str
    title: str
    estimate_hours: int = Field(gt=0)
