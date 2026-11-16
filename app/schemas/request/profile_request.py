from pydantic import BaseModel, Field
from app.schemas.enums.experience_level import ExperienceLevel
from app.schemas.enums.technical_role import TechnicalRole


class ProfileRequest(BaseModel):
    id: str
    name: str
    role: TechnicalRole
    experience_level: ExperienceLevel
    hourly_rate: float = Field(gt=0)
