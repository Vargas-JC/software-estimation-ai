from app.core.config import ARTIFACTS_DIR
from app.ml.predictor import RiskPredictor
from app.services.assignment_service import AssignmentService
from app.services.risk_service import RiskService

risk_predictor = RiskPredictor(artifacts_dir=ARTIFACTS_DIR)
risk_service = RiskService(predictor=risk_predictor)
assignment_service = AssignmentService()

__all__ = ["AssignmentService", "RiskService", "assignment_service", "risk_service"]
