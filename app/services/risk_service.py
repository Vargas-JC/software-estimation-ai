from app.core.config import ARTIFACTS_DIR, DEFAULT_DATASET_PATH
from app.ml.predictor import RiskPredictor
from app.ml.train import train_and_evaluate
from app.schemas import RiskPredictionRequest, RiskPredictionResponse


class RiskService:
    def __init__(self, predictor: RiskPredictor):
        self._predictor = predictor

    def load_models(self) -> None:
        reg_path = ARTIFACTS_DIR / "contingency_regressor.joblib"
        csv_updated = (
            DEFAULT_DATASET_PATH.exists()
            and reg_path.exists()
            and DEFAULT_DATASET_PATH.stat().st_mtime > reg_path.stat().st_mtime
        )

        # Reentrena automaticamente al reiniciar si no hay modelo o si se actualizo el CSV.
        if not reg_path.exists() or csv_updated:
            train_and_evaluate()

        self._predictor.load()

    def is_ready(self) -> bool:
        return self._predictor.is_loaded

    def predict(self, request: RiskPredictionRequest) -> RiskPredictionResponse:
        return self._predictor.predict(request)

    def get_metrics(self) -> dict:
        return self._predictor.get_metrics()
