import os
from pathlib import Path

APP_ENV = os.getenv("APP_ENV", "dev").lower()
IS_PROD = APP_ENV == "prod"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(BASE_DIR / "artifacts")))
DEFAULT_DATASET_PATH = BASE_DIR / "data" / "initial_projects.csv"
DOCS_URL = None if IS_PROD else "/docs"
REDOC_URL = None if IS_PROD else "/redoc"
