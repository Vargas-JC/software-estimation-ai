from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn

from app.api.v1.health import router as health_router
from app.api.v1.router import api_v1_router
from app.core.config import DOCS_URL, HOST, IS_PROD, PORT, REDOC_URL
from app.core.exceptions import register_exception_handlers
from app.services import risk_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    risk_service.load_models()
    yield


app = FastAPI(
    title="InScope AI Service",
    version="1.0.0",
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL,
    lifespan=lifespan,
)
register_exception_handlers(app)
app.include_router(health_router)
app.include_router(api_v1_router)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=not IS_PROD)
