from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class BadRequestException(AppException):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=400)


class UnprocessableEntityException(AppException):
    def __init__(self, message: str):
        super().__init__(message=message, status_code=422)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppException)
    async def app_exception_handler(_: Request, exc: AppException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})
