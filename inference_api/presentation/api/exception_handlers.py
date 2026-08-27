import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from domain.exceptions import InferenceAPIError, InvalidInputFormatError

logger = logging.getLogger(__name__)


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(InferenceAPIError)
    async def inference_exception_handler(request: Request, exc: InferenceAPIError):
        status_code = 400
        if isinstance(exc, InvalidInputFormatError):
            status_code = 422

        logger.error(f"Inference API Error: {exc}")
        return JSONResponse(
            status_code=status_code,
            content={"error": str(exc), "type": exc.__class__.__name__},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected internal server error occurred."},
        )
