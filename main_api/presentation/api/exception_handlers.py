from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from domain.exceptions import EventSnapError, EventNotFoundError, NoMatchesFoundError


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(EventSnapError)
    async def domain_exception_handler(request: Request, exc: EventSnapError):
        status_code = 400
        if isinstance(exc, (EventNotFoundError, NoMatchesFoundError)):
            status_code = 404

        content = {"error": str(exc), "type": exc.__class__.__name__}
        if hasattr(exc, "details") and exc.details:
            content["details"] = exc.details

        return JSONResponse(
            status_code=status_code,
            content=content,
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"error": "An unexpected internal server error occurred."},
        )
