import logging
import time

import grpc
from grpc import aio

from domain.exceptions import (
    InferenceAPIError,
    InvalidInputFormatError,
    ModelExecutionError,
)

logger = logging.getLogger(__name__)


class LoggingInterceptor(aio.ServerInterceptor):
    """
    Logs incoming gRPC requests and the time taken to process them.
    """

    async def intercept_service(self, continuation, handler_call_details: grpc.HandlerCallDetails):
        handler = await continuation(handler_call_details)
        if handler is None:
            return handler

        orig_fn = handler.unary_unary
        if orig_fn is None:
            return handler

        async def wrapped(request, context: aio.ServicerContext):
            method = handler_call_details.method
            img_count = len(request.images) if hasattr(request, "images") else 0
            if img_count > 0:
                logger.info(f"[gRPC] Started {method} for {img_count} images")
            else:
                logger.info(f"[gRPC] Started {method}")
            start_time = time.perf_counter()
            try:
                return await orig_fn(request, context)
            finally:
                elapsed = time.perf_counter() - start_time
                logger.info(f"[gRPC] Finished {method} - Time taken: {elapsed:.4f}s")

        return handler._replace(unary_unary=wrapped)


class ExceptionInterceptor(aio.ServerInterceptor):
    """
    Catches domain and unexpected exceptions from gRPC handlers and converts
    them to appropriate gRPC status codes, mirroring the FastAPI exception
    handlers in presentation/api/exception_handlers.py.
    """

    async def intercept_service(self, continuation, handler_call_details: grpc.HandlerCallDetails):
        handler = await continuation(handler_call_details)
        if handler is None:
            return handler

        # We only need to wrap unary-unary handlers (all current RPCs are unary).
        orig_fn = handler.unary_unary
        if orig_fn is None:
            return handler

        async def wrapped(request, context: aio.ServicerContext):
            try:
                return await orig_fn(request, context)
            except InvalidInputFormatError as exc:
                logger.error(f"[gRPC] Invalid input: {exc}")
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            except ModelExecutionError as exc:
                logger.error(f"[gRPC] Model execution error: {exc}")
                await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            except InferenceAPIError as exc:
                logger.error(f"[gRPC] Inference API error: {exc}")
                await context.abort(grpc.StatusCode.INTERNAL, str(exc))
            except Exception as exc:
                logger.error(f"[gRPC] Unexpected server error: {exc}", exc_info=True)
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    "An unexpected internal server error occurred.",
                )

        return handler._replace(unary_unary=wrapped)
