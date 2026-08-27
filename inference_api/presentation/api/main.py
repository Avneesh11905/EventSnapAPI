import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
import uvicorn
from fastapi.middleware.gzip import GZipMiddleware
from grpc import aio as grpc_aio
from grpc_reflection.v1alpha import reflection

from infrastructure.di_container import get_container
from presentation.api.routes.inference import router
from presentation.api.exception_handlers import add_exception_handlers
from presentation.grpc.servicer import FaceInferenceServicer
from presentation.grpc.interceptors import ExceptionInterceptor, LoggingInterceptor
from presentation.grpc.proto import inference_pb2, inference_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRPC_PORT = 50051


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the lifecycle of the gRPC server alongside Uvicorn.

    SO_REUSEPORT is REQUIRED: with Gunicorn's --workers 2, two Uvicorn worker
    processes are forked.  Each worker calls this lifespan startup, so two
    processes attempt to bind 0.0.0.0:50051.  SO_REUSEPORT lets the Linux
    kernel allow this and load-balances incoming gRPC connections across both
    workers transparently.
    """
    container = app.state.container

    grpc_server = grpc_aio.server(
        interceptors=[LoggingInterceptor(), ExceptionInterceptor()],
        options=[
            ("grpc.so_reuseport", 1),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
        ],
    )

    inference_pb2_grpc.add_FaceInferenceServicer_to_server(
        FaceInferenceServicer(container), grpc_server
    )

    service_names = (
        inference_pb2.DESCRIPTOR.services_by_name["FaceInference"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, grpc_server)

    grpc_server.add_insecure_port(f"0.0.0.0:{GRPC_PORT}")
    await grpc_server.start()
    logger.info(f"🔌 gRPC server started on 0.0.0.0:{GRPC_PORT} (SO_REUSEPORT enabled)")

    yield

    logger.info("Shutting down gRPC server (grace=5s)...")
    await grpc_server.stop(grace=5)
    logger.info("gRPC server stopped.")


# Build the FastAPI app with the lifespan context manager.
app = FastAPI(
    title="Eventsnap Inference API",
    description="Dual-protocol inference server: HTTP/JSON + gRPC",
    lifespan=lifespan,
)

# Initialise DI container and attach it to app state so lifespan can access it.
container = get_container()
app.state.container = container

app.add_middleware(GZipMiddleware, minimum_size=1000)

add_exception_handlers(app)

app.include_router(router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "grpc_port": GRPC_PORT}


if __name__ == "__main__":
    port = 5000
    logger.info("=" * 60)
    logger.info("🚀 Eventsnap Inference Server — dual-protocol mode")
    logger.info(f"   HTTP  →  http://0.0.0.0:{port}/")
    logger.info(f"   gRPC  →  0.0.0.0:{GRPC_PORT}")
    logger.info(f"   Docs  →  http://0.0.0.0:{port}/docs")
    logger.info("=" * 60)
    uvicorn.run(app="presentation.api.main:app", host="0.0.0.0", port=port)
