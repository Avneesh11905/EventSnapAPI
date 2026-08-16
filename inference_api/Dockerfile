# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies using uv.lock and pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Stage 2: Final Runtime Image
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS final

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and system dependencies for OpenCV runtime
RUN apt-get update && \
    apt-get install -y software-properties-common curl procps libgl1 libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copy local source code
COPY --chown=appuser:appgroup . .

# Download Hugging Face ONNX Models inside the image
RUN /app/.venv/bin/python download_models.py

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 5000

CMD ["uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
