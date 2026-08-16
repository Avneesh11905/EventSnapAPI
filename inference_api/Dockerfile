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
FROM python:3.11-slim AS final

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl procps libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# Copy ONLY download_models.py first to cache the model download layer
COPY --chown=appuser:appgroup download_models.py .

# Download ONNX Models (this layer will now cache perfectly unless download_models.py changes)
RUN /app/.venv/bin/python download_models.py

# Now copy the rest of the local source code
COPY --chown=appuser:appgroup . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 5000

CMD ["uvicorn", "presentation.api.main:app", "--host", "0.0.0.0", "--port", "5000", "--workers", "1"]
