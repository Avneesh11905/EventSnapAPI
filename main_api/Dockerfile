# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install system dependencies for building PostgreSQL drivers
RUN apt-get update && \
    apt-get install -y gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Stage 2: Development Environment
FROM builder AS development

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates bash \
    && curl -1sLf 'https://artifacts-cli.infisical.com/setup.deb.sh' | bash \
    && apt-get update && apt-get install -y infisical \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

COPY . .
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app


# Stage 3: Production Builder
FROM builder AS production

# Install dependencies using uv.lock and pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev


# Stage 4: Final Runtime Image
FROM python:3.14-slim-trixie AS final

# Install runtime PostgreSQL dependencies, utilities, OpenCV libraries, and Infisical CLI
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl procps libgl1 libglib2.0-0 \
    && curl -1sLf 'https://artifacts-cli.infisical.com/setup.deb.sh' | bash \
    && apt-get update && apt-get install -y infisical \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

WORKDIR /app

# Copy virtual environment from production builder
COPY --from=production --chown=appuser:appgroup /app/.venv /app/.venv

# Copy local source code
COPY --chown=appuser:appgroup . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["infisical", "run", "--", "gunicorn", "presentation.api.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--forwarded-allow-ips", "*"]
