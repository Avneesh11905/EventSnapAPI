FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for building PostgreSQL drivers if needed
RUN apt-get update && \
    apt-get install -y gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Copy dependencies first to leverage Docker cache
RUN pip install --default-timeout=1000 --retries=10 --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
