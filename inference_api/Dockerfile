FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment and add it to PATH
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install necessary system dependencies for OpenCV (libgl, libglib)
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install the inference requirements, plus FastAPI & Uvicorn for local testing
RUN pip install --default-timeout=1000 --retries=10 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Start the local testing server for the inference API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
