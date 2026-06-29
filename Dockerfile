# Runtime image with CUDA-enabled PyTorch for RunPod GPU inference.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Keep Python logs immediate and avoid writing .pyc files inside the container.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Runtime files are expected on the mounted RunPod workspace volume.
ENV WIFI_ANALYZER_WORKSPACE=/workspace/wifi-log-analyzer
ENV WIFI_ANALYZER_JOB_WORKERS=1

WORKDIR /app

# Install only small OS utilities needed for debugging and health checks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source to improve Docker layer caching.
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code, tests, scripts, and bundled sample/runtime artifacts.
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY README.md RUNPOD_DEPLOYMENT.md CI_CD.md ./

EXPOSE 8000

# Start the FastAPI app when the container launches.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
