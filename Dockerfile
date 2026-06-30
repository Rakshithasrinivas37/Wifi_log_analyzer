# Runtime image with CUDA-enabled PyTorch for RunPod GPU inference.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Keep Python logs immediate and avoid writing .pyc files inside the container.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=80

# Runtime files are expected on the mounted RunPod workspace volume.
ENV WIFI_ANALYZER_WORKSPACE=/workspace/Wifi_log_analyzer
ENV WIFI_ANALYZER_JOB_WORKERS=1

WORKDIR /app

# Match this to the TensorRT-LLM version used when building the TRT engine.
ARG TRT_LLM_VERSION=0.12.0

# Install small OS utilities plus OpenMPI headers needed by TensorRT-LLM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TensorRT-LLM in the serverless image; keep it out of local requirements.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "tensorrt_llm==${TRT_LLM_VERSION}"

# Install app dependencies before copying source to improve Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, tests, scripts, and bundled sample/runtime artifacts.
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY data ./data
COPY models ./models
COPY README.md RUNPOD_DEPLOYMENT.md CI_CD.md ./

EXPOSE 80

# RunPod load-balancing checks /ping before routing inference traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/ping" || exit 1

# Start FastAPI on port 80 for RunPod load-balancing serverless endpoints.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
