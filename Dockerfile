# CUDA runtime with Ubuntu 22.04's Python 3.10, which matches the
# TensorRT-LLM 0.12.0 Linux wheel published by NVIDIA.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

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

# Install Python 3.10, small OS utilities, and OpenMPI headers needed by TRT-LLM.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        curl \
        libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Make shell commands and health checks use the Python 3.10 runtime explicitly.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Install TensorRT-LLM from NVIDIA's wheel index; keep it out of local requirements.
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        "tensorrt-llm==${TRT_LLM_VERSION}"

# Install app dependencies before copying source to improve Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code, tests, scripts, and bundled sample/runtime artifacts.
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY k8s ./k8s
COPY data ./data
COPY models ./models
COPY README.md RUNPOD_DEPLOYMENT.md CI_CD.md ./

EXPOSE 80

# RunPod load-balancing checks /ping before routing inference traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/ping" || exit 1

# Start FastAPI on port 80 for RunPod load-balancing serverless endpoints.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
