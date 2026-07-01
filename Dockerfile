# CUDA runtime with Ubuntu 22.04's Python 3.10 for cluster inference jobs.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Keep Python logs immediate and avoid writing .pyc files inside the container.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Kubernetes mounts runtime outputs here. Project files are baked into /app.
ENV WIFI_ANALYZER_APP_DIR=/app
ENV WIFI_ANALYZER_WORKSPACE=/workspace

# Default two-node cluster mapping; Kubernetes can override these env values.
ENV WORLD_SIZE=2
ENV CLUSTER_LOG_FILES=data/inputs/wifi_logs.txt,data/inputs/wifi_logs-1.txt
ENV CLUSTER_PCAP_FILES=data/inputs/wifi_logs.pcap,data/inputs/wifi_logs-1.pcap
ENV CLUSTER_MODEL_DIR=models/flan-t5-log-lora-model
ENV CLUSTER_INFERENCE_DEVICE=cpu
ENV CLUSTER_INFERENCE_DTYPE=fp32
ENV CLUSTER_BATCH_SIZE=4
ENV CLUSTER_RUN_PCAP=true
ENV CLUSTER_RUN_GROQ=false
ENV CLUSTER_MERGE_OUTPUTS=false

WORKDIR /app

# Install Python 3.10 and small OS utilities needed by inference and PCAP parsing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        curl \
        libpcap-dev \
    && rm -rf /var/lib/apt/lists/*

# Make shell commands and health checks use the Python 3.10 runtime explicitly.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

# Keep packaging tools current before installing app dependencies.
RUN python -m pip install --upgrade pip setuptools wheel

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
COPY README.md RUNPOD_DEPLOYMENT.md CI_CD.md GKE_CLUSTER.md ./

# Start one Kubernetes Indexed Job worker. JOB_COMPLETION_INDEX selects the file.
CMD ["python", "-u", "/app/scripts/cluster_handler.py"]
