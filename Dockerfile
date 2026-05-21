# 1. Base Layer: NVIDIA CUDA Runtime
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 2. Environment Settings
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 3. Install System Dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Environment Provisioning Files
COPY requirements.txt setup_env.sh ./

# 5. Execute Setup (BioNeMo & Python Packages)
RUN chmod +x setup_env.sh && ./setup_env.sh

# 6. Copy Pipeline Source Code and Config
COPY config.yaml main.py ./
COPY pipeline/ ./pipeline/

# 7. Define the Execution Command
ENTRYPOINT ["python3", "main.py"]