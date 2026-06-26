# 1. Use the official, pre-configured NVIDIA BioNeMo image as your base layer
FROM nvcr.io/nvidia/clara/bionemo-framework:2.0

# 2. Set your production runtime directory inside the container
WORKDIR /app

# 3. Copy your project requirements and install remaining standard packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your actual pipeline modules, configuration files, and orchestrator script
COPY config.yaml main.py ./
COPY pipeline/ ./pipeline/

# 5. Set the runtime entry execution layout
CMD ["python", "main.py"]