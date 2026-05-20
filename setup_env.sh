#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "🧬 Step 1: Installing fundamental dependencies from requirements.txt..."
pip install -r requirements.txt

echo "🗂️ Step 2: Fetching external NVIDIA BioNeMo Framework..."
if [ ! -d "bionemo-framework" ]; then
    git clone https://github.com/NVIDIA/bionemo-framework.git
else
    echo " -> BioNeMo framework folder already exists. Skipping clone."
fi

echo "📦 Step 3: Installing modular single-cell core sub-packages..."
# Install using editable mode (-e) to map system pathing boundaries
pip install -q -e bionemo-framework/sub-packages/bionemo-core
pip install -q -e bionemo-framework/sub-packages/bionemo-scdl

echo "🔥 Step 4: Compiling hardware-accelerated TransformerEngine layers..."
pip install --no-build-isolation transformer_engine[pytorch]

echo "🚀 Environment successfully provisioned for In-Silico Inference!"