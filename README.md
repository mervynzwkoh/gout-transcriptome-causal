# Causal Modeling of Dose-Dependent Genetic Perturbations in Gout-Related Urate Transporters

An end-to-end digital biology pipeline utilizing **NVIDIA BioNeMo (TransformerEngine)** to simulate single-cell transcriptomic shifts following targeted in-silico genetic knockouts.

## 🧬 Pipeline Architecture

The project is decoupled into four modular production workflows:

1. **`01_data_ingestion.ipynb`**: Streams single-cell screens from the Xaira X-Atlas/Orion dataset via Hugging Face.
2. **`02_preprocessing.ipynb`**: Filters, normalizes, and serializes expression matrices into Rank-Value token sequences.
3. **`03_model_fine_tuning.ipynb`**: Implements a custom PyTorch regression wrapper over an unrolled, hardware-accelerated **TransformerEngine BERT backbone**.
4. **`04_in_silico_inference.ipynb`**: Maps Ensembl IDs to token spaces, verifies native context expression, and executes pad-token masking mutations to observe transcriptomic state deltas ($\Delta$).

## 📊 Phase 5 Inference & Critical Analysis

### The Experiment

We executed a simulated zero-expression knockout of the $ABCG2$ transporter (`Token ID: 4803`) on a natively expressing target cell vector across a 2,048 sequence memory envelope.

### Results

- **Baseline Cell State Prediction:** `2.3223`
- **Perturbed Cell State Prediction:** `2.3223`
- **Computed Causal Shift ($\Delta$):** `0.0000`

### Technical Post-Mortem & Error Analysis

While the infrastructure and execution paths successfully validated, the computed absolute delta returned `0.0000`. This represents a classic **Attention Dilution** artifact caused by two explicit project constraints:

1. **Severe Underfitting ($N=35$):** The 106M parameter TransformerEngine backbone was restricted to a pilot size of 35 single-cell samples for 3 training epochs. Consequently, the model optimization path collapsed into predicting a global dataset mean rather than learning fine-grained attention maps for individual gene configurations.
2. **Feature Imbalance Bounds:** Masking a single transporter token alters exactly $1 / 2048$ ($<0.05\%$) of the data arrays. Without robustly fine-tuned self-attention matrix parameters to prioritize $ABCG2$, the signal was diluted by the background transcriptome layers.

## 🚀 Generalization Guide (How to run any gene)

To execute this simulation on an entirely different genetic target (e.g., $URAT1$ or an inflammatory cytokine):

1. Open `notebooks/01_data_ingestion.ipynb`.
2. Navigate to the **Global Configuration Boundary** cell.
3. Update `TARGET_GENE_SYMBOL` and `TARGET_ENSEMBL_ID` with your desired target.
4. Run **Runtime -> Run All**; the down-stream tokenizer dictionary and embedding masks will dynamically resolve the new sequence shapes automatically.

## ⚙️ Environment Provisioning & Installation

This project leverages a decoupled configuration structure. Environment installation logic has been stripped out of the core Jupyter runtimes to maintain pipeline hygiene and ensure local/cloud execution compatibility.

### Prerequisites

- Linux environment or a high-compute cloud environment (e.g., Google Colab with an A100/V100 Tensor Core GPU)
- Python 3.10+
- Active NVIDIA CUDA Driver workspace (required for TransformerEngine acceleration)

### Local / Terminal Setup

To initialize and provision the entire framework, clone this repository, change directories into the workspace root, and execute the automated environment bootstrap script:

\`\`\`bash

# 1. Clone the causal discovery pipeline

git clone https://github.com/mervynzwkoh/gout-transcriptome-causal.git
cd gout-transcriptome-causal

# 2. Provision libraries, clone BioNeMo, and compile CUDA-capable TransformerEngine components

./setup_env.sh
\`\`\`

### Google Colab Fast-Track Execution

If you are running this pipeline inside an ephemeral Google Colab notebook instance, create a single execution block at the very top of your workspace to cleanly configure the stateless machine:

\`\`\`python

# Run this cell at the absolute beginning of your runtime session to bridge file volumes

!git clone https://github.com/mervynzwkoh/gout-transcriptome-causal.git
%cd gout-transcriptome-causal
!bash setup_env.sh
\`\`\`
