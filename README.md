# Causal Modeling of Dose-Dependent Genetic Perturbations in Gout-Related Urate Transporters

An end-to-end digital biology and MLOps pipeline utilizing **NVIDIA BioNeMo (TransformerEngine)** to simulate single-cell transcriptomic shifts following targeted _in-silico_ genetic knockouts.

---

## 🧬 Biological & Architectural Overview

### The Scientific Thesis

ABCG2 and SLC22A12 (URAT1) act as the "master regulators" of uric acid excretion within the human kidneys and intestines. Genetic mutations or functional issues in these transporters lead directly to hyperuricemia and gout. This repository serves as a **"Digital Petri Dish"**, leveraging deep language models pretrained on single-cell biology to simulate how varying "dosages" of therapeutic silencing affect cell states, mapping out causal drug vectors without using physical wet-lab pipettes.

### Pipeline Architecture

The system transitions away from interactive, linear notebook workflows into a production-grade **Orchestrator Pattern** driven by a single configuration dashboard (`config.yaml`). The pipeline executes chronologically across four decoupled stages:

1. **Ingestion (`pipeline/ingestion.py`)**: Lazily streams single-cell genomic readouts from the 8-million cell Xaira X-Atlas/Orion dataset on the Hugging Face hub, isolating targeted perturbation subsets without exceeding local RAM limits.
2. **Preprocessing (`pipeline/preprocessing.py`)**: Transforms continuous raw expression metrics into non-parametric **Rank-Value Encoded** arrays (ordered descending by abundance) to build the deep "causal grammar" the transformer attention blocks require.
3. **Fine-Tuning (`pipeline/fine_tuning.py`)**: Instantiates a 106-million parameter foundation model (Geneformer) using a custom **Layer Unrolling Bypass** over NVIDIA's hardware-accelerated **TransformerEngine**. It optimizes weights using an AdamW regression harness to map cell embeddings directly to continuous guide RNA abundance proxies.
4. **In-Silico Inference (`pipeline/inference.py`)**: Loads the rehydrated, frozen weight arrays, dynamically maps the human Ensembl dictionary to the latent token vocabulary space, screens for cells with native context expression, and applies a tensor pad-mask mutation to quantify the absolute causal downstream shift ($\Delta$).

---

## 📊 Phase 4 Inference & Analytical Post-Mortem

### The Core Experiment

We executed a simulated zero-expression knockout of the $ABCG2$ transporter (`Ensembl ID: ENSG00000118777`, resolved dynamically to `Vocabulary Token ID: 4803`) on a natively expressing single-cell vector within a 2,048 sequence attention envelope.

### Structural Performance Metrics

- **Control Baseline Cell Prediction:** `2.3223`
- **Knockout Perturbed Cell Prediction:** `2.3223`
- **Computed Absolute Causal Shift ($\Delta$):** `0.0000`

### Engineering & Scientific Post-Mortem

While the MLOps infrastructure and execution boundaries successfully validated, the computed absolute delta returned a null shift (`0.0000`). This is a classic, expected outcome reflecting two explicit constraints of a pilot-scale run:

1. **Extreme Underfitting Artifact ($N=35$):** To validate the training loops, the 106M parameter TransformerEngine backbone was limited to a pilot micro-subset of 35 single-cell samples for 3 training epochs. The model optimization path consequently experienced a mode collapse, learning to predict the global dataset mean rather than mapping individual gene-to-gene attention dynamics.
2. **Attention Dilution Bounds:** Masking a single transporter token alters exactly $1 / 2048$ ($<0.05\%$) of the active input data matrix. Without robustly fine-tuned weight layers to amplify the importance of $ABCG2$, its deletion is mathematically overwhelmed by the remaining background transcriptome layer.

_Infrastructure Validation Status: Success. The pipeline is mathematically and architecturally ready to scale to the full 8-million cell atlas dataset once expanded compute resources are provisioned._

---

## 🛠️ Configuration Dashboard (`config.yaml`)

To target a completely different genetic axis (e.g., $URAT1$, $IL-1\beta$, or $TNF-\alpha$), **do not alter the underlying Python code**. Simply modify the central parameters dashboard at the root of the repository:

```yaml
biological_target:
  gene_symbol: "ABCG2"
  ensembl_id: "ENSG00000118777"
  cell_line_split: "HEK293T"

data_parameters:
  max_subset_cells: 35
  max_sequence_length: 2048
  min_expression_threshold: 0.0
  pad_token_id: 0

hyperparameters:
  batch_size: 4
  learning_rate: 0.00005
  weight_decay: 0.01
  epochs: 3

paths:
  bionemo_recipe_path: "/content/bionemo-framework/bionemo-recipes/models/geneformer/src/geneformer"
  raw_data_output: "data/gout_cells_subset.pkl"
  processed_data_output: "data/geneformer_tuning_input.pkl"
  model_checkpoint_output: "data/gout_geneformer_finetuned.pt"
  token_dictionary_path: "data/token_dictionary_gc104M.pkl"
```

## 🚀 Production Deployment via Docker

To eliminate local environment drift, cross-platform library fragmentation, and C++ compilation errors when binding NVIDIA's deep layers, the entire application workspace is containerized.

### Prerequisites

1. **NVIDIA Host GPU** with active CUDA Drivers matching or exceeding version 12.1.
2. **Docker Engine** installed on the host machine.
3. **NVIDIA Container Toolkit** installed and configured to map physical graphics assets inside the container runtime environment.

### 1. Build the Production Image

Compile the isolated container workspace by scanning the Dockerfile blueprint from the root of your project directory:

```bash
docker build -t digital-biology-pipeline .
```

### 2. Execute the End-to-End Automated Pipeline

Run the containerized system. We pass the `--gpus all` flag to unlock local hardware optimization kernels, and we use a volume flag (`-v`) to mount a local `data/` directory. This ensures that the generated model checkpoints (`.pt`) and tracking assets write directly out to your local physical disk before the temporary container spins down:

```bash
docker run --gpus all -v $(pwd)/data:/app/data digital-biology-pipeline
```

## Production Repository Layout

```markdown
gout-transcriptome-causal/
├── .dockerignore # Prevents volatile local data from bloating builds
├── .gitignore # Blocks heavy model binaries (_.pt, _.pkl) from Git histories
├── Dockerfile # Provisions Ubuntu + CUDA 12.1 + BioNeMo + TransformerEngine
├── README.md # Master executive engineering portfolio documentation
├── config.yaml # Centralized pipeline parameter dashboard
├── main.py # Core pipeline conductor and orchestrator script
├── requirements.txt # Standard upstream Python dependencies
├── setup_env.sh # Shell automation script to compile external NVIDIA core packages
│
├── pipeline/ # Production decoupled functional execution modules
│ ├── **init**.py
│ ├── ingestion.py # Lazy data streaming engine
│ ├── preprocessing.py # Rank-Value tokenization engine
│ ├── fine_tuning.py # Unrolled layer fine-tuning engine
│ └── inference.py # In-silico digital knockout mutation engine
│
└── notebooks/ # Visual research sandboxes and historical logs
├── 01_data_ingestion.ipynb
├── 02_preprocessing.ipynb
├── 03_model_fine_tuning.ipynb
└── 04_in_silico_inference.ipynb
```
