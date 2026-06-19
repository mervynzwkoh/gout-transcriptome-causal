# pipeline/inference.py
import os
import sys
import pickle
import torch
import urllib.request
from pipeline.fine_tuning import GeneformerForRegression

def run_inference(config):
    print("[Phase 4] Launching in-silico causal inference perturbation...")
    
    # 1. Fetch Dynamic Dependency Paths
    recipe_dir = config['paths']['bionemo_recipe_path']
    if recipe_dir not in sys.path:
        sys.path.append(recipe_dir)
        
    from modeling_bert_te import BertModel, TEBertConfig

    # 2. Dynamic Token Resolution Blueprint
    dict_path = config['paths']['token_dictionary_path']
    target_ensembl = config['biological_target']['ensembl_id']
    target_symbol = config['biological_target']['gene_symbol']
    
    if not os.path.exists(dict_path):
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        print("   Token dictionary missing from data volume. Streaming from cloud hub...")
        url = "https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/token_dictionary_gc104M.pkl"
        urllib.request.urlretrieve(url, dict_path)
        
    with open(dict_path, "rb") as f:
        gene_token_dict = pickle.load(f)
        
    target_token_id = gene_token_dict.get(target_ensembl)
    if target_token_id is None:
        raise ValueError(f"CRITICAL Error: ID mapping {target_ensembl} not found in vocabulary bounds.")
    
    print(f"   Feature target resolved: {target_symbol} maps to integer ID {target_token_id}")

    # 3. Model Weight Grid Rehydration
    model_config = TEBertConfig(
        hidden_size=768, num_attention_heads=12, num_hidden_layers=6,
        vocab_size=40000, max_position_embeddings=config['data_parameters']['max_sequence_length']
    )
    model_config.position_embedding_type = "absolute"
    model_config.is_decoder = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneformerForRegression(model_config, BertModel)
    
    checkpoint = config['paths']['model_checkpoint_output']
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Error: Fine-tuned weight file missing at path: {checkpoint}")
        
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # 4. Deserializing Profiling Layers and Searching for Native Candidate Expression
    with open(config['paths']['processed_data_output'], 'rb') as f:
        data = pickle.load(f)
        
    target_cell_idx = -1
    for idx, sequence in enumerate(data['tokens']):
        if target_token_id in sequence:
            target_cell_idx = idx
            break
            
    if target_cell_idx == -1:
        print(f"   WARNING: Target {target_symbol} is not naturally expressed in the current dataset.")
        print("   Execution halted: Expand data subsets to resolve contextual attention connections.")
        return

    print(f"   Context located. Native expression identified at profile offset index: {target_cell_idx}")
    
    # 5. Executing the In-Silico Tensor Knockout Mutation (Optimized Array Alignment)
    max_len = config['data_parameters']['max_sequence_length']
    pad_id = config['data_parameters'].get('pad_token_id', 0)
    
    # Extract baseline sequence up to maximum bounds
    baseline_sequence = data['tokens'][target_cell_idx][:max_len]
    
    # Filter out the target token entirely, then right-pad to preserve exact dimensions
    perturbed_sequence = [token for token in baseline_sequence if token != target_token_id]
    if len(perturbed_sequence) < max_len:
        perturbed_sequence.extend([pad_id] * (max_len - len(perturbed_sequence)))

    # Package parallel sequences into native hardware configurations
    baseline_tensor = torch.tensor([baseline_sequence], dtype=torch.long).to(device)
    perturbed_tensor = torch.tensor([perturbed_sequence], dtype=torch.long).to(device)

    # 6. Evaluate Divergent Trajectories
    with torch.no_grad():
        baseline_prediction = model(baseline_tensor).item()
        perturbed_prediction = model(perturbed_tensor).item()
        
    delta_shift = perturbed_prediction - baseline_prediction

    # 7. Print Terminal Matrix Summary Reporting
    print("\n===================================================")
    print("      IN-SILICO PERTURBATION EXPERIMENT ANALYSIS")
    print("===================================================")
    print(f"  Target Genetic Axis:     {target_symbol} ({target_ensembl})")
    print(f"  Control Baseline State:  {baseline_prediction:.4f}")
    print(f"  Knockout Perturbed State: {perturbed_prediction:.4f}")
    print(f"  Delta Causal Shift (Δ):  {delta_shift:.4f}")
    print("===================================================\n")
    print("Pipeline complete. Automated execution cycle exited successfully.")