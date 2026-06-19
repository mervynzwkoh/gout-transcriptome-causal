import os
import pickle
from datasets import load_dataset

def run_ingestion(config):
    print(f"[Phase 1] Starting data ingestion for target: {config['biological_target']['gene_symbol']}")
    
    dataset_stream = load_dataset(
        "Xaira-Therapeutics/X-Atlas-Orion", 
        streaming=True, 
        split=config['biological_target']['cell_line_split']
    )
    
    filtered_cells = []
    for cell in dataset_stream:
        if cell.get('gene_target') == config['biological_target']['gene_symbol']:
            filtered_cells.append(cell)
        if len(filtered_cells) >= config['data_parameters']['max_subset_cells']:
            break
            
    os.makedirs(os.path.dirname(config['paths']['raw_data_output']), exist_ok=True)
    with open(config['paths']['raw_data_output'], 'wb') as f:
        pickle.dump(filtered_cells, f)
        
    print(f"Ingestion successful. Subset written to {config['paths']['raw_data_output']}\n")