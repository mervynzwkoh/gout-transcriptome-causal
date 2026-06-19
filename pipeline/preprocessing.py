import os
import pickle
import pandas as pd

def run_preprocessing(config):
    print("⚙️ [Phase 2] Executing rank-value tokenization...")
    
    with open(config['paths']['raw_data_output'], 'rb') as f:
        gout_cells = pickle.load(f)
    
    max_len = config['data_parameters']['max_sequence_length']
    pad_id = config['data_parameters'].get('pad_token_id', 0)
        
    def preprocess_for_geneformer(cell_data):
        temp_df = pd.DataFrame({
            'token': cell_data['gene_token_id'], 
            'count': cell_data['gene_expression']
        })
        temp_df = temp_df[temp_df['count'] > config['data_parameters']['min_expression_threshold']]
        temp_df = temp_df.sort_values(by='count', ascending=False)
        tokens = temp_df['token'].tolist()[:max_len]
        
        # Enforce uniform dimensions: pad right if sequence is shorter than max_len
        if len(tokens) < max_len:
            tokens.extend([pad_id] * (max_len - len(tokens)))
            
        return tokens

    tokenized_cells = [preprocess_for_geneformer(cell) for cell in gout_cells]
    
    fine_tuning_data = {
        "tokens": tokenized_cells,
        "dosage": [cell['num_features'] for cell in gout_cells]
    }

    with open(config['paths']['processed_data_output'], 'wb') as f:
        pickle.dump(fine_tuning_data, f)
        
    print(f"✅ Preprocessing successful. Data serialized to {config['paths']['processed_data_output']}\n")