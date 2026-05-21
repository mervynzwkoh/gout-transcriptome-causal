import os
import pickle
import pandas as pd

def run_preprocessing(config):
    print("⚙️ [Phase 2] Executing rank-value tokenization...")
    
    with open(config['paths']['raw_data_output'], 'rb') as f:
        gout_cells = pickle.load(f)
        
    def preprocess_for_geneformer(cell_data):
        temp_df = pd.DataFrame({
            'token': cell_data['gene_token_id'], 
            'count': cell_data['gene_expression']
        })
        temp_df = temp_df[temp_df['count'] > config['data_parameters']['min_expression_threshold']]
        temp_df = temp_df.sort_values(by='count', ascending=False)
        return temp_df['token'].tolist()[:config['data_parameters']['max_sequence_length']]

    tokenized_cells = [preprocess_for_geneformer(cell) for cell in gout_cells]
    
    fine_tuning_data = {
        "tokens": tokenized_cells,
        "dosage": [cell['num_features'] for cell in gout_cells]
    }

    with open(config['paths']['processed_data_output'], 'wb') as f:
        pickle.dump(fine_tuning_data, f)
        
    print(f"✅ Preprocessing successful. Data serialized to {config['paths']['processed_data_output']}\n")