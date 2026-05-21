# pipeline/fine_tuning.py
import os
import sys
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Custom Unrolled Architecture Definition inside the module space
class GeneformerForRegression(nn.Module):
    def __init__(self, config_obj, bert_model_class):
        super().__init__()
        # Instantiate NVIDIA's highly optimized TransformerEngine backbone
        self.bert = bert_model_class(config_obj)
        self.regression_head = nn.Linear(config_obj.hidden_size, 1)

    def forward(self, input_ids):
        # Step 1: Step inside the model to pull initial embeddings directly
        hidden_states = self.bert.embeddings(input_ids=input_ids)
        batch_size, seq_length = input_ids.shape
        
        # Step 2: Construct the strict hardware-level log-space Attention Mask
        attention_mask = torch.zeros(
            (batch_size, 1, 1, seq_length),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # Step 3: Layer Unrolling Bypass Loop (Evading Hugging Face argument mismatch)
        for layer_module in self.bert.encoder.layer:
            layer_outputs = layer_module(hidden_states, attention_mask)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

        # Step 4: Isolate cellular high-dimensional vector representations ([CLS] token)
        cell_embedding = hidden_states[:, 0, :]
        
        # Step 5: Final continuous state score projection
        return self.regression_head(cell_embedding)


class GoutDosageDataset(Dataset):
    def __init__(self, processed_data_path):
        with open(processed_data_path, 'rb') as f:
            data = pickle.load(f)
        self.tokens = data['tokens']
        self.dosage = data['dosage']

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.tokens[idx], dtype=torch.long),
            "target": torch.tensor([self.dosage[idx]], dtype=torch.float)
        }


def run_fine_tuning(config):
    print("🔥 [Phase 3] Initiating hardware-accelerated fine-tuning script...")
    
    # 1. Force CUDA launch synchronization for precise tracing logs
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 2. Append recipe blueprints dynamically to sys.path
    recipe_dir = config['paths']['bionemo_recipe_path']
    if recipe_dir not in sys.path:
        sys.path.append(recipe_dir)
        
    from modeling_bert_te import BertModel, TEBertConfig

    # 3. Compile Architecture Configurations
    model_config = TEBertConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        vocab_size=40000,
        max_position_embeddings=config['data_parameters']['max_sequence_length']
    )
    model_config.position_embedding_type = "absolute"
    model_config.is_decoder = False

    # 4. Instantiate and Map Model to Compute Target Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   📡 Routing compute tensors to device target: {device}")
    
    model = GeneformerForRegression(model_config, BertModel)
    model.to(device)

    # 5. Connect Dataset Conveyor Belts
    dataset = GoutDosageDataset(config['paths']['processed_data_output'])
    loader = DataLoader(
        dataset, 
        batch_size=config['hyperparameters']['batch_size'], 
        shuffle=True
    )

    # 6. Configure Optimization Targets
    optimizer = AdamW(
        model.parameters(), 
        lr=float(config['hyperparameters']['learning_rate']), 
        weight_decay=config['hyperparameters']['weight_decay']
    )
    criterion = nn.MSELoss()

    # 7. Core Training Loop Execution
    epochs = config['hyperparameters']['epochs']
    print(f"   🚀 Training network model weights for {epochs} epochs over {len(dataset)} samples...")
    
    for epoch in range(epochs):
        model.train()
        total_epoch_loss = 0.0
        
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            
        avg_loss = total_epoch_loss / len(loader)
        print(f"     ├── Epoch [{epoch+1}/{epochs}] | Average Dataset MSE Loss: {avg_loss:.4f}")

    # 8. Extract State Dictionary and Serialize Checkpoint Out to Disk Volume
    os.makedirs(os.path.dirname(config['paths']['model_checkpoint_output']), exist_ok=True)
    torch.save(model.state_dict(), config['paths']['model_checkpoint_output'])
    print(f"✅ Fine-tuning complete. Checkpoint written to {config['paths']['model_checkpoint_output']}\n")