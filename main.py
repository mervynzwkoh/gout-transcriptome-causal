import yaml
from pipeline.ingestion import run_ingestion
from pipeline.preprocessing import run_preprocessing
from pipeline.fine_tuning import run_fine_tuning
from pipeline.inference import run_inference    

def main():
    print("Initializing End-to-End Digital Biology Pipeline...\n")
    
    # 1. Load the central dashboard
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Execute the sequence
    run_ingestion(config)
    run_preprocessing(config)
    run_fine_tuning(config)
    run_inference(config)
    
    print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()