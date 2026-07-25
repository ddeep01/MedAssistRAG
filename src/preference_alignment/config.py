from pathlib import Path

#MODEL_NAME = "llama3"
#MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "models" /"dpo_aligned_model"

TRAINING_CONFIG = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-5,
    "num_train_epochs": 1,
    "logging_steps": 10,
    "save_steps": 50
}