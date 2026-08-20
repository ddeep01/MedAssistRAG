from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

INPUT_FILE = "data/finetune_data.json"
OUTPUT_DIR = "data/tokenized"


# ============================================================
# Load Tokenizer
# ============================================================

print(f"Loading tokenizer: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# TinyLlama does not have a pad token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load Dataset
# ============================================================

dataset = load_dataset(
    "json",
    data_files=INPUT_FILE
)


# ============================================================
# Tokenization
# ============================================================

def tokenize(example):

    prompt = example["prompt"]
    response = example["response"]

    # Full causal-LM training sequence
    text = prompt + response + tokenizer.eos_token

    tokenized = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=768
    )

    # Causal LM labels are the same token IDs
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# ============================================================
# Apply Tokenization
# ============================================================

tokenized = dataset.map(
    tokenize,
    remove_columns=[
        "prompt",
        "response"
    ]
)


# ============================================================
# Save
# ============================================================

tokenized.save_to_disk(
    OUTPUT_DIR
)

print("✅ TinyLlama tokenization complete!")
print(f"✅ Saved tokenized dataset to: {OUTPUT_DIR}")