from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

INPUT_FILE = "data/finetune_data.json"
OUTPUT_DIR = "data/tokenized"


def tokenize_example(example, tokenizer):
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


def main():
    import os
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input dataset '{INPUT_FILE}' not found. Run prepare_finetune_data.py first.")
        return

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from: {INPUT_FILE}")
    dataset = load_dataset("json", data_files=INPUT_FILE)

    tokenized = dataset.map(
        lambda ex: tokenize_example(ex, tokenizer),
        remove_columns=["prompt", "response"]
    )

    tokenized.save_to_disk(OUTPUT_DIR)
    print("✅ TinyLlama tokenization complete!")
    print(f"✅ Saved tokenized dataset to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()