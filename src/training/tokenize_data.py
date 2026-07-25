from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("json", data_files="data/finetune_data.json")

def tokenize(example):
    inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

    labels = tokenizer(
        example["output"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized = dataset.map(tokenize, remove_columns=["input", "output"])

tokenized.save_to_disk("data/tokenized")

print("✅ Tokenization done!")