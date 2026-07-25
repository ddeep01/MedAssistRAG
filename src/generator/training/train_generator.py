from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

from datasets import Dataset

from src.generator.training.train_data import load_generator_data


def format_data(data, tokenizer):
    texts = []

    for item in data:
        text = item["prompt"] + item["response"]
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def tokenize_function(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # 🔥 ADD LABELS (THIS FIXES YOUR ERROR)
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


def main():
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(base_model)

    # 🔥 LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # CPU training (safe for Mac)
    model = model.to("cpu")

    print("Loading dataset...")
    data = load_generator_data(
        "data/processed/qa_dataset_large.csv",
        max_samples=5000
    )

    dataset = format_data(data, tokenizer)

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    training_args = TrainingArguments(
        output_dir="models/generator_finetuned",
        per_device_train_batch_size=2,
        num_train_epochs=5,   # you can increase to 10
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False
    )

    print("Training started...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    print("Saving model...")
    model.save_pretrained("models/generator_finetuned")
    tokenizer.save_pretrained("models/generator_finetuned")

    print("Done!")


if __name__ == "__main__":
    main()