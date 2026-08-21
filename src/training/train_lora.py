from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model
)

from datasets import load_from_disk


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATASET_PATH = "data/tokenized"

OUTPUT_DIR = "models/tinyllama-lora"

FINAL_MODEL_DIR = "models/tinyllama-lora-final"


def main():
    import os
    if not os.path.exists(DATASET_PATH):
        print(f"[Error] Tokenized dataset '{DATASET_PATH}' not found. Run tokenize_data.py first.")
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading TinyLlama...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)

    print("\n===== Trainable Parameters =====")
    model.print_trainable_parameters()

    print("\nLoading tokenized dataset...")
    dataset = load_from_disk(DATASET_PATH)
    print(dataset)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=data_collator
    )

    print("\n========================================")
    print("Starting TinyLlama LoRA Fine-Tuning")
    print("========================================")

    trainer.train()

    print("\nSaving LoRA adapter...")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    print("\n========================================")
    print("✅ TinyLlama LoRA Training Complete!")
    print(f"✅ Adapter saved to: {FINAL_MODEL_DIR}")
    print("========================================")


if __name__ == "__main__":
    main()