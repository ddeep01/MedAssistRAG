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


# ============================================================
# Load Tokenizer
# ============================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# Load TinyLlama
# ============================================================

print("Loading TinyLlama...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model.config.pad_token_id = tokenizer.pad_token_id


# ============================================================
# LoRA Configuration
# ============================================================

lora_config = LoraConfig(

    # Low-rank dimension
    r=8,

    # LoRA scaling
    lora_alpha=16,

    # Transformer attention modules
    target_modules=[
        "q_proj",
        "v_proj"
    ],

    # Regularization
    lora_dropout=0.1,

    # Do not train bias
    bias="none",

    # TinyLlama is a causal language model
    task_type="CAUSAL_LM"
)


# ============================================================
# Apply LoRA
# ============================================================

print("Applying LoRA...")

model = get_peft_model(
    model,
    lora_config
)


# ============================================================
# Check Trainable Parameters
# ============================================================

print("\n===== Trainable Parameters =====")

model.print_trainable_parameters()


# ============================================================
# Load Tokenized Dataset
# ============================================================

print("\nLoading tokenized dataset...")

dataset = load_from_disk(
    DATASET_PATH
)

print(dataset)


# ============================================================
# Data Collator
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ============================================================
# Training Arguments
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # Batch size
    per_device_train_batch_size=2,

    # Number of passes through dataset
    num_train_epochs=3,

    # Learning rate for LoRA
    learning_rate=2e-4,

    # Logging
    logging_steps=10,

    # Save checkpoints
    save_steps=100,

    save_total_limit=2,

    # Mac-safe setting
    fp16=False,

    # Disable external reporting
    report_to="none",

    # Keep all dataset columns
    remove_unused_columns=False
)


# ============================================================
# Trainer
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=dataset["train"],

    data_collator=data_collator
)


# ============================================================
# Train
# ============================================================

print("\n========================================")
print("Starting TinyLlama LoRA Fine-Tuning")
print("========================================")

trainer.train()


# ============================================================
# Save LoRA Adapter
# ============================================================

print("\nSaving LoRA adapter...")

model.save_pretrained(
    FINAL_MODEL_DIR
)

tokenizer.save_pretrained(
    FINAL_MODEL_DIR
)


print("\n========================================")
print("✅ TinyLlama LoRA Training Complete!")
print(f"✅ Adapter saved to: {FINAL_MODEL_DIR}")
print("========================================")