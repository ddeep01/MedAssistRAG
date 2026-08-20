from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model
)

from datasets import Dataset

from src.generator.training.train_data import load_generator_data


# ============================================================
# Configuration
# ============================================================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATA_PATH = "data/processed/qa_dataset_large.csv"

OUTPUT_DIR = "models/generator_finetuned"

MAX_SAMPLES = 5000

MAX_LENGTH = 768


# ============================================================
# Prepare Dataset
# ============================================================

def format_data(data):
    return Dataset.from_list(data)


# ============================================================
# Tokenization
# ============================================================

def tokenize_function(
    example,
    tokenizer
):

    prompt = example["prompt"]
    response = example["response"]

    # Full training sequence
    full_text = (
        prompt
        + response
        + tokenizer.eos_token
    )

    # Tokenize prompt separately
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

    # Tokenize full prompt + answer
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # --------------------------------------------------------
    # Create labels
    # --------------------------------------------------------

    labels = input_ids.copy()

    prompt_length = len(
        prompt_tokens["input_ids"]
    )

    # Ignore prompt tokens during loss calculation
    labels[:prompt_length] = [
        -100
    ] * prompt_length

    # Ignore padding tokens
    labels = [
        label if attention_mask[i] == 1 else -100
        for i, label in enumerate(labels)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("TinyLlama LoRA Fine-Tuning")
    print("=" * 60)

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    # TinyLlama has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading TinyLlama...")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

    print("Configuring LoRA...")

    peft_config = LoraConfig(

        r=8,

        lora_alpha=16,

        target_modules=[
            "q_proj",
            "v_proj"
        ],

        lora_dropout=0.05,

        bias="none",

        task_type="CAUSAL_LM"
    )

    # --------------------------------------------------------
    # Apply LoRA
    # --------------------------------------------------------

    model = get_peft_model(
        model,
        peft_config
    )

    print("\nTrainable parameters:")

    model.print_trainable_parameters()

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    model = model.to("cpu")

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("\nLoading generator dataset...")

    data = load_generator_data(
        DATA_PATH,
        max_samples=MAX_SAMPLES
    )

    print(
        f"Training samples: {len(data)}"
    )

    # --------------------------------------------------------
    # Convert to Hugging Face Dataset
    # --------------------------------------------------------

    dataset = format_data(data)

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print("\nTokenizing dataset...")

    dataset = dataset.map(
        lambda example: tokenize_function(
            example,
            tokenizer
        ),
        remove_columns=[
            "prompt",
            "response"
        ]
    )

    # --------------------------------------------------------
    # PyTorch format
    # --------------------------------------------------------

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    # --------------------------------------------------------
    # Training arguments
    # --------------------------------------------------------

    training_args = TrainingArguments(

        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=2,

        num_train_epochs=5,

        learning_rate=2e-4,

        logging_steps=50,

        save_steps=500,

        save_total_limit=2,

        remove_unused_columns=False,

        report_to="none",

        fp16=False
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...")

    trainer.train()

    # --------------------------------------------------------
    # Save adapter
    # --------------------------------------------------------

    print("\nSaving TinyLlama LoRA adapter...")

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("✅ TinyLlama LoRA training complete!")
    print(f"✅ Saved to: {OUTPUT_DIR}")
    print("=" * 60)


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()