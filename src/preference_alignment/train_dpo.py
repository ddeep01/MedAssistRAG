from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
import torch

from src.preference_alignment.config import MODEL_NAME, OUTPUT_DIR, TRAINING_CONFIG
from src.preference_alignment.format_data import load_and_format_dpo_data


def main():
    # ✅ Force CPU (avoids MPS crash)
    torch.set_default_device("cpu")

    print("🚀 Loading dataset...")
    dataset = load_and_format_dpo_data("data/processed/dpo_data.jsonl")

    # 🔥 Reduce dataset (VERY IMPORTANT)
    dataset = dataset.select(range(1000))

    print("📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # 🔥 LoRA setup
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("⚙️ Setting up DPO trainer...")

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        save_steps=TRAINING_CONFIG["save_steps"],
        bf16=False
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    print("🔥 Training started...")
    trainer.train()

    print("💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("✅ DPO training complete!")


if __name__ == "__main__":
    main()