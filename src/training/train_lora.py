from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

model_name = "google/flan-t5-large"

# 🔹 Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 🔹 FIXED LoRA config for T5
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # ✅ FIX
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)

# 🔥 IMPORTANT: Check trainable params
model.print_trainable_parameters()

# 🔹 Load dataset
dataset = load_from_disk("data/tokenized")

# 🔹 Training config
training_args = TrainingArguments(
    output_dir="models/lora",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=False  # Mac safe
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"]
)

# 🔹 Train
trainer.train()

# 🔹 Save
model.save_pretrained("models/lora-final")

print("✅ Training complete!")