import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from peft import PeftModel


class LLM:

    def __init__(self):

        # ====================================================
        # Base TinyLlama Model
        # ====================================================

        self.base_model = (
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )

        # ====================================================
        # Fine-Tuned LoRA Adapter
        # ====================================================

        self.adapter_path = (
            "models/generator_finetuned"
        )

        print("Loading TinyLlama tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        print("Loading TinyLlama base model...")

        model = AutoModelForCausalLM.from_pretrained(
            self.base_model
        )

        # ====================================================
        # Load LoRA Adapter
        # ====================================================

        print("Loading LoRA adapter...")

        model = PeftModel.from_pretrained(
            model,
            self.adapter_path
        )

        # ====================================================
        # CPU
        # ====================================================

        self.model = model.to("cpu")

        self.model.eval()

        print("✅ TinyLlama LoRA generator loaded!")


    # ========================================================
    # Raw Generation
    # ========================================================

    def generate_raw(self, prompt):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )

        inputs = {
            key: value.to("cpu")
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=200,

                do_sample=True,

                temperature=0.7,

                top_p=0.9,

                pad_token_id=self.tokenizer.pad_token_id,

                eos_token_id=self.tokenizer.eos_token_id
            )

        # ----------------------------------------------------
        # Only decode generated portion
        # ----------------------------------------------------

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[
            0,
            input_length:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()


    # ========================================================
    # Medical Generation
    # ========================================================

    def generate(
        self,
        question,
        context
    ):

        prompt = f"""### Instruction:
You are a medical assistant.

Use ONLY the provided context to answer the question.

If the answer cannot be found in the context, say "I don't know."

### Context:
{context}

### Question:
{question}

### Answer:
"""

        return self.generate_raw(prompt)