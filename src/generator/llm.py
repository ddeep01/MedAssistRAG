
import ollama


class LLM:
    def __init__(self):
        self.model = "llama3"

    def generate_raw(self, prompt):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def generate(self, question, context):
        prompt = f"""
You are a medical assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
"""
        return self.generate_raw(prompt)

'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


class LLM:
    def __init__(self):
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Paths
        sft_path = "models/generator_finetuned"
        dpo_path = "models/dpo_aligned_model"

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        print("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(base_model)

        # 🔥 STEP 1: Load SFT adapter
        print("Loading SFT adapter...")
        model = PeftModel.from_pretrained(model, sft_path)

        ## 🔥 STEP 2: Load DPO adapter on top
        print("Loading DPO adapter...")
        model = PeftModel.from_pretrained(model, dpo_path)

        # Move to CPU (safe for Mac)
        self.model = model.to("cpu")
        self.model.eval()

    def generate_raw(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to("cpu") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def generate(self, question, context):
        prompt = f"""
            You are a medical assistant.

            Use ONLY the context below to answer the question.
            If the answer is not in the context, say "I don't know."

            Context:
            {context}

            Question:
            {question}

            Answer:
        """
        return self.generate_raw(prompt)
'''