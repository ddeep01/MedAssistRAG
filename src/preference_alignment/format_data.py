from datasets import load_dataset

def load_and_format_dpo_data(path):
    dataset = load_dataset("json", data_files=path, split="train")

    def format_example(example):
        prompt = f"<|user|>\n{example['prompt']}\n<|assistant|>\n"

        return {
            "prompt": prompt[:512],
            "chosen": example["chosen"][:512],
            "rejected": example["rejected"][:512]
        }

    dataset = dataset.map(format_example)
    return dataset