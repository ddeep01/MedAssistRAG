import os
import xml.etree.ElementTree as ET
import pandas as pd
from datasets import load_dataset, load_from_disk

# =========================
# CLEAN FUNCTION
# =========================
def clean_text(text):
    if text is None:
        return ""
    text = text.replace("Key Points", "")
    text = text.replace("\n", " ")
    return text.strip()


# =========================
# PUBMEDQA
# =========================
def load_pubmedqa():
    ds = load_from_disk("data/raw/pubmedqa")

    data = []
    for item in ds["train"]:
        q = item.get("question", "")
        a = item.get("long_answer", "")

        data.append({
            "question": clean_text(q),
            "answer": clean_text(a),
            "source": "pubmedqa"
        })

    return pd.DataFrame(data)


# =========================
# MEDQUAD (XML)
# =========================
def load_medquad():
    base_path = "data/raw/medquad"
    data = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".xml"):
                file_path = os.path.join(root, file)

                try:
                    tree = ET.parse(file_path)
                    root_elem = tree.getroot()

                    qapairs = root_elem.find("QAPairs")
                    if qapairs is None:
                        continue

                    for qa in qapairs.findall("QAPair"):
                        question = qa.find("Question")
                        answer = qa.find("Answer")

                        if question is not None and answer is not None:
                            if question.text and answer.text:
                                data.append({
                                    "question": clean_text(question.text),
                                    "answer": clean_text(answer.text),
                                    "source": "medquad"
                                })

                except Exception as e:
                    print(f"Error: {file_path}")

    return pd.DataFrame(data)


# =========================
# MEDMCQA (BIG DATASET)
# =========================
def load_medmcqa():
    dataset = load_dataset("medmcqa")

    data = []
    for item in dataset["train"]:
        q = item.get("question", "")

        correct = item.get("cop", 0)

        options = [
            item.get("opa", ""),
            item.get("opb", ""),
            item.get("opc", ""),
            item.get("opd", "")
        ]

        if correct < len(options):
            a = options[correct]
        else:
            a = options[0]

        data.append({
            "question": clean_text(q),
            "answer": clean_text(a),
            "source": "medmcqa"
        })

    return pd.DataFrame(data)


# =========================
# MERGE EVERYTHING
# =========================
def merge_all():
    print("Loading PubMedQA...")
    df_pubmed = load_pubmedqa()
    print("PubMedQA:", len(df_pubmed))

    print("Loading MedQuAD...")
    df_medquad = load_medquad()
    print("MedQuAD:", len(df_medquad))

    print("Loading MedMCQA...")
    df_medmcqa = load_medmcqa()
    print("MedMCQA:", len(df_medmcqa))

    # Merge datasets
    df = pd.concat([df_pubmed, df_medquad, df_medmcqa])

    # Remove duplicates
    df = df.drop_duplicates(subset=["question"])

    # Save
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/qa_dataset_large.csv", index=False)

    print("\n✅ FINAL DATASET SIZE:", len(df))
    print("Saved to: data/processed/qa_dataset_large.csv")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    merge_all()