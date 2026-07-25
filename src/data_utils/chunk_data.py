from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(df):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = []

    for _, row in df.iterrows():
        texts = splitter.split_text(row["text"])
        for t in texts:
            chunks.append(t)

    return chunks