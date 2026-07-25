from src.pipeline.rag_pipeline import RAGPipeline

rag = RAGPipeline()

while True:
    query = input("\nAsk medical question: ")
    if query.lower() in ["exit", "quit"]:
        break

    result = rag.query(query)

    print("\n--- RETRIEVED CONTEXT ---")
    for d in result["sources"]:
        print("-", str(d)[:150])

    print("\n--- ANSWER ---")
    print(result["answer"])