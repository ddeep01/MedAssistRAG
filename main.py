from src.pipeline.rag_pipeline import RAGPipeline


def main():
    rag = RAGPipeline()

    print("[MedAssistRAG Interactive Medical Assistant]")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input("Ask medical question: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = rag.query(query)

        print("\n--- RETRIEVED CONTEXT ---")
        for d in result["sources"]:
            print("-", str(d)[:150])

        print("\n--- ANSWER ---")
        print(result["answer"])
        print("-" * 50)


if __name__ == "__main__":
    main()