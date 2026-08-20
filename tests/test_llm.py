from src.generator.llm import LLM


def main():
    llm = LLM()
    response = llm.generate("What is diabetes?")
    print(response)


if __name__ == "__main__":
    main()