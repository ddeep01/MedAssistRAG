from src.generator.llm import LLM

llm = LLM()

response = llm.generate("What is diabetes?")
print(response)