from src.llm.provider import create_llm

llm = create_llm(provider="openai")
try:
    print(llm.invoke("Hello, are you there?"))
except Exception as e:
    print("OpenAI Error:", e)
