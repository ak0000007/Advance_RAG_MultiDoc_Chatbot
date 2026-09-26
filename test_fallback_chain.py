from src.llm.provider import create_llm
from src.graph.nodes import AnswerGrade

llm = create_llm(provider="google", fallback_provider="openai")
structured_llm = llm.with_structured_output(AnswerGrade)
print("Type of structured_llm:", type(structured_llm))
print("Has fallbacks?", hasattr(structured_llm, "fallbacks"))
