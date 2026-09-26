from pydantic import BaseModel
from src.llm.provider import create_llm

class TestSchema(BaseModel):
    name: str

llm = create_llm(provider="google", fallback_provider=None) # No fallback
structured_llm = llm.with_structured_output(TestSchema)
print("Primary Structured LLM type:", type(structured_llm))
