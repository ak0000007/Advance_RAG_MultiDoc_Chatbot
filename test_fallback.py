from pydantic import BaseModel
from src.llm.provider import create_llm

class TestSchema(BaseModel):
    name: str

structured_llm = create_llm(provider="google", fallback_provider=None, structured_output=TestSchema) # No fallback
print("Primary Structured LLM type:", type(structured_llm))
