from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class RetrievalGrade(BaseModel):
    """
    Structured result produced by the retrieval grader.
    """

    relevant: bool = Field(
        description=(
            "Whether the retrieved documents contain "
            "useful information for answering the question."
        )
    )

    reason: str = Field(
        description=(
            "Brief explanation of why the retrieved "
            "documents are or are not relevant."
        )
    )

def build_retrieval_grader(llm):
    """Builds the retrieval grader chain."""
    parser = PydanticOutputParser(pydantic_object=RetrievalGrade)

    grader_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a retrieval relevance grader inside a
Retrieval-Augmented Generation system.

Your task is to determine whether the retrieved
documents contain information relevant to answering
the user's question.

IMPORTANT:

- Judge the retrieved context.
- Do not answer the question yourself.
- The documents do not need to contain the complete answer.
- They must contain useful information that can help answer
  the question.
- Do not use outside knowledge.
- Do not invent information.

Return the required structured format.

{format_instructions}
                """,
            ),
            (
                "human",
                """
Question:

{question}

Retrieved context:

{context}

Determine whether the retrieved context is relevant.
                """,
            ),
        ]
    )

    # Note: `llm` here should already be configured with `with_structured_output(RetrievalGrade)`
    # by dependencies.py, but we'll try to add it if it's not.
    try:
        structured_llm = llm.with_structured_output(RetrievalGrade)
    except AttributeError:
        structured_llm = llm  # Already configured via factory
        
    return grader_prompt | structured_llm
