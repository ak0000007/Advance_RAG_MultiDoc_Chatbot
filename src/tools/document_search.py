"""
LangChain tools backed by the project's existing
retrieval infrastructure.
"""

import logging
from langchain_core.tools import tool
from langchain_core.output_parsers import PydanticOutputParser
from src.rag.grading import RetrievalGrade

logger = logging.getLogger(__name__)

def build_document_search_tool(retriever, grader_llm, rewriter_chain, max_attempts: int):
    """
    Create a LangChain Tool backed by the project's
    existing hybrid + reranking retriever.
    Includes self-correcting retrieval logic.
    """

    @tool
    async def search_documents(query: str) -> str:
        """
        Search the indexed project documents for information
        relevant to the user's question.

        Use this tool when information from the available
        documents is required.
        """
        current_query = query
        
        for attempt in range(1, max_attempts + 1):
            documents = await retriever.ainvoke(
                {
                    "question": current_query,
                    "metadata_filter": None, # TODO: must be scoped per user before multi-tenant use
                }
            )

            if not documents:
                return (
                    "No relevant documents were found "
                    "for this query."
                )

            results = []
            for index, document in enumerate(documents, start=1):
                source = document.metadata.get("source", "unknown")
                results.append(
                    f"[Document {index}]\n"
                    f"Source: {source}\n"
                    f"{document.page_content}"
                )
            context = "\n\n".join(results)

            try:
                parser = PydanticOutputParser(pydantic_object=RetrievalGrade)
                grade = await grader_llm.ainvoke(
                    {
                        "question": query,
                        "context": context,
                        "format_instructions": parser.get_format_instructions(),
                    }
                )
                relevant = grade.relevant
                reason = grade.reason
            except Exception as e:
                logger.warning(f"Retrieval grading failed: {e}. Defaulting to relevant.")
                relevant = True
                reason = "Grading failed, assumed relevant."

            logger.info(f"Retrieval Attempt {attempt} | Query: {current_query} | Relevant: {relevant} | Reason: {reason}")

            if relevant:
                return context

            if attempt < max_attempts:
                current_query = await rewriter_chain.ainvoke(
                    {
                        "question": query,
                        "reason": reason,
                    }
                )

        return "Note: Relevance of these documents could not be fully confirmed.\n\n" + context

    return search_documents
