"""
LangChain tools backed by the project's existing
retrieval infrastructure.
"""

from langchain_core.tools import tool


def build_document_search_tool(retriever):
    """
    Create a LangChain Tool backed by the project's
    existing hybrid + reranking retriever.
    """

    @tool
    def search_documents(query: str) -> str:
        """
        Search the indexed project documents for information
        relevant to the user's question.

        Use this tool when information from the available
        documents is required.
        """

        documents = retriever.invoke(
            {
                "question": query,
                "metadata_filter": None,
            }
        )

        if not documents:
            return (
                "No relevant documents were found "
                "for this query."
            )

        results = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            source = document.metadata.get(
                "source",
                "unknown",
            )

            results.append(
                f"[Document {index}]\n"
                f"Source: {source}\n"
                f"{document.page_content}"
            )

        return "\n\n".join(results)

    return search_documents