from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings


class QdrantStore:
    """
    Qdrant vector-store wrapper.

    Responsible only for:
    - connecting to Qdrant
    - creating/loading a collection
    - exposing the LangChain VectorStore
    """

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "multidoc_rag",
        url: str | None = None,
        api_key: str | None = None,
        path: str | None = None,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name

        # Remote Qdrant
        if url:
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=collection_name,
                url=url,
                api_key=api_key,
            )

        # Local persistent Qdrant
        elif path:
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=collection_name,
                path=path,
            )

        else:
            raise ValueError(
                "Provide either 'url' or 'path'."
            )

    def as_retriever(
        self,
        search_kwargs: dict | None = None,
    ):
        """
        Return a LangChain retriever.
        """

        return self.vector_store.as_retriever(
            search_kwargs={"k": 4}
        )