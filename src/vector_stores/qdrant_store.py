from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings
from qdrant_client.http import models


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

        # Remote Qdrant with local fallback
        if url:
            use_remote = False
            try:
                import socket
                from urllib.parse import urlparse
                p = urlparse(url)
                host = p.hostname or "localhost"
                port = p.port or (443 if p.scheme == "https" else 80)
                sock = socket.create_connection((host, port), timeout=0.5)
                sock.close()
                use_remote = True
            except Exception:
                use_remote = False

            if use_remote:
                try:
                    self.vector_store = QdrantVectorStore.construct_instance(
                        embedding=embeddings,
                        collection_name=collection_name,
                        client_options={"url": url, "api_key": api_key, "timeout": 2.0},
                    )
                    self.vector_store.client.get_collections()
                except Exception:
                    use_remote = False

            if not use_remote:
                fallback_path = path or "./qdrant_data"
                self.vector_store = QdrantVectorStore.construct_instance(
                    embedding=embeddings,
                    collection_name=collection_name,
                    client_options={"path": fallback_path},
                )

        # Local persistent Qdrant
        elif path:
            self.vector_store = QdrantVectorStore.construct_instance(
                embedding=embeddings,
                collection_name=collection_name,
                client_options={"path": path},
            )

        else:
            raise ValueError(
                "Provide either 'url' or 'path'."
            )

    def as_retriever(
        self,
        search_kwargs: dict | None = None,
        metadata_filter: dict | None = None,
    ):
        """
        Return a LangChain retriever.
        Optional metadata_filter applies exact match filtering on document metadata.
        """
        kwargs = search_kwargs.copy() if search_kwargs else {"k": 4}

        if metadata_filter:
            must_conditions = [
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value)
                )
                for key, value in metadata_filter.items()
            ]
            kwargs["filter"] = models.Filter(must=must_conditions)

        return self.vector_store.as_retriever(
            search_kwargs=kwargs
        )

    def as_dynamic_retriever(self, base_search_kwargs: dict | None = None):
        """
        Returns a LangChain Runnable that dynamically applies metadata_filter 
        from the input payload at query time.
        """
        from langchain_core.runnables import RunnableLambda

        def _retrieve(inputs: dict | str):
            if isinstance(inputs, str):
                question = inputs
                metadata_filter = None
            else:
                question = inputs.get("question", "")
                metadata_filter = inputs.get("metadata_filter", None)
            
            # Delegate to the Qdrant-specific retriever builder
            retriever = self.as_retriever(
                search_kwargs=base_search_kwargs, 
                metadata_filter=metadata_filter
            )
            return retriever.invoke(question)

        return RunnableLambda(_retrieve)