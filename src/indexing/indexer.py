import hashlib
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.indexing import index
from langchain_core.vectorstores import VectorStore
from langchain_community.indexes import SQLRecordManager


# DocumentIndexer manages document loading, ID generation, hashing, and writing to vector database.
class DocumentIndexer:

    # Initializes DocumentIndexer with a vector store and a SQL database to track indexed states.
    def __init__(
        self,
        vector_store: VectorStore,
        db_url: str = "sqlite:///record_manager.db",
        namespace: str = "multidoc_rag",
    ):
        self.vector_store = vector_store

        self.record_manager = SQLRecordManager(
            namespace=namespace,
            db_url=db_url,
        )

        self.record_manager.create_schema()

    # Generates a deterministic unique ID for a document using a SHA-256 hash of its source and system key.
    @staticmethod
    def create_document_id(
        source: str,
        external_id: str,
    ) -> str:
        raw_id = f"{source}:{external_id}"

        return hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

    # Computes a SHA-256 hash of the document content to identify future text updates.
    @staticmethod
    def calculate_content_hash(
        content: str,
    ) -> str:
        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    # Processes raw documents to generate and inject deterministic IDs and content hashes into metadata.
    def prepare_documents(
        self,
        documents: Iterable[Document],
        source_system: str = "local_files",
    ) -> list[Document]:
        prepared_documents = []

        for doc in documents:
            source = doc.metadata.get(
                "source",
                "unknown",
            )

            document_id = self.create_document_id(
                source=source_system,
                external_id=source,
            )

            content_hash = self.calculate_content_hash(
                doc.page_content
            )

            doc.metadata["source_system"] = source_system
            doc.metadata["document_id"] = document_id
            doc.metadata["content_hash"] = content_hash

            prepared_documents.append(doc)

        return prepared_documents

    # Automatically prepares and indexes documents into the vector store while keeping state synchronized.
    def index_documents(
        self,
        documents: Iterable[Document],
        source_system: str = "local_files",
        cleanup: str = "incremental",
        source_id_key: str = "source",
    ):
        documents = self.prepare_documents(
            documents=documents,
            source_system=source_system,
        )

        result = index(
            docs_source=documents,
            record_manager=self.record_manager,
            vector_store=self.vector_store,
            cleanup=cleanup,
            source_id_key=source_id_key,
        )

        return result