from dotenv import load_dotenv
load_dotenv()

from src.telemetry import setup_langsmith
setup_langsmith()

import asyncio
import hashlib
import uuid
from pathlib import Path

from src.ingestion.loader import MultiDocumentLoader
from src.chunking.splitter import DocumentChunker
from src.vector_stores.qdrant_store import QdrantStore
from src.retrieval.bm25_store import BM25Store
from src.embeddings.embedding import BGEEmbeddings
from src.config import settings


def generate_chunk_id(doc, chunk_idx: int) -> str:
    """Generate deterministic RFC 4122 UUIDv5 based on source, index, and content hash."""
    source = doc.metadata.get("source", "unknown")
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_idx}:{content_hash}"))


def main():
    print("Loading documents from data/...")
    loader = MultiDocumentLoader(data_directory="./data")
    docs = loader.load_directory()
    
    print("Chunking documents...")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(docs)

    print(f"Generating deterministic IDs for {len(chunks)} chunks...")
    doc_ids = []
    for idx, chunk in enumerate(chunks):
        cid = generate_chunk_id(chunk, idx)
        chunk.id = cid
        chunk.metadata["chunk_id"] = cid
        doc_ids.append(cid)
    
    print(f"Embedding and storing {len(chunks)} chunks in Qdrant (idempotent upsert)...")
    bge = BGEEmbeddings()
    embeddings = bge.get_embeddings()
    if settings.qdrant_url:
        qdrant = QdrantStore(
            embeddings=embeddings, 
            collection_name="multidoc_rag", 
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
    else:
        qdrant = QdrantStore(
            embeddings=embeddings, 
            collection_name="multidoc_rag", 
            path="./qdrant_data"
        )
    qdrant.vector_store.add_documents(chunks, ids=doc_ids)

    print(f"Indexing and persisting {len(chunks)} chunks in BM25Store...")
    bm25_path = Path(settings.bm25_store_path)
    bm25_store = BM25Store.load(bm25_path) if bm25_path.exists() else BM25Store()
    bm25_store.add_documents(chunks)
    bm25_store.save(bm25_path)

    print(f"Ingestion complete. Qdrant & BM25 synchronized at {bm25_path}.")


if __name__ == "__main__":
    main()
