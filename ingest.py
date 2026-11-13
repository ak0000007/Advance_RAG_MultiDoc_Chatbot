from dotenv import load_dotenv
load_dotenv()

from src.telemetry import setup_langsmith
setup_langsmith()

import argparse
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


def generate_chunk_id(doc, chunk_idx: int, tenant_id: str = "default") -> str:
    """Generate deterministic RFC 4122 UUIDv5 based on tenant, source, index, and content hash."""
    source = doc.metadata.get("source", "unknown")
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}:{source}:{chunk_idx}:{content_hash}"))


def main():
    parser = argparse.ArgumentParser(description="Ingest documents with deterministic IDs and tenant isolation.")
    parser.add_argument("--data-dir", default="./data", help="Directory with documents to index")
    parser.add_argument("--tenant-id", default="default", help="Tenant ID for document isolation")
    args = parser.parse_args()

    print(f"Loading documents from {args.data_dir} for tenant '{args.tenant_id}'...")
    loader = MultiDocumentLoader(data_directory=args.data_dir)
    docs = loader.load_directory()
    
    print(f"Loaded {len(docs)} document objects. Chunking documents...")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(docs)

    print(f"Generating deterministic IDs and stamping tenant '{args.tenant_id}' for {len(chunks)} chunks...")
    doc_ids = []
    for idx, chunk in enumerate(chunks):
        chunk.metadata["tenant_id"] = args.tenant_id
        cid = generate_chunk_id(chunk, idx, tenant_id=args.tenant_id)
        chunk.id = cid
        chunk.metadata["chunk_id"] = cid
        if "title" not in chunk.metadata:
            candidate_title = chunk.metadata.get("file_name") or Path(chunk.metadata.get("source", "")).name
            if candidate_title:
                chunk.metadata["title"] = candidate_title
        doc_ids.append(cid)
    
    print(f"Embedding and storing {len(chunks)} chunks in Qdrant (idempotent upsert)...")
    bge = BGEEmbeddings()
    embeddings = bge.get_embeddings()
    if settings.qdrant_url:
        qdrant = QdrantStore(
            embeddings=embeddings, 
            collection_name="multidoc_rag", 
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
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

    print(f"Ingestion complete. Qdrant & BM25 synchronized at {bm25_path} (tenant='{args.tenant_id}').")


if __name__ == "__main__":
    main()
