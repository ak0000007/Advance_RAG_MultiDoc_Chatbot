import asyncio
from src.ingestion.loader import MultiDocumentLoader
from src.chunking.splitter import DocumentChunker
from src.vector_stores.qdrant_store import QdrantStore
from src.embeddings.embedding import BGEEmbeddings
from src.config import settings

def main():
    print("Loading documents from data/...")
    loader = MultiDocumentLoader(data_directory="./data")
    docs = loader.load_directory()
    
    print("Chunking documents...")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(docs)
    
    print(f"Embedding and storing {len(chunks)} chunks in Qdrant...")
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
    qdrant.vector_store.add_documents(chunks)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
