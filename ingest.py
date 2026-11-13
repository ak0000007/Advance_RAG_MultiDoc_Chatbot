from dotenv import load_dotenv
load_dotenv()

from src.telemetry import setup_langsmith
setup_langsmith()

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path

from src.ingestion.loader import MultiDocumentLoader
from src.chunking.splitter import DocumentChunker
from src.vector_stores.qdrant_store import QdrantStore
from src.retrieval.bm25_store import BM25Store
from src.embeddings.embedding import BGEEmbeddings
from src.config import settings

logger = logging.getLogger(__name__)


def normalize_salesforce_id(sf_id: str) -> str:
    """
    Validate and normalize a 15-character or 18-character Salesforce ID into 18 characters.
    Rejects malformed IDs with a clear ValueError.
    """
    if not isinstance(sf_id, str):
        raise ValueError(f"Salesforce ID must be a string, got {type(sf_id).__name__}")
    
    clean_id = sf_id.strip()
    if len(clean_id) == 18:
        if not clean_id.isalnum():
            raise ValueError(f"Salesforce ID must be alphanumeric, got: '{clean_id}'")
        return clean_id
    
    if len(clean_id) == 15:
        if not clean_id.isalnum():
            raise ValueError(f"Salesforce ID must be alphanumeric, got: '{clean_id}'")
        lookup = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        suffix = ""
        for i in range(3):
            block = clean_id[i * 5 : (i + 1) * 5]
            val = sum((1 << bit) for bit, char in enumerate(block) if char.isupper())
            suffix += lookup[val]
        return clean_id + suffix
    
    raise ValueError(
        f"Invalid Salesforce ID length ({len(clean_id)}). Must be 15 or 18 alphanumeric characters: '{clean_id}'"
    )


def load_sf_manifest(manifest_path: str | Path) -> dict[str, dict]:
    """
    Load and validate a Salesforce document manifest from JSON or CSV.
    Returns mapping: normalized_filename -> {
        'sf_account_id': str | None,
        'allowed_roles': list[str],
        'sf_scope': 'account' | 'org' | 'none'
    }
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Salesforce manifest file not found: {path}")

    manifest_entries: list[dict] = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                manifest_entries = data
            elif isinstance(data, dict):
                manifest_entries = [
                    {"file": k, **v} if isinstance(v, dict) else {"file": k}
                    for k, v in data.items()
                ]
            else:
                raise ValueError("JSON manifest must be an array of objects or an object map.")
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            manifest_entries = list(reader)
    else:
        raise ValueError(f"Unsupported manifest format: {path.suffix}. Must be .json or .csv")

    validated_map: dict[str, dict] = {}
    for row_idx, row in enumerate(manifest_entries, start=1):
        filename = str(row.get("file") or "").strip()
        if not filename:
            raise ValueError(f"Manifest row {row_idx} is missing required 'file' property: {row}")

        file_key = Path(filename).name.lower()
        scope = str(row.get("scope") or row.get("sf_scope") or "").strip().lower()
        raw_account_id = row.get("sf_account_id") or row.get("account_id")
        
        # Parse allowed_roles (list in JSON or semicolon/comma-separated in CSV)
        raw_roles = row.get("allowed_roles")
        allowed_roles: list[str] = []
        if isinstance(raw_roles, list):
            allowed_roles = [str(r).strip() for r in raw_roles if str(r).strip()]
        elif isinstance(raw_roles, str) and raw_roles.strip():
            delim = ";" if ";" in raw_roles else ","
            allowed_roles = [r.strip() for r in raw_roles.split(delim) if r.strip()]

        if scope == "org" and not raw_account_id:
            validated_map[file_key] = {
                "sf_account_id": None,
                "allowed_roles": allowed_roles,
                "sf_scope": "org",
            }
        elif raw_account_id:
            normalized_id = normalize_salesforce_id(str(raw_account_id))
            validated_map[file_key] = {
                "sf_account_id": normalized_id,
                "allowed_roles": allowed_roles,
                "sf_scope": "account",
            }
        else:
            raise ValueError(
                f"Manifest row {row_idx} for '{filename}' must specify either 'sf_account_id' or 'scope': 'org'."
            )

    return validated_map


def generate_chunk_id(
    doc,
    chunk_idx: int,
    tenant_id: str = "default",
    sf_account_id: str | None = None,
) -> str:
    """Generate deterministic RFC 4122 UUIDv5 based on tenant, sf_account_id, source, index, and content hash."""
    source = doc.metadata.get("source", "unknown")
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    if sf_account_id:
        seed = f"{tenant_id}:{sf_account_id}:{source}:{chunk_idx}:{content_hash}"
    else:
        seed = f"{tenant_id}:{source}:{chunk_idx}:{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def main():
    parser = argparse.ArgumentParser(description="Ingest documents with deterministic IDs and tenant isolation.")
    parser.add_argument("--data-dir", default="./data/txt", help="Directory with documents to index")
    parser.add_argument("--tenant-id", default="default", help="Tenant ID for document isolation")
    parser.add_argument("--sf-manifest", default=None, help="Path to Salesforce document manifest (JSON or CSV)")
    parser.add_argument("--recreate", action="store_true", help="Recreate indexes fresh instead of appending")
    args = parser.parse_args()

    manifest_map = {}
    if args.sf_manifest:
        print(f"Loading Salesforce document manifest from {args.sf_manifest}...")
        manifest_map = load_sf_manifest(args.sf_manifest)
        print(f"Loaded and validated manifest for {len(manifest_map)} files.")

    print(f"Loading documents from {args.data_dir} for tenant '{args.tenant_id}'...")
    loader = MultiDocumentLoader(data_directory=args.data_dir)
    docs = loader.load_directory()
    
    print(f"Loaded {len(docs)} document objects. Chunking documents...")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.split_documents(docs)

    print(f"Generating deterministic IDs and stamping metadata for {len(chunks)} chunks...")
    doc_ids = []
    unmatched_files = set()

    for idx, chunk in enumerate(chunks):
        chunk.metadata["tenant_id"] = args.tenant_id
        source_name = Path(chunk.metadata.get("source", "")).name.lower()
        file_name = Path(chunk.metadata.get("file_name", "")).name.lower()

        # Check manifest match
        manifest_entry = manifest_map.get(file_name) or manifest_map.get(source_name)
        if manifest_entry:
            chunk.metadata["sf_account_id"] = manifest_entry["sf_account_id"]
            chunk.metadata["allowed_roles"] = manifest_entry["allowed_roles"]
            chunk.metadata["sf_scope"] = manifest_entry["sf_scope"]
        else:
            # Fail closed: documents without manifest entry are invisible to SF auth
            chunk.metadata["sf_account_id"] = None
            chunk.metadata["allowed_roles"] = []
            chunk.metadata["sf_scope"] = "none"
            raw_source = chunk.metadata.get("file_name") or chunk.metadata.get("source") or "unknown"
            unmatched_files.add(raw_source)

        cid = generate_chunk_id(
            chunk,
            idx,
            tenant_id=args.tenant_id,
            sf_account_id=chunk.metadata["sf_account_id"],
        )
        chunk.id = cid
        chunk.metadata["chunk_id"] = cid
        if "title" not in chunk.metadata:
            candidate_title = chunk.metadata.get("file_name") or Path(chunk.metadata.get("source", "")).name
            if candidate_title:
                chunk.metadata["title"] = candidate_title
        doc_ids.append(cid)

    if unmatched_files and args.sf_manifest:
        print(
            f"[WARNING] {len(unmatched_files)} files without manifest entries (marked sf_scope='none', invisible to SF requests):"
        )
        for uf in sorted(unmatched_files):
            print(f"  - {uf}")
    
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
    bm25_store = BM25Store() if args.recreate or not bm25_path.exists() else BM25Store.load(bm25_path)
    bm25_store.add_documents(chunks)
    bm25_store.save(bm25_path)

    print(f"Ingestion complete. Qdrant & BM25 synchronized at {bm25_path} (tenant='{args.tenant_id}').")


if __name__ == "__main__":
    main()
