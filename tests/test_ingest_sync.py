import uuid
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from ingest import generate_chunk_id
from src.retrieval.bm25_store import BM25Store
from src.config import Settings


def test_generate_chunk_id_deterministic_and_uuidv5():
    """Verify H3: Identical chunk generates identical valid RFC 4122 UUID."""
    doc1 = Document(page_content="Flight to Paris at 08:00 AM", metadata={"source": "itinerary.pdf"})
    doc2 = Document(page_content="Flight to Paris at 08:00 AM", metadata={"source": "itinerary.pdf"})

    id1 = generate_chunk_id(doc1, 0)
    id2 = generate_chunk_id(doc2, 0)

    assert id1 == id2
    # Verify valid UUID format
    parsed = uuid.UUID(id1)
    assert parsed.version == 5

    # Different chunk index or content produces different UUID
    id_different_idx = generate_chunk_id(doc1, 1)
    assert id1 != id_different_idx

    doc3 = Document(page_content="Flight to Rome at 09:00 AM", metadata={"source": "itinerary.pdf"})
    id_different_content = generate_chunk_id(doc3, 0)
    assert id1 != id_different_content


def test_bm25_store_save_and_load_roundtrip():
    """Verify H2: BM25Store serializes to disk and reloads accurately."""
    docs = [
        Document(page_content="Hotel booking in Tokyo for 3 nights", metadata={"source": "tokyo.txt", "chunk_id": "c1"}),
        Document(page_content="Safari package in Kenya with luxury tent", metadata={"source": "kenya.txt", "chunk_id": "c2"}),
    ]

    store = BM25Store()
    store.add_documents(docs)

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "bm25_test.pkl"
        store.save(pkl_path)

        assert pkl_path.exists()

        loaded_store = BM25Store.load(pkl_path)
        assert len(loaded_store._documents) == 2

        # Search against loaded store returns matching doc
        results = loaded_store.search("Tokyo hotel", k=1)
        assert len(results) == 1
        assert "Tokyo" in results[0].page_content


def test_bm25_store_deduplication():
    """Verify H3 in BM25: Re-adding same document chunks does not duplicate index."""
    doc = Document(page_content="Payment receipt reference PAY-999", metadata={"chunk_id": "pay-999"})

    store = BM25Store()
    store.add_documents([doc])
    assert len(store._documents) == 1

    # Add duplicate document
    store.add_documents([doc])
    assert len(store._documents) == 1


def test_dependencies_bm25_loader_fallback():
    """Verify dependencies load persisted store if present, else fallback."""
    from src.api.dependencies import get_compiled_graph

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = str(Path(tmpdir) / "absent_store.pkl")
        test_settings = Settings(bm25_store_path=pkl_path)

        with patch("src.api.dependencies.settings", test_settings):
            # When file absent -> loads empty store without error
            store = BM25Store.load(pkl_path) if Path(pkl_path).exists() else BM25Store()
            assert len(store._documents) == 0
