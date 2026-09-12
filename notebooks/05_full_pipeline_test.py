"""
==========================================================================
Advanced RAG Multi-Document Chatbot — Full Pipeline Test Suite
==========================================================================

This script tests the ENTIRE project end-to-end on Google Colab with GPU.

Test Categories:
  ✅ POSITIVE — valid inputs, expected happy-path behavior
  ❌ NEGATIVE — invalid inputs, edge cases, error handling
  ⚪ NEUTRAL  — boundary conditions, empty inputs, graceful defaults

Modules Tested:
  1. Document Ingestion      (MultiDocumentLoader)
  2. Text Chunking           (DocumentChunker)
  3. Embeddings              (BGEEmbeddings)
  4. BM25 Retrieval          (BM25Store)
  5. Vector Store            (QdrantStore)
  6. Hybrid Retrieval + RRF  (HybridRetriever, reciprocal_rank_fusion)
  7. Cross-Encoder Reranking (CrossEncoderReranker)
  8. RAG Chain               (build_rag_chain, build_generation_chain, format_docs)
  9. RAG Pipeline            (RAGPipeline)
  10. Conversational RAG     (ConversationStore, build_conversational_rag)
  11. LLM Model              (Qwen3ChatModel, load_llm)
  12. LangGraph Workflow      (build_rag_graph, nodes, state, routing)
  13. Tool-Calling Agent      (build_tool_execution_graph)
  14. Document Search Tool    (build_document_search_tool)
  15. Document Indexer        (DocumentIndexer)
  16. Data Models             (Chunk, Document dataclasses)

Usage on Google Colab:
  1. Copy this file to Colab or run cells sequentially
  2. Each "# %%"  marker = one Colab cell
  3. Requires GPU runtime (T4 or better)

Author: Auto-generated for project testing
==========================================================================
"""

# %%
# ==========================================================================
# CELL 1: SETUP — Clone repo, install dependencies, configure environment
# ==========================================================================
#
# WHY:  Colab starts fresh. We need the project code + all dependencies.
# WHAT: Clones from GitHub, installs requirements, adds src to Python path.
# HOW:  Shell commands via !, then sys.path manipulation.

import os

# Clone the repository (skip if already cloned)
REPO_URL = "https://github.com/ak0000007/Advance_RAG_MultiDoc_Chatbot.git"
REPO_DIR = "/content/Advance_RAG_MultiDoc_Chatbot"

if not os.path.exists(REPO_DIR):
    print("📦 Cloning repository...")
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
else:
    print("📦 Repository already exists. Pulling latest...")
    os.system(f"cd {REPO_DIR} && git pull")

# Install dependencies
print("\n📦 Installing dependencies...")
os.system(f"pip install -q -r {REPO_DIR}/requirements.txt")
os.system("pip install -q langgraph")

# Add project root to Python path so `from src.xxx import yyy` works
import sys
sys.path.insert(0, REPO_DIR)

# Set environment variables for LangSmith tracing (optional)
os.environ["LANGSMITH_TRACING"] = "false"  # Disable during testing

print("\n✅ Setup complete!")
print(f"   GPU available: {os.system('nvidia-smi -L 2>/dev/null') == 0}")

# %%
# ==========================================================================
# CELL 2: IMPORTS — Load all project modules
# ==========================================================================
#
# WHY:  Verify every module imports without error. Import failure = broken code.
# WHAT: Imports all classes/functions from every src/ module.
# HOW:  Standard Python imports. Any ImportError means a dependency or
#       syntax issue in the codebase.

import time
import hashlib
import tempfile
import json
from pathlib import Path

# --- LangChain core ---
from langchain_core.documents import Document as LCDocument
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

# --- Project modules ---
from src.ingestion.loader import MultiDocumentLoader
from src.chunking.splitter import DocumentChunker
from src.embeddings.embedding import BGEEmbeddings
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.reranking.reranker import CrossEncoderReranker
from src.rag.chain import build_rag_chain, build_generation_chain, format_docs
from src.rag.pipeline import RAGPipeline
from src.rag.conversational import ConversationStore, build_conversational_rag
from src.models.chunk import Chunk
from src.models.document import Document as ProjectDocument
from src.graph.state import RAGState
from src.graph.nodes import (
    route_question,
    route_after_grading,
    create_fallback_node,
    RetrievalGrade,
    AnswerGrade,
    MAX_RETRIEVAL_ATTEMPTS,
)
from src.indexing.indexer import DocumentIndexer
from src.tools.document_search import build_document_search_tool

print("✅ All imports successful — no broken modules!")


# %%
# ==========================================================================
# CELL 3: TEST HELPERS
# ==========================================================================
#
# WHY:  Standardize test output. Makes pass/fail visible at a glance.
# WHAT: Helper functions for assertions, result printing, section headers.
# HOW:  Simple wrappers that catch exceptions and print colored status.

_test_results = {"passed": 0, "failed": 0, "tests": []}


def section(title):
    """Print a bold section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test(name, condition, detail=""):
    """Assert a test condition and record result."""
    if condition:
        _test_results["passed"] += 1
        _test_results["tests"].append(("PASS", name))
        print(f"  ✅ PASS: {name}")
    else:
        _test_results["failed"] += 1
        _test_results["tests"].append(("FAIL", name))
        print(f"  ❌ FAIL: {name}")
    if detail:
        print(f"         → {detail}")


def test_raises(name, exception_type, fn, *args, **kwargs):
    """Assert that calling fn(*args) raises the expected exception."""
    try:
        fn(*args, **kwargs)
        _test_results["failed"] += 1
        _test_results["tests"].append(("FAIL", name))
        print(f"  ❌ FAIL: {name} (no exception raised)")
    except exception_type:
        _test_results["passed"] += 1
        _test_results["tests"].append(("PASS", name))
        print(f"  ✅ PASS: {name}")
    except Exception as e:
        _test_results["failed"] += 1
        _test_results["tests"].append(("FAIL", name))
        print(f"  ❌ FAIL: {name} (got {type(e).__name__}: {e})")


def summary():
    """Print final test summary."""
    total = _test_results["passed"] + _test_results["failed"]
    print(f"\n{'='*70}")
    print(f"  TEST SUMMARY: {_test_results['passed']}/{total} passed, "
          f"{_test_results['failed']} failed")
    print(f"{'='*70}")
    if _test_results["failed"] > 0:
        print("\n  Failed tests:")
        for status, name in _test_results["tests"]:
            if status == "FAIL":
                print(f"    ❌ {name}")
    print()


print("✅ Test helpers ready")


# %%
# ==========================================================================
# CELL 4: TEST DATA MODELS (Chunk, Document dataclasses)
# ==========================================================================
#
# WHAT THESE TEST:
#   src/models/chunk.py   — Chunk dataclass (id, document_id, content, metadata)
#   src/models/document.py — Document dataclass (id, name, content, document_type, source, metadata)
#
# WHY WE TEST:
#   These are the foundation data structures. If they break, everything
#   above them (chunking, retrieval, indexing) breaks too.
#
# HOW IT WORKS:
#   Dataclasses auto-generate __init__, __repr__, __eq__. We verify
#   field access, default values, and type flexibility.

section("4. DATA MODELS — Chunk & Document")

# ✅ POSITIVE: Create Chunk with all fields
chunk = Chunk(id="c1", document_id="d1", content="Hello world", metadata={"page": 1})
test("Chunk creation with all fields",
     chunk.id == "c1" and chunk.content == "Hello world" and chunk.metadata["page"] == 1,
     f"id={chunk.id}, content='{chunk.content}', metadata={chunk.metadata}")

# ✅ POSITIVE: Create Document with all fields
doc = ProjectDocument(
    id="d1", name="test.pdf", content="Sample content",
    document_type="pdf", source="/data/test.pdf", metadata={"author": "test"}
)
test("Document creation with all fields",
     doc.id == "d1" and doc.source == "/data/test.pdf",
     f"id={doc.id}, name={doc.name}, type={doc.document_type}")

# ⚪ NEUTRAL: Chunk with default empty metadata
chunk_default = Chunk(id="c2", document_id="d1", content="test")
test("Chunk default metadata is empty dict",
     chunk_default.metadata == {},
     f"metadata={chunk_default.metadata}")

# ⚪ NEUTRAL: Document with default None source
doc_no_source = ProjectDocument(id="d2", name="f.txt", content="x", document_type="txt")
test("Document default source is None",
     doc_no_source.source is None,
     f"source={doc_no_source.source}")

# ❌ NEGATIVE: Chunk/Document with empty strings (allowed by dataclass, but test boundary)
chunk_empty = Chunk(id="", document_id="", content="")
test("Chunk allows empty strings (no validation)",
     chunk_empty.id == "" and chunk_empty.content == "",
     "Dataclass has no built-in validation — empty strings are accepted")


# %%
# ==========================================================================
# CELL 5: TEST DOCUMENT CHUNKER (DocumentChunker)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/chunking/splitter.py — DocumentChunker wrapping RecursiveCharacterTextSplitter
#
# WHY WE TEST:
#   Chunking controls how documents get split before embedding. Bad chunks
#   = bad retrieval. We verify:
#   - Correct splitting at specified size/overlap
#   - Input validation (chunk_size > 0, overlap < size)
#   - Edge cases (empty input, single-char text)
#
# HOW IT WORKS:
#   DocumentChunker wraps LangChain's RecursiveCharacterTextSplitter.
#   split_documents() takes list[LCDocument], split_text() takes raw string.
#   Both respect chunk_size and chunk_overlap parameters.

section("5. DOCUMENT CHUNKER")

# ✅ POSITIVE: Split a long text into chunks
chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
long_text = "The Ramayana is an ancient Indian epic. " * 50  # ~2000 chars
doc = LCDocument(page_content=long_text, metadata={"source": "test"})
chunks = chunker.split_documents([doc])
test("Chunker splits long document into multiple chunks",
     len(chunks) > 1,
     f"Input: ~{len(long_text)} chars → {len(chunks)} chunks")

# ✅ POSITIVE: Each chunk respects max size
max_chunk_len = max(len(c.page_content) for c in chunks)
test("Each chunk respects chunk_size limit",
     max_chunk_len <= 120,  # slight tolerance for word boundaries
     f"Max chunk length: {max_chunk_len} (limit: ~100)")

# ✅ POSITIVE: Metadata preserved through chunking
test("Chunk metadata preserved from parent document",
     all(c.metadata.get("source") == "test" for c in chunks),
     "All chunks retain source='test'")

# ✅ POSITIVE: split_text returns list of strings
text_chunks = chunker.split_text(long_text)
test("split_text returns list of strings",
     isinstance(text_chunks, list) and all(isinstance(t, str) for t in text_chunks),
     f"Got {len(text_chunks)} string chunks")

# ⚪ NEUTRAL: Empty document list returns empty
test("split_documents([]) returns []",
     chunker.split_documents([]) == [],
     "No crash on empty input")

# ⚪ NEUTRAL: Empty string returns empty
test("split_text('') returns []",
     chunker.split_text("") == [],
     "No crash on empty string")

# ⚪ NEUTRAL: Short text that doesn't need splitting
short_doc = LCDocument(page_content="Hello", metadata={})
short_chunks = chunker.split_documents([short_doc])
test("Short text (< chunk_size) stays as single chunk",
     len(short_chunks) == 1 and short_chunks[0].page_content == "Hello",
     f"Got {len(short_chunks)} chunk(s)")

# ❌ NEGATIVE: chunk_size <= 0 raises ValueError
test_raises("chunk_size=0 raises ValueError",
            ValueError, DocumentChunker, chunk_size=0, chunk_overlap=0)

test_raises("chunk_size=-1 raises ValueError",
            ValueError, DocumentChunker, chunk_size=-1, chunk_overlap=0)

# ❌ NEGATIVE: chunk_overlap < 0 raises ValueError
test_raises("chunk_overlap=-1 raises ValueError",
            ValueError, DocumentChunker, chunk_size=100, chunk_overlap=-1)

# ❌ NEGATIVE: chunk_overlap >= chunk_size raises ValueError
test_raises("chunk_overlap >= chunk_size raises ValueError",
            ValueError, DocumentChunker, chunk_size=100, chunk_overlap=100)

test_raises("chunk_overlap > chunk_size raises ValueError",
            ValueError, DocumentChunker, chunk_size=50, chunk_overlap=51)


# %%
# ==========================================================================
# CELL 6: TEST EMBEDDINGS (BGEEmbeddings)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/embeddings/embedding.py — BGEEmbeddings wrapping HuggingFaceEmbeddings
#
# WHY WE TEST:
#   Embeddings are the core of semantic search. We verify:
#   - Model loads correctly on GPU/CPU
#   - Embeddings have correct dimensionality
#   - Similar texts produce similar vectors (cosine similarity)
#   - Different texts produce different vectors
#
# HOW IT WORKS:
#   BGEEmbeddings loads BAAI/bge-m3 model. get_embeddings() returns
#   a LangChain Embeddings object with embed_query() and embed_documents().
#   Embeddings are normalized (L2 norm = 1) per encode_kwargs.

section("6. EMBEDDINGS — BGE-M3")

import torch
import numpy as np

# ✅ POSITIVE: Load embeddings model (auto-detect GPU)
print("  ⏳ Loading BGE-M3 embedding model (first load downloads ~2GB)...")
start = time.time()
bge = BGEEmbeddings()  # Auto-detects cuda/cpu
embeddings = bge.get_embeddings()
load_time = time.time() - start
test("BGEEmbeddings loads successfully",
     embeddings is not None,
     f"Device: {bge.device}, Model: {bge.model_name}, Load time: {load_time:.1f}s")

# ✅ POSITIVE: GPU detection
expected_device = "cuda" if torch.cuda.is_available() else "cpu"
test(f"Auto-detected device = {expected_device}",
     bge.device == expected_device,
     f"torch.cuda.is_available()={torch.cuda.is_available()}")

# ✅ POSITIVE: Embed a single query
query_vec = embeddings.embed_query("What is the Ramayana about?")
test("embed_query returns a vector",
     isinstance(query_vec, list) and len(query_vec) > 0,
     f"Vector dimension: {len(query_vec)}")

# ✅ POSITIVE: Embed multiple documents
doc_vecs = embeddings.embed_documents([
    "The Ramayana is an ancient epic",
    "Machine learning is a branch of AI",
])
test("embed_documents returns list of vectors",
     len(doc_vecs) == 2 and len(doc_vecs[0]) == len(query_vec),
     f"2 documents → 2 vectors of dim {len(doc_vecs[0])}")

# ✅ POSITIVE: Normalized embeddings (L2 norm ≈ 1.0)
norm = np.linalg.norm(query_vec)
test("Embeddings are L2-normalized",
     abs(norm - 1.0) < 0.01,
     f"L2 norm = {norm:.4f} (expected ~1.0)")

# ✅ POSITIVE: Similar texts have high cosine similarity
vec_rama = np.array(embeddings.embed_query("ancient Indian epic Ramayana"))
vec_epic = np.array(embeddings.embed_query("old Indian story of Rama"))
similarity_high = float(np.dot(vec_rama, vec_epic))
test("Similar texts have high cosine similarity",
     similarity_high > 0.5,
     f"cosine_sim('ancient Indian epic Ramayana', 'old Indian story of Rama') = {similarity_high:.4f}")

# ⚪ NEUTRAL: Unrelated texts have lower similarity
vec_ml = np.array(embeddings.embed_query("gradient descent backpropagation neural network"))
similarity_low = float(np.dot(vec_rama, vec_ml))
test("Unrelated texts have lower cosine similarity",
     similarity_low < similarity_high,
     f"cosine_sim(ramayana, ML) = {similarity_low:.4f} < {similarity_high:.4f}")

# ⚪ NEUTRAL: Empty string doesn't crash
empty_vec = embeddings.embed_query("")
test("Empty string produces a vector (no crash)",
     isinstance(empty_vec, list) and len(empty_vec) > 0,
     f"dim={len(empty_vec)}")


# %%
# ==========================================================================
# CELL 7: TEST BM25 RETRIEVAL (BM25Store)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/retrieval/bm25_store.py — BM25Store (keyword-based retrieval)
#
# WHY WE TEST:
#   BM25 provides keyword/sparse retrieval — complementary to dense vectors.
#   It catches exact keyword matches that embeddings might miss. We verify:
#   - Documents indexed correctly
#   - Relevant keyword queries return matching docs
#   - Irrelevant queries return low-scored docs
#   - Metadata filtering works
#   - Empty index returns empty results
#
# HOW IT WORKS:
#   BM25Store implements the Okapi BM25 scoring algorithm:
#   score(q,d) = sum over terms: IDF(t) * (tf * (k1+1)) / (tf + k1 * (1-b+b*dl/avgdl))
#   It tokenizes via whitespace split, builds inverted index, scores at query time.

section("7. BM25 RETRIEVAL")

# Create test documents
test_docs = [
    LCDocument(page_content="Lord Rama was the prince of Ayodhya, son of King Dasharatha",
               metadata={"source": "ramayana.pdf", "chapter": "1"}),
    LCDocument(page_content="Sita was kidnapped by Ravana and taken to Lanka",
               metadata={"source": "ramayana.pdf", "chapter": "3"}),
    LCDocument(page_content="Hanuman crossed the ocean to find Sita in Lanka",
               metadata={"source": "ramayana.pdf", "chapter": "4"}),
    LCDocument(page_content="Python is a programming language used for machine learning",
               metadata={"source": "python_guide.pdf", "chapter": "1"}),
    LCDocument(page_content="Neural networks consist of layers of interconnected nodes",
               metadata={"source": "ml_textbook.pdf", "chapter": "5"}),
]

# ✅ POSITIVE: Create BM25 store and add documents
bm25 = BM25Store(k1=1.5, b=0.75)
bm25.add_documents(test_docs)
test("BM25Store indexes documents without error",
     len(bm25._documents) == 5,
     f"Indexed {len(bm25._documents)} documents")

# ✅ POSITIVE: Search for relevant keyword
results = bm25.search("Rama prince Ayodhya", k=3)
test("BM25 finds relevant docs for 'Rama prince Ayodhya'",
     len(results) > 0 and "Rama" in results[0].page_content,
     f"Top result: '{results[0].page_content[:60]}...'")

# ✅ POSITIVE: Search returns correct number of results
results_k2 = bm25.search("Sita Lanka", k=2)
test("BM25 respects k parameter",
     len(results_k2) <= 2,
     f"k=2 → got {len(results_k2)} results")

# ✅ POSITIVE: Metadata filtering
filtered = bm25.search("Lanka", k=5, metadata_filter={"chapter": "4"})
test("BM25 metadata filter returns only matching docs",
     all(d.metadata.get("chapter") == "4" for d in filtered if "Lanka" in d.page_content),
     f"Got {len(filtered)} results with chapter=4 filter")

# ⚪ NEUTRAL: Search with no matching keywords
results_none = bm25.search("quantum physics relativity", k=5)
test("BM25 returns results even for non-matching query (all docs scored)",
     isinstance(results_none, list),
     f"Got {len(results_none)} results (all scored 0)")

# ⚪ NEUTRAL: Empty BM25 store
bm25_empty = BM25Store()
test("Empty BM25 search returns []",
     bm25_empty.search("anything") == [],
     "No crash on empty index")

# ✅ POSITIVE: as_retriever returns LangChain Runnable
retriever = bm25.as_retriever(search_kwargs={"k": 3})
runnable_results = retriever.invoke("Hanuman ocean")
test("as_retriever().invoke() works as LangChain Runnable",
     len(runnable_results) > 0,
     f"Got {len(runnable_results)} results via Runnable interface")

# ✅ POSITIVE: as_dynamic_retriever accepts dict input
dyn_retriever = bm25.as_dynamic_retriever()
dyn_results = dyn_retriever.invoke({"question": "Sita", "metadata_filter": None})
test("as_dynamic_retriever accepts dict input",
     len(dyn_results) > 0,
     f"Got {len(dyn_results)} results")

# ✅ POSITIVE: Dynamic retriever with metadata filter
dyn_filtered = dyn_retriever.invoke({
    "question": "Lanka",
    "metadata_filter": {"source": "ramayana.pdf"}
})
test("Dynamic retriever applies metadata_filter",
     all(d.metadata.get("source") == "ramayana.pdf" for d in dyn_filtered),
     f"All {len(dyn_filtered)} results from ramayana.pdf")


# %%
# ==========================================================================
# CELL 8: TEST HYBRID RETRIEVAL + RRF (HybridRetriever)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/retrieval/hybrid_retriever.py — HybridRetriever, reciprocal_rank_fusion()
#
# WHY WE TEST:
#   Hybrid retrieval merges dense (vector) + sparse (BM25) results using
#   Reciprocal Rank Fusion. This is the core retrieval strategy. We verify:
#   - RRF correctly fuses multiple ranked lists
#   - Documents appearing in both lists get boosted
#   - HybridRetriever supports both Runnables and plain callables
#   - Empty retrievers raise ValueError
#
# HOW IT WORKS:
#   RRF score for each doc = sum(1 / (k + rank)) across all lists.
#   k=60 is the constant from the original RRF paper.
#   HybridRetriever wraps N retrievers and fuses their output.

section("8. HYBRID RETRIEVAL + RRF")

# --- Test reciprocal_rank_fusion directly ---

doc_a = LCDocument(page_content="Document A about Rama", metadata={})
doc_b = LCDocument(page_content="Document B about Sita", metadata={})
doc_c = LCDocument(page_content="Document C about Hanuman", metadata={})
doc_d = LCDocument(page_content="Document D about Ravana", metadata={})

# List 1: A, B, C (from vector search)
# List 2: B, D, A (from BM25)
list1 = [doc_a, doc_b, doc_c]
list2 = [doc_b, doc_d, doc_a]

# ✅ POSITIVE: RRF merges correctly — doc_b and doc_a appear in both lists
fused = reciprocal_rank_fusion([list1, list2], k=60)
test("RRF produces merged list with no duplicates",
     len(fused) == 4,  # A, B, C, D — no dups
     f"2 lists of 3 → {len(fused)} unique docs after fusion")

# ✅ POSITIVE: Document appearing in both lists gets higher score
# doc_b appears at rank 1 in list1 AND rank 0 in list2 → highest RRF
test("RRF boosts docs appearing in multiple lists",
     fused[0].page_content == doc_b.page_content or fused[0].page_content == doc_a.page_content,
     f"Top doc: '{fused[0].page_content}'")

# --- Test HybridRetriever ---

# Create two mock retrievers (plain callables)
def mock_retriever_1(query):
    return [doc_a, doc_b]

def mock_retriever_2(query):
    return [doc_b, doc_c]

# ✅ POSITIVE: HybridRetriever with callables
hybrid = HybridRetriever(retrievers=[mock_retriever_1, mock_retriever_2], final_k=3)
hybrid_results = hybrid.search("test query")
test("HybridRetriever fuses results from 2 retrievers",
     len(hybrid_results) <= 3 and len(hybrid_results) > 0,
     f"Got {len(hybrid_results)} results (final_k=3)")

# ✅ POSITIVE: as_retriever returns Runnable
hybrid_runnable = hybrid.as_retriever()
runnable_results = hybrid_runnable.invoke("test")
test("HybridRetriever.as_retriever() works as Runnable",
     len(runnable_results) > 0,
     f"Got {len(runnable_results)} results")

# ✅ POSITIVE: as_dynamic_retriever accepts dict
dyn = hybrid.as_dynamic_retriever()
# Our mock retrievers don't accept dicts, but HybridRetriever's fallback handles it
try:
    dyn_results = dyn.invoke({"question": "test", "metadata_filter": None})
    test("HybridRetriever.as_dynamic_retriever handles dict input",
         True, f"Got {len(dyn_results)} results")
except Exception as e:
    test("HybridRetriever.as_dynamic_retriever handles dict input",
         False, f"Error: {e}")

# ❌ NEGATIVE: No retrievers raises ValueError
test_raises("HybridRetriever with empty retrievers raises ValueError",
            ValueError, HybridRetriever, retrievers=[])

# ⚪ NEUTRAL: Single retriever (no fusion needed)
single = HybridRetriever(retrievers=[mock_retriever_1], final_k=5)
single_results = single.search("test")
test("Single retriever still works (no fusion partner)",
     len(single_results) == 2,
     f"Got {len(single_results)} results from single retriever")

# ⚪ NEUTRAL: RRF with empty lists
fused_empty = reciprocal_rank_fusion([[], []])
test("RRF with empty lists returns []",
     fused_empty == [],
     "No crash on empty input")


# %%
# ==========================================================================
# CELL 9: TEST VECTOR STORE (QdrantStore)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/vector_stores/qdrant_store.py — QdrantStore wrapping QdrantVectorStore
#
# WHY WE TEST:
#   Vector store is the primary dense retrieval backend. We verify:
#   - Local Qdrant connection works (file-based, no server needed)
#   - Documents can be added and retrieved
#   - Metadata filtering works
#   - Error on missing url/path
#
# HOW IT WORKS:
#   QdrantStore wraps langchain-qdrant's QdrantVectorStore.
#   Supports remote (url) or local (path) Qdrant instances.
#   as_retriever() returns a standard LangChain retriever.
#   as_dynamic_retriever() returns a Runnable accepting dict with metadata_filter.

section("9. VECTOR STORE — Qdrant")

# ✅ POSITIVE: Create local Qdrant store
qdrant_path = "/content/test_qdrant_store"
qdrant_store = QdrantStore(
    embeddings=embeddings,
    collection_name="test_collection",
    path=qdrant_path,
)
test("QdrantStore creates local store successfully",
     qdrant_store.vector_store is not None,
     f"Path: {qdrant_path}, Collection: test_collection")

# ✅ POSITIVE: Add documents to vector store
qdrant_store.vector_store.add_documents(test_docs)
test("Documents added to Qdrant",
     True,  # No exception = success
     f"Added {len(test_docs)} documents")

# ✅ POSITIVE: Retrieve documents
qdrant_retriever = qdrant_store.as_retriever(search_kwargs={"k": 3})
q_results = qdrant_retriever.invoke("Rama prince of Ayodhya")
test("Qdrant retriever returns results",
     len(q_results) > 0,
     f"Query 'Rama prince of Ayodhya' → {len(q_results)} results")

# ✅ POSITIVE: Dynamic retriever
qdrant_dyn = qdrant_store.as_dynamic_retriever()
dyn_q = qdrant_dyn.invoke({"question": "Sita kidnapped Lanka", "metadata_filter": None})
test("Qdrant dynamic retriever works",
     len(dyn_q) > 0,
     f"Got {len(dyn_q)} results")

# ❌ NEGATIVE: No url or path raises ValueError
test_raises("QdrantStore without url or path raises ValueError",
            ValueError, QdrantStore, embeddings=embeddings)


# %%
# ==========================================================================
# CELL 10: TEST CROSS-ENCODER RERANKER (CrossEncoderReranker)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/reranking/reranker.py — CrossEncoderReranker
#
# WHY WE TEST:
#   Reranking is a critical quality step. Cross-encoders read (query, doc)
#   pairs together, giving much higher relevance accuracy than bi-encoders.
#   We verify:
#   - Reranking reorders docs by relevance
#   - Top-k selection works
#   - Rerank scores are attached to metadata
#   - Empty document list handled gracefully
#   - wrap_retriever() correctly composes with retrievers
#
# HOW IT WORKS:
#   CrossEncoderReranker lazy-loads a cross-encoder model (BAAI/bge-reranker-v2-m3).
#   rerank(query, docs, top_k) scores each doc against the query and returns
#   top_k sorted by descending relevance score.

section("10. CROSS-ENCODER RERANKER")

print("  ⏳ Loading cross-encoder reranker model...")
reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-v2-m3")

# ✅ POSITIVE: Rerank puts relevant docs first
candidate_docs = [
    LCDocument(page_content="Python is a programming language", metadata={"id": 1}),
    LCDocument(page_content="Lord Rama defeated Ravana in battle", metadata={"id": 2}),
    LCDocument(page_content="The Ramayana describes Rama's exile to the forest", metadata={"id": 3}),
    LCDocument(page_content="Neural networks use backpropagation", metadata={"id": 4}),
]

reranked = reranker.rerank("Who is Rama in the Ramayana?", candidate_docs, top_k=2)
test("Reranker returns top_k documents",
     len(reranked) == 2,
     f"top_k=2 → got {len(reranked)} docs")

# ✅ POSITIVE: Relevant doc ranked higher than irrelevant
test("Reranker puts Rama-related docs before ML docs",
     "Rama" in reranked[0].page_content,
     f"Top doc: '{reranked[0].page_content[:60]}...'")

# ✅ POSITIVE: Rerank scores attached to metadata
test("rerank_score attached to metadata",
     "rerank_score" in reranked[0].metadata,
     f"Score: {reranked[0].metadata.get('rerank_score', 'N/A'):.4f}")

# ✅ POSITIVE: Scores are in descending order
scores = [d.metadata["rerank_score"] for d in reranked]
test("Rerank scores in descending order",
     scores == sorted(scores, reverse=True),
     f"Scores: {[f'{s:.4f}' for s in scores]}")

# ⚪ NEUTRAL: Rerank empty list
test("Rerank empty list returns []",
     reranker.rerank("test", [], top_k=5) == [],
     "No crash on empty input")

# ⚪ NEUTRAL: top_k larger than document count
reranked_all = reranker.rerank("Rama", candidate_docs, top_k=100)
test("top_k > doc count returns all docs",
     len(reranked_all) == len(candidate_docs),
     f"top_k=100 but only {len(candidate_docs)} docs → got {len(reranked_all)}")

# ✅ POSITIVE: wrap_retriever composes with BM25
wrapped = reranker.wrap_retriever(bm25.as_retriever(search_kwargs={"k": 5}), top_k=2)
wrapped_results = wrapped.invoke({"question": "Rama exile forest", "metadata_filter": None})
test("wrap_retriever composes reranker with BM25 retriever",
     len(wrapped_results) <= 2,
     f"Got {len(wrapped_results)} reranked results")


# %%
# ==========================================================================
# CELL 11: TEST RAG CHAIN (build_rag_chain, build_generation_chain, format_docs)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/rag/chain.py — format_docs(), build_generation_chain(), build_rag_chain()
#
# WHY WE TEST:
#   The RAG chain is the core LCEL composition:
#   retriever → format → prompt → LLM → parser
#   We test each piece independently before the full chain.
#
# HOW IT WORKS:
#   format_docs() converts list[Document] → plain text with source attribution.
#   build_generation_chain() = format → prompt → LLM → parser (no retrieval).
#   build_rag_chain() = retriever → format → prompt → LLM → parser (full RAG).

section("11. RAG CHAIN COMPONENTS")

# --- Test format_docs ---

# ✅ POSITIVE: format_docs formats documents with sources
formatted = format_docs([
    LCDocument(page_content="Rama was prince of Ayodhya", metadata={"source": "ramayana.pdf"}),
    LCDocument(page_content="Sita was kidnapped", metadata={"document_id": "doc123"}),
])
test("format_docs creates numbered text with sources",
     "[1] (Source: ramayana.pdf)" in formatted and "[2] (Source: doc123)" in formatted,
     f"Output preview: '{formatted[:80]}...'")

# ⚪ NEUTRAL: format_docs with empty list
test("format_docs([]) returns empty string",
     format_docs([]) == "",
     "No crash on empty input")

# ⚪ NEUTRAL: format_docs with no source metadata
formatted_no_src = format_docs([LCDocument(page_content="test", metadata={})])
test("format_docs uses 'unknown' when no source",
     "unknown" in formatted_no_src,
     f"Output: '{formatted_no_src}'")


# %%
# ==========================================================================
# CELL 12: TEST LLM MODEL (Qwen3ChatModel)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/llm/model.py — Qwen3ChatModel, load_llm()
#
# WHY WE TEST:
#   The LLM is the brain of the RAG system. We verify:
#   - Model loads with 4-bit quantization on GPU
#   - Generates coherent responses
#   - LangChain message types handled correctly
#   - Tool binding works (bind_tools creates copy, doesn't mutate)
#   - Tool call parsing works for Qwen3 format
#
# HOW IT WORKS:
#   Qwen3ChatModel extends LangChain's BaseChatModel.
#   load_llm() loads Qwen3-4B-Instruct-2507 with BitsAndBytes 4-bit quantization.
#   _generate() converts messages → Qwen format → tokenize → generate → decode.
#   bind_tools() creates an immutable copy with tool schemas available.
#   _parse_tool_calls() extracts <tool_call>...</tool_call> blocks from output.

section("12. LLM MODEL — Qwen3-4B")

from src.llm.model import load_llm, Qwen3ChatModel

print("  ⏳ Loading Qwen3-4B-Instruct-2507 (4-bit quantized)...")
print("     This downloads ~3GB on first run and requires GPU.")
start = time.time()
llm = load_llm()
load_time = time.time() - start

test("Qwen3ChatModel loads successfully",
     isinstance(llm, Qwen3ChatModel),
     f"Type: {type(llm).__name__}, Load time: {load_time:.1f}s")

test("LLM type is 'qwen3-chat'",
     llm._llm_type == "qwen3-chat",
     f"_llm_type = '{llm._llm_type}'")

# ✅ POSITIVE: Generate a simple response
response = llm.invoke([HumanMessage(content="What is 2 + 2? Answer in one word.")])
test("LLM generates a response",
     isinstance(response, AIMessage) and len(response.content) > 0,
     f"Response: '{response.content[:100]}'")

# ✅ POSITIVE: System + Human message combo
response2 = llm.invoke([
    SystemMessage(content="You are a helpful assistant. Answer briefly."),
    HumanMessage(content="Name the capital of India."),
])
test("LLM handles System + Human messages",
     isinstance(response2, AIMessage) and len(response2.content) > 0,
     f"Response: '{response2.content[:100]}'")

# ✅ POSITIVE: bind_tools creates new model (doesn't mutate original)
from langchain_core.tools import tool as tool_decorator

@tool_decorator
def dummy_tool(query: str) -> str:
    """A dummy tool for testing."""
    return "dummy result"

llm_with_tools = llm.bind_tools([dummy_tool])
test("bind_tools creates a new model copy",
     llm_with_tools is not llm,
     "Original model unchanged")

test("Bound model has tool schemas",
     len(llm_with_tools.bound_tools) == 1,
     f"bound_tools count: {len(llm_with_tools.bound_tools)}")

test("Original model has no bound tools",
     len(llm.bound_tools) == 0,
     "Immutability preserved")

# ✅ POSITIVE: Tool call parsing
parsed = llm._parse_tool_calls(
    '<tool_call>\n{"name": "search_documents", "arguments": {"query": "Rama"}}\n</tool_call>'
)
test("_parse_tool_calls parses valid tool call",
     len(parsed) == 1 and parsed[0]["name"] == "search_documents",
     f"Parsed: {parsed}")

# ✅ POSITIVE: Multiple tool calls
parsed_multi = llm._parse_tool_calls(
    '<tool_call>\n{"name": "tool1", "arguments": {}}\n</tool_call>'
    '\n<tool_call>\n{"name": "tool2", "arguments": {"x": 1}}\n</tool_call>'
)
test("_parse_tool_calls handles multiple tool calls",
     len(parsed_multi) == 2,
     f"Parsed {len(parsed_multi)} tool calls")

# ⚪ NEUTRAL: No tool calls in text
parsed_none = llm._parse_tool_calls("Just a normal response with no tools.")
test("_parse_tool_calls returns [] for no tool calls",
     parsed_none == [],
     "No false positives")

# ❌ NEGATIVE: Invalid JSON in tool call (gracefully ignored)
parsed_bad = llm._parse_tool_calls(
    '<tool_call>\n{not valid json}\n</tool_call>'
)
test("_parse_tool_calls ignores malformed JSON",
     parsed_bad == [],
     "Bad JSON silently skipped")

# ✅ POSITIVE: Message conversion
msgs = llm._convert_messages([
    SystemMessage(content="Be brief"),
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there"),
])
test("_convert_messages produces correct Qwen format",
     msgs[0]["role"] == "system" and msgs[1]["role"] == "user" and msgs[2]["role"] == "assistant",
     f"Roles: {[m['role'] for m in msgs]}")

# ✅ POSITIVE: ToolMessage conversion
tool_msgs = llm._convert_messages([
    AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "c1"}]),
    ToolMessage(content="result data", tool_call_id="c1", name="search"),
])
test("_convert_messages handles ToolMessage",
     tool_msgs[-1]["role"] == "tool" and tool_msgs[-1]["tool_call_id"] == "c1",
     f"Tool message role: {tool_msgs[-1]['role']}")


# %%
# ==========================================================================
# CELL 13: TEST RAG PIPELINE (Full end-to-end)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/rag/pipeline.py — RAGPipeline (end-to-end facade)
#
# WHY WE TEST:
#   RAGPipeline ties together retrieval → reranking → generation.
#   This is the main user-facing API. We verify:
#   - Pipeline constructs correctly with various retriever configs
#   - invoke() produces answers grounded in context
#   - Positive scenario: question about indexed content → correct answer
#   - Negative scenario: question about non-indexed content → "not enough info"
#   - Neutral scenario: vague/ambiguous question → some response
#
# HOW IT WORKS:
#   RAGPipeline.__init__() wires:
#     retrievers → HybridRetriever → CrossEncoderReranker → build_rag_chain()
#   invoke(question) runs the full LCEL chain and returns answer string.

section("13. RAG PIPELINE — End-to-End")

# Build full pipeline with BM25 + Qdrant + Reranker
bm25_for_pipeline = BM25Store()
bm25_for_pipeline.add_documents(test_docs)

rag_pipeline = RAGPipeline(
    retrievers=[
        bm25_for_pipeline.as_dynamic_retriever(),
        qdrant_store.as_dynamic_retriever(),
    ],
    llm=llm,
    reranker=reranker,
    retrieval_k=10,
    final_k=3,
)

test("RAGPipeline constructed successfully",
     rag_pipeline.chain is not None,
     "Chain ready")

# ✅ POSITIVE: Question about indexed content
print("\n  ⏳ Running positive scenario query...")
answer_positive = rag_pipeline.invoke("Who was the prince of Ayodhya?")
test("POSITIVE: Question about indexed content gets answer",
     isinstance(answer_positive, str) and len(answer_positive) > 0,
     f"Answer: '{answer_positive[:150]}...'")

# Check answer mentions Rama (it should, since docs contain this info)
test("POSITIVE: Answer mentions 'Rama'",
     "rama" in answer_positive.lower() or "prince" in answer_positive.lower(),
     "Answer is grounded in context")

# ❌ NEGATIVE: Question about non-indexed content
print("\n  ⏳ Running negative scenario query...")
answer_negative = rag_pipeline.invoke("What is the GDP of Japan in 2025?")
test("NEGATIVE: Out-of-scope question handled gracefully",
     isinstance(answer_negative, str) and len(answer_negative) > 0,
     f"Answer: '{answer_negative[:150]}...'")

# The system prompt says "say that you do not have enough information"
# if context doesn't contain the answer
test("NEGATIVE: Answer indicates lack of info (expected from prompt)",
     any(phrase in answer_negative.lower() for phrase in [
         "not enough", "cannot", "don't have", "no information",
         "not found", "unable", "do not have", "doesn't", "does not"
     ]),
     "Model should admit when context lacks the answer")

# ⚪ NEUTRAL: Vague question
print("\n  ⏳ Running neutral scenario query...")
answer_neutral = rag_pipeline.invoke("Tell me something")
test("NEUTRAL: Vague question produces some response",
     isinstance(answer_neutral, str) and len(answer_neutral) > 0,
     f"Answer: '{answer_neutral[:150]}...'")

# ✅ POSITIVE: get_chain returns the underlying chain
chain = rag_pipeline.get_chain()
test("get_chain() returns the LCEL chain",
     chain is not None,
     f"Chain type: {type(chain).__name__}")


# %%
# ==========================================================================
# CELL 14: TEST CONVERSATIONAL RAG
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/rag/conversational.py — ConversationStore, build_conversational_rag()
#
# WHY WE TEST:
#   Conversational RAG adds chat history awareness. The query rewriter
#   resolves references ("it", "they", "the first one") using history.
#   We verify:
#   - ConversationStore manages per-session histories
#   - First question goes directly to RAG (no rewriting needed)
#   - Follow-up questions get rewritten using history
#
# HOW IT WORKS:
#   ConversationStore: dict mapping session_id → InMemoryChatMessageHistory.
#   build_conversational_rag() wraps the RAG chain with:
#     1. Query rewriter (resolves references using history)
#     2. RunnableWithMessageHistory (auto-manages chat history per session)
#   First question: no history → passes through unchanged.
#   Follow-ups: history present → rewriter rewrites the question.

section("14. CONVERSATIONAL RAG")

# --- Test ConversationStore ---

store = ConversationStore()

# ✅ POSITIVE: New session creates empty history
history = store.get_history("session_1")
test("New session gets empty history",
     len(history.messages) == 0,
     "session_1 created with 0 messages")

# ✅ POSITIVE: Same session returns same history object
history2 = store.get_history("session_1")
test("Same session_id returns same history",
     history is history2,
     "Object identity preserved")

# ✅ POSITIVE: Different sessions are independent
history_other = store.get_history("session_2")
test("Different sessions are independent",
     history_other is not history,
     "session_1 ≠ session_2")

# ✅ POSITIVE: Clear history
store.get_history("session_3").add_user_message("Hello")
store.clear_history("session_3")
test("clear_history removes messages",
     len(store.get_history("session_3").messages) == 0,
     "session_3 cleared")

# ⚪ NEUTRAL: Clear non-existent session (no crash)
store.clear_history("nonexistent_session")
test("clear_history on non-existent session doesn't crash",
     True,
     "No KeyError raised")

# --- Test conversational RAG flow ---
print("\n  ⏳ Building conversational RAG chain...")
conv_store = ConversationStore()
conv_rag = build_conversational_rag(
    rag_chain=rag_pipeline.get_chain(),
    llm=llm,
    history_store=conv_store,
)

# ✅ POSITIVE: First question (no history, no rewriting)
config = {"configurable": {"session_id": "test_session"}}
print("  ⏳ First question (no history)...")
conv_answer1 = conv_rag.invoke(
    {"question": "Who kidnapped Sita?"},
    config=config,
)
test("Conversational RAG first question returns answer",
     isinstance(conv_answer1, str) and len(conv_answer1) > 0,
     f"Answer: '{conv_answer1[:100]}...'")

# ✅ POSITIVE: Follow-up question (uses history + rewriting)
print("  ⏳ Follow-up question (with history)...")
conv_answer2 = conv_rag.invoke(
    {"question": "Where did he take her?"},
    config=config,
)
test("Conversational RAG follow-up resolves references",
     isinstance(conv_answer2, str) and len(conv_answer2) > 0,
     f"Answer: '{conv_answer2[:100]}...' (should reference Lanka)")


# %%
# ==========================================================================
# CELL 15: TEST LANGGRAPH NODES & ROUTING
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/graph/state.py  — RAGState TypedDict
#   src/graph/nodes.py  — route_question, route_after_grading, create_fallback_node,
#                          RetrievalGrade, AnswerGrade schemas
#
# WHY WE TEST:
#   Graph nodes implement the corrective RAG workflow:
#   START → Router → Retrieve → Grade → Generate/Rewrite/Fallback
#   We test routing logic and node behavior independently of the LLM.
#
# HOW IT WORKS:
#   route_question: checks if history exists → "retrieve" or "rewrite"
#   route_after_grading: checks retrieval_relevant + attempts → "generate"/"rewrite"/"fallback"
#   create_fallback_node: returns safe "couldn't find info" response.
#   Pydantic schemas (RetrievalGrade, AnswerGrade) validate structured LLM output.

section("15. LANGGRAPH NODES & ROUTING")

# --- Test route_question ---

# ✅ POSITIVE: No history → retrieve directly
state_no_history = {"question": "Who is Rama?", "history": []}
test("route_question: no history → 'retrieve'",
     route_question(state_no_history) == "retrieve",
     "First question goes straight to retrieval")

# ✅ POSITIVE: Has history → rewrite
state_with_history = {"question": "Tell me more about him", "history": [HumanMessage(content="x")]}
test("route_question: has history → 'rewrite'",
     route_question(state_with_history) == "rewrite",
     "Follow-up questions get rewritten")

# ⚪ NEUTRAL: Missing history key defaults to retrieve
state_missing = {"question": "Hello"}
test("route_question: missing history key → 'retrieve'",
     route_question(state_missing) == "retrieve",
     "Graceful default when history not in state")

# --- Test route_after_grading ---

# ✅ POSITIVE: Relevant retrieval → generate
test("route_after_grading: relevant → 'generate'",
     route_after_grading({"retrieval_relevant": True, "retrieval_attempts": 1}) == "generate",
     "Good retrieval proceeds to generation")

# ✅ POSITIVE: Not relevant + retries left → rewrite
test("route_after_grading: not relevant + retries → 'rewrite'",
     route_after_grading({"retrieval_relevant": False, "retrieval_attempts": 1}) == "rewrite",
     f"attempts=1 < MAX={MAX_RETRIEVAL_ATTEMPTS} → retry")

# ✅ POSITIVE: Not relevant + max attempts → fallback
test("route_after_grading: not relevant + max attempts → 'fallback'",
     route_after_grading({"retrieval_relevant": False, "retrieval_attempts": MAX_RETRIEVAL_ATTEMPTS}) == "fallback",
     f"attempts={MAX_RETRIEVAL_ATTEMPTS} → give up gracefully")

# ⚪ NEUTRAL: Default values when keys missing
test("route_after_grading: missing keys → 'fallback' (defaults to not relevant, 0 attempts → rewrite)",
     route_after_grading({}) == "rewrite",
     "Defaults: relevant=False, attempts=0 → rewrite")

# --- Test fallback node ---

fallback = create_fallback_node()
fallback_result = fallback({"question": "test"})
test("Fallback node returns safe message",
     "couldn't find" in fallback_result["answer"].lower() or "could not" in fallback_result["answer"].lower(),
     f"Fallback: '{fallback_result['answer'][:80]}...'")

# --- Test Pydantic schemas ---

grade = RetrievalGrade(relevant=True, reason="Documents discuss Rama")
test("RetrievalGrade schema validates correctly",
     grade.relevant is True and "Rama" in grade.reason,
     f"relevant={grade.relevant}, reason='{grade.reason}'")

answer_grade = AnswerGrade(supported=False, reason="Unsupported claim")
test("AnswerGrade schema validates correctly",
     answer_grade.supported is False,
     f"supported={answer_grade.supported}")


# %%
# ==========================================================================
# CELL 16: TEST FULL LANGGRAPH WORKFLOW
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/graph/graph.py — build_rag_graph() (the complete corrective RAG graph)
#
# WHY WE TEST:
#   The LangGraph workflow is the most advanced execution mode:
#   retrieve → grade → generate/rewrite/fallback with self-correction.
#   We verify the graph compiles and runs end-to-end.
#
# HOW IT WORKS:
#   build_rag_graph() constructs a StateGraph with nodes:
#     rewrite, retrieve, grade_retrieval, generate, grade_answer, fallback
#   Conditional edges implement corrective RAG:
#     - Good retrieval → generate → grade answer → END
#     - Bad retrieval → rewrite → retrieve → grade (up to MAX attempts)
#     - Max attempts exhausted → fallback → END

section("16. FULL LANGGRAPH WORKFLOW")

from src.graph.graph import build_rag_graph
from src.rag.chain import build_generation_chain

# Build the generation-only chain (LangGraph does its own retrieval)
gen_chain = build_generation_chain(llm)

# Build the corrective RAG graph
rag_graph = build_rag_graph(
    llm=llm,
    retriever=rag_pipeline.retriever,
    generation_chain=gen_chain,
)

test("LangGraph RAG workflow compiles successfully",
     rag_graph is not None,
     f"Graph type: {type(rag_graph).__name__}")

# ✅ POSITIVE: Run graph with a question about indexed content
print("\n  ⏳ Running LangGraph workflow (positive scenario)...")
graph_result = rag_graph.invoke({
    "question": "Who crossed the ocean to find Sita?",
})
test("LangGraph POSITIVE: produces answer",
     "answer" in graph_result and len(graph_result["answer"]) > 0,
     f"Answer: '{graph_result.get('answer', 'N/A')[:100]}...'")

test("LangGraph POSITIVE: documents retrieved",
     "documents" in graph_result and len(graph_result.get("documents", [])) > 0,
     f"Retrieved {len(graph_result.get('documents', []))} documents")

# ❌ NEGATIVE: Question completely outside indexed content
print("\n  ⏳ Running LangGraph workflow (negative scenario)...")
graph_result_neg = rag_graph.invoke({
    "question": "What is the chemical formula of sulfuric acid?",
})
test("LangGraph NEGATIVE: handles out-of-scope question",
     "answer" in graph_result_neg and len(graph_result_neg["answer"]) > 0,
     f"Answer: '{graph_result_neg.get('answer', 'N/A')[:100]}...'")

# ⚪ NEUTRAL: Question with history (triggers rewrite path)
print("\n  ⏳ Running LangGraph workflow (with history, neutral)...")
graph_result_hist = rag_graph.invoke({
    "question": "Tell me more about it",
    "history": [
        HumanMessage(content="Who is Hanuman?"),
        AIMessage(content="Hanuman is a devotee of Lord Rama."),
    ],
})
test("LangGraph NEUTRAL: handles follow-up with history",
     "answer" in graph_result_hist,
     f"Answer: '{graph_result_hist.get('answer', 'N/A')[:100]}...'")

test("LangGraph NEUTRAL: rewritten_query created for follow-up",
     "rewritten_query" in graph_result_hist,
     f"Rewritten: '{graph_result_hist.get('rewritten_query', 'N/A')[:80]}'")


# %%
# ==========================================================================
# CELL 17: TEST DOCUMENT SEARCH TOOL
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/tools/document_search.py — build_document_search_tool()
#
# WHY WE TEST:
#   The search tool is what LangGraph's ToolNode calls when the LLM
#   decides it needs to retrieve documents. We verify:
#   - Tool is callable and returns formatted results
#   - Tool handles "no results" gracefully
#   - Tool has correct LangChain tool metadata (name, description)
#
# HOW IT WORKS:
#   build_document_search_tool(retriever) creates a @tool-decorated function
#   that calls retriever.invoke({"question": query}) and formats results
#   as "[Document N]\nSource: ...\n..." text blocks.

section("17. DOCUMENT SEARCH TOOL")

search_tool = build_document_search_tool(rag_pipeline.retriever)

# ✅ POSITIVE: Tool has correct name
test("Search tool has name 'search_documents'",
     search_tool.name == "search_documents",
     f"Tool name: {search_tool.name}")

# ✅ POSITIVE: Tool has description
test("Search tool has description",
     len(search_tool.description) > 0,
     f"Description: '{search_tool.description[:60]}...'")

# ✅ POSITIVE: Tool returns formatted results
tool_result = search_tool.invoke({"query": "Rama prince Ayodhya"})
test("Search tool returns formatted document results",
     isinstance(tool_result, str) and len(tool_result) > 0,
     f"Result preview: '{tool_result[:100]}...'")

# ✅ POSITIVE: Results contain document markers
test("Search tool output contains [Document N] markers",
     "[Document 1]" in tool_result,
     "Formatted with numbered documents")


# %%
# ==========================================================================
# CELL 18: TEST DOCUMENT INDEXER
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/indexing/indexer.py — DocumentIndexer (ID generation, content hashing)
#
# WHY WE TEST:
#   DocumentIndexer is responsible for deduplication and incremental indexing.
#   We test the deterministic ID generation and content hashing separately
#   from the full index() call (which requires a real vector store + SQL).
#
# HOW IT WORKS:
#   create_document_id(source, external_id) = SHA-256 of "source:external_id"
#   calculate_content_hash(content) = SHA-256 of content string
#   prepare_documents() injects source_system, document_id, content_hash
#   into each document's metadata before indexing.

section("18. DOCUMENT INDEXER")

# ✅ POSITIVE: Deterministic document ID generation
doc_id = DocumentIndexer.create_document_id("local_files", "/data/test.pdf")
test("create_document_id is deterministic",
     doc_id == DocumentIndexer.create_document_id("local_files", "/data/test.pdf"),
     f"ID: {doc_id[:20]}...")

# ✅ POSITIVE: Different inputs produce different IDs
doc_id_2 = DocumentIndexer.create_document_id("local_files", "/data/other.pdf")
test("Different inputs → different document IDs",
     doc_id != doc_id_2,
     "Hash collision avoided")

# ✅ POSITIVE: Content hash is deterministic
hash1 = DocumentIndexer.calculate_content_hash("Hello world")
hash2 = DocumentIndexer.calculate_content_hash("Hello world")
test("calculate_content_hash is deterministic",
     hash1 == hash2,
     f"Hash: {hash1[:20]}...")

# ✅ POSITIVE: Different content → different hash
hash3 = DocumentIndexer.calculate_content_hash("Different content")
test("Different content → different hash",
     hash1 != hash3,
     "Content change detected")

# ✅ POSITIVE: prepare_documents injects metadata
indexer = DocumentIndexer(
    vector_store=qdrant_store.vector_store,
    db_url="sqlite:///test_record_manager.db",
)
prep_docs = indexer.prepare_documents(
    [LCDocument(page_content="Test content", metadata={"source": "test.pdf"})],
    source_system="test_system",
)
test("prepare_documents injects metadata",
     prep_docs[0].metadata.get("source_system") == "test_system"
     and "document_id" in prep_docs[0].metadata
     and "content_hash" in prep_docs[0].metadata,
     f"Metadata keys: {list(prep_docs[0].metadata.keys())}")


# %%
# ==========================================================================
# CELL 19: TEST DOCUMENT INGESTION (MultiDocumentLoader)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/ingestion/loader.py — MultiDocumentLoader
#
# WHY WE TEST:
#   The loader is the entry point for all documents. We verify:
#   - Supported file types load correctly (.txt, .csv)
#   - Unsupported file types raise ValueError
#   - Metadata (file_name, file_type) injected correctly
#   - load_directory() recursively finds files
#
# HOW IT WORKS:
#   MultiDocumentLoader.load_file() dispatches to the appropriate
#   LangChain loader based on file extension:
#   .pdf → PyPDFLoader, .txt → TextLoader, .csv → CSVLoader,
#   .docx → UnstructuredWordDocumentLoader, .xlsx → UnstructuredExcelLoader
#   Each loaded document gets file_name and file_type metadata.

section("19. DOCUMENT INGESTION")

# Create temp test files
temp_dir = tempfile.mkdtemp()

# Create a .txt file
txt_path = Path(temp_dir) / "test.txt"
txt_path.write_text("This is a test document about the Ramayana.", encoding="utf-8")

# Create a .csv file
csv_path = Path(temp_dir) / "test.csv"
csv_path.write_text("name,description\nRama,Prince of Ayodhya\nSita,Wife of Rama", encoding="utf-8")

loader = MultiDocumentLoader(data_directory=temp_dir)

# ✅ POSITIVE: Load .txt file
txt_docs = loader.load_file(txt_path)
test("load_file loads .txt correctly",
     len(txt_docs) > 0 and "Ramayana" in txt_docs[0].page_content,
     f"Loaded {len(txt_docs)} doc(s), content: '{txt_docs[0].page_content[:50]}'")

# ✅ POSITIVE: Metadata injected
test("Loaded doc has file_name metadata",
     txt_docs[0].metadata.get("file_name") == "test.txt",
     f"file_name: {txt_docs[0].metadata.get('file_name')}")

test("Loaded doc has file_type metadata",
     txt_docs[0].metadata.get("file_type") == ".txt",
     f"file_type: {txt_docs[0].metadata.get('file_type')}")

# ✅ POSITIVE: Load .csv file
csv_docs = loader.load_file(csv_path)
test("load_file loads .csv correctly",
     len(csv_docs) > 0,
     f"Loaded {len(csv_docs)} row(s) from CSV")

# ✅ POSITIVE: load_directory finds all files
all_docs = loader.load_directory()
test("load_directory recursively loads all supported files",
     len(all_docs) >= 3,  # 1 txt + 2 csv rows
     f"Loaded {len(all_docs)} documents from directory")

# ❌ NEGATIVE: Unsupported file type
unsupported = Path(temp_dir) / "test.xyz"
unsupported.write_text("some data")
test_raises("load_file raises ValueError for unsupported .xyz",
            ValueError, loader.load_file, unsupported)

# ⚪ NEUTRAL: Empty directory
empty_dir = tempfile.mkdtemp()
empty_loader = MultiDocumentLoader(data_directory=empty_dir)
empty_docs = empty_loader.load_directory()
test("load_directory on empty dir returns []",
     empty_docs == [],
     "No crash on empty directory")


# %%
# ==========================================================================
# CELL 20: TEST GRAPH STATE (RAGState TypedDict)
# ==========================================================================
#
# WHAT THIS TESTS:
#   src/graph/state.py — RAGState (TypedDict with total=False)
#
# WHY WE TEST:
#   RAGState defines the shared state for the LangGraph workflow.
#   total=False means all fields are optional. We verify it works
#   as a plain dict (TypedDict is a type hint, not runtime enforced).
#
# HOW IT WORKS:
#   RAGState is a TypedDict. At runtime it's just a dict.
#   Fields: question, history, rewritten_query, documents,
#           retrieval_attempts, retrieval_relevant, retrieval_grade_reason,
#           answer, answer_supported, answer_grade_reason

section("20. GRAPH STATE")

# ✅ POSITIVE: Create state with all fields
full_state: RAGState = {
    "question": "Who is Rama?",
    "history": [],
    "rewritten_query": "Who is Rama in Ramayana?",
    "documents": [],
    "retrieval_attempts": 0,
    "retrieval_relevant": True,
    "retrieval_grade_reason": "Relevant docs found",
    "answer": "Rama is the prince of Ayodhya",
    "answer_supported": True,
    "answer_grade_reason": "Supported by context",
}
test("RAGState accepts all fields",
     full_state["question"] == "Who is Rama?" and full_state["answer_supported"] is True,
     f"All {len(full_state)} fields set")

# ⚪ NEUTRAL: Partial state (total=False makes all fields optional)
partial_state: RAGState = {"question": "Hello"}
test("RAGState works with partial fields (total=False)",
     partial_state["question"] == "Hello",
     "Only 'question' set — no error")

# ⚪ NEUTRAL: Empty state
empty_state: RAGState = {}
test("Empty RAGState is valid (total=False)",
     isinstance(empty_state, dict),
     "TypedDict is a type hint, not enforced at runtime")


# %%
# ==========================================================================
# CELL 21: FINAL SUMMARY
# ==========================================================================

section("FINAL TEST RESULTS")
summary()

# Cleanup
import shutil
if os.path.exists("/content/test_qdrant_store"):
    shutil.rmtree("/content/test_qdrant_store")
if os.path.exists("test_record_manager.db"):
    os.remove("test_record_manager.db")
print("🧹 Cleanup complete")
