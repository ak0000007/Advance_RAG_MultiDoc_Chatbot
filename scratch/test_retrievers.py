import sys
import os
sys.path.append(os.getcwd())

from langchain_core.documents import Document
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.reranker import CrossEncoderReranker
from src.rag.chain import build_rag_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

# Mock Langchain standard retriever (expects string)
def mock_standard_retriever(query):
    if not isinstance(query, str):
        # Emulate the dict replace error
        query.replace("a", "b")
    return [Document(page_content=f"Standard result for {query}")]

standard_runnable = RunnableLambda(mock_standard_retriever)

# Mock dynamic retriever (expects dict)
def mock_dynamic_retriever(inputs):
    if not isinstance(inputs, dict):
        # Emulate dict missing error
        inputs.get("question")
    return [Document(page_content=f"Dynamic result for {inputs['question']}")]

dynamic_runnable = RunnableLambda(mock_dynamic_retriever)

mock_llm = RunnableLambda(lambda x: AIMessage(content="mock answer"))

print("Testing chain with standard retriever...")
chain1 = build_rag_chain(standard_runnable, mock_llm)
res1 = chain1.invoke({"question": "test1"})
print("Chain 1 output:", res1)

print("Testing chain with dynamic retriever...")
chain2 = build_rag_chain(dynamic_runnable, mock_llm)
res2 = chain2.invoke({"question": "test2"})
print("Chain 2 output:", res2)

print("Testing hybrid with standard retriever...")
hybrid1 = HybridRetriever([standard_runnable]).as_dynamic_retriever()
res3 = hybrid1.invoke({"question": "test3"})
print("Hybrid 1 output:", len(res3))

print("Testing hybrid with dynamic retriever...")
hybrid2 = HybridRetriever([dynamic_runnable]).as_dynamic_retriever()
res4 = hybrid2.invoke({"question": "test4"})
print("Hybrid 2 output:", len(res4))

print("Testing reranker with standard retriever...")
reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2", device="cpu")
# monkey patch the model load to avoid downloading
reranker._load_model = lambda: None
reranker._model = type("MockModel", (), {"predict": lambda self, pairs: [1.0 for _ in pairs]})()

reranked1 = reranker.wrap_retriever(standard_runnable)
res5 = reranked1.invoke({"question": "test5"})
print("Reranked 1 output:", len(res5))

print("Testing reranker with dynamic retriever...")
reranked2 = reranker.wrap_retriever(dynamic_runnable)
res6 = reranked2.invoke({"question": "test6"})
print("Reranked 2 output:", len(res6))

