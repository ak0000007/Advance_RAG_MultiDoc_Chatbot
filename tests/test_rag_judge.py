"""Deterministic unit tests for LLM Judge evaluation.
Uses mock LLMs to verify abstention detection, false abstention detection,
and answer faithfulness without requiring external LLM API calls.
"""
from unittest.mock import MagicMock
from langchain_core.documents import Document
from evals.rag.runner import evaluate_llm_judge


def make_mock_llm(return_text: str):
    mock = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = return_text
    mock.invoke.return_value = mock_resp
    return mock


def test_judge_unanswerable_true_abstention():
    """Unanswerable query where model correctly abstains -> abstention=1.0, false_abstention=0.0."""
    llm = make_mock_llm("I do not have enough information to answer this question.")
    res = evaluate_llm_judge(
        query="What is the discount code for rocket flights to the Moon?",
        retrieved_docs=[Document(page_content="Corporate travel policy...")],
        is_unanswerable=True,
        expected_answer="",
        llm=llm,
    )
    assert res["abstention"] == 1.0
    assert res["false_abstention"] == 0.0
    assert res["faithfulness"] == 1.0


def test_judge_unanswerable_hallucination():
    """Unanswerable query where model hallucinates an answer -> abstention=0.0."""
    llm = make_mock_llm("The moon flight discount code is APOLLO-2024 with 50% discount.")
    res = evaluate_llm_judge(
        query="What is the discount code for rocket flights to the Moon?",
        retrieved_docs=[Document(page_content="Corporate travel policy...")],
        is_unanswerable=True,
        expected_answer="",
        llm=llm,
    )
    assert res["abstention"] == 0.0
    assert res["false_abstention"] == 0.0
    assert res["faithfulness"] == 0.0


def test_judge_answerable_successful_response():
    """Answerable query where model provides grounded answer -> faithfulness=1.0, false_abstention=0.0."""
    llm = make_mock_llm("Domestic travel meal allowance is capped at $75 per day.")
    res = evaluate_llm_judge(
        query="What is the daily meal allowance for domestic travel?",
        retrieved_docs=[Document(page_content="Domestic travel meal allowance is $75 per day.")],
        is_unanswerable=False,
        expected_answer="Domestic meal allowance is $75 per day.",
        llm=llm,
    )
    assert res["abstention"] == 0.0
    assert res["false_abstention"] == 0.0
    assert res["faithfulness"] == 1.0


def test_judge_answerable_false_abstention():
    """Answerable query where model improperly refuses -> false_abstention=1.0."""
    llm = make_mock_llm("I do not know the answer based on the provided documents.")
    res = evaluate_llm_judge(
        query="What is the standard domestic hotel lodging cap?",
        retrieved_docs=[Document(page_content="Standard domestic lodging is capped at $250 per night.")],
        is_unanswerable=False,
        expected_answer="Standard domestic lodging cap is $250.",
        llm=llm,
    )
    assert res["abstention"] == 0.0
    assert res["false_abstention"] == 1.0
    assert res["faithfulness"] == 0.0
