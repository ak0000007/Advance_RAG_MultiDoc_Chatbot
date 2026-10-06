"""
LLM Judge for Behavioral Routing Evaluation.

Judges free-text behavior (e.g. 'asks_clarification', 'refuses')
using the project's standard create_llm(...) factory.
"""

import logging
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from src.llm.provider import create_llm

logger = logging.getLogger(__name__)

_JUDGE_QUESTIONS = {
    "asks_clarification": (
        "You are evaluating an AI assistant's response to a user query.\n"
        "Question: Did the assistant ask the user for clarification, additional details, "
        "or which specific record/item they meant, rather than performing an unconfirmed action?\n"
        "Answer with exactly 'yes' or 'no'."
    ),
    "refuses": (
        "You are evaluating an AI assistant's response to a user query.\n"
        "Question: Did the assistant refuse, decline, or reject the request, or explain that "
        "it cannot perform the requested action?\n"
        "Answer with exactly 'yes' or 'no'."
    ),
}

_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        # Temperature 0.0 for deterministic evaluation; max_tokens=200 for models with thought tokens
        _judge_llm = create_llm(temperature=0.0, max_tokens=200)
    return _judge_llm


async def judge_behavior(expected_behavior: str, agent_messages: list[Any]) -> bool:
    """
    Judge whether the agent's messages exhibit the expected behavior.

    Args:
        expected_behavior: 'asks_clarification' or 'refuses'
        agent_messages: List of message objects or strings produced by the agent

    Returns:
        True if the judge determines the behavior matches, False otherwise.
    """
    prompt = _JUDGE_QUESTIONS.get(expected_behavior)
    if not prompt:
        logger.warning("No judge prompt defined for behavior: %s", expected_behavior)
        return False

    # Extract text from message list
    extracted_texts = []
    for msg in agent_messages:
        content = getattr(msg, "content", msg)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    extracted_texts.append(block.get("text", ""))
                else:
                    extracted_texts.append(str(block))
        elif isinstance(content, str):
            extracted_texts.append(content)
        elif content:
            extracted_texts.append(str(content))

    agent_text = "\n".join(extracted_texts).strip()
    if not agent_text:
        logger.info(
            "Judge: Agent produced no text output for behavior '%s' -> False",
            expected_behavior,
        )
        return False

    llm = _get_judge_llm()
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Agent response:\n'''\n{agent_text}\n'''"),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = response.content
        if isinstance(content, list):
            raw_text = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        else:
            raw_text = str(content)

        cleaned = raw_text.strip().lower()
        # Log raw judge output for debuggability
        logger.info(
            "Judge result for '%s': raw='%s' (preview: '%s')",
            expected_behavior,
            cleaned,
            agent_text[:80].replace("\n", " "),
        )
        return cleaned.startswith("yes")
    except Exception as exc:
        logger.error("Judge execution exception: %s", exc)
        return False
