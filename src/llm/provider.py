from __future__ import annotations

from importlib import import_module
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def list_providers() -> list[str]:
    """
    Return supported LLM providers.
    """

    return [
        "local",
        "google",
    ]


def create_llm(
    provider: str = "local",
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create a LangChain BaseChatModel.

    The rest of the application should call this function
    instead of importing a provider-specific model directly.

    Supported providers:

        local  -> local Qwen3
        google -> Gemini API
    """

    provider = provider.lower().strip()

    # ---------------------------------------------------------
    # Local Qwen3
    # ---------------------------------------------------------

    if provider == "local":

        from .model import load_llm

        return load_llm(
            model_id=model or "Qwen/Qwen3-4B-Instruct-2507",
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

    # ---------------------------------------------------------
    # Google Gemini
    # ---------------------------------------------------------

    if provider == "google":

        try:
            ChatGoogleGenerativeAI = import_module(
                "langchain_google_genai"
            ).ChatGoogleGenerativeAI

        except ModuleNotFoundError as exc:

            raise ModuleNotFoundError(
                "Gemini support requires "
                "'langchain-google-genai'. "
                "Install it with:\n\n"
                "pip install langchain-google-genai"
            ) from exc

        init_kwargs: dict[str, Any] = {
            "model": model or DEFAULT_GEMINI_MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if api_key is not None:
            init_kwargs["google_api_key"] = api_key

        init_kwargs.update(kwargs)

        return ChatGoogleGenerativeAI(
            **init_kwargs
        )

    # ---------------------------------------------------------
    # Unknown provider
    # ---------------------------------------------------------

    raise ValueError(
        f"Unknown LLM provider: '{provider}'. "
        f"Supported providers: {list_providers()}"
    )
