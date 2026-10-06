from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from src.config import settings


# =========================================================
# Default configuration
# =========================================================

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"


# =========================================================
# Provider errors
# =========================================================


class ModelProviderError(Exception):
    """Base exception for model-provider errors."""


class ModelRateLimitError(ModelProviderError):
    """
    Raised when the primary provider is temporarily unavailable
    because of quota/rate limiting.
    """


# =========================================================
# Provider list
# =========================================================


def list_providers() -> list[str]:
    """
    Return supported model providers.
    """

    return [
        "google",
        "deepseek",
        "openai",
        "local",
    ]


# =========================================================
# Google Gemini
# =========================================================


def _create_gemini(
    *,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create the Gemini chat model.
    """

    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
        )
    except ImportError as exc:
        raise ImportError(
            "Gemini support requires "
            "'langchain-google-genai'. "
            "Install it with:\n\n"
            "pip install -U langchain-google-genai"
        ) from exc

    gemini_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_retries": 1,
    }

    google_api_key = (
        api_key
        or settings.google_api_key
    )

    if google_api_key:
        gemini_kwargs["google_api_key"] = google_api_key

    gemini_kwargs.update(kwargs)

    base_model = ChatGoogleGenerativeAI(
        **gemini_kwargs
    )

    return base_model


# =========================================================
# DeepSeek
# =========================================================


def _create_deepseek(
    *,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create the DeepSeek model.

    DeepSeek's API is OpenAI-compatible.

    Context caching is handled automatically by the
    DeepSeek API; no explicit cache flag is required.
    """

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "DeepSeek support requires "
            "'langchain-openai'. "
            "Install it with:\n\n"
            "pip install -U langchain-openai"
        ) from exc

    deepseek_api_key = (
        api_key
        or settings.deepseek_api_key
    )

    if not deepseek_api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not configured."
        )

    return ChatOpenAI(
        model=model,
        api_key=deepseek_api_key,
        base_url="https://api.deepseek.com",
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


# =========================================================
# OpenAI
# =========================================================


def _create_openai(
    *,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create an OpenAI-compatible model.

    This is not currently part of our primary/fallback path,
    but keeping it here makes the provider layer extensible.
    """

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI support requires "
            "'langchain-openai'."
        ) from exc

    openai_api_key = (
        api_key
        or settings.openai_api_key
    )

    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured."
        )

    return ChatOpenAI(
        model=model,
        api_key=openai_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


# =========================================================
# Local Qwen
# =========================================================


def _create_local(
    *,
    model: str | None,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """
    Keep the existing local Qwen implementation available
    for development/testing.
    """

    from .model import load_llm

    return load_llm(
        model_id=model or "Qwen/Qwen3-4B-Instruct-2507",
        temperature=temperature,
        max_new_tokens=max_tokens,
    )


# =========================================================
# Single provider factory
# =========================================================


def _create_single_llm(
    provider: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 600,
    **kwargs: Any,
) -> BaseChatModel:

    provider = provider.lower().strip()

    if provider == "google":

        return _create_gemini(
            model=model or DEFAULT_GEMINI_MODEL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "deepseek":

        return _create_deepseek(
            model=model or DEFAULT_DEEPSEEK_MODEL,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "openai":

        return _create_openai(
            model=model or "gpt-4o-mini", ## use gpt-4.1-nano  for simple tasks and gpt-4.1-mini for more Agentic task tasks
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if provider == "local":

        return _create_local(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. "
        f"Supported providers: {list_providers()}"
    )


# =========================================================
# Public factory
# =========================================================


def create_llm(
    provider: str = "google",
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 600,
    fallback_provider: str | None = "openai",
    fallback_model: str | None = None,
    fallback_api_key: str | None = None,
    structured_output: type | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create the project's model.

    Default architecture:

        Gemini
           ↓
        rate/quota failure
           ↓
        DeepSeek
    """
    from langchain_core.runnables import Runnable

    primary_llm = _create_single_llm(
        provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    
    if structured_output:
        primary_llm = primary_llm.with_structured_output(structured_output)

    if not fallback_provider:
        return primary_llm

    fallback_llm = _create_single_llm(
        fallback_provider,
        model=fallback_model,
        api_key=fallback_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    
    if structured_output:
        fallback_llm = fallback_llm.with_structured_output(structured_output)

    # Use native exceptions for fallbacks
    exceptions = (Exception,)
    try:
        from google.api_core.exceptions import ResourceExhausted, TooManyRequests
        exceptions = (ResourceExhausted, TooManyRequests, Exception)
    except ImportError:
        pass

    return primary_llm.with_fallbacks([fallback_llm], exceptions_to_handle=exceptions)