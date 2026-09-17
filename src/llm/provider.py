from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

def list_providers() -> list[str]:
    """Return supported LLM providers."""
    return ["local", "google", "deepseek", "openai"]

class RateLimitException(Exception):
    """Specific exception for 429 Rate Limits so fallback doesn't trigger on other errors."""
    pass

def _create_single_llm(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    **kwargs: Any,
) -> BaseChatModel:
    """Internal factory to create a single LLM instance."""
    provider = provider.lower().strip()

    if provider == "local":
        from .model import load_llm
        return load_llm(
            model_id=model or "Qwen/Qwen3-4B-Instruct-2507",
            temperature=temperature,
            max_new_tokens=max_tokens,
        )

    if provider == "google":
        try:
            ChatGoogleGenerativeAI = import_module("langchain_google_genai").ChatGoogleGenerativeAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install with: pip install langchain-google-genai") from exc
            
        class StrictRateLimitGemini(ChatGoogleGenerativeAI):
            def _check_429(self, e: Exception):
                err = str(e).lower()
                if "429" in err or "quota" in err or "exhausted" in err or "too many requests" in err:
                    raise RateLimitException(str(e))
                raise e

            def _generate(self, *args, **kwargs):
                try:
                    return super()._generate(*args, **kwargs)
                except Exception as e:
                    self._check_429(e)

            async def _agenerate(self, *args, **kwargs):
                try:
                    return await super()._agenerate(*args, **kwargs)
                except Exception as e:
                    self._check_429(e)

        init_kwargs = {
            "model": model or DEFAULT_GEMINI_MODEL,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if api_key is not None:
            init_kwargs["google_api_key"] = api_key
            
        init_kwargs.update(kwargs)
        return StrictRateLimitGemini(**init_kwargs)

    if provider == "deepseek":
        try:
            ChatOpenAI = import_module("langchain_openai").ChatOpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install with: pip install langchain-openai") from exc
        
        return ChatOpenAI(
            model=model or DEFAULT_DEEPSEEK_MODEL,
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
    if provider == "openai":
        try:
            ChatOpenAI = import_module("langchain_openai").ChatOpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Install with: pip install langchain-openai") from exc
            
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    raise ValueError(f"Unknown LLM provider: '{provider}'. Supported: {list_providers()}")


def create_llm(
    provider: str = "google",
    *,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 300,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Create a LangChain BaseChatModel with an optional automatic fallback.
    """
    
    # 1. Create the primary LLM
    primary_llm = _create_single_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    # 2. Chain fallback if requested
    if fallback_provider:
        fallback_llm = _create_single_llm(
            provider=fallback_provider,
            model=fallback_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # ONLY fallback on Rate Limits, pass other errors through!
        return primary_llm.with_fallbacks([fallback_llm], exceptions_to_handle=(RateLimitException,))
        
    return primary_llm
