"""
Model pricing registry and token cost calculation.
Rates expressed in USD per 1,000,000 tokens (input, output).
"""

from typing import Tuple

PRICING: dict[str, tuple[float, float]] = {
    # Gemini models (USD per 1M input / 1M output)
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    # DeepSeek models
    "deepseek-chat": (0.27, 1.10),
    "deepseek-coder": (0.27, 1.10),
    # OpenAI models
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic models
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
}

DEFAULT_PRICING: tuple[float, float] = (0.00, 0.00)


def _normalize_model_name(model: str) -> str:
    """Normalize model string to match pricing keys."""
    raw = model.strip().lower()
    # Strip common provider prefixes like 'models/', 'openai/', 'google/'
    for prefix in ("models/", "google/", "openai/", "anthropic/", "deepseek/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]

    # Strip version dates or -latest suffixes if present
    # e.g. gpt-4o-mini-2024-07-18 -> gpt-4o-mini
    return raw


def get_model_pricing(model: str) -> tuple[float, float, bool]:
    """
    Look up pricing for a model name.
    Returns (input_rate_per_1m, output_rate_per_1m, is_known).
    """
    norm = _normalize_model_name(model)
    if norm in PRICING:
        rate = PRICING[norm]
        return rate[0], rate[1], True

    # Check prefix matches (e.g. 'gemini-1.5-flash-8b' matches 'gemini-1.5-flash')
    for key, rate in PRICING.items():
        if norm.startswith(key):
            return rate[0], rate[1], True

    return DEFAULT_PRICING[0], DEFAULT_PRICING[1], False


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> tuple[float, bool]:
    """
    Calculate USD cost given model, input tokens, and output tokens.
    Returns (cost_usd, is_pricing_estimated).
    """
    inp_rate, out_rate, known = get_model_pricing(model)
    cost = (input_tokens * inp_rate / 1_000_000.0) + (output_tokens * out_rate / 1_000_000.0)
    # Round to 6 decimal places (micro-dollars)
    return round(cost, 6), not known
