# core/llm.py
# Centralised LLM client — all API calls go through here.
#
# Supports DeepSeek (OpenAI-compatible) as the default provider.
# Switch provider by changing LLM_PROVIDER in config.py.

from __future__ import annotations

from openai import OpenAI

from config import LLM_API_BASE, LLM_API_KEY_ENV

import os

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazy-initialise and return the shared OpenAI-compatible client."""
    global _client
    if _client is None:
        api_key = os.environ.get(LLM_API_KEY_ENV, "")
        if not api_key:
            raise RuntimeError(
                f"LLM API key not set. Export {LLM_API_KEY_ENV} in your environment."
            )
        _client = OpenAI(api_key=api_key, base_url=LLM_API_BASE)
    return _client


def chat(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
) -> str:
    """
    Send a single-turn chat completion and return the assistant's text.

    This is the only function the rest of the codebase should call for LLM
    inference. Keeps the provider-specific details (API format, auth, caching)
    in one place.

    Args:
        model:      Model identifier (e.g. "deepseek-chat").
        system:     System prompt text.
        user:       User message text.
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's response text (stripped).

    Raises:
        Exception on API errors — callers should handle gracefully.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content.strip()
