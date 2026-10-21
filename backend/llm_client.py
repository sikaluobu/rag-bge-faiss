"""Generation side: call the DeepSeek chat completion endpoint.

The API key is read from the DEEPSEEK_API_KEY environment variable. It is
never written into a source file, so this module is safe to publish.
"""

from __future__ import annotations

from typing import Optional

from . import config

_client = None


def _get_client():
    """Create the OpenAI-compatible client on first use."""
    global _client
    if _client is not None:
        return _client

    if not config.LLM_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your key, "
            "or export DEEPSEEK_API_KEY=<key> before starting the service."
        )

    from openai import OpenAI

    _client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    return _client


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the context does not contain the answer, say that you don't know."
)


def generate_answer(prompt: str) -> Optional[str]:
    """Return the model's answer, or None when the API call fails."""
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=config.LLM_TEMPERATURE,
        )
    except Exception as exc:  # network / quota / auth errors should not crash the API
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    return response.choices[0].message.content.strip()
