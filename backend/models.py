"""
OpenBioLLM connection via OpenAI-compatible vLLM endpoint.
"""

from __future__ import annotations
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        gateway_url = os.environ.get("GATEWAY_URL", "").rstrip("/")
        api_key = os.environ.get("API_KEY", "sk-bhaskera-alice")
        if not gateway_url:
            raise RuntimeError("GATEWAY_URL environment variable is not set")
        _client = OpenAI(base_url=f"{gateway_url}", api_key=api_key)
    return _client


def get_model_name() -> str:
    return os.environ.get("MODEL_NAME", "aaditya/OpenBioLLM-Llama3-8B")


EMPTY_RETRY_ATTEMPTS = 4


def generate(
    prompt: str,
    max_tokens: int = 512,
    system: str | None = None,
) -> str:
    """
    Call the LLM and return the raw text response.

    The OpenBioLLM-Llama3-8B deployment has no chat_template configured on
    its tokenizer, so it falls back to a plain role/content format with no
    assistant-priming tokens. This makes the model frequently predict EOS
    immediately (empty completion). Per the server maintainer, the
    per-request `temperature` field is not wired through to the sampling
    params — it's fixed by configs/openbiollm.yaml's inference.temperature
    and only changes if that file is edited and the server restarted. So:
      - no system role is sent (folded into the user message instead)
      - `temperature` is accepted for API compatibility but never sent to
        the server, since it has no effect there
      - empty completions are retried plainly (do_sample=true still gives
        each retry a fresh chance even at a fixed temperature)
    """
    client = _get_client()
    model = get_model_name()

    user_content = f"{system}\n\n{prompt}" if system else prompt
    messages = [{"role": "user", "content": user_content}]

    for attempt in range(1 + EMPTY_RETRY_ATTEMPTS):
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
        )
        tokens = response.usage.completion_tokens if response.usage else 0
        if tokens > 0:
            content = response.choices[0].message.content or ""
            logger.debug(
                "model ok (internal attempt %d/%d): tokens=%d content=%r",
                attempt + 1, 1 + EMPTY_RETRY_ATTEMPTS, tokens, content[:80],
            )
            return content
        logger.warning(
            "empty response (internal attempt %d/%d) — model returned 0 tokens",
            attempt + 1, 1 + EMPTY_RETRY_ATTEMPTS,
        )

    logger.error("all %d internal retries exhausted — returning empty string", 1 + EMPTY_RETRY_ATTEMPTS)
    return ""


def health_check() -> dict:
    """Verify connectivity to the gateway."""
    gateway_url = os.environ.get("GATEWAY_URL", "NOT SET")
    model = get_model_name()
    try:
        _ = _get_client()
        return {"status": "ok", "model": model, "gateway": gateway_url}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "model": model, "gateway": gateway_url}
