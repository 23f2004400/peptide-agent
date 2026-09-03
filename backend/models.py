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


# Reasoning models (e.g. DeepSeek-R1) emit a <think>...</think> block before
# the sequence; cutting generation at the first newline (as done below for
# every other model) would truncate that reasoning. IS_REASONING_MODEL=true
# forces this regardless of MODEL_NAME; otherwise inferred from the name.
_REASONING_MODEL_MARKERS = ("deepseek-r1", "qwq", "r1-distill")


def is_reasoning_model() -> bool:
    model_name = get_model_name().lower()
    return (
        os.environ.get("IS_REASONING_MODEL", "false").lower() == "true"
        or any(marker in model_name for marker in _REASONING_MODEL_MARKERS)
    )


EMPTY_RETRY_ATTEMPTS = 4


def generate(
    prompt: str,
    max_tokens: int = 512,
    system: str | None = None,
    assistant_primer: str = "",
    temperature: float | None = None,
) -> str:
    """
    Call the LLM and return the raw text response.

    assistant_primer: if non-empty, added as a partial assistant message so
    the model is forced to CONTINUE from that text rather than starting fresh.
    This prevents the model from outputting "Explanation:" style preambles —
    when the assistant turn already begins with "KLL", the model continues
    with more amino acid letters.  The primer is prepended to the returned
    string so the caller sees the complete sequence.

    temperature: optional sampling temperature forwarded to the API. Left
    unset (None) by default so existing call sites keep the server's default
    behavior unchanged; callers that want varied/diverse completions across
    repeated calls (e.g. the agent's edit step) can pass an explicit value.
    """
    client = _get_client()
    model = get_model_name()

    user_content = f"{system}\n\n{prompt}" if system else prompt
    messages: list[dict] = [{"role": "user", "content": user_content}]
    if assistant_primer:
        messages.append({"role": "assistant", "content": assistant_primer})

    create_kwargs = dict(model=model, messages=messages, max_tokens=max_tokens)
    if temperature is not None:
        create_kwargs["temperature"] = temperature
    if not is_reasoning_model():
        create_kwargs["stop"] = ["\n"]

    for attempt in range(1 + EMPTY_RETRY_ATTEMPTS):
        response = client.chat.completions.create(**create_kwargs)
        tokens = response.usage.completion_tokens if response.usage else 0
        if tokens > 0:
            content = response.choices[0].message.content or ""
            logger.debug(
                "model ok (internal attempt %d/%d): tokens=%d content=%r",
                attempt + 1, 1 + EMPTY_RETRY_ATTEMPTS, tokens, content[:80],
            )
            return assistant_primer + content
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
