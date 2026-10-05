"""LLM client: one `complete()` over two wire protocols (OpenAI Chat Completions, Anthropic Messages).

No temperature or other sampling parameters are sent (some models reject them); per-model knobs go in
LLM_EXTRA_BODY, which is passed through as the SDK's `extra_body`.
"""

import functools
import logging

import anthropic
import openai

from app.config import Settings, get_settings

TIMEOUT_SECONDS = 60.0

log = logging.getLogger("app.llm")


class LLMError(Exception):
    """The model call failed or the model refused. The message is safe to store in query_logs.error (readable by
    KB editors); the provider's own error text only goes to the server log."""


@functools.lru_cache
def _openai(base_url: str, api_key: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=base_url, api_key=api_key or "unused", timeout=TIMEOUT_SECONDS, max_retries=1)


@functools.lru_cache
def _anthropic(base_url: str, api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=base_url, api_key=api_key or "unused", timeout=TIMEOUT_SECONDS, max_retries=1)


def _finished(text: str, hit_limit: bool, max_tokens: int) -> str:
    """A reply cut off at the token limit (reasoning models can spend all of it before answering) or with no text
    is a failure, not an answer and not a refusal: rag.ask turns it into a 502 instead of "insufficient"."""
    if hit_limit:
        what = "a truncated answer" if text.strip() else "no answer"
        raise LLMError(f"The language model hit the token limit (LLM_MAX_TOKENS={max_tokens}) and returned {what}")
    if not text.strip():
        raise LLMError("The language model returned an empty answer")
    return text


def complete(system: str, user: str, settings: Settings | None = None) -> tuple[str, dict[str, int]]:
    """Returns (text, {"prompt_tokens": int, "completion_tokens": int}). Raises LLMError."""
    s = settings or get_settings()
    if not (s.llm_base_url and s.llm_model):
        raise LLMError("LLM_BASE_URL and LLM_MODEL are not configured")
    key = s.llm_api_key.get_secret_value()
    extra = s.llm_extra_body or None
    try:
        if s.llm_protocol == "openai":
            r = _openai(s.llm_base_url, key).chat.completions.create(
                model=s.llm_model,
                max_tokens=s.llm_max_tokens,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body=extra,
            )
            choice = r.choices[0]
            if choice.finish_reason == "content_filter" or getattr(choice.message, "refusal", None):
                raise LLMError("The model refused to answer")
            usage = {
                "prompt_tokens": r.usage.prompt_tokens if r.usage else 0,
                "completion_tokens": r.usage.completion_tokens if r.usage else 0,
            }
            return _finished(choice.message.content or "", choice.finish_reason == "length", s.llm_max_tokens), usage
        r = _anthropic(s.llm_base_url, key).messages.create(
            model=s.llm_model,
            max_tokens=s.llm_max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            extra_body=extra,
        )
        if r.stop_reason == "refusal":
            raise LLMError("The model refused to answer")
        text = "".join(b.text for b in r.content if b.type == "text")
        usage = {"prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens}
        return _finished(text, r.stop_reason == "max_tokens", s.llm_max_tokens), usage
    except LLMError:
        raise
    except (openai.OpenAIError, anthropic.AnthropicError) as e:
        log.error("LLM request failed: %s: %s", type(e).__name__, e)
        status = getattr(e, "status_code", None)
        why = f"HTTP {status}" if status else type(e).__name__
        raise LLMError(f"The language model request failed ({why})") from e
