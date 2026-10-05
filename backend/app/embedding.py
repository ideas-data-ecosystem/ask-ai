"""Embedding client for an OpenAI-compatible /embeddings endpoint, with the canary guard.

Shared by ingestion (mode "document") and queries (mode "query"). Every embedding request carries a fixed
canary sentence as its first input; its vector must match the knowledge base's stored canary for that mode
(embedding_canary for documents, embedding_query_canary for questions), so a router that silently falls back to
another model (even one with the same dimension) fails loudly instead of corrupting the index or the answers.

Error messages are safe to show to non-admins (they land in documents.error and query_logs.error): the provider's
response body and exception text only go to the server log.
"""

import functools
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from app.config import Settings, get_settings

log = logging.getLogger("app.embedding")

CANARY_TEXT = "Pegawai negeri sipil wajib menaati ketentuan peraturan perundang-undangan."
CANARY_MIN_COSINE = 0.99
TIMEOUT_SECONDS = 60.0
BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0)  # waits before retries 1..4 on 429, 5xx and transport errors

Mode = Literal["document", "query"]


class EmbeddingError(Exception):
    """The embedding call failed. Transient unless `permanent` (retrying cannot help)."""

    permanent = False


class EmbeddingConfigError(EmbeddingError):
    permanent = True


class EmbeddingMismatch(EmbeddingError):
    """Vectors do not belong to the expected model: canary drift or wrong dimension."""

    permanent = True


@dataclass(frozen=True)
class Embedded:
    vectors: list[list[float]]
    canary: list[float]  # this mode's canary vector (the reference to store on a KB's first index)
    tokens: int


@functools.cache
def _client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT_SECONDS)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _retry_wait(delay: float, retry_after: str | None) -> float:
    try:
        return min(max(float(retry_after), delay), 30.0) if retry_after else delay
    except ValueError:
        return delay


def _call(inputs: list[str], mode: Mode, settings: Settings) -> tuple[list[list[float]], int]:
    """One POST /embeddings with retry and backoff. Returns (vectors in input order, tokens)."""
    if not (settings.embedding_base_url and settings.embedding_model):
        raise EmbeddingConfigError("EMBEDDING_BASE_URL and EMBEDDING_MODEL are not configured")
    prefix = settings.embedding_doc_prefix if mode == "document" else settings.embedding_query_prefix
    extra = settings.embedding_doc_extra_body if mode == "document" else settings.embedding_query_extra_body
    body: dict[str, Any] = {"model": settings.embedding_model, "input": [prefix + t for t in inputs], **extra}
    headers = {"Authorization": f"Bearer {settings.embedding_api_key.get_secret_value()}"}
    url = settings.embedding_base_url.rstrip("/") + "/embeddings"

    attempts = len(BACKOFF_SECONDS) + 1
    for delay in (*BACKOFF_SECONDS, None):
        retry_after = None
        try:
            r = _client().post(url, json=body, headers=headers)
        except httpx.TransportError as e:
            problem, detail = type(e).__name__, f"{type(e).__name__}: {e}"
        else:
            if r.status_code < 400:
                try:
                    payload = r.json()
                    vectors = [d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"])]
                    tokens = int((payload.get("usage") or {}).get("total_tokens") or 0)
                except (ValueError, KeyError, TypeError) as e:
                    log.error("unexpected /embeddings response: %s: %s", type(e).__name__, e)
                    raise EmbeddingError("Unexpected /embeddings response (malformed payload)") from e
                if len(vectors) != len(inputs):
                    raise EmbeddingError(f"Sent {len(inputs)} inputs, got {len(vectors)} vectors")
                return vectors, tokens
            problem = f"HTTP {r.status_code}"
            detail = f"{problem}: {r.text[:300]}"
            if r.status_code != 429 and r.status_code < 500:
                log.error("embedding request rejected: %s", detail)
                raise EmbeddingError(f"Embedding request rejected ({problem})")
            retry_after = r.headers.get("Retry-After")
        if delay is None:
            log.error("embedding request failed after %d attempts: %s", attempts, detail)
            raise EmbeddingError(f"Embedding request failed after {attempts} attempts ({problem})")
        wait = _retry_wait(delay, retry_after)
        log.warning("embedding request failed (%s); retrying in %.0fs", detail, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def _same_space(s: Settings) -> bool:
    return (s.embedding_doc_prefix, s.embedding_doc_extra_body) == (
        s.embedding_query_prefix,
        s.embedding_query_extra_body,
    )


def _check_dim(vec: Sequence[float], settings: Settings) -> None:
    if len(vec) != settings.embedding_dim:
        raise EmbeddingMismatch(f"Embedding has {len(vec)} dimensions but EMBEDDING_DIM is {settings.embedding_dim}")


def embed(
    texts: Sequence[str],
    mode: Mode = "document",
    canary: Sequence[float] | None = None,
    settings: Settings | None = None,
) -> Embedded:
    """Embed `texts` in `mode`. Every request carries CANARY_TEXT as input[0] in that same mode.

    `canary` is the knowledge base's stored canary for this mode (None on its first index, in which case the first
    canary seen becomes the reference). Raises EmbeddingMismatch when the canary drifts below cosine 0.99 or a
    vector's length differs from EMBEDDING_DIM, EmbeddingError for anything else.
    """
    s = settings or get_settings()
    # psycopg returns the stored canary as a pgvector Vector
    ref = (
        (canary.to_list() if hasattr(canary, "to_list") else [float(x) for x in canary]) if canary is not None else None
    )
    tokens = 0
    vectors: list[list[float]] = []
    for i in range(0, max(len(texts), 1), s.embedding_batch):  # at least one request, so the canary is always checked
        got, used = _call([CANARY_TEXT, *texts[i : i + s.embedding_batch]], mode, s)
        tokens += used
        seen = got.pop(0)
        _check_dim(seen, s)
        if ref is None:
            ref = [float(x) for x in seen]
        else:
            sim = _cosine(seen, ref) if len(ref) == len(seen) else 0.0
            if sim < CANARY_MIN_COSINE:
                raise EmbeddingMismatch(
                    f"Embedding canary drifted (cosine {sim:.3f} < {CANARY_MIN_COSINE}): "
                    "the embedding endpoint is not serving the model this knowledge base was indexed with"
                )
        for v in got:
            _check_dim(v, s)
        vectors += got
    assert ref is not None
    return Embedded(vectors, ref, tokens)


def query_canary(doc_canary: Sequence[float], settings: Settings | None = None) -> list[float]:
    """The canary vector in query mode, to store next to the document-mode one on a KB's first index.

    Same prefix and extra body in both modes means the same vector, so no request is needed; otherwise one
    query-mode request captures it.
    """
    s = settings or get_settings()
    if _same_space(s):
        return [float(x) for x in doc_canary]
    return embed([], "query", canary=None, settings=s).canary
