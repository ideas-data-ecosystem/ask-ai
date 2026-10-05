"""Test doubles for the embedding and LLM clients (no network) and a helper that marks a KB as indexed."""

from types import SimpleNamespace

from app import embedding, llm
from app.config import fingerprint

from .conftest import DIM

CANARY_VEC = [1.0] + [0.0] * (DIM - 1)
ORTHOGONAL_VEC = [1.0, -1.0] * (DIM // 2)  # cosine 0 against the all-equal vectors of conftest.make_kb
MODEL = "fake-embed"


def install_embed(monkeypatch, query_vec=None, canary=None, by_text=None):
    """Replace the HTTP call of app.embedding; embed() itself (batching, canary guard, dimension check) stays real.

    The returned state controls the vectors: `canary` for the canary sentence, `by_text[text]` for a listed text,
    `query_vec` for every other text.
    """
    state = SimpleNamespace(
        query_vec=query_vec or [1.0] * DIM, canary=canary or CANARY_VEC, by_text=by_text or {}, calls=[]
    )

    def vector(text):
        return state.canary if text == embedding.CANARY_TEXT else state.by_text.get(text, state.query_vec)

    def fake_call(inputs, mode, settings):
        state.calls.append((list(inputs), mode))
        return [list(vector(t)) for t in inputs], len(inputs)

    monkeypatch.setattr(embedding, "_call", fake_call)
    return state


def install_llm(monkeypatch, reply="Jawaban [1]."):
    """`reply` is the text of every call, or a function (system, user) -> text. `error` makes every call fail.
    `calls` records (system, user) and `settings` the Settings of each call."""
    state = SimpleNamespace(reply=reply, error=None, calls=[], settings=[])

    def complete(system, user, settings=None):
        state.calls.append((system, user))
        state.settings.append(settings)
        if state.error:
            raise llm.LLMError(state.error)
        text = state.reply(system, user) if callable(state.reply) else state.reply
        return text, {"prompt_tokens": 10, "completion_tokens": 5}

    monkeypatch.setattr(llm, "complete", complete)
    return state


def rag_settings(settings, **updates):
    # Reset everything the developer's .env may customise, so the fingerprint matches mark_indexed's default setup.
    defaults = {
        "embedding_model": MODEL,
        "embedding_doc_prefix": "",
        "embedding_query_prefix": "",
        "embedding_doc_extra_body": {},
        "embedding_query_extra_body": {},
        "llm_model": "fake-llm",
        "llm_base_url": "http://llm.invalid",
        "llm_extra_body": {},
        "llm_rewrite_extra_body": {},
    }
    return settings.model_copy(update={**defaults, **updates})


def mark_indexed(pool, kb, model=MODEL, canary=CANARY_VEC):
    """The KB as the worker leaves it after indexing with `model` (default setup: no prefixes, no extra bodies)."""
    from pgvector import Vector

    vec = Vector(canary) if canary else None
    with pool.connection() as conn:
        conn.execute(
            "UPDATE knowledge_bases SET embedding_model = %s, embedding_fingerprint = %s, embedding_canary = %s, "
            "embedding_query_canary = %s WHERE id = %s",
            [model, fingerprint(model), vec, vec, kb.id],
        )


def drain_jobs(pool, settings) -> int:
    """Run the worker loop body until the queue is empty; returns how many claims were processed."""
    from app.ingest import claim_job, process_job

    n = 0
    while True:
        with pool.connection() as conn:
            job = claim_job(conn)
        if job is None:
            return n
        process_job(pool, job, settings)
        n += 1


def clear_queue(pool) -> None:
    """The test database is shared by the whole session: drop jobs other tests left behind."""
    with pool.connection() as conn:
        conn.execute("DELETE FROM ingestion_jobs WHERE status IN ('queued', 'running')")
