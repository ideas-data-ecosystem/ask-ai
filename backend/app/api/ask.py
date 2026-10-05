"""POST /kbs/{kb_id}/ask: question answering over one knowledge base, with optional earlier turns as context."""

import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app import rag
from app.auth import COOKIE_NAME, authenticate, authorize_kb
from app.config import Settings, app_settings

router = APIRouter()

MAX_ASKS_IN_FLIGHT = 2  # per user; each one can sit minutes on the embedding and LLM calls
_in_flight: Counter[UUID] = Counter()
_in_flight_lock = threading.Lock()


@contextmanager
def _ask_slot(user_id: UUID) -> Iterator[None]:
    """429 when the user already has MAX_ASKS_IN_FLIGHT questions being answered.

    ponytail: counted in this process only, so N uvicorn workers allow N times the limit. A shared counter
    (a Postgres advisory lock or Redis) replaces it if the api ever runs with several workers.
    """
    with _in_flight_lock:
        if _in_flight[user_id] >= MAX_ASKS_IN_FLIGHT:
            raise HTTPException(429, "Too many questions in progress; wait for one to finish")
        _in_flight[user_id] += 1
    try:
        yield
    finally:
        with _in_flight_lock:
            _in_flight[user_id] -= 1
            if _in_flight[user_id] <= 0:
                del _in_flight[user_id]


def _clean(text: str) -> str:
    """Trimmed, without NUL characters: Postgres text cannot store them (the query log write would fail)."""
    return text.replace("\x00", "").strip()


class HistoryTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo such as "answr" is a 422, not a silently empty answer

    question: str
    answer: str


class AskIn(BaseModel):
    question: str
    # Earlier turns, the most recent last. Absent, null or [] is a single-turn question.
    history: Annotated[list[HistoryTurn], Field(max_length=rag.MAX_HISTORY_TURNS)] | None = None


class CitationOut(BaseModel):
    n: int
    document_id: UUID
    filename: str
    category: str | None
    page_start: int | None
    page_end: int | None
    heading: str | None
    snippet: str


class AskOut(BaseModel):
    answer: str
    insufficient: bool
    citations: list[CitationOut]


@router.post("/kbs/{kb_id}/ask", response_model=AskOut)
def ask(kb_id: UUID, body: AskIn, request: Request, settings: Settings = Depends(app_settings)):
    """Takes no `get_conn` or `kb_access` dependency: those would hold a pooled connection for the whole request,
    LLM latency included, so the session and membership checks use a connection that is released right away and
    `rag.ask` borrows one only while it reads or writes the database."""
    pool = request.app.state.pool
    with pool.connection() as conn:
        user = authenticate(conn, request.cookies.get(COOKIE_NAME))
        authorize_kb(conn, user, kb_id, "viewer")
    question = _clean(body.question)
    if not 1 <= len(question) <= settings.max_question_chars:
        raise HTTPException(422, f"The question must be between 1 and {settings.max_question_chars} characters")
    # Over-long turns are cut, not refused: an answer this server wrote can be longer than the cap.
    history = [
        (_clean(t.question)[: settings.max_question_chars], _clean(t.answer)[: rag.HISTORY_ANSWER_CHARS])
        for t in body.history or []
    ]
    if any(not q for q, _ in history):
        raise HTTPException(422, "Every history turn needs a non-empty question")
    with _ask_slot(user.id):
        return rag.ask(pool, settings, kb_id, user.id, question, history)
