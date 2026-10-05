"""Ingestion jobs and query logs (admins and KB editors)."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from psycopg import Connection
from pydantic import BaseModel

from app.auth import Role, kb_access
from app.db import get_conn

router = APIRouter()


class Job(BaseModel):
    id: int
    kb_id: UUID
    document_id: UUID
    filename: str
    status: Literal["queued", "running", "done", "failed"]
    attempts: int
    error: str | None
    stats: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class Retrieved(BaseModel):
    chunk_id: int
    document_id: UUID
    vec_sim: float | None
    fts_rank: float | None
    fused: float


class QueryLog(BaseModel):
    id: int
    kb_id: UUID
    user_id: UUID | None
    user_email: str | None
    question: str
    answer: str | None
    insufficient: bool
    retrieved: list[Retrieved]
    model: str | None
    latency_ms: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    error: str | None
    rewritten_question: str | None
    history_turns: int
    created_at: datetime


# stats also carries the embedding model and fingerprint of the run (used by the reindex logic); the contract does
# not expose them.
JOB_SQL = (
    "SELECT j.id, j.kb_id, j.document_id, d.filename, j.status, j.attempts, j.error, "
    "j.stats - ARRAY['embedding_model', 'embedding_fingerprint'] AS stats, j.created_at, j.started_at, j.finished_at "
    "FROM ingestion_jobs j JOIN documents d ON d.id = j.document_id AND d.kb_id = j.kb_id"
)


@router.get("/kbs/{kb_id}/jobs", response_model=list[Job])
def list_jobs(
    kb_id: UUID,
    status: Literal["queued", "running", "done", "failed"] | None = None,
    document_id: UUID | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: Role = Depends(kb_access("editor")),
    conn: Connection = Depends(get_conn),
):
    return conn.execute(
        JOB_SQL + " WHERE j.kb_id = %(kb)s AND (%(status)s::text IS NULL OR j.status = %(status)s) "
        "AND (%(doc)s::uuid IS NULL OR j.document_id = %(doc)s) ORDER BY j.id DESC LIMIT %(limit)s OFFSET %(offset)s",
        {"kb": kb_id, "status": status, "doc": document_id, "limit": limit, "offset": offset},
    ).fetchall()


@router.get("/kbs/{kb_id}/queries", response_model=list[QueryLog])
def list_queries(
    kb_id: UUID,
    insufficient: bool | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    _: Role = Depends(kb_access("editor")),
    conn: Connection = Depends(get_conn),
):
    return conn.execute(
        "SELECT q.id, q.kb_id, q.user_id, u.email AS user_email, q.question, q.answer, q.insufficient, q.retrieved, "
        "q.model, q.latency_ms, q.prompt_tokens, q.completion_tokens, q.error, q.rewritten_question, q.history_turns, "
        "q.created_at "
        "FROM query_logs q LEFT JOIN users u ON u.id = q.user_id "
        "WHERE q.kb_id = %(kb)s AND (%(ins)s::boolean IS NULL OR q.insufficient = %(ins)s) "
        "ORDER BY q.id DESC LIMIT %(limit)s OFFSET %(offset)s",
        {"kb": kb_id, "ins": insufficient, "limit": limit, "offset": offset},
    ).fetchall()
