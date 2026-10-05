"""Knowledge bases: CRUD, stats and members."""

import re
import shutil
import unicodedata
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from psycopg import Connection
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, StringConstraints

from app.auth import Role, User, current_user, kb_access, require_admin
from app.config import Settings, app_settings
from app.db import get_conn, kb_scope
from app.ingest import maybe_switch_model, stray_setup

router = APIRouter()

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$", max_length=64)]
Description = Annotated[str, StringConstraints(max_length=2000)]


def slugify(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:64].strip("-")


class KBIn(BaseModel):
    name: Name
    description: Description = ""
    slug: Slug | None = None  # derived from name when absent


class KBPatch(BaseModel):
    """Absent or null fields are left unchanged."""

    name: Name | None = None
    description: Description | None = None
    status: Literal["active", "archived"] | None = None


class KB(BaseModel):
    id: UUID
    slug: str
    name: str
    description: str
    status: Literal["active", "archived"]
    embedding_model: str | None
    role: Role  # the caller's role on this KB
    doc_count: int
    last_indexed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class StatusCounts(BaseModel):
    queued: int
    processing: int
    ready: int
    failed: int
    total: int


class JobCounts(BaseModel):
    queued: int
    running: int
    done: int
    failed: int


class KBStats(BaseModel):
    kb_id: UUID
    embedding_model: str | None
    llm_model: str | None  # global LLM_MODEL setting, not per KB
    documents: StatusCounts
    jobs: JobCounts
    chunks: int
    pages: int
    ocr_pages: int
    last_indexed_at: datetime | None


class Member(BaseModel):
    user_id: UUID
    email: str
    name: str
    role: Literal["editor", "viewer"]


class MemberIn(BaseModel):
    role: Literal["editor", "viewer"]


_KB_SQL = """
SELECT kb.id, kb.slug, kb.name, kb.description, kb.status, kb.embedding_model,
       CASE WHEN %(admin)s THEN 'admin' ELSE m.role END AS role,
       (SELECT count(*) FROM documents d WHERE d.kb_id = kb.id) AS doc_count,
       (SELECT max(d.indexed_at) FROM documents d WHERE d.kb_id = kb.id) AS last_indexed_at,
       kb.created_at, kb.updated_at
FROM knowledge_bases kb
LEFT JOIN kb_members m ON m.kb_id = kb.id AND m.user_id = %(uid)s
"""


def ensure_active(conn: Connection, kb_id: UUID) -> None:
    """409 when the KB is archived (it still serves reads, but takes no uploads, reindex or questions)."""
    row = conn.execute("SELECT status FROM knowledge_bases WHERE id = %s", [kb_id]).fetchone()
    if row and row["status"] == "archived":
        raise HTTPException(409, "Knowledge base is archived")


def _get_kb(conn: Connection, user: User, kb_id: UUID) -> dict:
    return conn.execute(
        _KB_SQL + " WHERE kb.id = %(kb)s", {"admin": user.is_admin, "uid": user.id, "kb": kb_id}
    ).fetchone()  # type: ignore[return-value]


@router.get("/kbs", response_model=list[KB])
def list_kbs(user: User = Depends(current_user), conn: Connection = Depends(get_conn)):
    """Admins see every KB; others see the KBs they are a member of."""
    return conn.execute(
        _KB_SQL + " WHERE %(admin)s OR m.user_id IS NOT NULL ORDER BY kb.name",
        {"admin": user.is_admin, "uid": user.id},
    ).fetchall()


@router.post("/kbs", status_code=201, response_model=KB)
def create_kb(body: KBIn, admin: User = Depends(require_admin), conn: Connection = Depends(get_conn)):
    slug = body.slug or slugify(body.name)
    if not slug:
        raise HTTPException(422, "Cannot derive a slug from the name; pass slug explicitly")
    try:
        kb_id = conn.execute(
            "INSERT INTO knowledge_bases (slug, name, description) VALUES (%s, %s, %s) RETURNING id",
            [slug, body.name, body.description],
        ).fetchone()["id"]  # type: ignore[index]
    except UniqueViolation:
        raise HTTPException(409, f"Slug '{slug}' is already in use") from None
    return _get_kb(conn, admin, kb_id)


@router.get("/kbs/{kb_id}", response_model=KB)
def get_kb(
    kb_id: UUID,
    _: Role = Depends(kb_access("viewer")),
    user: User = Depends(current_user),
    conn: Connection = Depends(get_conn),
):
    return _get_kb(conn, user, kb_id)


@router.patch("/kbs/{kb_id}", response_model=KB)
def patch_kb(
    kb_id: UUID,
    body: KBPatch,
    _: Role = Depends(kb_access("admin")),
    user: User = Depends(current_user),
    conn: Connection = Depends(get_conn),
):
    changes = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if changes:
        sets = ", ".join(f"{col} = %({col})s" for col in changes)  # keys come from KBPatch fields
        conn.execute(
            f"UPDATE knowledge_bases SET {sets}, updated_at = now() WHERE id = %(id)s", {**changes, "id": kb_id}
        )
    return _get_kb(conn, user, kb_id)


@router.delete("/kbs/{kb_id}", status_code=204)
def delete_kb(
    kb_id: UUID,
    _: Role = Depends(kb_access("admin")),
    conn: Connection = Depends(get_conn),
    settings: Settings = Depends(app_settings),
):
    """Cascades to documents, chunks, jobs, logs and members; then removes the KB's uploaded files."""
    conn.execute("DELETE FROM knowledge_bases WHERE id = %s", [kb_id])  # autocommit: committed here
    shutil.rmtree(settings.storage_dir / "uploads" / str(kb_id), ignore_errors=True)  # seeded files are never touched


@router.get("/kbs/{kb_id}/stats", response_model=KBStats)
def kb_stats(
    kb_id: UUID,
    _: Role = Depends(kb_access("viewer")),
    conn: Connection = Depends(get_conn),
    settings: Settings = Depends(app_settings),
):
    kb = conn.execute("SELECT embedding_model FROM knowledge_bases WHERE id = %s", [kb_id]).fetchone()
    docs = conn.execute(
        "SELECT count(*) AS total, "
        "count(*) FILTER (WHERE status = 'queued') AS queued, "
        "count(*) FILTER (WHERE status = 'processing') AS processing, "
        "count(*) FILTER (WHERE status = 'ready') AS ready, "
        "count(*) FILTER (WHERE status = 'failed') AS failed, "
        "coalesce(sum(page_count), 0) AS pages, coalesce(sum(ocr_pages), 0) AS ocr_pages, "
        "max(indexed_at) AS last_indexed_at FROM documents WHERE kb_id = %s",
        [kb_id],
    ).fetchone()
    jobs = conn.execute(
        "SELECT count(*) FILTER (WHERE status = 'queued') AS queued, "
        "count(*) FILTER (WHERE status = 'running') AS running, "
        "count(*) FILTER (WHERE status = 'done') AS done, "
        "count(*) FILTER (WHERE status = 'failed') AS failed FROM ingestion_jobs WHERE kb_id = %s",
        [kb_id],
    ).fetchone()
    with kb_scope(conn, kb_id):
        chunks = conn.execute("SELECT count(*) AS n FROM document_chunks WHERE kb_id = %s", [kb_id]).fetchone()["n"]  # type: ignore[index]
    return {
        "kb_id": kb_id,
        "embedding_model": kb["embedding_model"],  # type: ignore[index]
        "llm_model": settings.llm_model or None,
        "documents": docs,
        "jobs": jobs,
        "chunks": chunks,
        "pages": docs["pages"],  # type: ignore[index]
        "ocr_pages": docs["ocr_pages"],  # type: ignore[index]
        "last_indexed_at": docs["last_indexed_at"],  # type: ignore[index]
    }


@router.post("/kbs/{kb_id}/reindex", status_code=202)
def reindex_kb(
    kb_id: UUID,
    _: Role = Depends(kb_access("editor")),
    conn: Connection = Depends(get_conn),
    settings: Settings = Depends(app_settings),
) -> dict[str, int]:
    """Queue a job for every document that has none queued or running.

    When the configured embedding setup (settings.embedding_fingerprint: model, prefixes, extra bodies) differs from
    the one the KB was indexed with, the stored canaries are reset so the first job captures the new setup's. The KB
    keeps answering "reindex required" (its stored setup stays the old one) until every document has been
    re-embedded, see ingest.maybe_switch_model.

    The same applies when the stored setup equals the configured one but some document's latest finished job used
    another (the operator reverted EMBEDDING_MODEL partway through a reindex): the KB is pointed back at that other
    setup first, so the forward path above applies and the next full pass heals it.
    """
    ensure_active(conn, kb_id)
    if not settings.embedding_model:
        raise HTTPException(409, "EMBEDDING_MODEL is not configured")
    fingerprint = settings.embedding_fingerprint
    reset = "embedding_canary = NULL, embedding_query_canary = NULL"
    with conn.transaction():
        kb = conn.execute(
            "SELECT embedding_fingerprint FROM knowledge_bases WHERE id = %s FOR UPDATE", [kb_id]
        ).fetchone()
        has_docs = conn.execute("SELECT EXISTS (SELECT 1 FROM documents WHERE kb_id = %s) AS x", [kb_id]).fetchone()[
            "x"
        ]  # type: ignore[index]
        if not has_docs:  # nothing is indexed, so there is no old vector space to protect
            conn.execute(
                f"UPDATE knowledge_bases SET embedding_model = NULL, embedding_fingerprint = NULL, {reset} WHERE id = %s",
                [kb_id],
            )
        elif kb["embedding_fingerprint"] not in (None, fingerprint):  # type: ignore[index]
            conn.execute(f"UPDATE knowledge_bases SET {reset} WHERE id = %s", [kb_id])
        elif stray := stray_setup(conn, kb_id, fingerprint):
            conn.execute(
                f"UPDATE knowledge_bases SET embedding_model = %s, embedding_fingerprint = %s, {reset} WHERE id = %s",
                [stray["model"], stray["fingerprint"], kb_id],
            )
        enqueued = conn.execute(
            "WITH d AS (UPDATE documents SET status = 'queued', error = NULL WHERE kb_id = %(kb)s AND NOT EXISTS "
            "(SELECT 1 FROM ingestion_jobs j WHERE j.document_id = documents.id AND j.kb_id = documents.kb_id "
            "AND j.status IN ('queued', 'running')) RETURNING id) "
            "INSERT INTO ingestion_jobs (kb_id, document_id) SELECT %(kb)s, id FROM d",
            {"kb": kb_id},
        ).rowcount
    maybe_switch_model(conn, kb_id, settings)  # every document may already be on the new setup
    return {"enqueued": enqueued}


@router.get("/kbs/{kb_id}/members", response_model=list[Member])
def list_members(kb_id: UUID, _: Role = Depends(kb_access("admin")), conn: Connection = Depends(get_conn)):
    return conn.execute(
        "SELECT m.user_id, u.email, u.name, m.role FROM kb_members m JOIN users u ON u.id = m.user_id "
        "WHERE m.kb_id = %s ORDER BY u.email",
        [kb_id],
    ).fetchall()


@router.put("/kbs/{kb_id}/members/{user_id}", response_model=Member)
def put_member(
    kb_id: UUID,
    user_id: UUID,
    body: MemberIn,
    _: Role = Depends(kb_access("admin")),
    conn: Connection = Depends(get_conn),
):
    """Add the user to the KB or change their role."""
    row = conn.execute(
        "WITH up AS (INSERT INTO kb_members (kb_id, user_id, role) "
        "SELECT %(kb)s, u.id, %(role)s FROM users u WHERE u.id = %(user)s "
        "ON CONFLICT (kb_id, user_id) DO UPDATE SET role = EXCLUDED.role RETURNING user_id, role) "
        "SELECT up.user_id, u.email, u.name, up.role FROM up JOIN users u ON u.id = up.user_id",
        {"kb": kb_id, "user": user_id, "role": body.role},
    ).fetchone()
    if row is None:
        raise HTTPException(404, "User not found")
    return row


@router.delete("/kbs/{kb_id}/members/{user_id}", status_code=204)
def delete_member(
    kb_id: UUID, user_id: UUID, _: Role = Depends(kb_access("admin")), conn: Connection = Depends(get_conn)
):
    if conn.execute("DELETE FROM kb_members WHERE kb_id = %s AND user_id = %s", [kb_id, user_id]).rowcount == 0:
        raise HTTPException(404, "Not a member")
