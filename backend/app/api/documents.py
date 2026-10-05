"""Documents: list, upload, delete, reindex and the file download."""

import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from psycopg import Connection
from psycopg_pool import ConnectionPool
from pydantic import BaseModel
from starlette.datastructures import UploadFile  # what the form parser makes (fastapi's is a subclass)

from app.api.kbs import ensure_active
from app.api.logs import JOB_SQL, Job
from app.auth import COOKIE_NAME, Role, authenticate, authorize_kb, kb_access
from app.config import Settings, app_settings
from app.db import get_conn
from app.ingest import ALLOWED_TYPES, DOC_COLUMNS, PermanentError, add_document, maybe_switch_model, resolve_stored

router = APIRouter()

MAX_CATEGORY_CHARS = 200
MAX_FILENAME_CHARS = 255
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class Document(BaseModel):
    id: UUID
    kb_id: UUID
    filename: str
    category: str | None
    mime: str | None
    size_bytes: int | None
    sha256: str
    status: Literal["queued", "processing", "ready", "failed"]
    page_count: int | None
    ocr_pages: int
    chunk_count: int
    error: str | None
    indexed_at: datetime | None
    created_at: datetime


def _display_filename(raw: str | None) -> str:
    """The client's filename as display data only: directory parts and control characters removed."""
    name = _CONTROL.sub("", os.path.basename((raw or "").replace("\\", "/"))).strip()
    return name[:MAX_FILENAME_CHARS]


def _content_matches(ext: str, head: bytes) -> bool:
    if ext == "pdf":
        return head.startswith(b"%PDF-")
    if ext == "docx":
        return head.startswith(b"PK")
    return b"\x00" not in head  # md, txt


def _write_upload(upload: UploadFile, dest: Path, max_bytes: int) -> tuple[str, int]:
    """Stream the upload to `dest` while hashing it; removes the partial file on any failure."""
    digest, size = hashlib.sha256(), 0
    try:
        with dest.open("wb") as out:
            while block := upload.file.read(1 << 20):
                size += len(block)
                if size > max_bytes:
                    raise HTTPException(413, f"File is larger than {max_bytes // (1 << 20)} MB")
                digest.update(block)
                out.write(block)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


@router.get("/kbs/{kb_id}/documents", response_model=list[Document])
def list_documents(
    kb_id: UUID,
    status: Literal["queued", "processing", "ready", "failed"] | None = None,
    category: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    _: Role = Depends(kb_access("viewer")),
    conn: Connection = Depends(get_conn),
):
    return conn.execute(
        f"SELECT {DOC_COLUMNS} FROM documents WHERE kb_id = %(kb)s "
        "AND (%(status)s::text IS NULL OR status = %(status)s) AND (%(cat)s::text IS NULL OR category = %(cat)s) "
        "ORDER BY category NULLS LAST, filename, id LIMIT %(limit)s OFFSET %(offset)s",
        {"kb": kb_id, "status": status, "cat": category, "limit": limit, "offset": offset},
    ).fetchall()


def _authorize_upload(request: Request, kb_id: UUID) -> None:
    """Session and editor role, on a connection that is released before the body is read (same helpers as ask)."""
    with request.app.state.pool.connection() as conn:
        authorize_kb(conn, authenticate(conn, request.cookies.get(COOKIE_NAME)), kb_id, "editor")


def _store_upload(
    pool: ConnectionPool, settings: Settings, kb_id: UUID, upload: UploadFile, category: str | None
) -> dict:
    with pool.connection() as conn:
        ensure_active(conn, kb_id)
        filename = _display_filename(upload.filename)
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in ALLOWED_TYPES:
            raise HTTPException(415, f"File type not allowed; use one of: {', '.join(sorted(ALLOWED_TYPES))}")
        category = _CONTROL.sub("", category or "").strip() or None
        if category and len(category) > MAX_CATEGORY_CHARS:
            raise HTTPException(422, f"Category is longer than {MAX_CATEGORY_CHARS} characters")

        doc_id = uuid4()
        relative = f"uploads/{kb_id}/{doc_id}.{ext}"
        dest = settings.storage_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        sha256, size = _write_upload(upload, dest, settings.max_upload_mb * (1 << 20))
        try:
            if size == 0:
                raise HTTPException(400, "The file is empty")
            with dest.open("rb") as f:
                if not _content_matches(ext, f.read(8192)):
                    raise HTTPException(415, "The file content does not match its extension")
            row = add_document(
                conn,
                kb_id=kb_id,
                doc_id=doc_id,
                filename=filename,
                category=category,
                storage_path=relative,
                mime=ALLOWED_TYPES[ext],
                size_bytes=size,
                sha256=sha256,
            )
            if row is None:
                raise HTTPException(409, "This file already exists in the knowledge base")
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        return row


@router.post(
    "/kbs/{kb_id}/documents",
    status_code=201,
    response_model=Document,
    openapi_extra={  # the form is read by hand (below), so FastAPI cannot derive the request body schema
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["file"],
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "category": {"type": "string", "maxLength": MAX_CATEGORY_CHARS},
                        },
                    }
                }
            },
        }
    },
)
async def upload_document(kb_id: UUID, request: Request, settings: Settings = Depends(app_settings)):
    """No client input reaches a path: the file is stored as uploads/<kb_id>/<doc_id>.<ext> with a server-made
    id and an extension from the allowlist. The original filename and the category are only database fields.

    Takes no `UploadFile`/`Form` parameter: FastAPI would parse the multipart body before running any dependency,
    so an anonymous caller could make the server spool a file to disk. Instead the session and the editor role are
    checked first, and only then is the form read, with a one-file, one-field cap.

    The body is capped before routing and auth by app.middleware.BodyLimitMiddleware (MAX_UPLOAD_MB + 1 MiB); the
    exact MAX_UPLOAD_MB limit on the file is enforced here while it streams to disk.
    """
    await run_in_threadpool(_authorize_upload, request, kb_id)
    async with request.form(max_files=1, max_fields=1) as form:  # the only fields are `file` and `category`
        upload, category = form.get("file"), form.get("category")
        if not isinstance(upload, UploadFile):
            raise RequestValidationError(
                [{"type": "missing", "loc": ("body", "file"), "msg": "Field required", "input": None}]
            )
        return await run_in_threadpool(
            _store_upload,
            request.app.state.pool,
            settings,
            kb_id,
            upload,
            category if isinstance(category, str) else None,
        )


@router.delete("/kbs/{kb_id}/documents/{doc_id}", status_code=204)
def delete_document(
    kb_id: UUID,
    doc_id: UUID,
    _: Role = Depends(kb_access("editor")),
    conn: Connection = Depends(get_conn),
    settings: Settings = Depends(app_settings),
):
    row = conn.execute(
        "DELETE FROM documents WHERE id = %s AND kb_id = %s RETURNING storage_path", [doc_id, kb_id]
    ).fetchone()  # autocommit: committed here, chunks and jobs cascade
    if row is None:
        raise HTTPException(404, "Document not found")
    # Only files the app created are ever removed; seeded files stay where they are.
    uploads = (settings.storage_dir / "uploads").resolve()
    path = (settings.storage_dir / row["storage_path"]).resolve()
    if path.is_relative_to(uploads):
        path.unlink(missing_ok=True)
    if settings.embedding_model:  # the deleted document may have been the last one still on the old model
        maybe_switch_model(conn, kb_id, settings)


@router.post("/kbs/{kb_id}/documents/{doc_id}/reindex", status_code=202, response_model=Job)
def reindex_document(
    kb_id: UUID, doc_id: UUID, _: Role = Depends(kb_access("editor")), conn: Connection = Depends(get_conn)
):
    ensure_active(conn, kb_id)
    with conn.transaction():
        if (
            conn.execute("SELECT 1 FROM documents WHERE id = %s AND kb_id = %s FOR UPDATE", [doc_id, kb_id]).fetchone()
            is None
        ):
            raise HTTPException(404, "Document not found")
        if conn.execute(
            "SELECT 1 FROM ingestion_jobs WHERE document_id = %s AND kb_id = %s AND status IN ('queued', 'running')",
            [doc_id, kb_id],
        ).fetchone():
            raise HTTPException(409, "The document already has a queued or running job")
        job_id = conn.execute(
            "INSERT INTO ingestion_jobs (kb_id, document_id) VALUES (%s, %s) RETURNING id", [kb_id, doc_id]
        ).fetchone()["id"]  # type: ignore[index]
        conn.execute(
            "UPDATE documents SET status = 'queued', error = NULL WHERE id = %s AND kb_id = %s", [doc_id, kb_id]
        )
    return conn.execute(JOB_SQL + " WHERE j.id = %s", [job_id]).fetchone()


@router.get("/kbs/{kb_id}/documents/{doc_id}/file")
def document_file(
    kb_id: UUID,
    doc_id: UUID,
    _: Role = Depends(kb_access("viewer")),
    conn: Connection = Depends(get_conn),
    settings: Settings = Depends(app_settings),
):
    """Serves the stored file inline; browsers open a PDF at a fragment such as #page=3."""
    row = conn.execute(
        "SELECT storage_path, filename, mime FROM documents WHERE id = %s AND kb_id = %s", [doc_id, kb_id]
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Document not found")
    try:
        path = resolve_stored(settings, row["storage_path"])
    except PermanentError:
        raise HTTPException(404, "File not found") from None
    if not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(
        path,
        media_type=row["mime"] or "application/octet-stream",
        filename=row["filename"],
        content_disposition_type="inline",
        headers={"X-Content-Type-Options": "nosniff"},
    )
