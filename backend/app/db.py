"""Connection pool, kb_scope (the only way to touch document_chunks), startup checks, migrations."""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

import psycopg
from fastapi import Request
from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row, tuple_row
from psycopg_pool import ConnectionPool

from app.config import Settings

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


class StartupCheckError(RuntimeError):
    """The database or role is not safe to run against."""


def open_pool(settings: Settings) -> ConnectionPool:
    """Pool of rag_app connections: autocommit (so conn.transaction() is a real transaction),
    dict rows, pgvector types registered. Run migrations first or register_vector fails."""
    pool = ConnectionPool(
        settings.app_dsn,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "row_factory": dict_row},
        configure=register_vector,
        check=ConnectionPool.check_connection,
        open=False,
    )
    pool.open(wait=True, timeout=30)
    return pool


def get_conn(request: Request) -> Iterator[Connection]:
    """FastAPI dependency: one pooled connection per request (cached across dependencies)."""
    with request.app.state.pool.connection() as conn:
        yield conn


@contextmanager
def kb_scope(conn: Connection, kb_id: UUID | str) -> Iterator[Connection]:
    """Transaction in which RLS exposes only this KB's document_chunks.

    All reads and writes of document_chunks must happen inside it. Outside a scope the
    policy sees no rows (fails closed). set_config(..., true) is transaction-local, so the
    pooled connection goes back with the setting cleared.
    """
    kb_id = UUID(str(kb_id))  # reject anything that is not a UUID before it reaches SQL
    if conn.info.transaction_status != TransactionStatus.IDLE:
        raise RuntimeError("kb_scope needs an idle connection (no open transaction)")
    with conn.transaction():
        conn.execute("SELECT set_config('app.kb_id', %s, true)", [str(kb_id)])
        yield conn


# The one policy document_chunks may have, as Postgres deparses it. Compared without whitespace, parentheses and case.
POLICY_EXPR = "kb_id = (NULLIF(current_setting('app.kb_id'::text, true), ''::text))::uuid"
# The newest migration's columns: an api or worker that starts before `app.cli migrate` finished must not serve.
LATEST_MIGRATION_COLUMNS = ("embedding_fingerprint", "embedding_query_canary")


def _norm(expr: str | None) -> str:
    return re.sub(r"[\s()]", "", expr or "").lower()


def check_db(conn: Connection, embedding_dim: int) -> None:
    """Refuse to run unless the isolation setup is intact: a role that cannot bypass RLS, cannot TRUNCATE or own
    document_chunks, RLS on and forced with exactly the one expected policy, matching dimension, migrations applied."""
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
        superuser, bypass = cur.fetchone()  # type: ignore[misc]
        if superuser or bypass:
            raise StartupCheckError(
                "Connected role is a superuser or has BYPASSRLS; RLS would not isolate knowledge bases."
            )
        cur.execute(
            "SELECT c.relrowsecurity, c.relforcerowsecurity, a.atttypmod, "
            "pg_has_role(current_user, c.relowner, 'MEMBER'), has_table_privilege(current_user, c.oid, 'TRUNCATE') "
            "FROM pg_class c JOIN pg_attribute a ON a.attrelid = c.oid AND a.attname = 'embedding' "
            "WHERE c.oid = 'document_chunks'::regclass"
        )
        rls, forced, db_dim, owns, can_truncate = cur.fetchone()  # type: ignore[misc]
        if not (rls and forced):
            raise StartupCheckError("Row-level security is not enabled and forced on document_chunks.")
        if owns:
            raise StartupCheckError(
                "The app role owns document_chunks (or belongs to its owner); an owner can drop RLS."
            )
        if can_truncate:
            raise StartupCheckError("The app role may TRUNCATE document_chunks, which ignores RLS.")
        cur.execute(
            "SELECT pg_get_expr(polqual, polrelid), pg_get_expr(polwithcheck, polrelid) "
            "FROM pg_policy WHERE polrelid = 'document_chunks'::regclass"
        )
        policies = cur.fetchall()
        if len(policies) != 1 or any(_norm(e) != _norm(POLICY_EXPR) for e in policies[0]):
            raise StartupCheckError(
                f"document_chunks must have exactly one policy with the expression {POLICY_EXPR!r}; found {policies!r}."
            )
        if db_dim != embedding_dim:
            raise StartupCheckError(
                f"document_chunks.embedding is vector({db_dim}) but EMBEDDING_DIM={embedding_dim}; "
                "add a migration or recreate the database volume."
            )
        cur.execute(
            "SELECT count(*) FROM pg_attribute WHERE attrelid = 'knowledge_bases'::regclass AND NOT attisdropped "
            "AND attname = ANY(%s)",
            [list(LATEST_MIGRATION_COLUMNS)],
        )
        if cur.fetchone()[0] != len(LATEST_MIGRATION_COLUMNS):  # type: ignore[index]
            raise StartupCheckError("The database schema is behind the code; run `python -m app.cli migrate` first.")


def migrate(owner_dsn: str, embedding_dim: int, rag_app_password: str) -> list[str]:
    """Apply numbered migrations/*.sql not yet in schema_migrations, as the owner. Returns applied names."""
    if not (isinstance(embedding_dim, int) and 0 < embedding_dim <= 16000):
        raise ValueError(f"EMBEDDING_DIM must be an int in 1..16000, got {embedding_dim!r}")
    applied = []
    # ponytail: no advisory lock, one migration runner at a time (the compose `migrate` service); add one if that changes.
    with psycopg.connect(owner_dsn, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        for path in sorted(MIGRATIONS_DIR.glob("[0-9]*.sql")):
            if path.name in done:
                continue
            sql = path.read_text().replace("{{EMBEDDING_DIM}}", str(embedding_dim))
            with conn.transaction():
                conn.execute("SELECT set_config('app.rag_password', %s, true)", [rag_app_password])
                conn.execute(sql)
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", [path.name])
            applied.append(path.name)
    return applied
