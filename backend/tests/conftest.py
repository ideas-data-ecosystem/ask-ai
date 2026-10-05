"""Fixtures: a throwaway database per session (EMBEDDING_DIM=8), migrated by the owner, used as rag_app.

Needs the compose db on 127.0.0.1:5433 and POSTGRES_PASSWORD / RAG_APP_PASSWORD in the environment or ../.env.
"""

import uuid
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from fastapi.testclient import TestClient
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.auth import hash_password
from app.config import Settings
from app.db import kb_scope, migrate, open_pool
from app.main import create_app

DIM = 8
PASSWORD = "correct-horse-1"


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    base = Settings(postgres_db="postgres", embedding_dim=DIM)
    dbname = f"chatai_test_{uuid.uuid4().hex[:8]}"
    s = base.model_copy(
        update={
            "postgres_db": dbname,
            "storage_dir": tmp_path_factory.mktemp("storage"),
            "static_dir": Path("/nonexistent"),
            "cookie_secure": False,
        }
    )
    with psycopg.connect(base.owner_dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{dbname}"')
    try:
        migrate(s.owner_dsn, DIM, s.rag_app_password.get_secret_value())
        yield s
    finally:
        with psycopg.connect(base.owner_dsn, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{dbname}" WITH (FORCE)')


@pytest.fixture
def owner_conn(settings):
    with psycopg.connect(settings.owner_dsn, autocommit=True, row_factory=dict_row) as conn:
        register_vector(conn)
        yield conn


@pytest.fixture
def pool(settings):
    p = open_pool(settings)
    yield p
    p.close()


@pytest.fixture
def app(settings):
    app = create_app(settings)
    with TestClient(app):  # runs the lifespan (pool + startup checks) for every client below
        yield app


@pytest.fixture
def anon(app):
    return TestClient(app)


@pytest.fixture
def login_as(app):
    def _login(user) -> TestClient:
        client = TestClient(app)
        r = client.post("/api/auth/login", json={"email": user.email, "password": user.password})
        assert r.status_code == 200, r.text
        return client

    return _login


@pytest.fixture
def make_user(pool):
    def _make(is_admin: bool = False, is_active: bool = True):
        email = f"u{uuid.uuid4().hex[:10]}@test.local"
        with pool.connection() as conn:
            uid = conn.execute(
                "INSERT INTO users (email, name, password_hash, is_admin, is_active) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                [email, "Test User", hash_password(PASSWORD), is_admin, is_active],
            ).fetchone()["id"]
        return SimpleNamespace(id=uid, email=email, password=PASSWORD)

    return _make


@pytest.fixture
def make_kb(pool):
    """Creates a KB with one document and `chunks` chunks (all containing the same overlapping text)."""

    def _make(chunks: int = 0):
        tag = uuid.uuid4().hex[:10]
        with pool.connection() as conn:
            kb_id = conn.execute(
                "INSERT INTO knowledge_bases (slug, name) VALUES (%s, %s) RETURNING id", [f"kb-{tag}", f"KB {tag}"]
            ).fetchone()["id"]
            doc_id = conn.execute(
                "INSERT INTO documents (kb_id, filename, storage_path, sha256, status) "
                "VALUES (%s, 'a.pdf', 'x', %s, 'ready') RETURNING id",
                [kb_id, tag],
            ).fetchone()["id"]
            if chunks:
                with kb_scope(conn, kb_id):
                    conn.cursor().executemany(
                        "INSERT INTO document_chunks (kb_id, document_id, chunk_index, content, embedding) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        [
                            (kb_id, doc_id, i, "Pasal 87 mengatur pemberhentian", Vector([0.1] * DIM))
                            for i in range(chunks)
                        ],
                    )
        return SimpleNamespace(id=kb_id, doc_id=doc_id)

    return _make


@pytest.fixture
def add_member(pool):
    def _add(kb, user, role: str) -> None:
        with pool.connection() as conn:
            conn.execute("INSERT INTO kb_members (kb_id, user_id, role) VALUES (%s, %s, %s)", [kb.id, user.id, role])

    return _add
