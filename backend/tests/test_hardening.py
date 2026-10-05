"""Regression tests for the review findings: request caps, CSRF, ask limits, ingestion caps, login lockout,
startup checks, cookie default, error text, unknown API paths, shared role password, cross-KB log access."""

import argparse
import asyncio
import json
import logging
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import openai
import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.conninfo import make_conninfo
from starlette.exceptions import HTTPException

from app import cli, embedding, ingest, llm, rag
from app.api import users
from app.auth import DUMMY_HASH
from app.config import Settings
from app.db import StartupCheckError, check_db, migrate
from app.embedding import EmbeddingError, embed
from app.main import create_app
from app.middleware import DEFAULT_BODY_BYTES, BodyLimitMiddleware

from .conftest import DIM
from .fakes import clear_queue, drain_jobs, install_embed, install_llm, mark_indexed, rag_settings

# --- S1: request bodies are capped before routing and auth --------------------------------------------------


def test_oversized_bodies_get_413_before_any_auth(settings):
    app = create_app(settings.model_copy(update={"max_upload_mb": 1}))  # upload cap: 1 MiB + 1 MiB
    kb = uuid.uuid4()
    with TestClient(app) as c:
        json_headers = {"content-type": "application/json"}
        huge = b'{"email":"a@b.c","password":"' + b"A" * (DEFAULT_BODY_BYTES + 10) + b'"}'
        r = c.post("/api/auth/login", content=huge, headers=json_headers)
        assert r.status_code == 413 and r.json() == {"detail": "Request body too large"}
        # a normal-sized body is not stopped: it reaches the login check
        small = c.post("/api/auth/login", json={"email": "a@b.c", "password": "x"})
        assert small.status_code == 401

        # upload route: the cap is MAX_UPLOAD_MB + 1 MiB, and it applies before the (missing) login
        url = f"/api/kbs/{kb}/documents"
        assert c.post(url, files={"file": ("a.txt", b"A" * (3 << 20))}).status_code == 413
        assert c.post(url, files={"file": ("a.txt", b"A" * (3 << 19))}).status_code == 401  # 1.5 MiB: reaches auth

        # a streamed body has no Content-Length: it is counted as it arrives
        def chunks():
            for _ in range(8):
                yield b"A" * (16 << 10)

        r = c.post("/api/auth/login", content=chunks(), headers=json_headers)
        assert r.status_code == 413


def test_body_limit_counts_bytes_when_there_is_no_content_length():
    async def reads_everything(scope, receive, send):
        while (await receive()).get("more_body"):
            pass

    app = BodyLimitMiddleware(reads_everything, upload_bytes=1 << 20)
    scope = {"type": "http", "method": "POST", "path": "/api/auth/login", "headers": []}
    pending = [{"type": "http.request", "body": b"A" * 40000, "more_body": True}] * 2

    async def receive():
        return pending.pop(0) if pending else {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        raise AssertionError("the guard must raise, not answer, once the app is reading")

    with pytest.raises(HTTPException) as e:
        asyncio.run(app(scope, receive, send))
    assert e.value.status_code == 413


# --- S3: cross-site state changes are refused ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("headers", "allowed"),
    [
        ({}, True),  # not a browser (curl, scripts)
        ({"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"}, True),
        ({"Sec-Fetch-Site": "none"}, True),
        ({"Origin": "http://testserver"}, True),
        ({"Origin": "https://evil.example"}, False),
        ({"Origin": "null"}, False),
        ({"Origin": "http://testserver.evil.example"}, False),
        ({"Sec-Fetch-Site": "cross-site"}, False),
        ({"Sec-Fetch-Site": "same-site"}, False),  # a sibling subdomain: same site, other origin
        ({"Sec-Fetch-Site": "same-origin", "Origin": "https://evil.example"}, False),
    ],
)
def test_state_changing_requests_labelled_cross_site_get_403(headers, allowed, anon):
    r = anon.post("/api/auth/logout", headers=headers)  # always 204 when it reaches the handler
    assert (r.status_code == 204) is allowed
    if not allowed:
        assert r.status_code == 403 and r.json() == {"detail": "Cross-site request rejected"}


def test_cross_site_upload_and_reads(login_as, make_user, make_kb, add_member, settings, app):
    app.state.settings = rag_settings(settings)
    kb = make_kb()
    editor = make_user()
    add_member(kb, editor, "editor")
    c = login_as(editor)
    evil = {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}
    url = f"/api/kbs/{kb.id}/documents"
    r = c.post(url, files={"file": ("a.txt", b"injected prompt text")}, headers=evil)
    assert r.status_code == 403
    assert c.patch(f"/api/kbs/{kb.id}", json={"name": "x"}, headers=evil).status_code == 403
    assert c.delete(f"/api/kbs/{kb.id}/documents/{uuid.uuid4()}", headers=evil).status_code == 403
    assert len(c.get(url).json()) == 1  # only the fixture's own document: nothing was uploaded
    assert c.get(url, headers=evil).status_code == 200  # reads are not state changes
    assert (
        c.post(url, files={"file": ("a.txt", b"honest text")}, headers={"Origin": "http://testserver"}).status_code
        == 201
    )


# --- S4: /ask holds no connection across slow calls, and one user cannot flood it ------------------------------


def test_ask_holds_no_pooled_connection_during_the_embedding_and_llm_calls(
    app, settings, monkeypatch, login_as, make_user, make_kb, add_member, pool
):
    app.state.settings = rag_settings(settings)
    kb = make_kb(chunks=2)
    mark_indexed(pool, kb)
    user = make_user()
    add_member(kb, user, "viewer")
    install_embed(monkeypatch)
    install_llm(monkeypatch)
    in_use: dict[str, int] = {}

    def borrowed() -> int:
        stats = app.state.pool.get_stats()
        return stats["pool_size"] - stats["pool_available"]

    for module, name, label in ((embedding, "_call", "embedding"), (llm, "complete", "llm")):
        inner = getattr(module, name)

        def spy(*args, _inner=inner, _label=label, **kwargs):
            in_use[_label] = borrowed()
            return _inner(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    r = login_as(user).post(f"/api/kbs/{kb.id}/ask", json={"question": "Apa isi Pasal 87?"})
    assert r.status_code == 200 and r.json()["insufficient"] is False
    assert in_use == {"embedding": 0, "llm": 0}


def test_a_user_gets_429_beyond_two_questions_in_flight(
    app, settings, monkeypatch, login_as, make_user, make_kb, add_member, pool
):
    app.state.settings = rag_settings(settings)
    kb = make_kb(chunks=2)
    mark_indexed(pool, kb)
    user, other = make_user(), make_user()
    add_member(kb, user, "viewer")
    add_member(kb, other, "viewer")
    install_embed(monkeypatch)
    release, entered = threading.Event(), threading.Semaphore(0)

    def slow(system, user_msg, settings=None):
        entered.release()
        assert release.wait(20)
        return "Jawaban [1].", {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(llm, "complete", slow)
    body = {"question": "Apa isi Pasal 87?"}
    url = f"/api/kbs/{kb.id}/ask"
    mine, theirs = ([login_as(u) for _ in range(3)] for u in (user, other))
    try:
        with ThreadPoolExecutor(3) as ex:
            first = [ex.submit(mine[i].post, url, json=body) for i in range(2)]
            assert entered.acquire(timeout=20) and entered.acquire(timeout=20)  # both are inside the LLM call
            third = mine[2].post(url, json=body)
            assert third.status_code == 429
            neighbour = ex.submit(theirs[0].post, url, json=body)  # the limit is per user
            assert entered.acquire(timeout=20)
            release.set()
            assert [f.result().status_code for f in first] == [200, 200]
            assert neighbour.result().status_code == 200
    finally:
        release.set()
    assert mine[2].post(url, json=body).status_code == 200  # the slots were given back


# --- S5: hostile documents -----------------------------------------------------------------------------------


def blank_pdf(path: Path, pages: int) -> None:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(pages):
        pdf.new_page(200, 200)
    pdf.save(str(path))


def test_docx_zip_bombs_are_refused_from_the_zip_index_before_parsing(monkeypatch, tmp_path, settings):
    import docx

    def opened(*args, **kwargs):
        raise AssertionError("python-docx must not be reached")

    monkeypatch.setattr(docx, "Document", opened)
    big = tmp_path / "big.docx"
    with zipfile.ZipFile(big, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"0" * (2 << 20))
    monkeypatch.setattr(ingest, "DOCX_MAX_UNCOMPRESSED", 1 << 20)
    with pytest.raises(ingest.PermanentError, match="unpacks to 2 MB"):
        ingest.read_document(big, "docx", settings)

    monkeypatch.setattr(ingest, "DOCX_MAX_UNCOMPRESSED", 100 << 20)
    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", b"0" * (20 << 20))  # compresses about 1000:1
    with pytest.raises(ingest.PermanentError, match="zip bomb"):
        ingest.read_document(bomb, "docx", settings)

    not_a_zip = tmp_path / "x.docx"
    not_a_zip.write_bytes(b"PK but not a zip")
    with pytest.raises(ingest.PermanentError, match="Cannot read DOCX"):
        ingest.read_document(not_a_zip, "docx", settings)


def test_pdf_page_and_ocr_page_caps_fail_before_the_work_starts(monkeypatch, tmp_path, settings):
    def no_ocr(path, indexes):
        raise AssertionError("OCR must not start")

    monkeypatch.setattr(ingest, "ocr_pdf_pages", no_ocr)
    path = tmp_path / "blank.pdf"
    blank_pdf(path, pages=3)
    with pytest.raises(ingest.PermanentError, match="3 pages; the limit is 2"):
        ingest.read_document(path, "pdf", settings.model_copy(update={"max_pdf_pages": 2}))
    with pytest.raises(ingest.PermanentError, match="3 pages need OCR; the limit is 2"):
        ingest.read_document(path, "pdf", settings.model_copy(update={"max_ocr_pages": 2}))


def test_a_pdf_over_the_page_cap_fails_its_job_once_with_a_clear_message(
    app, pool, settings, monkeypatch, login_as, make_user, tmp_path
):
    s = app.state.settings = rag_settings(settings, max_pdf_pages=2)
    clear_queue(pool)
    install_embed(monkeypatch)
    admin = login_as(make_user(is_admin=True))
    kb = admin.post("/api/kbs", json={"name": "Page cap"}).json()["id"]
    blank_pdf(tmp_path / "long.pdf", pages=3)
    r = admin.post(f"/api/kbs/{kb}/documents", files={"file": ("long.pdf", (tmp_path / "long.pdf").read_bytes())})
    assert r.status_code == 201
    assert drain_jobs(pool, s) == 1  # permanent: no retries
    doc = admin.get(f"/api/kbs/{kb}/documents").json()[0]
    assert doc["status"] == "failed" and "limit is 2" in doc["error"] and "MAX_PDF_PAGES" in doc["error"]


# --- S6: login lockout ---------------------------------------------------------------------------------------


def test_a_burst_of_wrong_logins_gets_exactly_five_password_checks(app, make_user, monkeypatch):
    user = make_user()
    checks = []
    real = users.verify_password

    def counting(password, stored):
        if stored != DUMMY_HASH:
            checks.append(1)
        return real(password, stored)

    monkeypatch.setattr(users, "verify_password", counting)

    def attempt(_):
        return TestClient(app).post("/api/auth/login", json={"email": user.email, "password": "wrong-pass-x"})

    with ThreadPoolExecutor(20) as ex:
        responses = list(ex.map(attempt, range(20)))
    assert {r.status_code for r in responses} == {401}
    assert len(checks) == 5  # the sixth and later saw the lock the fifth set


# --- S7: the startup check covers the whole isolation setup ----------------------------------------------------

POLICY = "kb_id = nullif(current_setting('app.kb_id', true), '')::uuid"


def healthy(pool):
    with pool.connection() as conn:
        check_db(conn, DIM)


def test_startup_check_rejects_an_extra_permissive_policy(pool, owner_conn):
    owner_conn.execute("CREATE POLICY leak ON document_chunks USING (true)")
    try:
        with pool.connection() as conn, pytest.raises(StartupCheckError, match="exactly one policy"):
            check_db(conn, DIM)
    finally:
        owner_conn.execute("DROP POLICY leak ON document_chunks")
    healthy(pool)


def test_startup_check_rejects_a_changed_policy_expression(pool, owner_conn):
    owner_conn.execute("ALTER POLICY kb_isolation ON document_chunks USING (true)")
    try:
        with pool.connection() as conn, pytest.raises(StartupCheckError, match="exactly one policy"):
            check_db(conn, DIM)
    finally:
        owner_conn.execute(f"ALTER POLICY kb_isolation ON document_chunks USING ({POLICY}) WITH CHECK ({POLICY})")
    healthy(pool)


def test_startup_check_rejects_a_truncate_grant(pool, owner_conn):
    owner_conn.execute("GRANT TRUNCATE ON document_chunks TO rag_app")
    try:
        with pool.connection() as conn, pytest.raises(StartupCheckError, match="TRUNCATE"):
            check_db(conn, DIM)
    finally:
        owner_conn.execute("REVOKE TRUNCATE ON document_chunks FROM rag_app")
    healthy(pool)


def test_startup_check_rejects_an_app_role_that_belongs_to_the_table_owner(pool, owner_conn):
    # a role is a member of itself, so the same check catches an app role that owns the table directly
    owner = owner_conn.execute("SELECT current_user AS u").fetchone()["u"]
    owner_conn.execute(f'GRANT "{owner}" TO rag_app')
    try:
        with pool.connection() as conn, pytest.raises(StartupCheckError, match="owns document_chunks"):
            check_db(conn, DIM)
    finally:
        owner_conn.execute(f'REVOKE "{owner}" FROM rag_app')
    healthy(pool)


def test_startup_check_refuses_a_schema_behind_the_code(pool, owner_conn):
    owner_conn.execute("ALTER TABLE knowledge_bases RENAME COLUMN embedding_query_canary TO old_name")
    try:
        with pool.connection() as conn, pytest.raises(StartupCheckError, match="app.cli migrate"):
            check_db(conn, DIM)
    finally:
        owner_conn.execute("ALTER TABLE knowledge_bases RENAME COLUMN old_name TO embedding_query_canary")
    healthy(pool)


# --- S2: only the migrate step needs the owner password -----------------------------------------------------


def test_the_owner_password_is_optional_and_only_migrate_needs_it(settings, monkeypatch):
    s = Settings(embedding_dim=DIM, postgres_password=None)
    assert s.app_dsn  # the app connects as rag_app without it
    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        s.owner_dsn  # noqa: B018
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    with pytest.raises(SystemExit, match="POSTGRES_PASSWORD"):
        cli.cmd_migrate(None)


def test_wait_ready_returns_when_the_database_is_ready_and_fails_after_the_timeout_when_not(settings, monkeypatch):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    cli.cmd_wait_ready(argparse.Namespace(timeout=5))  # migrated, passes the checks: returns at once
    monkeypatch.setattr(cli, "get_settings", lambda: settings.model_copy(update={"embedding_dim": DIM + 1}))
    with pytest.raises(SystemExit, match="not ready after 0s.*EMBEDDING_DIM"):
        cli.cmd_wait_ready(argparse.Namespace(timeout=0))


# --- S8: the cookie is Secure unless the operator opts out --------------------------------------------------


def test_cookie_is_secure_by_default_and_the_example_env_only_offers_the_local_http_opt_out(app, settings, make_user):
    assert Settings.model_fields["cookie_secure"].default is True
    app.state.settings = settings.model_copy(update={"cookie_secure": True})
    user = make_user()
    r = TestClient(app).post("/api/auth/login", json={"email": user.email, "password": "correct-horse-1"})
    assert r.status_code == 200 and "secure" in r.headers["set-cookie"].lower()
    example = (Path(__file__).resolve().parents[2] / ".env.example").read_text().splitlines()
    assert "COOKIE_SECURE=false" not in example  # copying the example to .env must not turn Secure off
    assert "#COOKIE_SECURE=false" in example  # the local-http opt-out stays one uncomment away


# --- S9: provider detail stays in the server log ---------------------------------------------------------------

LEAK = "sk-LEAK-internal-endpoint-detail"
TEXT = b"Pasal 3\nPNS wajib menaati kewajiban dan menghindari larangan yang ditentukan dalam peraturan perundang-undangan.\n"


def test_embedding_http_errors_do_not_carry_the_provider_body(monkeypatch, settings, caplog):
    monkeypatch.setattr(
        embedding,
        "_client",
        lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(400, text=LEAK))),
    )
    s = settings.model_copy(update={"embedding_base_url": "http://emb.invalid/v1", "embedding_model": "m"})
    with caplog.at_level(logging.ERROR, logger="app.embedding"), pytest.raises(EmbeddingError) as e:
        embed(["a"], "document", settings=s)
    assert LEAK not in str(e.value) and "HTTP 400" in str(e.value)
    assert LEAK in caplog.text  # the detail is still available to the operator


def test_llm_errors_do_not_reach_the_query_log_but_stay_in_the_server_log(
    monkeypatch, pool, settings, owner_conn, make_kb, caplog
):
    class Failing:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise openai.OpenAIError(f"invalid key {LEAK}")

    monkeypatch.setattr(llm, "_openai", lambda base_url, api_key: Failing)
    install_embed(monkeypatch)
    kb = make_kb(chunks=2)
    mark_indexed(pool, kb)
    with caplog.at_level(logging.ERROR, logger="app.llm"), pytest.raises(Exception) as e:
        rag.ask(pool, rag_settings(settings), kb.id, None, "Apa isi Pasal 87?")
    assert e.value.status_code == 502
    logged = owner_conn.execute("SELECT error FROM query_logs WHERE kb_id = %s", [kb.id]).fetchone()["error"]
    assert LEAK not in logged and "language model request failed" in logged
    assert LEAK in caplog.text


def test_unexpected_errors_are_generic_in_the_query_log_and_in_documents(
    app, pool, settings, owner_conn, monkeypatch, login_as, make_user, make_kb
):
    kb = make_kb(chunks=1)
    mark_indexed(pool, kb)

    def boom(*args, **kwargs):
        raise RuntimeError(f"connection string with {LEAK}")

    monkeypatch.setattr(rag, "search", boom)
    with pytest.raises(RuntimeError):
        rag.ask(pool, rag_settings(settings), kb.id, None, "Apa itu?")
    logged = owner_conn.execute("SELECT error FROM query_logs WHERE kb_id = %s", [kb.id]).fetchone()["error"]
    assert logged == "Internal error (RuntimeError)"

    s = app.state.settings = rag_settings(settings)
    clear_queue(pool)
    monkeypatch.setattr(ingest, "read_document", boom)
    admin = login_as(make_user(is_admin=True))
    new_kb = admin.post("/api/kbs", json={"name": "Generic error"}).json()["id"]
    admin.post(f"/api/kbs/{new_kb}/documents", files={"file": ("a.txt", b"isi")})
    assert drain_jobs(pool, s) == 3  # unexpected errors are retried
    doc = admin.get(f"/api/kbs/{new_kb}/documents").json()[0]
    job = admin.get(f"/api/kbs/{new_kb}/jobs").json()[0]
    assert doc["error"] == job["error"] == "Unexpected error (RuntimeError); see the server log"


def test_provider_text_never_reaches_documents_error_through_the_worker(
    app, pool, settings, monkeypatch, login_as, make_user
):
    s = app.state.settings = rag_settings(settings, embedding_base_url="http://emb.invalid/v1")
    clear_queue(pool)
    monkeypatch.setattr(embedding.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        embedding,
        "_client",
        lambda: httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503, text=LEAK))),
    )
    admin = login_as(make_user(is_admin=True))
    kb = admin.post("/api/kbs", json={"name": "Provider text"}).json()["id"]
    admin.post(f"/api/kbs/{kb}/documents", files={"file": ("a.txt", TEXT)})
    assert drain_jobs(pool, s) == 3
    doc = admin.get(f"/api/kbs/{kb}/documents").json()[0]
    assert doc["status"] == "failed" and LEAK not in doc["error"] and "HTTP 503" in doc["error"]


# --- C7: unknown /api paths are 404 whatever the method -------------------------------------------------------


@pytest.mark.parametrize("with_spa", [False, True])
def test_unknown_api_paths_are_404_for_every_method(with_spa, settings, tmp_path):
    (tmp_path / "index.html").write_text("<html>spa</html>")
    s = settings.model_copy(update={"static_dir": tmp_path if with_spa else Path("/nonexistent")})
    with TestClient(create_app(s)) as c:
        for method in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            r = c.request(method, "/api/nope/at/all")
            assert (r.status_code, r.json()) == (404, {"detail": "Not Found"}), method
        assert c.get("/api/health").status_code == 200  # real routes still win
        assert c.get("/api/openapi.json").status_code == 200
        if with_spa:
            assert c.get("/some/spa/route").text == "<html>spa</html>"


# --- C10: a second database must not reset the shared role's password; logs stay inside their KB ---------------


def test_migrating_another_database_keeps_the_shared_role_password(settings):
    base = settings.model_copy(update={"postgres_db": "postgres"})
    name = f"chatai_test_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(base.owner_dsn, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    try:
        scratch = settings.model_copy(update={"postgres_db": name})
        applied = migrate(scratch.owner_dsn, DIM, "some-other-password")
        assert applied[0] == "001_init.sql" and len(applied) >= 2
        psycopg.connect(settings.app_dsn).close()  # the password the running app uses still works
        other = make_conninfo(
            host=settings.db_host, port=settings.db_port, dbname=name, user="rag_app", password="some-other-password"
        )
        with pytest.raises(psycopg.OperationalError):
            psycopg.connect(other)
    finally:
        with psycopg.connect(base.owner_dsn, autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_jobs_and_query_logs_of_another_kb_are_not_readable_through_document_id(
    login_as, make_user, make_kb, add_member, owner_conn
):
    a, b = make_kb(), make_kb()
    owner_conn.execute("INSERT INTO ingestion_jobs (kb_id, document_id) VALUES (%s, %s)", [b.id, b.doc_id])
    own_job = owner_conn.execute(
        "INSERT INTO ingestion_jobs (kb_id, document_id) VALUES (%s, %s) RETURNING id", [a.id, a.doc_id]
    ).fetchone()["id"]
    logs = {
        kb.id: owner_conn.execute(
            "INSERT INTO query_logs (kb_id, question) VALUES (%s, 'q') RETURNING id", [kb.id]
        ).fetchone()["id"]
        for kb in (a, b)
    }
    editor = make_user()
    add_member(a, editor, "editor")
    c = login_as(editor)
    assert c.get(f"/api/kbs/{a.id}/jobs", params={"document_id": str(b.doc_id)}).json() == []
    assert [j["id"] for j in c.get(f"/api/kbs/{a.id}/jobs", params={"document_id": str(a.doc_id)}).json()] == [own_job]
    got = c.get(f"/api/kbs/{a.id}/queries", params={"document_id": str(b.doc_id)}).json()
    assert [q["id"] for q in got] == [logs[a.id]]  # B's log is not there
    assert c.get(f"/api/kbs/{b.id}/jobs").status_code == 403 and c.get(f"/api/kbs/{b.id}/queries").status_code == 403
    admin = login_as(make_user(is_admin=True))  # control: B's own logs are there for someone who may read them
    assert len(admin.get(f"/api/kbs/{b.id}/jobs", params={"document_id": str(b.doc_id)}).json()) == 1
    assert [q["id"] for q in admin.get(f"/api/kbs/{b.id}/queries").json()] == [logs[b.id]]


def test_fingerprint_covers_model_prefixes_and_extra_bodies_of_both_modes(settings):
    base = rag_settings(settings)
    assert base.embedding_fingerprint == "fake-embed||{}||{}"
    changed = [
        {"embedding_model": "other"},
        {"embedding_doc_prefix": "passage: "},
        {"embedding_doc_extra_body": {"input_type": "passage"}},
        {"embedding_query_prefix": "query: "},
        {"embedding_query_extra_body": {"input_type": "query"}},
    ]
    prints = {rag_settings(settings, **c).embedding_fingerprint for c in changed}
    assert len(prints) == len(changed) and base.embedding_fingerprint not in prints
    assert json.loads(base.embedding_fingerprint.split("|")[2]) == {}
