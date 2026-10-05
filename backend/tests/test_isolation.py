import psycopg
import pytest
from pgvector import Vector

from app.db import StartupCheckError, check_db, kb_scope

from .conftest import DIM
from .fakes import install_embed, install_llm, mark_indexed, rag_settings


def count(conn, where: str = "", *params) -> int:
    return conn.execute(f"SELECT count(*) AS n FROM document_chunks {where}", params).fetchone()["n"]


def test_scope_sees_only_own_rows(pool, make_kb):
    a, b = make_kb(chunks=3), make_kb(chunks=2)
    with pool.connection() as conn:
        with kb_scope(conn, a.id):
            assert count(conn) == 3
            assert count(conn, "WHERE kb_id = %s", b.id) == 0  # naming B inside A's scope still sees nothing
        with kb_scope(conn, b.id):
            assert count(conn) == 2


def test_no_scope_sees_nothing(pool, make_kb):
    make_kb(chunks=2)
    with pool.connection() as conn:
        assert count(conn) == 0


def test_setting_is_cleared_after_scope(pool, make_kb):
    kb = make_kb(chunks=1)
    with pool.connection() as conn:
        with kb_scope(conn, kb.id):
            assert conn.execute("SELECT current_setting('app.kb_id', true) AS v").fetchone()["v"] == str(kb.id)
        assert conn.execute("SELECT current_setting('app.kb_id', true) AS v").fetchone()["v"] == ""
        assert count(conn) == 0


def test_scope_needs_idle_connection_and_uuid(pool, make_kb):
    kb = make_kb()
    with pool.connection() as conn:
        with conn.transaction(), pytest.raises(RuntimeError), kb_scope(conn, kb.id):
            pass
        with pytest.raises(ValueError), kb_scope(conn, "x' OR '1'='1"):
            pass


def insert_chunk(conn, kb_id, doc_id):
    conn.execute(
        "INSERT INTO document_chunks (kb_id, document_id, chunk_index, content, embedding) VALUES (%s, %s, 0, 'x', %s)",
        [kb_id, doc_id, Vector([0.2] * DIM)],
    )


def test_chunk_kb_must_match_its_document(pool, make_kb):
    a, b = make_kb(), make_kb()
    with pool.connection() as conn, pytest.raises(psycopg.errors.ForeignKeyViolation), kb_scope(conn, b.id):
        insert_chunk(conn, b.id, a.doc_id)  # passes RLS (kb B), but document belongs to A


def test_cannot_write_chunk_into_another_scope(pool, make_kb):
    a, b = make_kb(), make_kb()
    with pool.connection() as conn, pytest.raises(psycopg.errors.InsufficientPrivilege), kb_scope(conn, b.id):
        insert_chunk(conn, a.id, a.doc_id)  # RLS WITH CHECK rejects a row for kb A inside scope B


def test_chunk_kb_and_document_ids_are_not_null(owner_conn, make_kb):
    # the owner bypasses RLS, so this reaches the column constraints (with MATCH SIMPLE a NULL would skip the FK)
    kb = make_kb()
    for kb_id, doc_id in ((None, kb.doc_id), (kb.id, None)):
        with pytest.raises(psycopg.errors.NotNullViolation):
            insert_chunk(owner_conn, kb_id, doc_id)


def test_app_role_cannot_truncate(pool):
    with pool.connection() as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("TRUNCATE document_chunks")


def test_startup_check(pool, owner_conn):
    with pool.connection() as conn:
        check_db(conn, DIM)
        with pytest.raises(StartupCheckError, match="EMBEDDING_DIM"):
            check_db(conn, DIM + 1)
    with pytest.raises(StartupCheckError, match="superuser"):
        check_db(owner_conn, DIM)  # POSTGRES_USER is a superuser in the pgvector image


def test_non_member_gets_403_and_listing_hides_kb(login_as, make_user, make_kb, add_member):
    a, b = make_kb(), make_kb()
    user = make_user()
    add_member(a, user, "viewer")
    c = login_as(user)
    assert c.get(f"/api/kbs/{a.id}").status_code == 200
    for path in ("", "/stats", "/members"):
        assert c.get(f"/api/kbs/{b.id}{path}").status_code == 403
    ids = {k["id"] for k in c.get("/api/kbs").json()}
    assert ids == {str(a.id)}


def test_viewer_and_editor_cannot_update_or_delete_kb(login_as, make_user, make_kb, add_member):
    kb = make_kb()
    for role in ("viewer", "editor"):
        user = make_user()
        add_member(kb, user, role)
        c = login_as(user)
        assert c.patch(f"/api/kbs/{kb.id}", json={"name": "x"}).status_code == 403
        assert c.delete(f"/api/kbs/{kb.id}").status_code == 403


def test_unknown_kb_is_404(login_as, make_user):
    import uuid

    c = login_as(make_user())
    assert c.get(f"/api/kbs/{uuid.uuid4()}").status_code == 404
    assert c.get(f"/api/kbs/{uuid.uuid4()}/stats").status_code == 404


def test_anonymous_is_401(anon, make_kb):
    kb = make_kb()
    assert anon.get(f"/api/kbs/{kb.id}").status_code == 401
    assert anon.get("/api/kbs").status_code == 401


# --- asking and documents across knowledge bases ----------------------------------------------------------


def test_ask_returns_citations_from_its_own_kb_only(
    app, pool, settings, owner_conn, monkeypatch, login_as, make_user, make_kb, add_member
):
    app.state.settings = rag_settings(settings, top_k=8)
    install_embed(monkeypatch)
    # overlapping text: both KBs hold three identical chunks, and the model cites every source it is shown
    install_llm(monkeypatch, "".join(f"[{n}]" for n in range(1, 9)))
    a, b = make_kb(chunks=3), make_kb(chunks=3)
    mark_indexed(pool, a)
    mark_indexed(pool, b)
    user = make_user()
    add_member(a, user, "viewer")
    out = login_as(user).post(f"/api/kbs/{a.id}/ask", json={"question": "Apa isi Pasal 87?"}).json()
    assert len(out["citations"]) == 3  # a leak from B would add up to three more
    assert {c["document_id"] for c in out["citations"]} == {str(a.doc_id)}
    log = owner_conn.execute("SELECT retrieved FROM query_logs WHERE kb_id = %s", [a.id]).fetchone()
    chunk_kbs = owner_conn.execute(
        "SELECT DISTINCT kb_id FROM document_chunks WHERE id = ANY(%s)", [[r["chunk_id"] for r in log["retrieved"]]]
    ).fetchall()
    assert [r["kb_id"] for r in chunk_kbs] == [a.id]


def test_non_member_cannot_ask_and_nothing_is_logged_or_sent(
    app, pool, settings, owner_conn, monkeypatch, login_as, make_user, make_kb, add_member, anon
):
    app.state.settings = rag_settings(settings)
    emb, llm_ = install_embed(monkeypatch), install_llm(monkeypatch)
    a, b = make_kb(chunks=2), make_kb(chunks=2)
    mark_indexed(pool, b)
    user = make_user()
    add_member(a, user, "viewer")
    c = login_as(user)
    assert c.post(f"/api/kbs/{b.id}/ask", json={"question": "Apa isi Pasal 87?"}).status_code == 403
    assert anon.post(f"/api/kbs/{b.id}/ask", json={"question": "Apa isi Pasal 87?"}).status_code == 401
    assert emb.calls == [] and llm_.calls == []
    assert owner_conn.execute("SELECT count(*) AS n FROM query_logs WHERE kb_id = %s", [b.id]).fetchone()["n"] == 0
    assert c.get(f"/api/kbs/{b.id}/queries").status_code == 403


def test_documents_of_another_kb_are_404_through_this_kbs_url(app, settings, login_as, make_user, make_kb, add_member):
    app.state.settings = rag_settings(settings)
    admin = login_as(make_user(is_admin=True))
    a, b = make_kb(), make_kb()
    doc = admin.post(f"/api/kbs/{b.id}/documents", files={"file": ("rahasia.txt", b"isi rahasia KB B")}).json()
    editor = make_user()
    add_member(a, editor, "editor")
    c = login_as(editor)
    paths = [f"/api/kbs/{a.id}/documents/{doc['id']}/file", f"/api/kbs/{a.id}/documents/{doc['id']}"]
    assert admin.get(f"/api/kbs/{b.id}/documents/{doc['id']}/file").status_code == 200  # control: the file is there
    assert c.get(paths[0]).status_code == 404
    assert c.post(f"{paths[1]}/reindex").status_code == 404
    assert c.delete(paths[1]).status_code == 404
    assert admin.get(f"/api/kbs/{b.id}/documents/{doc['id']}/file").status_code == 200  # still there after the attempts
    # and B's own URLs are closed to a member of A only
    assert c.get(f"/api/kbs/{b.id}/documents/{doc['id']}/file").status_code == 403
    assert c.delete(f"/api/kbs/{b.id}/documents/{doc['id']}").status_code == 403
