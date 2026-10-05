import argparse
import uuid

from starlette.requests import Request

from app import cli

from .fakes import clear_queue, drain_jobs, install_embed, rag_settings


def test_admin_creates_and_edits_kb(login_as, make_user):
    c = login_as(make_user(is_admin=True))
    r = c.post("/api/kbs", json={"name": "Regulasi ASN Ünik", "description": "d"})
    assert r.status_code == 201
    kb = r.json()
    assert kb["slug"] == "regulasi-asn-unik" and kb["role"] == "admin" and kb["doc_count"] == 0
    assert c.post("/api/kbs", json={"name": "Other", "slug": "regulasi-asn-unik"}).status_code == 409
    assert c.post("/api/kbs", json={"name": "???"}).status_code == 422  # no slug derivable
    r = c.patch(f"/api/kbs/{kb['id']}", json={"name": "Renamed", "status": "archived", "description": None})
    assert r.json()["name"] == "Renamed" and r.json()["status"] == "archived" and r.json()["description"] == "d"
    assert c.patch(f"/api/kbs/{kb['id']}", json={"status": "bogus"}).status_code == 422


def test_non_admin_cannot_create_kb(login_as, make_user):
    assert login_as(make_user()).post("/api/kbs", json={"name": "x"}).status_code == 403


def test_stats_count_chunks_through_scope(login_as, make_user, make_kb):
    a, _b = make_kb(chunks=3), make_kb(chunks=5)
    s = login_as(make_user(is_admin=True)).get(f"/api/kbs/{a.id}/stats").json()
    assert s["chunks"] == 3 and s["documents"]["total"] == 1 and s["documents"]["ready"] == 1
    assert s["jobs"] == {"queued": 0, "running": 0, "done": 0, "failed": 0}
    assert "llm_model" in s  # the configured LLM_MODEL (or null), shown next to the embedding model


def test_members_are_managed_by_admins_only(login_as, make_user, make_kb, add_member):
    kb, editor, viewer = make_kb(), make_user(), make_user()
    add_member(kb, editor, "editor")
    admin = login_as(make_user(is_admin=True))
    assert admin.put(f"/api/kbs/{kb.id}/members/{viewer.id}", json={"role": "viewer"}).json()["role"] == "viewer"
    assert admin.put(f"/api/kbs/{kb.id}/members/{viewer.id}", json={"role": "editor"}).json()["role"] == "editor"
    assert admin.put(f"/api/kbs/{kb.id}/members/{viewer.id}", json={"role": "admin"}).status_code == 422
    assert admin.put(f"/api/kbs/{kb.id}/members/{uuid.uuid4()}", json={"role": "viewer"}).status_code == 404
    assert {m["email"] for m in admin.get(f"/api/kbs/{kb.id}/members").json()} == {editor.email, viewer.email}
    assert admin.delete(f"/api/kbs/{kb.id}/members/{viewer.id}").status_code == 204
    assert admin.delete(f"/api/kbs/{kb.id}/members/{viewer.id}").status_code == 404
    # neither an editor nor a viewer may list, add, change or remove members
    v = make_user()
    add_member(kb, v, "viewer")
    for c in (login_as(editor), login_as(v)):
        assert c.get(f"/api/kbs/{kb.id}/members").status_code == 403
        assert c.put(f"/api/kbs/{kb.id}/members/{v.id}", json={"role": "editor"}).status_code == 403
        assert c.delete(f"/api/kbs/{kb.id}/members/{v.id}").status_code == 403


def test_delete_kb_cascades_and_removes_only_its_uploads(login_as, make_user, make_kb, settings, owner_conn):
    kb, other = make_kb(chunks=2), make_kb(chunks=1)
    mine = settings.storage_dir / "uploads" / str(kb.id)
    theirs = settings.storage_dir / "uploads" / str(other.id)
    seeded = settings.storage_dir / "00 - Dasar" / "seed.pdf"
    for p in (mine / "d.pdf", theirs / "d.pdf", seeded):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    assert login_as(make_user(is_admin=True)).delete(f"/api/kbs/{kb.id}").status_code == 204
    n = lambda tbl, k: owner_conn.execute(f"SELECT count(*) AS n FROM {tbl} WHERE kb_id = %s", [k.id]).fetchone()["n"]
    assert n("document_chunks", kb) == 0 and n("documents", kb) == 0  # superuser bypasses RLS: real counts
    assert n("document_chunks", other) == 1
    assert not mine.exists() and (theirs / "d.pdf").exists() and seeded.exists()


# --- documents: upload, file, delete, reindex, logs -------------------------------------------------------

PDF = b"%PDF-1.4\n"


def editor_of(kb, make_user, add_member, login_as, role="editor"):
    user = make_user()
    add_member(kb, user, role)
    return login_as(user)


def kb_files(settings, kb_id):
    d = settings.storage_dir / "uploads" / str(kb_id)
    return sorted(p.name for p in d.glob("*")) if d.exists() else []


def test_same_filename_with_different_bytes_makes_two_documents_and_two_files(
    login_as, make_user, make_kb, add_member, settings
):
    kb = make_kb()
    c = editor_of(kb, make_user, add_member, login_as)
    url = f"/api/kbs/{kb.id}/documents"
    d1 = c.post(url, files={"file": ("Perpres.pdf", PDF + b"one")}).json()
    d2 = c.post(url, files={"file": ("Perpres.pdf", PDF + b"two")}).json()
    assert d1["id"] != d2["id"] and d1["filename"] == d2["filename"] == "Perpres.pdf"
    assert d1["status"] == "queued" and d1["size_bytes"] == len(PDF) + 3 and len(d1["sha256"]) == 64
    assert kb_files(settings, kb.id) == sorted([f"{d1['id']}.pdf", f"{d2['id']}.pdf"])
    # the same bytes under any name are a conflict and leave no stray file behind
    r = c.post(url, files={"file": ("other-name.pdf", PDF + b"one")})
    assert r.status_code == 409 and len(kb_files(settings, kb.id)) == 2
    assert len(c.get(url).json()) == 3  # the two uploads and the fixture's own document


def test_filename_and_category_never_reach_a_path(login_as, make_user, make_kb, add_member, settings):
    kb = make_kb()
    c = editor_of(kb, make_user, add_member, login_as)
    url = f"/api/kbs/{kb.id}/documents"
    r1 = c.post(url, files={"file": ("../../../evil.pdf", PDF + b"1")}, data={"category": "../../etc"})
    r2 = c.post(url, files={"file": ("..\\..\\evil2.pdf", PDF + b"2")}, data={"category": "/abs/path"})
    assert r1.status_code == r2.status_code == 201
    assert r1.json()["filename"] == "evil.pdf" and r2.json()["filename"] == "evil2.pdf"
    assert r1.json()["category"] == "../../etc"  # a label only
    ids = {r1.json()["id"], r2.json()["id"]}
    assert kb_files(settings, kb.id) == sorted(f"{i}.pdf" for i in ids)
    root = settings.storage_dir
    assert (
        not list(root.rglob("evil*"))
        and not list(root.parent.glob("evil*"))
        and not list(root.parent.parent.glob("evil*"))
    )
    assert c.post(url, files={"file": ("a.pdf", PDF + b"3")}, data={"category": "x" * 201}).status_code == 422


def test_upload_validation(app, login_as, make_user, make_kb, add_member, settings):
    kb = make_kb()
    c = editor_of(kb, make_user, add_member, login_as)
    url = f"/api/kbs/{kb.id}/documents"
    assert c.post(url, files={"file": ("a.exe", b"MZ")}).status_code == 415
    assert c.post(url, files={"file": ("noext", b"x")}).status_code == 415
    assert c.post(url, files={"file": ("a.pdf", b"not a pdf at all")}).status_code == 415  # content must match
    assert c.post(url, files={"file": ("a.pdf", b"")}).status_code == 400
    assert c.post(url).status_code == 422
    app.state.settings = settings.model_copy(update={"max_upload_mb": 1})
    big = PDF + b"0" * (1 << 20)
    assert c.post(url, files={"file": ("big.pdf", big)}).status_code == 413
    assert kb_files(settings, kb.id) == []  # nothing left from the rejected uploads
    assert c.post(url, files={"file": ("A.MD", b"# ok")}).status_code == 201  # extension is case-insensitive
    viewer = editor_of(kb, make_user, add_member, login_as, role="viewer")
    assert viewer.post(url, files={"file": ("a.pdf", PDF + b"v")}).status_code == 403
    assert viewer.get(url).status_code == 200


def test_upload_is_authorized_before_the_form_is_parsed(
    monkeypatch, anon, login_as, make_user, make_kb, add_member, settings
):
    parsed = []
    original = Request.form

    def spy(self, **kwargs):
        parsed.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(Request, "form", spy)
    kb = make_kb()
    url = f"/api/kbs/{kb.id}/documents"
    files = {"file": ("a.txt", b"isi")}
    broken = {"content": b"not multipart at all", "headers": {"content-type": "multipart/form-data; boundary=x"}}
    stranger = login_as(make_user())  # logged in, member of nothing
    attempts = {
        "anonymous": anon.post(url, files=files),
        "anonymous, malformed body": anon.post(url, **broken),
        "non-member": stranger.post(url, files=files),
        "viewer": editor_of(kb, make_user, add_member, login_as, role="viewer").post(url, files=files),
        "unknown kb": stranger.post(f"/api/kbs/{uuid.uuid4()}/documents", files=files),
    }
    assert {who: r.status_code for who, r in attempts.items()} == {
        "anonymous": 401,
        "anonymous, malformed body": 401,  # a malformed form would be 400 had it been parsed first
        "non-member": 403,
        "viewer": 403,
        "unknown kb": 404,
    }
    assert parsed == [] and kb_files(settings, kb.id) == []

    # control: the spy is wired to the route, and an editor's form is read with the one-file, one-field caps
    editor = editor_of(kb, make_user, add_member, login_as)
    assert editor.post(url, files=files).status_code == 201
    assert parsed == [{"max_files": 1, "max_fields": 1}]


def test_upload_takes_one_file_and_one_field(login_as, make_user, make_kb, add_member, settings):
    kb = make_kb()
    c = editor_of(kb, make_user, add_member, login_as)
    url = f"/api/kbs/{kb.id}/documents"
    two_files = [("file", ("a.txt", b"one")), ("file", ("b.txt", b"two"))]
    assert c.post(url, files=two_files).status_code == 400
    assert c.post(url, files={"file": ("a.txt", b"x")}, data={"category": "c", "other": "d"}).status_code == 400
    missing = c.post(url, data={"file": "text, not a file part"})
    assert missing.status_code == 422 and missing.json()["detail"][0]["loc"] == ["body", "file"]
    assert kb_files(settings, kb.id) == []
    assert c.post(url, files={"file": ("a.txt", b"x")}, data={"category": "c"}).status_code == 201


def test_file_download_is_inline_with_its_content_type(login_as, make_user, make_kb, add_member):
    kb = make_kb()
    editor = editor_of(kb, make_user, add_member, login_as)
    doc = editor.post(f"/api/kbs/{kb.id}/documents", files={"file": ("Aturan Disiplin.pdf", PDF + b"body")}).json()
    viewer = editor_of(kb, make_user, add_member, login_as, role="viewer")
    r = viewer.get(f"/api/kbs/{kb.id}/documents/{doc['id']}/file")
    assert r.status_code == 200 and r.content == PDF + b"body"
    assert r.headers["content-type"] == "application/pdf" and r.headers["x-content-type-options"] == "nosniff"
    assert (
        r.headers["content-disposition"].startswith("inline")
        and "Aturan%20Disiplin.pdf" in r.headers["content-disposition"]
    )
    assert viewer.get(f"/api/kbs/{kb.id}/documents/{uuid.uuid4()}/file").status_code == 404


def test_delete_removes_uploaded_files_but_never_seeded_ones(login_as, make_user, make_kb, add_member, settings, pool):
    kb = make_kb()
    c = editor_of(kb, make_user, add_member, login_as)
    doc = c.post(f"/api/kbs/{kb.id}/documents", files={"file": ("a.pdf", PDF + b"del")}).json()
    stored = settings.storage_dir / "uploads" / str(kb.id) / f"{doc['id']}.pdf"
    seeded = settings.storage_dir / "seeded-corpus" / "keep.pdf"
    seeded.parent.mkdir(parents=True, exist_ok=True)
    seeded.write_bytes(PDF)
    with pool.connection() as conn:
        seeded_id = conn.execute(
            "INSERT INTO documents (kb_id, filename, storage_path, sha256) VALUES (%s, 'keep.pdf', %s, 'seededsha') "
            "RETURNING id",
            [kb.id, "seeded-corpus/keep.pdf"],
        ).fetchone()["id"]
    assert stored.exists()
    assert c.delete(f"/api/kbs/{kb.id}/documents/{doc['id']}").status_code == 204
    assert not stored.exists()
    assert c.delete(f"/api/kbs/{kb.id}/documents/{doc['id']}").status_code == 404
    assert c.delete(f"/api/kbs/{kb.id}/documents/{seeded_id}").status_code == 204
    assert seeded.exists()  # the row is gone, the user's file is not


def test_reindex_a_document_and_a_failed_job_is_visible(
    app, pool, login_as, make_user, make_kb, add_member, settings, monkeypatch
):
    app.state.settings = s = rag_settings(settings)
    install_embed(monkeypatch)
    clear_queue(pool)
    kb = make_kb()
    c = editor_of(kb, make_user, add_member, login_as)
    doc = c.post(f"/api/kbs/{kb.id}/documents", files={"file": ("rusak.pdf", PDF + b"broken")}).json()
    base = f"/api/kbs/{kb.id}/documents/{doc['id']}"
    assert c.post(f"{base}/reindex").status_code == 409  # the upload's own job is still queued
    assert drain_jobs(pool, s) == 1
    d = next(d for d in c.get(f"/api/kbs/{kb.id}/documents").json() if d["id"] == doc["id"])
    assert d["status"] == "failed" and "Cannot open PDF" in d["error"]  # permanent: one attempt, no retries
    job = c.get(f"/api/kbs/{kb.id}/jobs", params={"status": "failed"}).json()[0]
    assert job["document_id"] == doc["id"] and job["filename"] == "rusak.pdf" and job["stats"] == {}
    r = c.post(f"{base}/reindex")
    assert r.status_code == 202 and r.json()["status"] == "queued" and r.json()["document_id"] == doc["id"]
    assert c.post(f"{base}/reindex").status_code == 409
    assert c.get(f"/api/kbs/{kb.id}/jobs", params={"document_id": doc["id"]}).json()[0]["id"] == r.json()["id"]


def test_kb_reindex_queues_every_document_without_an_active_job(
    app, login_as, make_user, make_kb, add_member, settings
):
    app.state.settings = rag_settings(settings)
    kb = make_kb()  # one fixture document without a job
    c = editor_of(kb, make_user, add_member, login_as)
    c.post(f"/api/kbs/{kb.id}/documents", files={"file": ("a.txt", b"isi a")})  # has its own queued job
    assert c.post(f"/api/kbs/{kb.id}/reindex").json() == {"enqueued": 1}
    assert c.post(f"/api/kbs/{kb.id}/reindex").json() == {"enqueued": 0}
    viewer = editor_of(kb, make_user, add_member, login_as, role="viewer")
    assert viewer.post(f"/api/kbs/{kb.id}/reindex").status_code == 403


def test_archived_kb_reads_but_rejects_upload_reindex_and_ask(app, login_as, make_user, make_kb, settings):
    app.state.settings = rag_settings(settings)
    kb = make_kb()
    admin = login_as(make_user(is_admin=True))
    doc = admin.post(f"/api/kbs/{kb.id}/documents", files={"file": ("a.txt", b"isi a")}).json()
    admin.patch(f"/api/kbs/{kb.id}", json={"status": "archived"})
    url = f"/api/kbs/{kb.id}"
    assert (
        admin.post(f"{url}/documents", files={"file": ("b.txt", b"isi b")}).json()["detail"]
        == "Knowledge base is archived"
    )
    assert admin.post(f"{url}/reindex").status_code == 409
    assert admin.post(f"{url}/documents/{doc['id']}/reindex").status_code == 409
    assert admin.post(f"{url}/ask", json={"question": "apa?"}).status_code == 409
    assert (
        admin.get(f"{url}/documents").status_code == 200
        and admin.get(f"{url}/documents/{doc['id']}/file").status_code == 200
    )
    assert admin.get(f"{url}/jobs").status_code == 200 and admin.get(f"{url}/queries").status_code == 200


def test_ask_validation_and_query_logs_for_editors(
    app, pool, login_as, make_user, make_kb, add_member, settings, monkeypatch
):
    app.state.settings = rag_settings(settings, max_question_chars=20)
    install_embed(monkeypatch)
    kb = make_kb()
    viewer = editor_of(kb, make_user, add_member, login_as, role="viewer")
    editor = editor_of(kb, make_user, add_member, login_as)
    ask = f"/api/kbs/{kb.id}/ask"
    assert viewer.post(ask, json={"question": "   "}).status_code == 422
    assert viewer.post(ask, json={"question": "x" * 21}).status_code == 422
    assert viewer.post(ask, json={}).status_code == 422
    out = viewer.post(ask, json={"question": "  Apa saja?  "})  # never indexed: a normal 200, insufficient
    assert out.status_code == 200 and out.json()["insufficient"] is True and out.json()["citations"] == []
    assert (
        viewer.get(f"/api/kbs/{kb.id}/queries").status_code == 403
        and viewer.get(f"/api/kbs/{kb.id}/jobs").status_code == 403
    )
    logs = editor.get(f"/api/kbs/{kb.id}/queries").json()
    assert len(logs) == 1 and logs[0]["question"] == "Apa saja?" and logs[0]["user_email"].endswith("@test.local")
    assert logs[0]["insufficient"] is True and logs[0]["retrieved"] == [] and logs[0]["error"] is None
    assert editor.get(f"/api/kbs/{kb.id}/queries", params={"insufficient": "false"}).json() == []


def test_seed_registers_files_in_place_and_is_idempotent(settings, owner_conn, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    root = settings.storage_dir / f"corpus-{uuid.uuid4().hex[:8]}"
    for rel, data in {
        "01 - Satu/a.pdf": PDF + b"a",
        "01 - Satu/copy-of-a.pdf": PDF + b"a",  # same content: one document
        "02 - Dua/b.pdf": PDF + b"b",
        "top.pdf": PDF + b"c",
        "01 - Satu/notes.zip": b"PK",
        ".hidden.pdf": PDF + b"h",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    (settings.storage_dir / "uploads" / "x").mkdir(parents=True, exist_ok=True)
    (settings.storage_dir / "uploads" / "x" / "u.pdf").write_bytes(PDF + b"u")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    args = argparse.Namespace(kb="Seed Test KB", directory=str(root))
    cli.cmd_seed(args)
    kb = owner_conn.execute("SELECT id FROM knowledge_bases WHERE slug = 'seed-test-kb'").fetchone()
    docs = owner_conn.execute(
        "SELECT filename, category, storage_path, status FROM documents WHERE kb_id = %s ORDER BY filename", [kb["id"]]
    ).fetchall()
    assert [(d["filename"], d["category"]) for d in docs] == [
        ("a.pdf", "01 - Satu"),
        ("b.pdf", "02 - Dua"),
        ("top.pdf", None),
    ]
    assert docs[0]["storage_path"] == f"{root.name}/01 - Satu/a.pdf"  # relative to STORAGE_DIR, in place
    assert {d["status"] for d in docs} == {"queued"}
    assert (
        owner_conn.execute("SELECT count(*) AS n FROM ingestion_jobs WHERE kb_id = %s", [kb["id"]]).fetchone()["n"] == 3
    )
    out = capsys.readouterr().out
    assert "3 registered" in out and "duplicate content" in out and "type not allowed" in out

    cli.cmd_seed(args)  # again: nothing new
    assert "registered" not in capsys.readouterr().out.replace("already registered", "")
    assert owner_conn.execute("SELECT count(*) AS n FROM documents WHERE kb_id = %s", [kb["id"]]).fetchone()["n"] == 3
    assert (
        owner_conn.execute("SELECT count(*) AS n FROM ingestion_jobs WHERE kb_id = %s", [kb["id"]]).fetchone()["n"] == 3
    )
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}  # the corpus is untouched

    cli.cmd_seed(argparse.Namespace(kb="Seed Test KB", directory=str(settings.storage_dir / "uploads")))
    assert "skipped" in capsys.readouterr().out  # uploads/ is never seeded


def test_seed_and_eval_find_the_kb_by_slug_never_by_a_name_that_is_not_unique(settings, owner_conn, monkeypatch):
    from eval import run_eval

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    tag = uuid.uuid4().hex[:8]
    name = f"Regulasi {tag}"
    # names are not unique: an older KB with the very same name but another slug must not receive the corpus
    decoy = owner_conn.execute(
        "INSERT INTO knowledge_bases (slug, name) VALUES (%s, %s) RETURNING id", [f"decoy-{tag}", name]
    ).fetchone()["id"]
    mine = owner_conn.execute(
        "INSERT INTO knowledge_bases (slug, name) VALUES (%s, %s) RETURNING id", [f"regulasi-{tag}", f"Other {tag}"]
    ).fetchone()["id"]
    root = settings.storage_dir / f"slug-corpus-{tag}"
    root.mkdir()
    (root / "a.pdf").write_bytes(PDF + b"slug")
    cli.cmd_seed(argparse.Namespace(kb=name, directory=str(root)))
    count = lambda kb: owner_conn.execute("SELECT count(*) AS n FROM documents WHERE kb_id = %s", [kb]).fetchone()["n"]
    assert (count(mine), count(decoy)) == (1, 0)
    assert run_eval.find_kb(owner_conn, name)["id"] == mine
    assert run_eval.find_kb(owner_conn, f"regulasi-{tag}")["id"] == mine  # a slug works too


def test_every_endpoint_in_the_contract_is_implemented(anon):
    import re
    from pathlib import Path

    contract = Path(__file__).resolve().parents[2] / "docs" / "api-contract.md"
    declared = re.findall(r"^### `(GET|POST|PUT|PATCH|DELETE) (/api/\S+)`", contract.read_text(), re.MULTILINE)
    assert len(declared) >= 25  # the parse found the whole contract
    implemented = {
        (method.upper(), path) for path, ops in anon.get("/api/openapi.json").json()["paths"].items() for method in ops
    }
    assert [d for d in declared if d not in implemented] == []
