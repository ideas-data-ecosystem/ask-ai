"""Admin CLI: python -m app.cli migrate | wait-ready | create-admin | seed"""

import argparse
import getpass
import sys
import time
from collections import Counter
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from app.api.kbs import slugify
from app.auth import hash_password
from app.config import get_settings
from app.db import StartupCheckError, check_db, migrate
from app.ingest import ALLOWED_TYPES, add_document, sha256_file


def cmd_migrate(args: argparse.Namespace) -> None:
    s = get_settings()
    try:
        dsn = s.owner_dsn
    except RuntimeError as e:  # POSTGRES_PASSWORD is only set on the migrate service
        sys.exit(str(e))
    applied = migrate(dsn, s.embedding_dim, s.rag_app_password.get_secret_value())
    print("applied: " + ", ".join(applied) if applied else "up to date")


def cmd_wait_ready(args: argparse.Namespace) -> None:
    """Block until the database is migrated and passes the startup checks, as the app role.

    compose runs it before the api and the worker: podman-compose starts them without waiting for the one-shot
    migrate service (it ignores `service_completed_successfully`), so they wait here instead.
    """
    s = get_settings()
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            with psycopg.connect(s.app_dsn) as conn:
                check_db(conn, s.embedding_dim)
            return
        except (psycopg.Error, StartupCheckError) as e:  # not migrated yet: no role, no table, schema behind
            if time.monotonic() >= deadline:
                sys.exit(f"The database is not ready after {args.timeout}s: {e}")
            time.sleep(2)


def cmd_create_admin(args: argparse.Namespace) -> None:
    password = args.password or getpass.getpass("Password: ")
    if len(password) < 8:
        sys.exit("Password must be at least 8 characters")
    email = args.email.strip().lower()
    # Upsert so a lost admin password can be reset the same way; also unlocks and ends old sessions.
    with psycopg.connect(get_settings().app_dsn, autocommit=True) as conn:
        user_id = conn.execute(
            "INSERT INTO users (email, name, password_hash, is_admin) VALUES (%s, %s, %s, true) "
            "ON CONFLICT (email) DO UPDATE SET password_hash = EXCLUDED.password_hash, is_admin = true, "
            "is_active = true, failed_logins = 0, locked_until = NULL RETURNING id",
            [email, args.name or email, hash_password(password)],
        ).fetchone()[0]
        conn.execute("DELETE FROM sessions WHERE user_id = %s", [user_id])
    print(f"admin ready: {email}")


def cmd_seed(args: argparse.Namespace) -> None:
    """Register the files under a directory where they are (no copy, no move) and queue one job per new file.

    The directory must be inside STORAGE_DIR. The subfolder path becomes the document category. Seeded files are
    never deleted by the app. Safe to repeat: content already in the KB (sha256) is skipped.
    """
    s = get_settings()
    storage, root = s.storage_dir.resolve(), Path(args.directory).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")
    if not root.is_relative_to(storage):
        sys.exit(f"{root} is not inside STORAGE_DIR ({storage}); set STORAGE_DIR to the corpus root")
    slug = slugify(args.kb)
    if not slug:
        sys.exit("Cannot derive a slug from the knowledge base name")
    counts: Counter[str] = Counter()
    with psycopg.connect(s.app_dsn, autocommit=True, row_factory=dict_row) as conn:
        # By slug only: names are not unique, so a name match could register the corpus into the wrong KB.
        kb = conn.execute("SELECT id FROM knowledge_bases WHERE slug = %s", [slug]).fetchone()
        if kb is None:
            kb = conn.execute(
                "INSERT INTO knowledge_bases (slug, name) VALUES (%s, %s) RETURNING id", [slug, args.kb]
            ).fetchone()
            print(f"created knowledge base {args.kb!r}")
        for path in sorted(root.rglob("*")):
            rel_storage = path.relative_to(storage)
            if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(storage):
                continue  # directories, symlinks and anything that resolves outside STORAGE_DIR
            ext = path.suffix.lstrip(".").lower()
            if rel_storage.parts[0] == "uploads" or any(p.startswith(".") for p in rel_storage.parts):
                counts["skipped (uploads or hidden)"] += 1
            elif ext not in ALLOWED_TYPES:
                counts["ignored (type not allowed)"] += 1
            else:
                sha = sha256_file(path)
                stored = rel_storage.as_posix()
                known = conn.execute(
                    "SELECT sha256 FROM documents WHERE kb_id = %s AND storage_path = %s", [kb["id"], stored]
                ).fetchone()
                if known:
                    changed = known["sha256"] != sha
                    counts[
                        "changed on disk (delete the old document to re-register)" if changed else "already registered"
                    ] += 1
                    continue
                row = add_document(
                    conn,
                    kb_id=kb["id"],
                    filename=path.name,
                    category=" / ".join(path.relative_to(root).parts[:-1]) or None,
                    storage_path=stored,
                    mime=ALLOWED_TYPES[ext],
                    size_bytes=path.stat().st_size,
                    sha256=sha,
                )
                counts["registered" if row else "duplicate content (already in this KB)"] += 1
    for what, n in sorted(counts.items()):
        print(f"{n:4d} {what}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply pending SQL migrations as the owner role").set_defaults(func=cmd_migrate)
    p = sub.add_parser("wait-ready", help="wait until migrations are applied and the startup checks pass")
    p.add_argument("--timeout", type=int, default=120, help="seconds to wait before failing (default 120)")
    p.set_defaults(func=cmd_wait_ready)
    p = sub.add_parser("create-admin", help="create (or reset) an admin user")
    p.add_argument("--email", required=True)
    p.add_argument("--name")
    p.add_argument("--password", help="omit to be prompted (avoids shell history)")
    p.set_defaults(func=cmd_create_admin)
    p = sub.add_parser("seed", help="register a directory of documents in a knowledge base and queue ingestion")
    p.add_argument("--kb", required=True, help="knowledge base name; found by its slug, created if missing")
    p.add_argument("directory", help="directory inside STORAGE_DIR, e.g. the corpus root")
    p.set_defaults(func=cmd_seed)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
