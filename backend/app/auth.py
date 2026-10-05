"""Passwords (scrypt), server-side sessions, lockout, and the FastAPI auth/KB-access dependencies."""

import hashlib
import hmac
import secrets
from collections.abc import Callable
from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from psycopg import Connection
from pydantic import BaseModel

from app.db import get_conn

COOKIE_NAME = "session"
SESSION_HOURS = 8
MAX_FAILED_LOGINS = 5
LOCK_MINUTES = 15
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}

Role = Literal["viewer", "editor", "admin"]
_RANK: dict[str, int] = {"viewer": 1, "editor": 2, "admin": 3}


class User(BaseModel):
    id: UUID
    email: str
    name: str
    is_admin: bool
    is_active: bool
    created_at: datetime
    locked_until: datetime | None = None


USER_COLUMNS = "id, email, name, is_admin, is_active, created_at, locked_until"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, digest = stored.split("$")
        expected = bytes.fromhex(digest)
        got = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected)
        )
    except ValueError:
        return False
    return hmac.compare_digest(got, expected)


DUMMY_HASH = hash_password(secrets.token_urlsafe(16))  # burned for unknown emails so timing does not leak them


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(conn: Connection, user_id: UUID) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute("DELETE FROM sessions WHERE expires_at < now()")  # opportunistic cleanup
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, now() + make_interval(hours => %s))",
        [_token_hash(token), user_id, SESSION_HOURS],
    )
    return token


def delete_session(conn: Connection, token: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token_hash = %s", [_token_hash(token)])


def delete_user_sessions(conn: Connection, user_id: UUID) -> None:
    conn.execute("DELETE FROM sessions WHERE user_id = %s", [user_id])


def authenticate(conn: Connection, token: str | None) -> User:
    """401 unless the token maps to an unexpired session of an active user (checked on every request)."""
    row = None
    if token:
        row = conn.execute(
            "SELECT u.id, u.email, u.name, u.is_admin, u.is_active, u.created_at, u.locked_until "
            "FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash = %s AND s.expires_at > now() AND u.is_active",
            [_token_hash(token)],
        ).fetchone()
    if row is None:
        raise HTTPException(401, "Not authenticated")
    return User.model_validate(row)


def current_user(request: Request, conn: Connection = Depends(get_conn)) -> User:
    return authenticate(conn, request.cookies.get(COOKIE_NAME))


def require_admin(user: User = Depends(current_user)) -> User:
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    return user


def authorize_kb(conn: Connection, user: User, kb_id: UUID, min_role: Role = "viewer") -> Role:
    """The caller's role on the KB (admin > editor > viewer). 404 if the KB does not exist, 403 if the caller is
    not a member (admins are implicit) or ranks below `min_role`."""
    row = conn.execute(
        "SELECT m.role FROM knowledge_bases kb "
        "LEFT JOIN kb_members m ON m.kb_id = kb.id AND m.user_id = %s WHERE kb.id = %s",
        [user.id, kb_id],
    ).fetchone()
    if row is None:
        raise HTTPException(404, "Knowledge base not found")
    role: str | None = "admin" if user.is_admin else row["role"]
    if role is None:
        raise HTTPException(403, "Not a member of this knowledge base")
    if _RANK[role] < _RANK[min_role]:
        raise HTTPException(403, f"Requires {min_role} role")
    return role  # type: ignore[return-value]


def kb_access(min_role: Role = "viewer") -> Callable[..., Role]:
    """Dependency factory for routes with a `{kb_id}` path parameter: `authorize_kb` on the request's connection.

    Use: `role: Role = Depends(kb_access("editor"))`. Routes that must not hold a connection for the whole request
    (ask) call `authenticate` and `authorize_kb` themselves on a short-lived one.
    """

    def dep(kb_id: UUID, user: User = Depends(current_user), conn: Connection = Depends(get_conn)) -> Role:
        return authorize_kb(conn, user, kb_id, min_role)

    return dep
