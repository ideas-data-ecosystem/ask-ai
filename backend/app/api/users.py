"""Auth (login/logout/me) and admin user management."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from psycopg import Connection
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, StringConstraints

from app.auth import (
    COOKIE_NAME,
    DUMMY_HASH,
    LOCK_MINUTES,
    MAX_FAILED_LOGINS,
    SESSION_HOURS,
    USER_COLUMNS,
    User,
    create_session,
    current_user,
    delete_session,
    delete_user_sessions,
    hash_password,
    require_admin,
    verify_password,
)
from app.config import Settings, app_settings
from app.db import get_conn

router = APIRouter()

Email = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True, min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+$"),
]
Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
NewPassword = Annotated[str, StringConstraints(min_length=8, max_length=128)]


class LoginIn(BaseModel):
    email: Email
    password: Annotated[str, StringConstraints(max_length=1024)]


class UserIn(BaseModel):
    email: Email
    name: Name
    password: NewPassword
    is_admin: bool = False


class UserPatch(BaseModel):
    """Absent or null fields are left unchanged."""

    name: Name | None = None
    is_active: bool | None = None
    is_admin: bool | None = None
    password: NewPassword | None = None


@router.post("/auth/login", response_model=User)
def login(
    body: LoginIn,
    response: Response,
    conn: Connection = Depends(get_conn),
    settings: Settings = Depends(app_settings),
):
    """Every failure is the same 401, a locked account included, so the response never reveals which emails exist.

    One atomic UPDATE reserves the attempt before the password is checked: it counts the try and sets the lock
    on the fifth, and matches no row while the account is locked. A burst of parallel guesses therefore gets at
    most 5 password checks per lock window, not one per request that read the row before the first update.

    ponytail: anyone can keep a known email locked by guessing wrong 5 times per window (account-level lockout,
    no per-IP key). Add an IP or backoff key, or a proxy rate limit, if that matters.
    """
    row = conn.execute(
        "UPDATE users SET "
        "failed_logins = CASE WHEN failed_logins + 1 >= %(max)s THEN 0 ELSE failed_logins + 1 END, "
        "locked_until = CASE WHEN failed_logins + 1 >= %(max)s "
        "THEN now() + make_interval(mins => %(mins)s) ELSE locked_until END "
        "WHERE email = %(email)s AND (locked_until IS NULL OR locked_until <= now()) "
        f"RETURNING {USER_COLUMNS}, password_hash",
        {"max": MAX_FAILED_LOGINS, "mins": LOCK_MINUTES, "email": body.email},
    ).fetchone()
    if row is None:  # unknown email, or locked: burn the same time as a real check
        verify_password(body.password, DUMMY_HASH)
        raise HTTPException(401, "Invalid email or password")
    if not verify_password(body.password, row["password_hash"]) or not row["is_active"]:
        raise HTTPException(401, "Invalid email or password")
    conn.execute("UPDATE users SET failed_logins = 0, locked_until = NULL WHERE id = %s", [row["id"]])
    token = create_session(conn, row["id"])
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )
    return User.model_validate({**row, "locked_until": None})


@router.post("/auth/logout", status_code=204)
def logout(request: Request, conn: Connection = Depends(get_conn), settings: Settings = Depends(app_settings)):
    """Idempotent: 204 even without a valid session."""
    if token := request.cookies.get(COOKIE_NAME):
        delete_session(conn, token)
    response = Response(status_code=204)
    response.delete_cookie(COOKIE_NAME, path="/", httponly=True, samesite="lax", secure=settings.cookie_secure)
    return response


@router.get("/auth/me", response_model=User)
def me(user: User = Depends(current_user)):
    return user


@router.get("/users", response_model=list[User], dependencies=[Depends(require_admin)])
def list_users(conn: Connection = Depends(get_conn)):
    return conn.execute(f"SELECT {USER_COLUMNS} FROM users ORDER BY created_at, email").fetchall()


@router.post("/users", status_code=201, response_model=User, dependencies=[Depends(require_admin)])
def create_user(body: UserIn, conn: Connection = Depends(get_conn)):
    try:
        return conn.execute(
            f"INSERT INTO users (email, name, password_hash, is_admin) VALUES (%s, %s, %s, %s) RETURNING {USER_COLUMNS}",
            [body.email, body.name, hash_password(body.password), body.is_admin],
        ).fetchone()
    except UniqueViolation:
        raise HTTPException(409, "Email already in use") from None


@router.patch("/users/{user_id}", response_model=User)
def patch_user(
    user_id: UUID,
    body: UserPatch,
    admin: User = Depends(require_admin),
    conn: Connection = Depends(get_conn),
):
    """Deactivating or resetting the password also deletes the user's sessions and clears the lockout."""
    changes = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}
    if admin.id == user_id and (changes.get("is_active") is False or changes.get("is_admin") is False):
        raise HTTPException(400, "Admins cannot deactivate or demote themselves")
    kill_sessions = "password" in changes or changes.get("is_active") is False
    sets: list[str] = []
    params: dict = {"id": user_id}
    if "password" in changes:
        sets += ["password_hash = %(password_hash)s", "failed_logins = 0", "locked_until = NULL"]
        params["password_hash"] = hash_password(changes.pop("password"))
    for col, val in changes.items():  # keys come from UserPatch fields, never from the client
        sets.append(f"{col} = %({col})s")
        params[col] = val
    if sets:
        row = conn.execute(
            f"UPDATE users SET {', '.join(sets)} WHERE id = %(id)s RETURNING {USER_COLUMNS}", params
        ).fetchone()
    else:
        row = conn.execute(f"SELECT {USER_COLUMNS} FROM users WHERE id = %s", [user_id]).fetchone()
    if row is None:
        raise HTTPException(404, "User not found")
    if kill_sessions:
        delete_user_sessions(conn, user_id)
    return row
