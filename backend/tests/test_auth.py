from fastapi.testclient import TestClient

from .conftest import PASSWORD


def login(client, user, password=PASSWORD):
    return client.post("/api/auth/login", json={"email": user.email, "password": password})


def test_login_sets_cookie(anon, make_user):
    user = make_user()
    r = login(anon, user)
    assert r.status_code == 200
    assert r.json()["email"] == user.email and "password_hash" not in r.json()
    cookie = r.headers["set-cookie"].lower()
    assert "session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie
    assert anon.get("/api/auth/me").json()["id"] == str(user.id)


def test_bad_credentials_are_401(anon, make_user):
    user = make_user()
    assert login(anon, user, "wrong-password").status_code == 401
    assert anon.post("/api/auth/login", json={"email": "nobody@test.local", "password": "x"}).status_code == 401
    assert anon.get("/api/auth/me").status_code == 401


def test_five_wrong_passwords_lock_the_account_and_a_lock_looks_like_a_wrong_password(anon, make_user, pool):
    user = make_user()
    ghost = {"email": "nobody@test.local", "password": "wrong-password"}
    for _ in range(5):
        assert login(anon, user, "wrong-password").status_code == 401
    r = login(anon, user)  # correct password, but locked: the same 401 as a wrong one, no Retry-After
    unknown = anon.post("/api/auth/login", json=ghost)
    assert (
        (r.status_code, r.json())
        == (unknown.status_code, unknown.json())
        == (401, {"detail": "Invalid email or password"})
    )
    assert "retry-after" not in r.headers
    with pool.connection() as conn:
        assert conn.execute("SELECT locked_until > now() AS l FROM users WHERE id = %s", [user.id]).fetchone()["l"]
        conn.execute("UPDATE users SET locked_until = now() - interval '1 second' WHERE id = %s", [user.id])
    assert login(anon, user).status_code == 200  # lock expired


def test_success_resets_the_failure_counter(anon, make_user):
    user = make_user()
    for _ in range(4):
        login(anon, user, "wrong-password")
    assert login(anon, user).status_code == 200
    for _ in range(4):
        assert login(anon, user, "wrong-password").status_code == 401  # would have locked at 5 total


def test_logout_invalidates_the_session_server_side(app, anon, make_user):
    user = make_user()
    login(anon, user)
    token = anon.cookies.get("session")
    assert anon.post("/api/auth/logout").status_code == 204
    replay = TestClient(app, cookies={"session": token})
    assert replay.get("/api/auth/me").status_code == 401


def test_expired_session_is_rejected(anon, make_user, pool):
    user = make_user()
    login(anon, user)
    with pool.connection() as conn:
        conn.execute("UPDATE sessions SET expires_at = now() - interval '1 second' WHERE user_id = %s", [user.id])
    assert anon.get("/api/auth/me").status_code == 401


def test_deactivating_a_user_kills_their_session(login_as, make_user):
    admin, user = make_user(is_admin=True), make_user()
    user_client, admin_client = login_as(user), login_as(admin)
    assert user_client.get("/api/auth/me").status_code == 200
    r = admin_client.patch(f"/api/users/{user.id}", json={"is_active": False})
    assert r.status_code == 200 and r.json()["is_active"] is False
    assert user_client.get("/api/auth/me").status_code == 401
    assert login(TestClient(user_client.app), user).status_code == 401


def test_password_reset_kills_sessions_and_unlocks(login_as, anon, make_user):
    admin, user = make_user(is_admin=True), make_user()
    user_client, admin_client = login_as(user), login_as(admin)
    for _ in range(5):
        login(anon, user, "wrong-password")
    assert admin_client.patch(f"/api/users/{user.id}", json={"password": "brand-new-pass"}).status_code == 200
    assert user_client.get("/api/auth/me").status_code == 401
    assert login(anon, user, "brand-new-pass").status_code == 200


def test_non_admin_cannot_manage_users(login_as, anon, make_user):
    user, other = make_user(), make_user()
    c = login_as(user)
    body = {"email": "new@test.local", "name": "New", "password": "longenough1"}
    assert c.post("/api/users", json=body).status_code == 403
    assert c.get("/api/users").status_code == 403
    assert c.patch(f"/api/users/{other.id}", json={"is_admin": True}).status_code == 403
    assert anon.post("/api/users", json=body).status_code == 401


def test_admin_creates_user_who_can_log_in(login_as, anon, make_user):
    admin = make_user(is_admin=True)
    c = login_as(admin)
    body = {"email": "  New.User@Test.Local ", "name": "New", "password": "longenough1"}
    r = c.post("/api/users", json=body)
    assert r.status_code == 201 and r.json()["email"] == "new.user@test.local" and r.json()["is_admin"] is False
    assert c.post("/api/users", json=body).status_code == 409
    assert c.post("/api/users", json={**body, "email": "x@test.local", "password": "short"}).status_code == 422
    assert (
        anon.post("/api/auth/login", json={"email": "new.user@test.local", "password": "longenough1"}).status_code
        == 200
    )


def test_admin_cannot_lock_themselves_out(login_as, make_user):
    admin = make_user(is_admin=True)
    c = login_as(admin)
    assert c.patch(f"/api/users/{admin.id}", json={"is_active": False}).status_code == 400
    assert c.patch(f"/api/users/{admin.id}", json={"is_admin": False}).status_code == 400
