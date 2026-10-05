"""FastAPI app factory. Run: uvicorn app.main:create_app --factory"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.api import ask, documents, kbs, logs, users
from app.config import Settings, get_settings
from app.db import check_db, open_pool
from app.middleware import BodyLimitMiddleware, SameOriginMiddleware


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        pool = open_pool(settings)
        try:
            with pool.connection() as conn:
                check_db(conn, settings.embedding_dim)  # raises StartupCheckError: the app refuses to start
        except BaseException:
            pool.close()
            raise
        app.state.pool = pool
        yield
        pool.close()

    app = FastAPI(
        title="IDEAS Ask",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.settings = settings
    app.add_middleware(SameOriginMiddleware)
    app.add_middleware(BodyLimitMiddleware, upload_bytes=(settings.max_upload_mb + 1) << 20)  # outermost: runs first

    @app.get("/api/health")
    def health() -> dict[str, str]:
        try:
            with app.state.pool.connection() as conn:
                conn.execute("SELECT 1")
        except Exception:  # noqa: BLE001
            raise HTTPException(503, "Database unavailable") from None
        return {"status": "ok"}

    app.include_router(users.router, prefix="/api")
    app.include_router(kbs.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(ask.router, prefix="/api")
    app.include_router(logs.router, prefix="/api")

    # Unknown /api paths are 404 for every method, not caught by the (GET-only) SPA fallback below, which would
    # answer a POST with 405. Registered after the real routes, which match first.
    @app.api_route(
        "/api/{rest:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    def api_not_found(rest: str) -> None:
        raise HTTPException(404, "Not Found")

    static = settings.static_dir
    if static.is_dir():
        root = static.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (root / path).resolve()
            if candidate.is_file() and candidate.is_relative_to(root):
                return FileResponse(candidate)
            index = root / "index.html"
            if not index.is_file():
                raise HTTPException(404, "Frontend not built")
            return FileResponse(index)

    return app
