"""Pure-ASGI guards that run before routing and authentication: a request body cap and a cross-site rejection.

Both answer with the API's error shape (`{"detail": ...}`). Pure ASGI (no BaseHTTPMiddleware) so a body is never
buffered by the guard itself and the cap applies to bytes as they arrive.
"""

import re
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

DEFAULT_BODY_BYTES = 64 << 10  # every JSON route; bodies are a few hundred bytes
_UPLOAD_PATH = re.compile(r"^/api/kbs/[^/]+/documents/?$")
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_SAME_ORIGIN_SITES = frozenset({"same-origin", "none"})  # "none": typed in the address bar, no page involved


class BodyLimitMiddleware:
    """413 when a request body is over the cap, before routing and before the login check.

    FastAPI parses a form or JSON body before it runs the auth dependency, so without this anyone could stream
    gigabytes to any URL. The cap is MAX_UPLOAD_MB + 1 MiB (multipart framing) on the upload route and 64 KiB
    elsewhere. A declared Content-Length over the cap is refused without reading; a body without one (chunked) or
    one that lies is counted as it is read.

    ponytail: put the same cap in the reverse proxy too, so the bytes never reach this process.
    """

    def __init__(self, app: ASGIApp, upload_bytes: int) -> None:
        self.app = app
        self.upload_bytes = upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_upload = scope["method"] == "POST" and _UPLOAD_PATH.match(scope["path"])
        limit = self.upload_bytes if is_upload else DEFAULT_BODY_BYTES
        declared = Headers(scope=scope).get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            await JSONResponse({"detail": "Request body too large"}, status_code=413)(scope, receive, send)
            return

        received = 0

        async def counted_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, "Request body too large")  # FastAPI re-raises it as the response
            return message

        await self.app(scope, counted_receive, send)


class SameOriginMiddleware:
    """403 for a state-changing /api request that the browser labels cross-site.

    SameSite=Lax only stops cross-*site* requests; a page on a sibling subdomain is same-site and still gets the
    cookie sent. So a non-GET/HEAD/OPTIONS request is refused when `Sec-Fetch-Site` is present and is not
    same-origin/none, or when `Origin` is present and is not this host. Requests without either header (curl,
    scripts) are not browser requests and are left to the session check.

    ponytail: compares Origin with the Host header, so a reverse proxy must pass Host through unchanged.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        path = scope.get("path", "")
        if (
            scope["type"] == "http"
            and scope["method"] not in _SAFE_METHODS
            and (path == "/api" or path.startswith("/api/"))
        ):
            headers = Headers(scope=scope)
            site, origin, host = headers.get("sec-fetch-site"), headers.get("origin"), headers.get("host")
            cross_site = site is not None and site not in _SAME_ORIGIN_SITES
            # "Origin: null" (sandboxed or privacy-stripped requests) has an empty netloc and never matches
            foreign_origin = origin is not None and (not host or urlsplit(origin).netloc.lower() != host.lower())
            if cross_site or foreign_origin:
                await JSONResponse({"detail": "Cross-site request rejected"}, status_code=403)(scope, receive, send)
                return
        await self.app(scope, receive, send)
