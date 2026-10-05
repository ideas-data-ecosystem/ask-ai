# API contract

Single source for the frontend. Every path is under `/api`. Everything below is implemented and tested (section headings show the phase it was built in); if an implementation needs a change, change this file in the same step.

## Conventions

- JSON in, JSON out (`Content-Type: application/json`), except the multipart upload and the file download.
- Auth: the cookie `session` (HttpOnly, SameSite=Lax, Secure unless `COOKIE_SECURE=false`, which is for local `http://` development only; 8 h). The SPA is same-origin, so use `credentials: "same-origin"`.
- CSRF: no token is needed. SameSite=Lax stops cross-*site* requests, but a page on a sibling subdomain is same-site and would still send the cookie, so the server also refuses a non-`GET`/`HEAD`/`OPTIONS` request to `/api` with `403 {"detail": "Cross-site request rejected"}` when the `Sec-Fetch-Site` header is present and is not `same-origin` or `none`, or when an `Origin` header is present and does not match the request's `Host`. Browsers send these headers on their own, so the SPA needs nothing; clients without them (curl, scripts) are not affected. A reverse proxy must forward `Host` unchanged. No state-changing `GET` exists.
- Request body limits, checked before routing and before login: `413 {"detail": "Request body too large"}` when a body is over 64 KiB on every route except the upload, whose limit is `MAX_UPLOAD_MB` + 1 MiB. Declared (`Content-Length`) and streamed (chunked) bodies are both counted.
- `error` fields (`Document.error`, `Job.error`, `QueryLog.error`) hold a short generic message (for example `Embedding request failed after 5 attempts (HTTP 503)`); the provider's own error text is only in the server log.
- Dev: Vite proxies `/api` to the FastAPI port, so cookies stay same-origin.
- IDs are UUID strings, except `job.id`, `query_log.id` and `chunk_id`, which are integers.
- Timestamps are ISO 8601 with a timezone (`2026-10-04T15:44:54.667670Z`). `null` is always an explicit JSON `null`, never an omitted key, in responses.
- PATCH bodies: omitted or `null` fields mean "leave unchanged".
- Interactive docs for what is implemented so far: `/api/docs`, schema `/api/openapi.json`.

### Roles

| Role | Who | Can |
|---|---|---|
| `admin` | `users.is_admin`; implicit on every KB | everything, sees all KBs |
| `editor` | KB member | viewer rights + documents (upload, delete, reindex), logs of that KB. Not members: those are admin only |
| `viewer` | KB member | read the KB, list/open documents, ask |

Rank: admin > editor > viewer. "viewer+" means viewer, editor or admin. A KB-level 404 is checked before 403: unknown KB id gives 404, a real KB the caller is not a member of gives 403.

### Shared error shape

Every error is JSON with `detail`:

```json
{ "detail": "Knowledge base not found" }
```

Validation errors (422) carry FastAPI's list form instead of a string:

```json
{ "detail": [ { "type": "string_too_short", "loc": ["body", "password"], "msg": "String should have at least 8 characters", "input": "abc" } ] }
```

Client rule: show `detail` when it is a string, otherwise join the `msg` values. Status codes used everywhere:

| Code | Meaning |
|---|---|
| 401 | no valid session (also deactivated user, expired or logged-out session): go to the login page |
| 403 | authenticated but not allowed (not a member, or role too low), or a cross-site state-changing request (see CSRF) |
| 404 | unknown KB, document, user or route |
| 409 | conflict (duplicate, archived KB, state conflict) |
| 413 | request body too large (any route), or an upload over `MAX_UPLOAD_MB` |
| 415 | upload type not allowed |
| 422 | invalid body, query, or path (for example a malformed UUID) |
| 429 | `ask` only: the caller already has 2 questions being answered |
| 502 | embedding or LLM upstream failed |
| 503 | `/api/health` only: database unavailable |

## Shared object shapes

```ts
type Role = "admin" | "editor" | "viewer";

interface User {
  id: string; email: string; name: string;
  is_admin: boolean; is_active: boolean;
  created_at: string; locked_until: string | null;   // non-null and in the future = currently locked
}

interface KB {
  id: string; slug: string; name: string; description: string;
  status: "active" | "archived";
  embedding_model: string | null;   // null until the first successful index
  role: Role;                       // the caller's role on this KB
  doc_count: number; last_indexed_at: string | null;
  created_at: string; updated_at: string;
}

interface Member { user_id: string; email: string; name: string; role: "editor" | "viewer"; }

interface Document {
  id: string; kb_id: string; filename: string; category: string | null;
  mime: string | null; size_bytes: number | null; sha256: string;
  status: "queued" | "processing" | "ready" | "failed";
  page_count: number | null; ocr_pages: number; chunk_count: number;
  error: string | null; indexed_at: string | null; created_at: string;
}

interface Job {
  id: number; kb_id: string; document_id: string; filename: string;
  status: "queued" | "running" | "done" | "failed";
  attempts: number; error: string | null;
  stats: { pages?: number; ocr_pages?: number; chunks?: number; tokens?: number; seconds?: number };  // {} until done
  created_at: string; started_at: string | null; finished_at: string | null;
}

interface Citation {
  n: number; document_id: string; filename: string; category: string | null;
  page_start: number | null; page_end: number | null; heading: string | null;
  snippet: string;                  // at most 300 characters of the chunk text
}

interface QueryLog {
  id: number; kb_id: string; user_id: string | null; user_email: string | null;
  question: string; answer: string | null; insufficient: boolean;
  retrieved: { chunk_id: number; document_id: string; vec_sim: number | null; fts_rank: number | null; fused: number }[];
  model: string | null; latency_ms: number | null;
  prompt_tokens: number | null; completion_tokens: number | null;
  error: string | null; created_at: string;
}
```

---

## Health (phase 1)

### `GET /api/health`
No auth. `200 {"status": "ok"}`; `503` when the database is unreachable.

## Auth (phase 1)

### `POST /api/auth/login`
No auth. Body: `{ "email": string, "password": string }` (email is trimmed and lowercased).
- `200` body `User`, header `Set-Cookie: session=...`.
- `401` `{"detail": "Invalid email or password"}` for an unknown email, wrong password, deactivated user, or a locked account (same message and no extra headers for all four, so the response never reveals which emails exist). The account is locked for 15 minutes after 5 consecutive wrong passwords; while locked even the correct password gets this `401`. `User.locked_until` (admin `GET /api/users`) shows an active lock. There is no `429` on login.
- `422` malformed email. `413` body over 64 KiB.

### `POST /api/auth/logout`
Any caller. `204`, no body. Deletes the server-side session and clears the cookie. Idempotent (also `204` without a session).

### `GET /api/auth/me`
Any logged-in user. `200` body `User`. `401` otherwise.

## Users (phase 1, admin only)

### `GET /api/users`
`200` body `User[]` ordered by creation. `401`, `403`.

### `POST /api/users`
Body: `{ "email": string, "name": string (1-200), "password": string (8-128), "is_admin"?: boolean = false }`.
- `201` body `User`.
- `409` `{"detail": "Email already in use"}`. `422` invalid fields. `401`, `403`.

### `PATCH /api/users/{user_id}`
Body (all optional): `{ "name"?: string, "is_active"?: boolean, "is_admin"?: boolean, "password"?: string (8-128) }`.
- `200` body `User`.
- Setting `is_active: false` or a new `password` deletes all of that user's sessions immediately; a new password also clears the lockout. Changing your own password therefore logs you out.
- `400` when an admin sets `is_active: false` or `is_admin: false` on their own account.
- `404` unknown user. `401`, `403`, `422`.

## Knowledge bases (phase 1)

### `GET /api/kbs`
Any logged-in user. `200` body `KB[]` ordered by name: all KBs for an admin, only KBs the caller is a member of otherwise.

### `POST /api/kbs`
Admin. Body: `{ "name": string (1-200), "description"?: string (<=2000) = "", "slug"?: string }`. `slug` matches `^[a-z0-9]+(-[a-z0-9]+)*$` (max 64); when omitted it is derived from the name (accents stripped, non-alphanumerics become `-`).
- `201` body `KB` (`role: "admin"`, `doc_count: 0`).
- `409` slug already used. `422` invalid, or no slug derivable from the name. `401`, `403`.

### `GET /api/kbs/{kb_id}`
viewer+. `200` body `KB`. `401`, `403`, `404`.

### `PATCH /api/kbs/{kb_id}`
Admin only (editors cannot rename or archive). Body (all optional): `{ "name"?: string, "description"?: string, "status"?: "active" | "archived" }`.
- `200` body `KB`.
- An archived KB still serves reads, document listing, file download and logs; it rejects upload, reindex and ask with `409 {"detail": "Knowledge base is archived"}`.
- `401`, `403`, `404`, `422`.

### `DELETE /api/kbs/{kb_id}`
Admin only. `204`. Cascades to documents, chunks, jobs, query logs and members, then removes the KB's uploaded files. Seeded files under `knowledges/` are never deleted. `401`, `403`, `404`.

### `GET /api/kbs/{kb_id}/stats`
viewer+. `200`:

```json
{
  "kb_id": "uuid",
  "embedding_model": "string | null",
  "llm_model": "string | null",
  "documents": { "total": 42, "queued": 0, "processing": 0, "ready": 42, "failed": 0 },
  "jobs": { "queued": 0, "running": 0, "done": 42, "failed": 0 },
  "chunks": 2012,
  "pages": 1806,
  "ocr_pages": 310,
  "last_indexed_at": "timestamp | null"
}
```
All counts are integers. `401`, `403`, `404`.

### `GET /api/kbs/{kb_id}/members`
Admin only. `200` body `Member[]` ordered by email. Admins are implicit and not listed unless they are also members. `401`, `403`, `404`.

### `PUT /api/kbs/{kb_id}/members/{user_id}`
Admin only. Body: `{ "role": "editor" | "viewer" }`. Adds the user or changes their role (upsert). `{user_id}` is a user UUID (an admin gets it from `GET /api/users`).
- `200` body `Member`.
- `404` unknown user. `422` bad role. `401`, `403`.

### `DELETE /api/kbs/{kb_id}/members/{user_id}`
Admin only. `204`. `404` when the user is not a member. `401`, `403`.

## Documents (phase 2)

### `GET /api/kbs/{kb_id}/documents`
viewer+. Query: `status?` (`queued|processing|ready|failed`), `category?` (exact match), `limit?` (1-1000, default 200), `offset?` (default 0). Ordered by `category` (nulls last) then `filename`. `200` body `Document[]`. `401`, `403`, `404`, `422`.

### `POST /api/kbs/{kb_id}/documents`
editor+. `multipart/form-data` with exactly one file per request:
- `file`: required. Extension (case-insensitive) in `pdf, docx, md, txt`.
- `category`: optional text, at most 200 characters.

The original filename and category are stored as data only. `201` body `Document` with `status: "queued"` (a job is enqueued in the same transaction).
- `400` empty file, or more than one file or form field (the body takes exactly `file` plus optional `category`). Login and KB role are checked before the body is parsed. `409` the same content (sha256) already exists in this KB, or the KB is archived. `413` the file is larger than `MAX_UPLOAD_MB`, or the whole request body is larger than `MAX_UPLOAD_MB` + 1 MiB (answered before the login check, so it also applies without a session). `415` extension not allowed. `422` missing `file`. `401`, `403`, `404`.
- The worker later fails the document (status `failed`, a clear `error`) when a PDF has more than `MAX_PDF_PAGES` pages (default 1000) or needs OCR on more than `MAX_OCR_PAGES` pages (default 300), or when a DOCX unpacks to more than 100 MB or is compressed more than 100:1. Retrying cannot help, so such a document is not retried automatically.

### `DELETE /api/kbs/{kb_id}/documents/{doc_id}`
editor+. `204`. Removes the document, its chunks and jobs, and its file only when it lives under `knowledges/uploads/`. `404` when the document does not exist in this KB (a document of another KB is also `404`). `401`, `403`.

### `POST /api/kbs/{kb_id}/documents/{doc_id}/reindex`
editor+. `202` body `Job` (new, `status: "queued"`). Also the retry action for `failed` documents.
- `409` the document already has a queued or running job, or the KB is archived. `404`, `401`, `403`.

### `GET /api/kbs/{kb_id}/documents/{doc_id}/file`
viewer+. `200` the stored file with its `Content-Type` and `Content-Disposition: inline`. A citation opens it with a URL fragment: `/api/kbs/{kb_id}/documents/{doc_id}/file#page=3` (browsers' PDF viewers honour it).
- `404` when the document does not exist in this KB or the file is missing on disk. `401`, `403`.

### `POST /api/kbs/{kb_id}/reindex`
editor+. Enqueues one job for every document of the KB that has no queued or running job. `202 {"enqueued": number}` (can be `0`). The KB keeps answering "reindex required" until the last job finishes when the embedding setup (model, prefixes or extra bodies) changed.
- `409` KB archived. `401`, `403`, `404`.

## Ask (phase 3)

### `POST /api/kbs/{kb_id}/ask`
viewer+. Body: `{ "question": string }`, trimmed length 1 to `MAX_QUESTION_CHARS` (default 1000). Single turn, no streaming.

`200`:

```json
{
  "answer": "Pemberhentian PNS diatur dalam ... [1][3]",
  "insufficient": false,
  "citations": [
    {
      "n": 1,
      "document_id": "uuid",
      "filename": "PP 94 Tahun 2021.pdf",
      "category": "08 - Berhenti dari ASN",
      "page_start": 12,
      "page_end": 12,
      "heading": "Pasal 87",
      "snippet": "..."
    }
  ]
}
```

- `answer` carries `[n]` markers; each marker present in `answer` has exactly one entry in `citations` with the same `n`. `citations` is ordered by `n` and contains only sources that were cited.
- When `insufficient` is `true` (evidence gate, the model's `INSUFFICIENT_CONTEXT`, or an answer without a valid citation), `answer` is exactly `Tidak ada informasi yang cukup di knowledge base ini` and `citations` is `[]`. This is a normal `200`, not an error.
- `409` the KB was indexed with a different embedding setup (model, prefixes or extra bodies) than the configured one, or its index is incomplete (`detail` starts with `Reindex required`), or the KB is archived.
- `502` the embedding or LLM call failed or the model refused; the attempt is still written to the query log with `error`.
- `429` the caller already has 2 questions in flight (per user, per server process): wait for one to finish. Not written to the query log.
- `422` empty or too long question. `401`, `403`, `404`.

## Logs (phase 3 for queries, phase 2 for jobs)

Both are editor+ (admins included; viewers get `403`). Newest first.

### `GET /api/kbs/{kb_id}/jobs` (phase 2)
Query: `status?` (`queued|running|done|failed`), `document_id?`, `limit?` (1-500, default 100), `offset?` (default 0). `200` body `Job[]` ordered by `id` descending. `401`, `403`, `404`, `422`.

### `GET /api/kbs/{kb_id}/queries` (phase 3)
Query: `insufficient?` (boolean), `limit?` (1-500, default 100), `offset?` (default 0). `200` body `QueryLog[]` ordered by `id` descending. `401`, `403`, `404`, `422`.

## Everything else

Any unknown `/api/...` path returns `404 {"detail": "Not Found"}` for every HTTP method (a known path with the wrong method also answers `404`). Non-API paths are served by the SPA (static files, falling back to `index.html`).
