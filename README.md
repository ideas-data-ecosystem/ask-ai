# IDEAS Ask

A web app that answers questions from isolated document collections (knowledge bases) and cites its sources. It is not a
generic chatbot: it never searches the web or falls back on the model's own knowledge, and when the sources do not
answer the question it says so. Every user logs in, and access is by knowledge base membership (admin, editor, viewer).
Isolation is enforced in the backend and in Postgres row-level security, not by prompting. Stack: FastAPI, Postgres 17
with pgvector (hybrid vector + full-text retrieval), an ingestion worker with OCR, and a Vue 3 SPA. Embeddings come from
any OpenAI-compatible `/embeddings` endpoint; the LLM speaks either the OpenAI Chat Completions or the Anthropic
Messages protocol. The HTTP contract is in `docs/api-contract.md`.

## Prerequisites

- podman and podman-compose (`podman compose ...`; there is no Docker in this setup)
- uv (Python 3.12 is managed by it)
- node and npm (for the SPA)
- For host runs of the worker only: tesseract with the `ind` language data (the container image already has it)

## First run

1. Configure.

   ```sh
   cp .env.example .env
   ```

   Compose refuses to start without `POSTGRES_PASSWORD`, `RAG_APP_PASSWORD` and `EMBEDDING_DIM` (the vector size of your
   embedding model; fixed when the database is first migrated). Ingestion and questions also need
   `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` and the `LLM_*` settings. Replace every `change-me`
   placeholder. Keep passwords to letters, digits, `-` and `_`. The session cookie is Secure by default; for a trial
   over plain `http://` uncomment `COOKIE_SECURE=false`, or login will not stick.

2. Build the SPA and create the uploads directory. Both are bind-mounted into the api container, so they must exist
   before the stack starts.

   ```sh
   (cd frontend && npm ci && npm run build)
   mkdir -p knowledges/uploads
   ```

   The directory must be writable by the image user (uid 10001). On Linux with rootless podman run
   `podman unshare chown 10001:10001 knowledges/uploads` (see the comment in `compose.yaml`).

3. Start the stack (db, one-shot migrate, api, worker). The API listens on `127.0.0.1:8000` (`API_PORT`).

   ```sh
   podman compose up -d --build
   ```

4. Create the first admin. Omit `--password` to be prompted. Other users are created by admins in the app.

   ```sh
   podman compose exec api python -m app.cli create-admin --email you@example.com --name "Your Name"
   ```

## Seeding the corpus

The PDFs under `knowledges/` are registered where they are (no copy, no move); each subfolder becomes the document
category. The command creates the knowledge base if it does not exist, skips `uploads/`, hidden files and unsupported
types, and is safe to repeat (content already in the KB is skipped).

```sh
podman compose exec api python -m app.cli seed --kb "Regulasi ASN" /data/knowledges
podman compose logs -f worker   # ingestion progress; the document list in the app shows each status
```

Four of the documents are scans and 544 pages need OCR, so the first ingestion is slow. Seeded files are never deleted by
the app.

## Local development

Only the database runs in podman; the api, worker and SPA run on the host. Host runs read the repo's `.env`
(`../.env` from `backend/`), so in it uncomment `STORAGE_DIR=../knowledges` and `STATIC_DIR=../frontend/dist` and set
`COOKIE_SECURE=false` (a browser does not send a Secure cookie over plain http).

```sh
podman compose up -d db                       # published on 127.0.0.1:5433
cd backend
uv run python -m app.cli migrate              # as the owner role; needs POSTGRES_PASSWORD in .env
uv run python -m app.cli create-admin --email you@example.com
uv run uvicorn app.main:create_app --factory  # http://127.0.0.1:8000, docs at /api/docs
uv run python -m app.worker                   # second terminal
cd ../frontend && npm ci && npm run dev       # Vite dev server, proxies /api to 127.0.0.1:8000
```

Tests need the db up on `127.0.0.1:5433` and `POSTGRES_PASSWORD` and `RAG_APP_PASSWORD` in the environment or `.env`.
They create and drop a throwaway database, and fake the embedding and LLM calls.

```sh
cd backend
uv run pytest -q
uvx ruff check . && uvx ruff format --check .
cd ../frontend && npm run check
```

## Evaluating retrieval

`backend/eval/golden.jsonl` holds in-scope questions with their expected source documents and out-of-scope questions that
must be declared insufficient. The run uses the real embedding and LLM endpoints from `.env` against an indexed KB:

```sh
cd backend && uv run python -m eval.run_eval --kb "Regulasi ASN" -v
```

It reports the top-k hit rate, the answered and insufficient rates, and the best-similarity range of in-scope versus
out-of-scope questions. Set `MIN_SIMILARITY` in `.env` from that range (the evidence gate: below it the app answers
"not enough information" without calling the LLM), then recreate the api (`podman compose up -d`). Optional flags:
`--no-llm` (retrieval and gate only) and `--other-kb "<name>"` (in-scope questions asked in an unrelated KB must be
insufficient).

## Embedding notes

- The embedding setup is fingerprinted: model, document prefix, query prefix and both extra bodies. Changing any of
  them means the stored vectors no longer match, so the KB answers "reindex required" until it is reindexed (the KB's
  reindex action in the app, or `POST /api/kbs/{id}/reindex`). It never serves a mix of two setups.
- `EMBEDDING_DIM` is fixed when the database is first migrated. A different size needs a new migration or a fresh
  database volume; the app refuses to start on a mismatch.
- Asymmetric models need to tell documents from questions: `EMBEDDING_DOC_PREFIX` / `EMBEDDING_QUERY_PREFIX` for text
  prefixes (e5, qwen3), `EMBEDDING_DOC_EXTRA_BODY` / `EMBEDDING_QUERY_EXTRA_BODY` for JSON merged into the request.
- NVIDIA Nemotron embed needs `input_type` through the extra-body variables (examples in `.env.example`) and
  `EMBEDDING_DIM=2048`.
- Turn off model fallback for the embedding route in the gateway. Every embedding request carries a fixed canary
  sentence and the call fails if its vector drifts from the one stored for the KB (cosine below 0.99), which catches a
  silent fallback to another model, but a gateway that never falls back is the real fix.

## Deployment checklist

For a pilot, keep the API on an internal network or VPN. Both published ports are bound to `127.0.0.1`; put a reverse
proxy in front rather than rebinding them.

- Reverse proxy with TLS.
- Request body limit in the proxy: `MAX_UPLOAD_MB` + 1 MiB on `POST /api/kbs/*/documents`, 64 KiB elsewhere (the app
  enforces the same caps, but the proxy keeps the bytes out of the process).
- Per-IP rate limits on `POST /api/auth/login` and `POST /api/kbs/*/documents`. The app only locks an account after 5
  failed logins and limits concurrent questions per user; it has no IP rate limit.
- Security headers from the proxy: HSTS, a Content-Security-Policy, `frame-ancestors` (or X-Frame-Options) and
  Referrer-Policy.
- Pass the `Host` header through unchanged. The CSRF check compares `Origin` with `Host` on every state-changing
  `/api` request.
- `COOKIE_SECURE=true`: leave it unset in `.env` (the default is secure) and do not ship `false`.
- Single worker: run exactly one `worker` container and one uvicorn process for the api. The worker resets jobs left
  `running` when it starts, and the per-user question limit is counted in memory per process.
- `RAG_APP_PASSWORD` is only applied when the `rag_app` role is first created; rotate it later with
  `ALTER ROLE rag_app PASSWORD ...` as the owner, then update `.env`.
