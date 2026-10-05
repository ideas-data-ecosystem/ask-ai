-- Initial schema. Run by app.db.migrate as the owner role.
-- The runner substitutes {{EMBEDDING_DIM}} (validated int) and sets the transaction-local
-- GUC app.rag_password (used below, never written to a file).

CREATE EXTENSION IF NOT EXISTS vector;

-- Application role: no superuser, no BYPASSRLS, no TRUNCATE (granted below table by table).
-- Roles are cluster-wide: the password is set only when the role is created. A later database on the same
-- cluster must not reset the password of a role another database's running app already uses.
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_app') THEN
    EXECUTE format(
      'CREATE ROLE rag_app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE PASSWORD %L',
      current_setting('app.rag_password'));
  ELSE
    ALTER ROLE rag_app LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;
  END IF;
END $$;

CREATE TABLE users (
  id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  email         text NOT NULL UNIQUE,
  name          text NOT NULL,
  password_hash text NOT NULL,
  is_admin      boolean NOT NULL DEFAULT false,
  is_active     boolean NOT NULL DEFAULT true,
  failed_logins integer NOT NULL DEFAULT 0,
  locked_until  timestamptz,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
  token_hash text PRIMARY KEY,
  user_id    uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at timestamptz NOT NULL
);
CREATE INDEX sessions_user_id_idx ON sessions (user_id);

CREATE TABLE knowledge_bases (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  slug             text NOT NULL UNIQUE,
  name             text NOT NULL,
  description      text NOT NULL DEFAULT '',
  status           text NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived')),
  embedding_model  text,
  embedding_canary vector({{EMBEDDING_DIM}}),
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE kb_members (
  kb_id   uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  user_id uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  role    text NOT NULL CHECK (role IN ('editor', 'viewer')),
  PRIMARY KEY (kb_id, user_id)
);
CREATE INDEX kb_members_user_id_idx ON kb_members (user_id);

CREATE TABLE documents (
  id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  kb_id        uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  filename     text NOT NULL,
  category     text,
  storage_path text NOT NULL,
  mime         text,
  size_bytes   bigint,
  sha256       text NOT NULL,
  status       text NOT NULL DEFAULT 'queued'
               CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
  page_count   integer,
  ocr_pages    integer NOT NULL DEFAULT 0,
  chunk_count  integer NOT NULL DEFAULT 0,
  error        text,
  indexed_at   timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (kb_id, sha256),
  UNIQUE (id, kb_id)
);

-- Both columns NOT NULL: with MATCH SIMPLE a NULL would skip the composite FK check.
-- ponytail: exact scan over one KB's rows via the kb_id btree (full recall, milliseconds at ~2k chunks).
-- No HNSW (2,000-dim cap, model not fixed) and no GIN (@@ is not leakproof, unusable under RLS).
-- Upgrade past ~100k chunks per KB: HNSW with iterative scan, or partition by KB.
CREATE TABLE document_chunks (
  id          bigserial PRIMARY KEY,
  kb_id       uuid NOT NULL,
  document_id uuid NOT NULL,
  chunk_index integer NOT NULL,
  page_start  integer,
  page_end    integer,
  heading     text,
  content     text NOT NULL,
  tsv         tsvector GENERATED ALWAYS AS (to_tsvector('indonesian', content)) STORED,
  embedding   vector({{EMBEDDING_DIM}}) NOT NULL,
  FOREIGN KEY (document_id, kb_id) REFERENCES documents (id, kb_id) ON DELETE CASCADE
);
CREATE INDEX document_chunks_kb_id_idx ON document_chunks (kb_id);

ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;
CREATE POLICY kb_isolation ON document_chunks
  USING      (kb_id = nullif(current_setting('app.kb_id', true), '')::uuid)
  WITH CHECK (kb_id = nullif(current_setting('app.kb_id', true), '')::uuid);

CREATE TABLE ingestion_jobs (
  id          bigserial PRIMARY KEY,
  kb_id       uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  document_id uuid NOT NULL,
  status      text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed')),
  attempts    integer NOT NULL DEFAULT 0,
  error       text,
  stats       jsonb NOT NULL DEFAULT '{}',
  created_at  timestamptz NOT NULL DEFAULT now(),
  started_at  timestamptz,
  finished_at timestamptz,
  FOREIGN KEY (document_id, kb_id) REFERENCES documents (id, kb_id) ON DELETE CASCADE
);
CREATE INDEX ingestion_jobs_kb_id_idx ON ingestion_jobs (kb_id, id DESC);

CREATE TABLE query_logs (
  id                bigserial PRIMARY KEY,
  kb_id             uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
  user_id           uuid REFERENCES users(id) ON DELETE SET NULL,
  question          text NOT NULL,
  retrieved         jsonb NOT NULL DEFAULT '[]',
  answer            text,
  insufficient      boolean NOT NULL DEFAULT false,
  model             text,
  latency_ms        integer,
  prompt_tokens     integer,
  completion_tokens integer,
  error             text,
  created_at        timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX query_logs_kb_id_idx ON query_logs (kb_id, id DESC);

-- No TRUNCATE (it ignores RLS), no DDL. schema_migrations is not granted.
GRANT SELECT, INSERT, UPDATE, DELETE ON
  users, sessions, knowledge_bases, kb_members, documents, document_chunks, ingestion_jobs, query_logs
  TO rag_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO rag_app;
