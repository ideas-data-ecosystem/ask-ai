-- Embedding fingerprint (model|doc prefix|doc extra body|query prefix|query extra body) and the query-mode canary.
-- A KB's vectors are only valid for the setup they were made with, so a KB stores the fingerprint it was indexed
-- with, next to the document-mode canary (embedding_canary) and now the query-mode canary, which every question's
-- embedding request is compared against.

ALTER TABLE knowledge_bases
  ADD COLUMN embedding_fingerprint  text,
  ADD COLUMN embedding_query_canary vector({{EMBEDDING_DIM}});

-- Rows from before this migration only know the bare model name. It never equals a real fingerprint (those contain
-- '|'), so such a KB answers "reindex required" instead of pretending to be unindexed.
UPDATE knowledge_bases SET embedding_fingerprint = embedding_model WHERE embedding_model IS NOT NULL;
UPDATE ingestion_jobs SET stats = stats || jsonb_build_object('embedding_fingerprint', stats->>'embedding_model')
  WHERE stats ? 'embedding_model';
