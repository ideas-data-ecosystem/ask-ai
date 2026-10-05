-- Conversational follow-ups: a question asked with earlier turns is first rewritten into a standalone question, and
-- retrieval runs on both. The log keeps the rewrite (NULL: no history, or the rewrite failed or was empty) and how
-- many history turns the request carried (0: a single-turn question, as before this migration).

ALTER TABLE query_logs
  ADD COLUMN rewritten_question text,
  ADD COLUMN history_turns      smallint NOT NULL DEFAULT 0;
