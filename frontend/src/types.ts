// Shapes from docs/api-contract.md. Keep in sync with the contract, not with the backend code.
export type Role = "admin" | "editor" | "viewer";

export interface User {
  id: string;
  email: string;
  name: string;
  is_admin: boolean;
  is_active: boolean;
  created_at: string;
  locked_until: string | null;
}

export interface KB {
  id: string;
  slug: string;
  name: string;
  description: string;
  status: "active" | "archived";
  embedding_model: string | null;
  role: Role;
  doc_count: number;
  last_indexed_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface Member {
  user_id: string;
  email: string;
  name: string;
  role: "editor" | "viewer";
}

export type DocStatus = "queued" | "processing" | "ready" | "failed";

export interface Doc {
  id: string;
  kb_id: string;
  filename: string;
  category: string | null;
  mime: string | null;
  size_bytes: number | null;
  sha256: string;
  status: DocStatus;
  page_count: number | null;
  ocr_pages: number;
  chunk_count: number;
  error: string | null;
  indexed_at: string | null;
  created_at: string;
}

export interface Job {
  id: number;
  kb_id: string;
  document_id: string;
  filename: string;
  status: "queued" | "running" | "done" | "failed";
  attempts: number;
  error: string | null;
  stats: { pages?: number; ocr_pages?: number; chunks?: number; tokens?: number; seconds?: number };
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface Citation {
  n: number;
  document_id: string;
  filename: string;
  category: string | null;
  page_start: number | null;
  page_end: number | null;
  heading: string | null;
  snippet: string;
}

// One earlier turn sent with a question (`history` of POST /ask): at most 4, the most recent last.
export interface AskTurn {
  question: string;
  answer: string;
}

export interface AskResponse {
  answer: string;
  insufficient: boolean;
  citations: Citation[];
}

export interface QueryLog {
  id: number;
  kb_id: string;
  user_id: string | null;
  user_email: string | null;
  question: string;
  answer: string | null;
  insufficient: boolean;
  retrieved: { chunk_id: number; document_id: string; vec_sim: number | null; fts_rank: number | null; fused: number }[];
  model: string | null;
  latency_ms: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  error: string | null;
  rewritten_question: string | null; // the standalone question searched next to `question`; null without history
  history_turns: number; // earlier turns the question carried (0: single-turn)
  created_at: string;
}

export interface KbStats {
  kb_id: string;
  embedding_model: string | null;
  llm_model: string | null; // global LLM_MODEL setting
  documents: { total: number; queued: number; processing: number; ready: number; failed: number };
  jobs: { queued: number; running: number; done: number; failed: number };
  chunks: number;
  pages: number;
  ocr_pages: number;
  last_indexed_at: string | null;
}

// One question in the Ask thread. `pending` while /ask is in flight. `restored`: reopened from the history list,
// shown but not sent as context.
export interface Turn {
  id: number;
  kbId: string;
  question: string;
  status: "pending" | "done" | "error";
  answer: string;
  insufficient: boolean;
  citations: Citation[];
  error: string;
  restored: boolean;
}

// A past question kept in sessionStorage (per user, per tab) so it can be reopened without asking again.
export interface HistoryEntry {
  id: string;
  kbId: string;
  kbName: string;
  question: string;
  answer: string;
  insufficient: boolean;
  citations: Citation[];
  at: string;
}
