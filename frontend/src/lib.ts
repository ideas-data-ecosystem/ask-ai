import type { Citation, Doc, HistoryEntry } from "./types.ts";

export type AnswerPart = { text: string } | { n: number };

// Split an answer into text and [n] markers. Only markers that have a citation become chips;
// anything else stays plain text (answers are rendered as text, never as HTML).
export function splitAnswer(answer: string, cited: Set<number>): AnswerPart[] {
  const parts: AnswerPart[] = [];
  let last = 0;
  for (const m of answer.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (!cited.has(n)) continue;
    if (m.index > last) parts.push({ text: answer.slice(last, m.index) });
    parts.push({ n });
    last = m.index + m[0].length;
  }
  if (last < answer.length) parts.push({ text: answer.slice(last) });
  return parts;
}

export function fmtDate(iso: string | null, empty = "-"): string {
  if (!iso) return empty;
  return new Date(iso).toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });
}

export function pageRange(c: Pick<Citation, "page_start" | "page_end">): string {
  if (c.page_start == null) return "-";
  return c.page_end != null && c.page_end !== c.page_start ? `${c.page_start}-${c.page_end}` : String(c.page_start);
}

// URL of the stored file; PDF viewers honour the #page fragment.
export function fileUrl(kbId: string, docId: string, page: number | null = null): string {
  return `/api/kbs/${kbId}/documents/${docId}/file` + (page != null ? `#page=${page}` : "");
}

export const MAX_QUESTION_CHARS = 1000; // contract default; the server stays the authority (422)

export const HISTORY_MAX = 50;

export const HISTORY_PREFIX = "chat-ai:history:";

// Remove every stored history key except `keep`. A getter, because even reading `window.localStorage` throws when
// storage is blocked.
export function wipeHistory(storage: () => Storage, keep = ""): void {
  try {
    const s = storage();
    for (let i = s.length - 1; i >= 0; i--) {
      const k = s.key(i);
      if (k?.startsWith(HISTORY_PREFIX) && k !== keep) s.removeItem(k);
    }
  } catch {
    /* storage blocked: nothing stored either */
  }
}

export const initial = (name: string) => (name.trim()[0] ?? "?").toUpperCase();

// "PP 94 Tahun 2021.pdf" -> "PP 94 Tahun 2021"
export const docTitle = (filename: string) => filename.replace(/\.[^./\\]+$/, "");

export interface SuggestionGroup {
  category: string;
  titles: string[];
}

// Empty-state cards: one per category (largest first), each with a few document titles. Ready documents only.
export function suggestionGroups(docs: Pick<Doc, "filename" | "category" | "status">[], maxGroups = 4, maxTitles = 4): SuggestionGroup[] {
  const by = new Map<string, string[]>();
  for (const d of docs) {
    if (d.status !== "ready") continue;
    const key = d.category ?? "Tanpa kategori";
    const list = by.get(key) ?? [];
    list.push(docTitle(d.filename));
    by.set(key, list);
  }
  return [...by]
    .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
    .slice(0, maxGroups)
    .map(([category, titles]) => ({ category, titles: titles.slice(0, maxTitles) }));
}

// Stored history is untrusted (old versions, hand edits): keep only well-formed entries and citations.
export function parseHistory(raw: string | null): HistoryEntry[] {
  let data: unknown;
  try {
    data = JSON.parse(raw ?? "[]");
  } catch {
    return [];
  }
  if (!Array.isArray(data)) return [];
  const strOrNull = (v: unknown) => v === null || typeof v === "string";
  const numOrNull = (v: unknown) => v === null || typeof v === "number";
  const okCitation = (c: Partial<Citation> | null): c is Citation =>
    !!c && typeof c.n === "number" && typeof c.document_id === "string" && typeof c.filename === "string" && strOrNull(c.category) && numOrNull(c.page_start) && numOrNull(c.page_end) && strOrNull(c.heading) && typeof c.snippet === "string";
  const ok = (e: Partial<HistoryEntry> | null): e is HistoryEntry =>
    !!e && typeof e.id === "string" && typeof e.kbId === "string" && typeof e.kbName === "string" && typeof e.question === "string" && typeof e.answer === "string" && typeof e.insufficient === "boolean" && Array.isArray(e.citations) && typeof e.at === "string";
  return data
    .filter(ok)
    .slice(0, HISTORY_MAX)
    .map((e) => ({ ...e, citations: e.citations.filter(okCitation) }));
}

// The async clipboard API needs https or localhost; on-prem http deployments fall back to execCommand.
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    /* fall through to the legacy path */
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.cssText = "position:fixed;opacity:0";
  document.body.append(ta);
  ta.select();
  try {
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    ta.remove();
  }
}
