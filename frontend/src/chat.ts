// State of the Ask page, shared with the sidebar ("Pertanyaan baru", history). No state library needed.
// The request lives here, not in the view, so an answer still lands when the user opens another page meanwhile.
import { reactive, watch } from "vue";
import { api, describeError } from "./api.ts";
import { HISTORY_MAX, HISTORY_PREFIX, parseHistory, wipeHistory } from "./lib.ts";
import { session } from "./session.ts";
import type { AskResponse, HistoryEntry, Turn } from "./types.ts";

export const chat = reactive({
  kbId: "",
  turns: [] as Turn[],
  draft: "",
  busy: false,
  wantFocus: false, // set by "Pertanyaan baru"; the Ask view focuses the composer and clears it
  history: [] as HistoryEntry[],
});

let gen = 0; // bumped whenever the thread is reset, so a late answer never lands in the new thread
let seq = 0;

// History holds answers and snippets from KBs the next user of a shared browser may not be allowed to read,
// so it lives in sessionStorage (gone with the tab) and is removed as soon as the user goes away.
const key = (userId: string) => HISTORY_PREFIX + userId;

// ponytail: earlier builds kept history in localStorage; drop this line once every browser has loaded a newer build.
wipeHistory(() => localStorage);

function saveHistory() {
  const id = session.me?.id;
  if (!id) return;
  try {
    sessionStorage.setItem(key(id), JSON.stringify(chat.history));
  } catch {
    /* private mode or quota: history just stays in memory */
  }
}

function resetThread() {
  gen++;
  chat.turns = [];
  chat.busy = false;
}

// A different (or no) user: drop the previous thread and load that user's history.
// Logout, a 401 (expiry, deactivation) or another login also removes every other user's stored history.
watch(
  () => session.me?.id,
  (id, prev) => {
    resetThread();
    chat.kbId = "";
    chat.draft = "";
    if (id || prev) wipeHistory(() => sessionStorage, id ? key(id) : ""); // not on boot, before /auth/me has answered
    let raw: string | null = null;
    try {
      raw = id ? sessionStorage.getItem(key(id)) : null;
    } catch {
      /* storage blocked */
    }
    chat.history = parseHistory(raw);
  },
  { immediate: true },
);

export function selectKb(id: string) {
  if (id === chat.kbId) return;
  chat.kbId = id;
  resetThread(); // a thread belongs to one KB
}

export function newThread() {
  resetThread();
  chat.draft = "";
  chat.wantFocus = true;
}

export function restore(e: HistoryEntry) {
  resetThread();
  chat.kbId = e.kbId;
  chat.turns.push({ id: ++seq, kbId: e.kbId, question: e.question, status: "done", answer: e.answer, insufficient: e.insufficient, citations: e.citations, error: "" });
}

export function removeHistory(id: string) {
  chat.history = chat.history.filter((e) => e.id !== id);
  saveHistory();
}

export async function ask(): Promise<void> {
  const question = chat.draft.trim();
  const kbId = chat.kbId;
  if (!question || !kbId || chat.busy) return;
  const g = gen;
  const uid = session.me?.id; // an answer that arrives after a user switch must not land in the next user's history
  chat.turns.push({ id: ++seq, kbId, question, status: "pending", answer: "", insufficient: false, citations: [], error: "" });
  const turn = chat.turns[chat.turns.length - 1]!; // the reactive proxy, so updates re-render
  chat.draft = "";
  chat.busy = true;
  try {
    const r = await api<AskResponse>(`/kbs/${kbId}/ask`, { method: "POST", json: { question } });
    const kbName = session.kbs?.find((k) => k.id === kbId)?.name ?? "";
    const entry: HistoryEntry = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 8), kbId, kbName, question, answer: r.answer, insufficient: r.insufficient, citations: r.citations, at: new Date().toISOString() };
    if (session.me?.id === uid) {
      chat.history = [entry, ...chat.history].slice(0, HISTORY_MAX);
      saveHistory();
    }
    if (g === gen) Object.assign(turn, { status: "done", answer: r.answer, insufficient: r.insufficient, citations: r.citations });
  } catch (e) {
    if (g === gen) {
      Object.assign(turn, { status: "error", error: describeError(e) });
      if (!chat.draft) chat.draft = question; // let the user retry without retyping
    }
  } finally {
    if (g === gen) chat.busy = false;
  }
}
