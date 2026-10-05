// Run with `npm run check` (Node strips types). No test framework on purpose.
import { ApiError, describeError, parseDetail } from "./api.ts";
import { CONTEXT_ANSWER_CHARS, CONTEXT_TURNS, contextTurns, docTitle, fileUrl, HISTORY_MAX, initial, MAX_QUESTION_CHARS, pageRange, parseHistory, splitAnswer, suggestionGroups, wipeHistory } from "./lib.ts";
import type { Turn } from "./types.ts";

function eq(actual: unknown, expected: unknown, label: string) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${label}: got ${JSON.stringify(actual)}, want ${JSON.stringify(expected)}`);
  }
}

eq(splitAnswer("A [1][3] b [2].", new Set([1, 3])), [{ text: "A " }, { n: 1 }, { n: 3 }, { text: " b [2]." }], "adjacent + unknown marker");
eq(splitAnswer("[12] x", new Set([12])), [{ n: 12 }, { text: " x" }], "multi-digit at start");
eq(splitAnswer("no markers", new Set()), [{ text: "no markers" }], "plain");
eq(splitAnswer("", new Set([1])), [], "empty");

eq(parseDetail({ detail: "Email already in use" }), "Email already in use", "string detail");
eq(parseDetail({ detail: [{ msg: "too short" }, { msg: "bad" }] }), "too short; bad", "422 list");
eq(parseDetail(null), "", "non-json body");

eq(describeError(new ApiError(409, "Knowledge base is archived")).includes("diarsipkan"), true, "409 archived");
eq(describeError(new ApiError(409, "Reindex required: model changed")).includes("diindex ulang"), true, "409 reindex");
eq(describeError(new ApiError(429, "")), "Masih ada 2 pertanyaan yang diproses; tunggu salah satunya selesai.", "429 ask in flight");
eq(describeError(new ApiError(503, "")).includes("503"), true, "5xx");
eq(describeError(new ApiError(502, "The embedding service failed")).includes("embedding"), true, "502 embedding");
eq(describeError(new ApiError(502, "The language model failed or refused to answer")).includes("Model bahasa"), true, "502 llm");
eq(describeError(new ApiError(502, "")).includes("Server API"), true, "502 from a proxy");
eq(describeError(new ApiError(0, "")).includes("terhubung"), true, "network");

eq(pageRange({ page_start: 3, page_end: 3 }), "3", "single page");
eq(pageRange({ page_start: 3, page_end: 5 }), "3-5", "range");
eq(pageRange({ page_start: null, page_end: null }), "-", "no page");
eq(fileUrl("k", "d", 12), "/api/kbs/k/documents/d/file#page=12", "file url with page");
eq(fileUrl("k", "d"), "/api/kbs/k/documents/d/file", "file url");

eq(docTitle("PP 94 Tahun 2021.pdf"), "PP 94 Tahun 2021", "title without extension");
eq(docTitle("a.b.docx"), "a.b", "only the last extension");
eq(docTitle("README"), "README", "no extension");
eq(initial("  faisal"), "F", "initial");
eq(initial(""), "?", "empty name");

const doc = (filename: string, category: string | null, status = "ready") => ({ filename, category, status: status as "ready" });
const groups = suggestionGroups([
  doc("a.pdf", "B"), doc("b.pdf", "B"), doc("c.pdf", "A"), doc("d.pdf", "B"), doc("e.pdf", "C"), doc("f.pdf", null), doc("g.pdf", "C"),
  doc("x.pdf", "B", "failed"), doc("y.pdf", "Z", "queued"), doc("h.pdf", "B"), doc("i.pdf", "B"),
]);
eq(groups.map((g) => g.category), ["B", "C", "A", "Tanpa kategori"], "largest first, ties by name, max 4, not-ready ignored");
eq(groups[0]?.titles, ["a", "b", "d", "h"], "at most 4 titles per card");
eq(suggestionGroups([doc("x.pdf", "A", "failed")]), [], "no ready documents");

const entry = { id: "1", kbId: "k", kbName: "n", question: "q", answer: "a", insufficient: false, citations: [], at: "t" };
eq(parseHistory(JSON.stringify([entry, { id: 2 }, null])), [entry], "drops malformed entries");
const cite = { n: 1, document_id: "d", filename: "f.pdf", category: null, page_start: 3, page_end: null, heading: null, snippet: "s" };
eq(parseHistory(JSON.stringify([{ ...entry, citations: [null, { ...cite, n: "1" }, { ...cite, snippet: undefined }, { ...cite, page_start: "3" }, cite] }])), [{ ...entry, citations: [cite] }], "drops malformed citations, keeps the entry");
eq(parseHistory("{not json"), [], "bad json");
eq(parseHistory(null), [], "nothing stored");
eq(parseHistory(JSON.stringify(Array.from({ length: HISTORY_MAX + 5 }, () => entry))).length, HISTORY_MAX, "capped");

const store = new Map([["chat-ai:history:a", "1"], ["other", "x"], ["chat-ai:history:b", "2"], ["chat-ai:history:c", "3"]]);
const fake = { get length() { return store.size; }, key: (i: number) => [...store.keys()][i] ?? null, removeItem: (k: string) => void store.delete(k) } as unknown as Storage;
wipeHistory(() => fake, "chat-ai:history:a");
eq([...store.keys()], ["chat-ai:history:a", "other"], "login: keeps the current user's history and foreign keys");
wipeHistory(() => fake);
eq([...store.keys()], ["other"], "logout: removes all history");
wipeHistory(() => { throw new Error("SecurityError"); }); // blocked storage must not throw

const turn = (question: string, over: Partial<Turn> = {}): Turn => ({ id: 0, kbId: "k", question, status: "done", answer: `a ${question}`, insufficient: false, citations: [], error: "", restored: false, ...over });
eq(contextTurns([], "k"), [], "no earlier turns: empty history");
eq(
  contextTurns([turn("q1"), turn("q2", { insufficient: true, answer: "Tidak ada informasi yang cukup di knowledge base ini" }), turn("q3", { status: "error", answer: "" }), turn("q4", { status: "pending" })], "k"),
  [{ question: "q1", answer: "a q1" }, { question: "q2", answer: "" }],
  "oldest first; a refused turn keeps its question with an empty answer; failed and pending turns are left out",
);
eq(contextTurns(["q1", "q2", "q3", "q4", "q5", "q6"].map((q) => turn(q)), "k").map((t) => t.question), ["q3", "q4", "q5", "q6"], "only the last 4");
eq(CONTEXT_TURNS, 4, "contract cap");
eq(contextTurns([turn("old", { restored: true }), turn("new")], "k"), [{ question: "new", answer: "a new" }], "a reopened history entry is not context");
eq(contextTurns([turn("other kb", { kbId: "x" }), turn("this kb")], "k").map((t) => t.question), ["this kb"], "never across KBs");
const [cut] = contextTurns([turn("q".repeat(MAX_QUESTION_CHARS + 9), { answer: "b".repeat(CONTEXT_ANSWER_CHARS + 9) })], "k");
eq([cut?.question.length, cut?.answer.length], [MAX_QUESTION_CHARS, CONTEXT_ANSWER_CHARS], "fields cut to the contract caps");

console.log("lib checks passed");
