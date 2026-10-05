"""Golden-set evaluation against a real, indexed knowledge base (real embedding and LLM endpoints).

    cd backend && uv run python -m eval.run_eval --kb "Regulasi ASN" [--no-llm] [--other-kb "Other KB"] [-v]
        [--repeat 3] [--llm-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}']
        [--rewrite-extra-body '{"chat_template_kwargs": {"enable_thinking": false}}']
        [--min-similarity 0.15] [--fusion-weight 0.8] [--golden x.jsonl] [--out x.json]

golden.jsonl lines are {"intent", "question", "expected": [label, ...]} for in-scope questions and
{"intent", "question", "expect_insufficient": true} for out-of-scope ones. A label {"file", "pasal", "text"} (any
subset) names the chunks that hold the answer: the filename starts with "file", the chunk heading "Pasal N" or
"Pasal N–M" covers "pasal" (Penjelasan chunks do not), and the content contains "text" (case and whitespace
insensitive). Labels survive a reindex; chunk ids do not. A follow-up also has "history": [{"question", "answer"}, ...]
(at most 4 earlier turns, oldest first; "" for an answer the app refused), sent the way the browser sends it: the
question is rewritten into a standalone one and retrieval runs on both.

Reports retrieval (label hit@k, MRR, the best-similarity distribution that calibrates MIN_SIMILARITY) and, unless
--no-llm, answers: each question that passes the gate is asked --repeat times, and every run is an answer with valid
citations, insufficient, or an LLM error (a 502 in the app, neither an answer nor a refusal). The flip rate is the
share of questions that reached the model whose runs did not all end the same way; an intent is consistent when
every phrasing in every run was answered. --llm-extra-body, --min-similarity and --fusion-weight override the
configured values for this run only, and --rewrite-extra-body replaces LLM_REWRITE_EXTRA_BODY.
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from psycopg import Connection
from psycopg_pool import ConnectionPool

from app import embedding, llm, rag
from app.api.kbs import slugify
from app.config import Settings

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"
RATE_LIMIT_WAITS = (30, 60, 90)  # seconds before each retry of an LLM call answered with HTTP 429 (free API tiers)
_PASAL = re.compile(r"Pasal (\d+)(?:–(\d+))?")


def load_golden(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_kb(conn: Connection, name: str) -> dict[str, Any]:
    """By slug only (names are not unique); the slug of a name or of a slug is the same."""
    kb = conn.execute(
        "SELECT id, name, embedding_fingerprint, embedding_query_canary FROM knowledge_bases WHERE slug = %s",
        [slugify(name)],
    ).fetchone()
    if kb is None:
        sys.exit(f"Knowledge base {name!r} not found")
    return kb


def matches(row: dict[str, Any], label: dict[str, Any]) -> bool:
    if "file" in label and not row["filename"].startswith(label["file"]):
        return False
    if "pasal" in label:
        m = _PASAL.fullmatch(row["heading"] or "")
        if not m or not int(m[1]) <= label["pasal"] <= int(m[2] or m[1]):
            return False
    return "text" not in label or " ".join(label["text"].split()).lower() in " ".join(row["content"].split()).lower()


def history_of(item: dict[str, Any]) -> list[tuple[str, str]]:
    return [(t["question"], t["answer"]) for t in item.get("history", [])]


def answer_once(settings: Settings, rows: list[dict], item: dict[str, Any]) -> dict[str, Any]:
    """One model call as rag.ask makes it; a 429 is waited out (it is a quota artefact, not the app's behaviour)."""
    for wait in (*RATE_LIMIT_WAITS, None):
        started = time.perf_counter()  # a rate-limited attempt and its wait are not latency
        try:
            text, usage, final = rag.generate(settings, rows, item["question"], history_of(item))
        except llm.LLMError as e:
            if wait is not None and getattr(e.__cause__, "status_code", None) == 429:
                time.sleep(wait)
                continue
            return {"outcome": "error", "error": str(e), "llm_seconds": round(time.perf_counter() - started, 2)}
        run: dict[str, Any] = {
            "outcome": "insufficient" if final is None else "answered",
            "answer": text,
            "completion_tokens": usage.get("completion_tokens"),
            "llm_seconds": round(time.perf_counter() - started, 2),
        }
        if final is not None:
            cited = [rows[n - 1] for n in final[1]]
            run["cited"] = [f"{r['filename']} | {r['heading']}" for r in cited]
            run["cites_label"] = any(matches(r, lb) for r in cited for lb in item.get("expected", []))
        return run
    raise AssertionError("unreachable")


def rewrite_once(settings: Settings, question: str, history: list[tuple[str, str]]) -> tuple[str | None, float]:
    """rag.rewrite with HTTP 429 waited out as for answers (rag.rewrite falls back to the question on any failure, so a
    quota artefact would otherwise look like a failed rewrite). Returns (rewrite, seconds without the waits)."""
    inner, waited = llm.complete, 0.0

    def patient(system: str, user: str, settings: Settings | None = None) -> tuple[str, dict[str, int]]:
        nonlocal waited
        for wait in (*RATE_LIMIT_WAITS, None):
            try:
                return inner(system, user, settings=settings)
            except llm.LLMError as e:
                if wait is None or getattr(e.__cause__, "status_code", None) != 429:
                    raise
                time.sleep(wait)
                waited += wait
        raise AssertionError("unreachable")

    started = time.perf_counter()
    llm.complete = patient  # ponytail: swaps a module global, fine for this sequential runner only
    try:
        rewritten = rag.rewrite(settings, question, history)
    finally:
        llm.complete = inner
    return rewritten, time.perf_counter() - started - waited


def run_question(
    pool: ConnectionPool,
    settings: Settings,
    kb: dict[str, Any],
    item: dict[str, Any],
    use_llm: bool,
    repeat: int = 1,
) -> dict:
    """Rewrite (follow-ups only), retrieve and gate once, then answer `repeat` times the way rag.ask does, without
    writing a query log. retrieval_seconds includes the rewrite call."""
    question, history = item["question"], history_of(item)
    rewritten, rewrite_seconds = rewrite_once(settings, question, history) if history else (None, 0.0)
    started = time.perf_counter()
    rows = rag.search(pool, settings, kb["id"], question, kb["embedding_query_canary"], rewritten)
    retrieval_seconds = round(time.perf_counter() - started + rewrite_seconds, 2)
    out: dict[str, Any] = {
        "intent": item.get("intent"),
        "question": question,
        "history_turns": len(history),
        "rewritten": rewritten,
        "rewrite_seconds": round(rewrite_seconds, 2) if history else None,
        "best_sim": rag.best_similarity(rows),
        "gate": rag.gate_passes(f"{question} {rewritten or ''}", rows, settings.min_similarity),
        "retrieved": [
            f"{r['vec_sim']:.3f} {'L' if r['fts_rank'] is not None else ' '} {r['filename']} | {r['heading']}"
            for r in rows
        ],
        "retrieval_seconds": retrieval_seconds,
    }
    if item.get("expected"):
        rank = next((i for i, r in enumerate(rows, 1) if any(matches(r, lb) for lb in item["expected"])), None)
        out.update(rank=rank, hit=rank is not None)
    if not out["gate"]:
        out["runs"] = [{"outcome": "insufficient", "llm_seconds": None}] * repeat
    elif use_llm:
        out["runs"] = [answer_once(settings, rows, item) for _ in range(repeat)]
    else:
        out["runs"] = [{"outcome": "answered", "llm_seconds": None}] * repeat  # --no-llm: the gate let it through
    return out


def p90(xs: list[float]) -> float | None:
    return statistics.quantiles(xs, n=10, method="inclusive")[-1] if len(xs) > 1 else (xs[0] if xs else None)


def evaluate(
    pool: ConnectionPool,
    settings: Settings,
    kb: dict[str, Any],
    golden: list[dict[str, Any]],
    use_llm: bool = True,
    other_kb: dict[str, Any] | None = None,
    repeat: int = 1,
) -> dict[str, Any]:
    inscope = [run_question(pool, settings, kb, g, use_llm, repeat) for g in golden if g.get("expected")]
    outscope = [run_question(pool, settings, kb, g, use_llm, repeat) for g in golden if g.get("expect_insufficient")]

    def share(results: list[dict], outcome: str) -> float | None:
        runs = [r["outcome"] for q in results for r in q["runs"]]
        return sum(o == outcome for o in runs) / len(runs) if runs else None

    reached = [q for q in inscope + outscope if q["gate"] and use_llm]
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for q in inscope:
        by_intent[q["intent"]].append(q)
    llm_secs = [r["llm_seconds"] for q in reached for r in q["runs"] if r["llm_seconds"] is not None]
    followups = [q for q in inscope + outscope if q["history_turns"]]
    report: dict[str, Any] = {
        "inscope": inscope,
        "outscope": outscope,
        "repeat": repeat,
        "hit_rate": sum(q["hit"] for q in inscope) / len(inscope) if inscope else None,
        "mrr": sum(1 / q["rank"] for q in inscope if q["rank"]) / len(inscope) if inscope else None,
        "answered_rate": share(inscope, "answered"),
        "outscope_insufficient_rate": share(outscope, "insufficient"),
        "wrongly_refused": sum(r["outcome"] == "insufficient" for q in inscope for r in q["runs"]),
        "cites_label": sum(r.get("cites_label", False) for q in inscope for r in q["runs"]),
        "errors": [
            (q["question"], r["error"]) for q in inscope + outscope for r in q["runs"] if r["outcome"] == "error"
        ],
        "flip_rate": sum(len({r["outcome"] for r in q["runs"]}) > 1 for q in reached) / len(reached)
        if reached
        else None,
        "intents_consistent": sum(
            all(r["outcome"] == "answered" for q in qs for r in q["runs"]) for qs in by_intent.values()
        ),
        "intents": len(by_intent),
        "llm_seconds": (statistics.median(llm_secs), p90(llm_secs)) if llm_secs else None,
        "followups": len(followups),
        "followup_answered_rate": share([q for q in inscope if q["history_turns"]], "answered"),
        "rewrite_failures": sum(q["rewritten"] is None for q in followups),
    }
    sims_in = [q["best_sim"] for q in inscope if q["best_sim"] is not None]
    sims_out = [q["best_sim"] for q in outscope if q["best_sim"] is not None]
    report["sim_inscope"] = (min(sims_in), statistics.median(sims_in)) if sims_in else None
    report["sim_outscope_max"] = max(sims_out) if sims_out else None
    if other_kb is not None:  # the in-scope questions asked of an unrelated KB must all come back insufficient
        cross = [run_question(pool, settings, other_kb, g, use_llm) for g in golden if g.get("expected")]
        report["cross_kb_insufficient_rate"] = share(cross, "insufficient")
    return report


def print_report(report: dict[str, Any], settings: Settings, verbose: bool) -> None:

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.0f}%"

    if verbose:
        for q in report["inscope"] + report["outscope"]:
            tag = f"r{q['rank']:<3}" if q.get("rank") else "MISS" if "rank" in q else "out "
            sim = "n/a" if q["best_sim"] is None else f"{q['best_sim']:.3f}"
            runs = "".join(r["outcome"][0].upper() for r in q["runs"])
            print(f"{tag} sim={sim} gate={'pass' if q['gate'] else 'stop'} runs={runs:5} {q['question']}")
            if q["history_turns"]:
                print(f"     rewritten ({q['history_turns']} turns): {q['rewritten']}")
    k = settings.top_k
    print(f"label hit@{k} / MRR (in-scope):      {pct(report['hit_rate'])} / {report['mrr'] or 0:.2f}")
    print(f"answered rate (in-scope runs):     {pct(report['answered_rate'])}")
    print(f"insufficient rate (out-of-scope):  {pct(report['outscope_insufficient_rate'])}")
    print(f"wrongly insufficient (in-scope):   {report['wrongly_refused']} runs")
    print(f"answers citing a labelled chunk:   {report['cites_label']} runs")
    print(f"flip rate (x{report['repeat']}):                  {pct(report['flip_rate'])}")
    print(f"intents answered in every run:     {report['intents_consistent']}/{report['intents']}")
    for question, error in report["errors"]:
        print(f"LLM error: {error} ({question})")
    if report["llm_seconds"]:
        print("LLM seconds: median {:.1f} / p90 {:.1f}".format(*report["llm_seconds"]))
    if report["followups"]:
        print(f"answered rate (in-scope follow-ups): {pct(report['followup_answered_rate'])}")
        print(f"rewrite fell back to the question:  {report['rewrite_failures']}/{report['followups']} follow-ups")
    if "cross_kb_insufficient_rate" in report:
        print(f"insufficient rate (other KB):      {pct(report['cross_kb_insufficient_rate'])}")
    if report["sim_inscope"] and report["sim_outscope_max"] is not None:
        lo, med = report["sim_inscope"]
        hi = report["sim_outscope_max"]
        print(f"best similarity: in-scope min {lo:.3f} / median {med:.3f}; out-of-scope max {hi:.3f}")
        if hi < lo:
            print(f"MIN_SIMILARITY between {hi:.3f} and {lo:.3f} separates them (now {settings.min_similarity}); ")
            print(
                "questions citing a legal reference (Pasal 87, PP 94) found by the lexical leg bypass the gate regardless."
            )
        else:
            print(
                f"the ranges overlap; MIN_SIMILARITY={settings.min_similarity} cannot separate them by similarity alone"
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval.run_eval", description=__doc__.split("\n\n")[0])
    parser.add_argument("--kb", required=True, help="knowledge base name or slug (looked up by slug)")
    parser.add_argument("--other-kb", help="an unrelated KB: in-scope questions asked there must be insufficient")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--no-llm", action="store_true", help="retrieval and evidence gate only")
    parser.add_argument("--repeat", type=int, default=1, help="model calls per question (flip rate needs > 1)")
    parser.add_argument("--llm-extra-body", type=json.loads, help="JSON object replacing LLM_EXTRA_BODY for this run")
    parser.add_argument("--rewrite-extra-body", type=json.loads, help="JSON object replacing LLM_REWRITE_EXTRA_BODY")
    parser.add_argument("--min-similarity", type=float, help="replaces MIN_SIMILARITY for this run")
    parser.add_argument("--fusion-weight", type=float, help="replaces FUSION_VECTOR_WEIGHT for this run")
    parser.add_argument("--out", type=Path, help="also write the full report as JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    for flag, value in (("--llm-extra-body", args.llm_extra_body), ("--rewrite-extra-body", args.rewrite_extra_body)):
        if value is not None and not isinstance(value, dict):
            parser.error(f"{flag} must be a JSON object")

    from pydantic import ValidationError

    from app.config import get_settings
    from app.db import open_pool

    try:
        settings = get_settings()
    except ValidationError as e:
        missing = sorted({str(err["loc"][0]).upper() for err in e.errors()})
        sys.exit(f"Missing configuration: {', '.join(missing)}. Copy .env.example to .env (or export them).")
    overrides = {
        "llm_extra_body": args.llm_extra_body,
        "llm_rewrite_extra_body": args.rewrite_extra_body,
        "min_similarity": args.min_similarity,
        "fusion_vector_weight": args.fusion_weight,
    }
    settings = settings.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    problems = []
    if not (settings.embedding_base_url and settings.embedding_model):
        problems.append("EMBEDDING_BASE_URL and EMBEDDING_MODEL")
    if not args.no_llm and not (settings.llm_base_url and settings.llm_model):
        problems.append("LLM_BASE_URL and LLM_MODEL (or pass --no-llm)")
    if problems:
        sys.exit(f"Not configured: {'; '.join(problems)}. The evaluation talks to the real endpoints.")

    pool = open_pool(settings)
    try:
        with pool.connection() as conn:
            kb = find_kb(conn, args.kb)
            if kb["embedding_fingerprint"] != settings.embedding_fingerprint or kb["embedding_query_canary"] is None:
                sys.exit(
                    f"{kb['name']!r} was indexed with another embedding setup than the configured one: reindex first"
                )
            other = find_kb(conn, args.other_kb) if args.other_kb else None
        try:
            report = evaluate(pool, settings, kb, load_golden(args.golden), not args.no_llm, other, args.repeat)
        except embedding.EmbeddingError as e:  # LLM errors are recorded per run
            sys.exit(f"Endpoint failure: {e}")
        print_report(report, settings, args.verbose)
        if args.out:
            meta = {
                "llm_extra_body": settings.llm_extra_body,
                "llm_rewrite_extra_body": settings.llm_rewrite_extra_body,
                "min_similarity": settings.min_similarity,
                "fusion_vector_weight": settings.fusion_vector_weight,
                "top_k": settings.top_k,
                "llm_max_tokens": settings.llm_max_tokens,
            }
            args.out.write_text(json.dumps({"settings": meta, **report}, indent=1, ensure_ascii=False, default=str))
    finally:
        pool.close()


if __name__ == "__main__":
    main()
