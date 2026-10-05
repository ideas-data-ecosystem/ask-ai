"""Golden-set evaluation against a real, indexed knowledge base (real embedding and LLM endpoints).

    cd backend && uv run python -m eval.run_eval --kb "Regulasi ASN" [--no-llm] [--other-kb "Other KB"] [-v]

Reports the top-k hit rate (does a retrieved chunk come from an expected document), how often in-scope
questions are answered instead of declared insufficient, how often out-of-scope ones are declared
insufficient, and the best-similarity distribution to calibrate MIN_SIMILARITY. golden.jsonl lines are
{"question", "expected_files": [filename prefixes], "page"} for in-scope and {"question", "expect_insufficient"}
for out-of-scope questions. `--no-llm` stops after retrieval and the evidence gate.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from psycopg import Connection
from psycopg_pool import ConnectionPool

from app import embedding, llm, rag
from app.api.kbs import slugify
from app.config import Settings

GOLDEN = Path(__file__).resolve().parent / "golden.jsonl"


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


def run_question(
    pool: ConnectionPool, settings: Settings, kb: dict[str, Any], item: dict[str, Any], use_llm: bool
) -> dict:
    """Retrieve, gate and (optionally) answer one question the way rag.ask does, without writing a query log."""
    question = item["question"]
    rows = rag.search(pool, settings, kb["id"], question, kb["embedding_query_canary"])
    best = rag.best_similarity(rows)
    out: dict[str, Any] = {
        "question": question,
        "best_sim": best,
        "gate": rag.gate_passes(question, rows, settings.min_similarity),
        "files": [r["filename"] for r in rows],
        "insufficient": True,
        "cited_files": [],
    }
    expected = item.get("expected_files")
    if expected:
        out["hit"] = any(f.startswith(tuple(expected)) for f in out["files"])
    if out["gate"] and use_llm:
        _, _, final = rag.generate(settings, rows, question)
        if final is not None:
            out["insufficient"] = False
            out["cited_files"] = [rows[n - 1]["filename"] for n in final[1]]
    elif out["gate"]:
        out["insufficient"] = False  # --no-llm: the gate let it through
    return out


def evaluate(
    pool: ConnectionPool,
    settings: Settings,
    kb: dict[str, Any],
    golden: list[dict[str, Any]],
    use_llm: bool = True,
    other_kb: dict[str, Any] | None = None,
) -> dict[str, Any]:
    inscope = [run_question(pool, settings, kb, g, use_llm) for g in golden if g.get("expected_files")]
    outscope = [run_question(pool, settings, kb, g, use_llm) for g in golden if g.get("expect_insufficient")]
    report: dict[str, Any] = {
        "inscope": inscope,
        "outscope": outscope,
        "hit_rate": sum(r["hit"] for r in inscope) / len(inscope) if inscope else None,
        "answered_rate": sum(not r["insufficient"] for r in inscope) / len(inscope) if inscope else None,
        "outscope_insufficient_rate": sum(r["insufficient"] for r in outscope) / len(outscope) if outscope else None,
    }
    sims_in = [r["best_sim"] for r in inscope if r["best_sim"] is not None]
    sims_out = [r["best_sim"] for r in outscope if r["best_sim"] is not None]
    report["sim_inscope"] = (min(sims_in), statistics.median(sims_in)) if sims_in else None
    report["sim_outscope_max"] = max(sims_out) if sims_out else None
    if other_kb is not None:  # the in-scope questions asked of an unrelated KB must all come back insufficient
        cross = [run_question(pool, settings, other_kb, g, use_llm) for g in golden if g.get("expected_files")]
        report["cross_kb_insufficient_rate"] = sum(r["insufficient"] for r in cross) / len(cross) if cross else None
    return report


def print_report(report: dict[str, Any], settings: Settings, verbose: bool) -> None:

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.0f}%"

    if verbose:
        for r in report["inscope"] + report["outscope"]:
            tag = "hit " if r.get("hit") else "MISS" if "hit" in r else "out "
            sim = "n/a" if r["best_sim"] is None else f"{r['best_sim']:.3f}"
            print(
                f"{tag} sim={sim} gate={'pass' if r['gate'] else 'stop'} insufficient={r['insufficient']!s:5} {r['question']}"
            )
            if "hit" in r and not r["hit"]:
                print(f"       retrieved: {sorted(set(r['files']))}")
    print(f"top-{settings.top_k} hit rate (in-scope):        {pct(report['hit_rate'])}")
    print(f"answered rate (in-scope):             {pct(report['answered_rate'])}")
    print(f"insufficient rate (out-of-scope):     {pct(report['outscope_insufficient_rate'])}")
    if "cross_kb_insufficient_rate" in report:
        print(f"insufficient rate (other KB):         {pct(report['cross_kb_insufficient_rate'])}")
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
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from pydantic import ValidationError

    from app.config import get_settings
    from app.db import open_pool

    try:
        settings = get_settings()
    except ValidationError as e:
        missing = sorted({str(err["loc"][0]).upper() for err in e.errors()})
        sys.exit(f"Missing configuration: {', '.join(missing)}. Copy .env.example to .env (or export them).")
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
            report = evaluate(pool, settings, kb, load_golden(args.golden), not args.no_llm, other)
        except (embedding.EmbeddingError, llm.LLMError) as e:
            sys.exit(f"Endpoint failure: {e}")
        print_report(report, settings, args.verbose)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
