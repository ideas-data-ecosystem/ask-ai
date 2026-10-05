# RAG best practices for IDEAS Ask

## Status (2026-10-05, after the changes this page proposed)

| Item | Status |
|---|---|
| 1. Golden set v2 | Done in part: intent groups with chunk labels, hit@k, MRR, flip rate over `--repeat`, plus follow-up scenarios with history. Not done: about 150 questions, a dev/held-out split, Ragas spot checks. |
| 2. Convex combination, no candidate floor | Done: `FUSION_VECTOR_WEIGHT` (0.7), the floor is gone (root cause 5). |
| 3. Rewrite and search with both | Done for follow-ups only: a question sent with `history` is rewritten (reasoning off via `LLM_REWRITE_EXTRA_BODY`) and both are searched, each chunk scored with the better of the two. Single-turn questions are not rewritten. The acronym table is not built. |
| 4. Calibrated gate | Done in part: `MIN_SIMILARITY` 0.15, and the gate sees the better similarity of question and rewrite. Not gated on a reranker. |
| 5. Reproducible generation | Done in part: a reply cut at `LLM_MAX_TOKENS` (now 2048) or empty is a 502, not "insufficient"; low-effort reasoning is the recommended setting. No retry. |
| 6. Language-aware lexical leg | Done: -kah/-lah/-pun particles and per-KB document frequency (`MAX_DF`). `heading` is not in `tsv` yet. |
| 7. Reranker on NVIDIA's API | Not done. |
| 8. Chunk cleanup | Not done. |

Also added: a question that names a Pasal ("Pasal 311 PP 11 Tahun 2017") ranks the chunks headed by that Pasal first,
limited to the documents of the regulation when it is named with its year; FTS ranking alone never brought that chunk
into the top 8 (root cause 3).

IDEAS Ask answers from one knowledge base ("Regulasi ASN": 42 regulation PDFs, 2,610 chunks). Paraphrases of one question get different outcomes. "naik pangkat" and "apakah ada cara untuk resign ?" are answered, "adakah cara untuk naik pangkat ?" and "adakah cara untuk naik resign ?" are refused, and "apa itu ASN ?" retrieves no definition chunk. This page ranks changes for this app, based on primary sources and on measurements against the live index on 2026-10-05. Numbers marked [M] were measured for this page with read-only queries against the database, the NVIDIA embedding, chat and rerank endpoints, and `python -m eval.run_eval --no-llm`. Every other number links to its source. Function names refer to `backend/app` on that date, while `rag.py` and `llm.py` were being edited.

## Root causes

1. Cosine similarity does not separate in-scope from off-topic questions. In the eval run the in-scope minimum is 0.175 ("adakah cara untuk naik pangkat ?") and the out-of-scope maximum is 0.268 ("Siapa presiden pertama Republik Indonesia?", which matches "Peraturan Presiden ini mulai berlaku") [M]. The earlier "off-topic at most 0.13" figure came from a mini-benchmark that mixed each answer chunk into 20 random chunks; the maximum over 2,610 chunks is higher. "naik pangkat" passes the gate only because an OCR-garbage chunk (Peraturan BKN 1/2019 Pasal 33, "penaaaaaaaaa pete 20X") scores 0.213, while the kenaikan pangkat chunks score 0.08 to 0.16 [M].
2. Wording moves the query vector more than the topic does. For one intent the best cosine is 0.175 for "adakah cara untuk naik pangkat ?", 0.214 for "bagaimana cara naik pangkat?" and 0.480 for "Bagaimana tata cara kenaikan pangkat PNS?" [M]. "resign" occurs in 0 chunks; the corpus says "permintaan sendiri" (71 chunks) and "mengundurkan diri" (45) [M].
3. RRF ranks by position, and Postgres FTS ranking has no document frequency: the ranking functions "do not use any global information" ([PostgreSQL 17, 12.3.3](https://www.postgresql.org/docs/17/textsearch-controls.html)). "adakah" stems to `ada`, the lexeme of "pengadaan", found in 30% of chunks [M]. For "apa itu ASN ?" the only lexical word is `asn` (17% of chunks), so FTS favours chunks that repeat "Pegawai ASN", and RRF puts the first "Aparatur Sipil Negara yang selanjutnya disingkat ASN" chunk at fused rank 10 although it has the best cosine (0.50) [M].
4. Generation is not reproducible. Three runs of "adakah cara untuk naik resign ?" with an identical retrieved set (same hash in `query_logs`) gave an empty answer after all 2,048 completion tokens (15.6 s), `INSUFFICIENT_CONTEXT` (30 tokens) and a cited answer (263 tokens) [M]. "ASN" was answered twice and refused twice [M].
5. (Fixed.) An intermediate `search()` dropped fused candidates below `MIN_SIMILARITY` (then 0.2). For "naik pangkat" it returned one chunk, the OCR-garbage one, and the gate passed [M]. A candidate floor equal to the gate threshold removes every relevant chunk whenever those chunks score under the gate. `search()` now returns the top `TOP_K` fused chunks with no floor; `MIN_SIMILARITY` is only the evidence gate on the best of them.

## Ranked changes

### 1. Build golden set v2 before tuning anything

Change `backend/eval/golden.jsonl` into intent groups: each intent asked three or four ways (formal, colloquial, -kah form, acronym only), labelled with expected chunk ids or (document, Pasal) pairs instead of file prefixes. "apa itu ASN ?" lists 16 expected files, so its file-level hit says nothing [M]. Add near-domain out-of-scope questions of the "presiden" and "pajak" kind, which score closest to in-scope ones.

Extend `run_eval.py` with hit@k and MRR on chunk labels, ID-based context recall (reference context ids found in the retrieved set, [Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)), refusal precision and recall on both classes averaged into F1, as in Trust-Score's grounded refusals ([Song et al., ICLR 2025](https://arxiv.org/abs/2409.11242)), and a flip rate: every question run three times, plus agreement across the phrasings of one intent. Ragas faithfulness (supported claims / all claims) and answer relevancy (mean cosine between the question and questions generated from the answer) are LLM-judged, so use them as spot checks; at this sample size judge noise adds to sampling noise.

Size: at 24 in-scope questions a 90% answered rate has a 95% interval of about ±12 points; at 100 it is about ±6 (normal approximation; see [Miller 2024](https://arxiv.org/abs/2411.00640) on error bars for evals). For off-topic leaks, zero leaks in n questions bounds the rate at 3/n with 95% confidence ([Hanley and Lippman-Hand, JAMA 1983](https://doi.org/10.1001/jama.1983.03330370053031)), so a claim under 5% needs 60 off-topic questions. Tune on a dev split, report on a held-out split.

- Effort M (about 150 questions to write). Risk none. Items 2 to 5 are tuned against it.

### 2. Fuse with a convex combination of normalised scores and drop the candidate floor

In `_SEARCH_SQL`/`retrieve`, score the union of both candidate lists with both signals, normalise per query (vector: min-max over candidates; FTS: score divided by the query's best score), and rank by `a * vec + (1 - a) * fts`, with `a` tuned on the dev split (start at 0.7 to 0.8). Remove the `vec_sim >= min_similarity` filter from `search()`.

Evidence: [Bruch, Gai and Ingber (ACM TOIS 2023)](https://arxiv.org/abs/2210.11934) find that convex combination outperforms RRF in-domain and out-of-domain, that RRF is sensitive to its parameter, that the choice of normalisation matters little, and that the single weight needs "only a small set of training examples". Simulation on the live candidate pools [M]:

- "apa itu ASN ?": first definition chunk at rank 10 with RRF, rank 1 with `a = 0.8`, and 6 of the top 8 are definition chunks.
- "adakah cara untuk naik resign ?": top-8 slots with cosine under 0.10 drop from 4 to 0; resignation chunks in the top 8 rise from 4 to 8.
- "adakah cara untuk naik pangkat ?": kenaikan pangkat chunks in the top 8 go from 4 (RRF) to 2 (`a = 0.8`) or 6 (`a = 0.6`), which is why `a` must come from item 1.

RRF with a low floor (0.10) also clears the junk for "naik resign" but leaves the ASN definition at rank 10 [M]. Effort S. Risk: `a` is tied to the embedding model and needs re-tuning when it changes.

### 3. Rewrite the question into the corpus register and search with both

Add `rewrite()` to `rag.py`: one chat call with reasoning off that returns one formal Indonesian query in regulation vocabulary, without answering and without adding entities the question does not contain. Embed `[question, rewrite]` in the existing single `embedding.embed` call, use the smaller of the two distances in the vector leg, the union of words in the lexical leg, and the larger similarity in the gate. Fall back to the original question when the call fails. Expand acronyms deterministically: at ingest, collect "<long form> yang selanjutnya disingkat <ACRONYM>" pairs (80 chunks contain the phrase [M]) into a per-KB table, and expand acronyms that appear in the question before embedding. Ketentuan umum hold such definitions by drafting rule ([UU 12/2011 Lampiran II](https://peraturan.bpk.go.id/Details/39188)).

Evidence: [Query2doc (EMNLP 2023)](https://arxiv.org/abs/2303.07678) reports BM25 gains of 3% to 15% on MS MARCO and TREC DL from LLM-written expansions, with gains for dense retrievers too; [Ma et al. (EMNLP 2023)](https://arxiv.org/abs/2305.14283) and [HyDE (ACL 2023)](https://arxiv.org/abs/2212.10496) adapt the query instead of the index. None of them evaluates Indonesian or legal text. Measured with `nemotron-3-super`, `enable_thinking: false`, temperature 0, 0.35 to 0.61 s per rewrite [M]:

- "adakah cara untuk naik pangkat ?" became "Prosedur Kenaikan Pangkat Aparatur Sipil Negara (ASN)": best cosine 0.175 to 0.522.
- "adakah cara untuk naik resign ?" became "Prosedur Pengajuan Pemberhentian atas Permintaan Sendiri untuk Aparatur Sipil Negara (ASN)": 0.307 to 0.467, with a resignation chunk at vector rank 5.
- "apa itu ASN ?": 0.502 to 0.591, definition at rank 1.
- Off-topic "Siapa presiden pertama Republik Indonesia?": 0.268 to 0.293.

On these four questions the in-scope minimum moved from 0.175 to 0.467 while the off-topic one stayed under 0.30. The pangkat rewrite also shows the failure mode: the added "Aparatur Sipil Negara (ASN)" pulled the vector toward Pasal 1 definition chunks, and kenaikan pangkat chunks fell from vector rank 8 to 53 [M]. Keeping the original question as a leg contains that drift.

- Effect: fixes the gate refusal for "adakah cara untuk naik pangkat ?" and the "resign" vocabulary gap.
- Effort M. Risk: 0.4 to 0.6 s added latency and a second upstream dependency.

### 4. Calibrate the gate and let the model refuse

Compute `gate_passes` on the larger similarity of question and rewrite (later on the reranker probability from item 7), set the threshold from the dev split so that in-scope recall stays near 100%, and leave the rest to `INSUFFICIENT_CONTEXT` plus the citation check in `finalize`. Log both similarities in `query_logs.retrieved`. Evidence: [Joren et al. (ICLR 2025)](https://arxiv.org/abs/2411.06037) find that large models often answer instead of abstaining when the context is insufficient, and that combining a context-sufficiency signal with the model's self-rated confidence improves accuracy among answered questions by 2% to 10%. Neither signal is enough alone; root cause 1 shows a single cosine threshold on the raw question cannot work here.

- Effect: "adakah cara untuk naik pangkat ?" passes (0.522 after rewrite); off-topic questions still face the sentinel and the citation check.
- Effort S. Risk: more off-topic questions reach the LLM, so the leak rate from item 1 must be watched.

### 5. Make generation reproducible and fail loud

In `llm.py` and `rag._answer`, treat empty content or `finish_reason == "length"` as an upstream error (one retry, then 502) instead of "insufficient", and cap reasoning with the `reasoning_budget` knob the [model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) documents, or raise `max_tokens`. The card recommends temperature 1.0 and top_p 0.95 "across all tasks"; this app sends no temperature, and the hosted endpoint accepts temperature 0 [M]. Choose by flip rate and answered rate from item 1. The same card lists English, French, German, Italian, Japanese, Spanish and Chinese, not Indonesian, so an LLM with documented Indonesian support is worth one run of the golden set.

- Effect: removes the three-way outcome on identical context seen for "adakah cara untuk naik resign ?".
- Effort S. Risk low.

### 6. Make the lexical leg language-aware instead of list-driven

In `lexical_words`, strip the particles the [snowball Indonesian stemmer](https://snowballstem.org/algorithms/indonesian/stemmer.html) removes (-kah, -lah, -pun) before the stoplist check, because the app's stoplist runs before stemming. That covers "adakah", "bolehkah" and every future -kah form without listing each. Then drop words whose stem appears in more than about 20% to 25% of the KB's chunks, from `ts_stat` (27 to 45 ms over all chunks [M], cached per KB and cleared on reindex). It acts as the missing IDF. Add `heading` to the `tsv` generated column with weight A in a migration. Real BM25 needs an extension that the `pgvector/pgvector:pg17` image does not ship.

- Effect: generalises the -kah fix now hand-listed in `STOPWORDS`; drops `ada` and `pns` (30% each) and keeps `asn` (17%) [M]. Little change for the three examples beyond the current list.
- Effort S. Risk: a frequent but meaningful word loses its lexical match; the vector leg still carries it.

### 7. Add a cross-encoder reranker using NVIDIA's actual API

`nvidia/llama-3.2-nv-rerankqa-1b-v2` and `nvidia/llama-nemotron-rerank-1b-v2`, both documented with 26 languages including Indonesian ([1](https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_2-nv-rerankqa-1b-v2), [2](https://docs.api.nvidia.com/nim/reference/nvidia-llama-nemotron-rerank-1b-v2)), return 410 Gone on the hosted API, end of life 2026-05-18 and 2026-08-25 [M]. `nvidia/llama-nemotron-rerank-vl-1b-v2` answers, at 1.0 to 1.2 s for 40 passages of 15,000 to 20,000 tokens [M]. Its [model card](https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2) supports text-only passages and reports 73.98% average Recall@5 over BEIR, MIRACL, MLQA and MLDR, without listing Indonesian by name.

The wire format ([NIM docs](https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/using-reranking.html)) is `POST https://ai.api.nvidia.com/v1/retrieval/nvidia/<model-slug>/reranking` with `{"model", "query": {"text"}, "passages": [{"text"}], "truncate": "END"}`, answering `{"rankings": [{"index", "logit"}]}`, up to 512 passages. `integrate.api.nvidia.com/v1/rerank` and `/v1/ranking` return 404 [M]. `rag.rerank()` posts Cohere-style `{query, documents, top_n}` to `<base>/rerank` and reads `results[].relevance_score`, so it does not fit; it needs a second request and response shape.

Measured on 40 RRF candidates [M]:

- "apa itu ASN ?": the top 3 are definition chunks moved up from RRF ranks 12, 10 and 22, including PermenPANRB 20/2025 Pasal 1 ("Aparatur Sipil Negara yang selanjutnya disingkat ASN", logit -0.97, p 0.27). The RRF top chunk, PermenPANRB 6/2024 Pasal 16 on planning, scores -4.5.
- "adakah cara untuk naik pangkat ?": the top 6 are all kenaikan pangkat; PermenPANRB 1/2023 Pasal 38 scores -4.56 against -7.2 for PP 49/2018 on PPPK procurement (RRF rank 3).
- "adakah cara untuk naik resign ?": the top 3 are dismissal and resignation provisions, among them PP 11/2017 Pasal 261 (from RRF rank 33) and Peraturan BKN 3/2020 Pasal 6 on resignation at one's own request (from rank 12).

The best logits for colloquial in-scope questions (-4.54, -6.75) overlap those for off-topic ones (-5.38, -7.25) [M], so use the reranker for ordering and gate on it only after calibrating with rewritten queries. Cross-encoder reranking: [Nogueira and Cho 2019](https://arxiv.org/abs/1901.04085); in [Anthropic's tests](https://www.anthropic.com/news/contextual-retrieval) it cut top-20 failure from 2.9% to 1.9%.

- Effort M. Risk: two hosted rerankers reached end of life this year; about 1 s per question. Self-hosting the text reranker NIM avoids the churn.

### 8. Clean and structure the chunks

- OCR garbage takes slots: Peraturan BKN 1/2019 Pasal 33 for "naik pangkat", scrambled PermenPANRB 6/2022 annex pages for "tarif pajak" [M]. Score each chunk's share of dictionary words in `chunking.py` or `ingest.py` and exclude or down-rank low scorers.
- Definitions: headings "Pasal 1" cover 131 chunks [M], and those chunks pull any query that mentions ASN (item 3 drift). Index each numbered definition as its own small unit, or rely on the acronym table from item 3.
- Penjelasan: 103 chunks carry "Penjelasan" headings [M]. Penjelasan is the official interpretation of the batang tubuh ([hukumonline](https://www.hukumonline.com/klinik/a/fungsi-dan-peran-penjelasanlampiran-suatu-peraturan-perundangundangan-cl6386)), so attach "Penjelasan Pasal N" when Pasal N of the same document is retrieved instead of letting it compete for a slot.
- Contextual prefix: `ingest._embed_text` already prepends title and heading. Anthropic's 50 to 100 token LLM-written context per chunk cut top-20 failure from 5.7% to 3.7% (2.9% with contextual BM25) on their datasets ([source](https://www.anthropic.com/news/contextual-retrieval)). It costs one LLM call per chunk; test it after items 2 and 3.
- Effort M to L. Risk: a reindex per change.

## What not to do

- Adopt a generic Indonesian stopword list wholesale. The 758-word list attributed to [Tala 2003](https://eprints.illc.uva.nl/740/) ([masdevid/ID-Stopwords](https://github.com/masdevid/ID-Stopwords)) contains "naik", "cara", "ada", "sendiri", "tahun", "hari", "lama", "baru" and "umum" [M]. In this corpus they carry meaning ("naik pangkat", "permintaan sendiri", "hari kerja").
- Patch one word or one threshold per failing question. The -kah list in `rag.py` fixes "adakah" and waits for the next form; particle stripping plus document frequency covers the class.
- Treat cosine as a relevance probability. Root cause 1 shows the overlap, and definition chunks score high for unrelated queries.
- Replace the question with its rewrite. The pangkat rewrite moved the relevant chunks from rank 8 to 53 [M].
- Use HyDE-style long hypothetical answers. The HyDE abstract notes the generated document "may contain false details"; for a closed legal corpus that puts model knowledge into retrieval, and a 6 to 30 token rewrite gave the gains above [M].
- Turn full reasoning on to fix refusals. It costs 4.2 to 11.8 s (earlier measurement, wiki "RAG and embedding gotchas"), and at `max_tokens` 2,048 it produced an empty answer that was logged as insufficient [M].
- Send 20 chunks because Anthropic found top-20 better than top-10. That held for their models; [Liu et al. (TACL 2024)](https://arxiv.org/abs/2307.03172) show accuracy drops for evidence in the middle of long contexts. Tune `top_k` (5, 8, 12) on the golden set.

## Sources

- Cormack, Clarke, Büttcher. Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods. SIGIR 2009. https://doi.org/10.1145/1571941.1572114
- Bruch, Gai, Ingber. An Analysis of Fusion Functions for Hybrid Retrieval. ACM TOIS 2023. https://arxiv.org/abs/2210.11934
- Gao, Ma, Lin, Callan. Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE). ACL 2023. https://arxiv.org/abs/2212.10496
- Wang, Yang, Wei. Query2doc. EMNLP 2023. https://arxiv.org/abs/2303.07678
- Ma, Gong, He, Zhao, Duan. Query Rewriting for Retrieval-Augmented Large Language Models. EMNLP 2023. https://arxiv.org/abs/2305.14283
- Nogueira, Cho. Passage Re-ranking with BERT. 2019. https://arxiv.org/abs/1901.04085
- Anthropic. Introducing Contextual Retrieval. 2024. https://www.anthropic.com/news/contextual-retrieval
- Joren et al. Sufficient Context: A New Lens on Retrieval Augmented Generation Systems. ICLR 2025. https://arxiv.org/abs/2411.06037
- Song et al. Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse. ICLR 2025. https://arxiv.org/abs/2409.11242
- Gao, Yen, Yu, Chen. Enabling Large Language Models to Generate Text with Citations (ALCE). EMNLP 2023. https://arxiv.org/abs/2305.14627 (even the best models lacked complete citation support 50% of the time on ELI5, which is why `finalize` checks markers)
- Liu et al. Lost in the Middle. TACL 2024. https://arxiv.org/abs/2307.03172
- Es et al. RAGAS. https://arxiv.org/abs/2309.15217 and metric docs https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
- Miller. Adding Error Bars to Evals. 2024. https://arxiv.org/abs/2411.00640
- Hanley, Lippman-Hand. If nothing goes wrong, is everything all right? JAMA 1983. https://doi.org/10.1001/jama.1983.03330370053031
- Snowball Indonesian stemmer (after Tala 2003). https://snowballstem.org/algorithms/indonesian/stemmer.html
- Tala. A Study of Stemming Effects on Information Retrieval in Bahasa Indonesia. ILLC MoL-2003-02. https://eprints.illc.uva.nl/740/
- PostgreSQL 17 documentation, 12.3.3 Ranking Search Results. https://www.postgresql.org/docs/17/textsearch-controls.html
- NVIDIA reranker references and NIM reranking API. https://docs.api.nvidia.com/nim/reference/nvidia-llama-nemotron-rerank-1b-v2, https://docs.nvidia.com/nim/nemo-retriever/text-reranking/latest/using-reranking.html, https://huggingface.co/nvidia/llama-nemotron-rerank-vl-1b-v2
- NVIDIA Nemotron 3 Super model card. https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
- UU 12/2011 tentang Pembentukan Peraturan Perundang-undangan (Lampiran II, amended by UU 13/2022). https://peraturan.bpk.go.id/Details/39188
