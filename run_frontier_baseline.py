"""Generate + evaluate the Section 6.8 frontier-LLM baselines.

Two techniques:
  - opus_tuned_rag    : Claude Opus 4.5 generator + Tuned RAG retrieval
  - gpt41_full_corpus : GPT-4.1 with full 868-page corpus, no retrieval

Each gets 3 RAGAS trials with gpt-4o-mini as judge.
Total cost target: ~$10-15.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference,
)

from rag_harness import HarnessConfig, RetrievalConfig, run_model, BENCHMARK_QUERIES, build_embeddings
from frontier_full_corpus import run_gpt41_full_corpus

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    anthropic_api_key=keys["ANTHROPIC_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
    # Option A configuration
    generator_provider="anthropic",
    generator_model="claude-opus-4-5",
    judge_provider="openai",
    judge_model="gpt-4o-mini",
)
retrieval_cfg = RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5)

# Load existing all_answers.json so we can merge our new techniques in
all_ans = json.load(open("all_answers.json"))

# ---------------------------------------------------------------------------
# Option A: Opus 4.5 + Tuned RAG retrieval
# ---------------------------------------------------------------------------
print("=" * 70)
print("OPTION A: Opus 4.5 + Tuned RAG retrieval")
print("=" * 70)

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from rag_harness import retrieve, get_or_build_store, _format_context

# Get the existing Tuned RAG store + LLM, then run all 5 queries
store, _ = get_or_build_store("BAAI/bge-base-en-v1.5", cfg)
opus_llm = ChatAnthropic(
    model="claude-opus-4-5",
    anthropic_api_key=cfg.anthropic_api_key,
    temperature=0,
    max_tokens=1024,
)

# Use the same cited-analyst system prompt as Tuned RAG
SYSTEM = (
    "You are a senior energy markets analyst at Lumina Energy Partners. "
    "Answer ONLY from the provided IEA report excerpts. "
    "For every key quantitative claim, cite the source report and page in square "
    "brackets, e.g. [Gas2025 p.42]. "
    "If the excerpts do not contain the answer, say so explicitly. "
    "Structure as an investment-committee briefing: "
    "(1) Headline finding, (2) supporting data with citations, "
    "(3) cross-sector linkages, (4) countervailing forces. "
    "Be concise; 200-350 words. Avoid speculation beyond the sources."
)

opus_answers, opus_contexts, opus_walls = [], [], []
for i, q in enumerate(BENCHMARK_QUERIES, 1):
    t0 = time.perf_counter()
    docs = retrieve(q, store, retrieval_cfg, cfg=cfg)
    context_str = _format_context(docs)
    resp = opus_llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Question:\n{q}\n\nIEA excerpts:\n{context_str}"),
    ])
    wall = time.perf_counter() - t0
    opus_answers.append(resp.content)
    opus_contexts.append([d.page_content for d in docs])
    opus_walls.append(round(wall, 2))
    print(f"  Q{i}: {wall:.1f}s, {len(resp.content)} chars, {len(docs)} chunks")

all_ans["opus_tuned_rag"] = {
    "answers": opus_answers,
    "contexts": opus_contexts,
    "wall_seconds": opus_walls,
}
print(f"  Mean wall: {sum(opus_walls)/len(opus_walls):.1f}s")

# ---------------------------------------------------------------------------
# Option B-GPT: GPT-4.1 + full untrimmed corpus
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("OPTION B-GPT: GPT-4.1 + full untrimmed corpus")
print("=" * 70)

gpt = run_gpt41_full_corpus(keys["OPENAI_API_KEY"])
all_ans["gpt41_full_corpus"] = {
    "answers": gpt["answers"],
    "contexts": [[] for _ in gpt["answers"]],
    "usage": gpt["usage"],
    "wall_seconds": gpt["wall_seconds"],
    "estimated_cost_usd": gpt["estimated_cost_usd"],
    "corpus_stats": gpt["corpus_stats"],
}
print(f"  GPT-4.1 cost: ${gpt['estimated_cost_usd']}")

# Save before RAGAS so we have the answers even if eval crashes
json.dump(all_ans, open("all_answers.json", "w"), indent=2)
print(f"\nSaved both techniques to all_answers.json")

# ---------------------------------------------------------------------------
# RAGAS evaluation: 3 trials per technique
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RAGAS EVALUATION: 3 trials per technique")
print("=" * 70)

judge = ChatOpenAI(model="gpt-4o-mini", api_key=cfg.openai_api_key, temperature=0)
emb = build_embeddings("BAAI/bge-base-en-v1.5", cfg)
N_TRIALS = 3

results = {}
for tech in ["opus_tuned_rag", "gpt41_full_corpus"]:
    has_context = tech == "opus_tuned_rag"
    metrics = [AnswerRelevancy()]
    if has_context:
        metrics += [Faithfulness(), LLMContextPrecisionWithoutReference()]

    print(f"\n=== {tech} ({'RAG' if has_context else 'no retrieval'}) ===")
    rows = []
    for trial in range(1, N_TRIALS + 1):
        ds = Dataset.from_dict({
            "question": BENCHMARK_QUERIES,
            "answer":   all_ans[tech]["answers"],
            "contexts": all_ans[tech]["contexts"] if has_context else [[] for _ in BENCHMARK_QUERIES],
        })
        try:
            r = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb).to_pandas()
            row = {
                "trial": trial,
                "answer_relevancy": float(r["answer_relevancy"].mean()),
            }
            if has_context:
                row["faithfulness"] = float(r["faithfulness"].mean())
                row["context_precision"] = float(r["llm_context_precision_without_reference"].mean())
            print(f"  trial {trial}: AR={row['answer_relevancy']:.3f}"
                  + (f"  faith={row['faithfulness']:.3f}  cp={row['context_precision']:.3f}"
                     if has_context else ""))
            rows.append(row)
        except Exception as e:
            print(f"  trial {trial}: ERROR {e}")

    results[tech] = rows

# Save eval results
for tech, rows in results.items():
    df = pd.DataFrame(rows)
    df.to_csv(f"{tech}_eval_raw.csv", index=False)
    print(f"\n=== {tech} (n={len(rows)}) ===")
    for m in ["answer_relevancy", "faithfulness", "context_precision"]:
        if m in df.columns and df[m].notna().any():
            print(f"  {m:<22s} {df[m].mean():.3f} ± {df[m].std(ddof=1):.3f}")
        else:
            print(f"  {m:<22s} N/A")

print("\n=== DONE ===")
