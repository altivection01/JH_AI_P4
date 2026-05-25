"""Full evaluation of graph_rag_cov vs graph_rag.

For each technique:
  - n=5 trials standard RAGAS (AR, Faithfulness, Context Precision) — judge=gpt-4.1
  - n=5 trials custom causal_correctness — judge=gpt-4.1

We only re-run graph_rag for standard RAGAS if its row is missing from
ragas_gpt41_raw.csv (it should already be there from the earlier sweep).
Causal correctness runs fresh for both techniques.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference,
)

# Limit concurrency so we stay under OpenAI tier-1 gpt-4.1 TPM (30K)
RAGAS_RUN_CFG = RunConfig(max_workers=2, timeout=600)

from rag_harness import HarnessConfig, BENCHMARK_QUERIES, build_embeddings
from causal_correctness import causal_correctness_batch

N_TRIALS = 5
keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
)
judge = ChatOpenAI(model="gpt-4.1", api_key=cfg.openai_api_key, temperature=0)
emb = build_embeddings("BAAI/bge-base-en-v1.5", cfg)
all_ans = json.load(open("all_answers.json"))

# =====================================================================
# Step 1: Standard RAGAS on graph_rag_cov (we already have graph_rag from
#         the previous gpt-4.1 sweep — ragas_gpt41_raw.csv)
# =====================================================================
print("=" * 70)
print("STEP 1: Standard RAGAS metrics on graph_rag_cov")
print("=" * 70)

cov_data = all_ans["graph_rag_cov"]
metrics = [AnswerRelevancy(), Faithfulness(), LLMContextPrecisionWithoutReference()]

ragas_rows = []
for trial in range(1, N_TRIALS + 1):
    t0 = time.perf_counter()
    ds = Dataset.from_dict({
        "question": BENCHMARK_QUERIES,
        "answer":   cov_data["answers"],
        "contexts": cov_data["contexts"],
    })
    try:
        r = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb,
                     run_config=RAGAS_RUN_CFG).to_pandas()
        row = {
            "technique": "graph_rag_cov", "trial": trial, "judge": "gpt-4.1",
            "answer_relevancy": float(r["answer_relevancy"].mean()),
            "faithfulness": float(r["faithfulness"].mean()),
            "context_precision": float(r["llm_context_precision_without_reference"].mean()),
            "wall_seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"  trial {trial}: AR={row['answer_relevancy']:.3f}  "
              f"faith={row['faithfulness']:.3f}  cp={row['context_precision']:.3f}")
        ragas_rows.append(row)
    except Exception as e:
        print(f"  trial {trial}: ERROR {e}")

# Append to existing ragas_gpt41_raw.csv if present
ragas_df = pd.DataFrame(ragas_rows)
try:
    existing = pd.read_csv("ragas_gpt41_raw.csv")
    combined = pd.concat([existing, ragas_df], ignore_index=True)
except FileNotFoundError:
    combined = ragas_df
combined.to_csv("ragas_gpt41_raw.csv", index=False)

# Rebuild aggregate
agg_rows = []
for tech in combined["technique"].unique():
    sub = combined[(combined["technique"] == tech)
                   & combined["answer_relevancy"].notna()]
    if len(sub) == 0:
        continue
    rec = {"technique": tech, "n_trials": int(len(sub)), "judge": "gpt-4.1"}
    for m in ["answer_relevancy", "faithfulness", "context_precision"]:
        if m in sub.columns and sub[m].notna().any():
            rec[f"{m}_mean"] = round(sub[m].mean(), 4)
            rec[f"{m}_std"] = round(sub[m].std(ddof=1), 4) if len(sub) > 1 else 0.0
        else:
            rec[f"{m}_mean"] = None
            rec[f"{m}_std"] = None
    agg_rows.append(rec)
pd.DataFrame(agg_rows).to_csv("ragas_gpt41_summary.csv", index=False)

# =====================================================================
# Step 2: Causal correctness on BOTH graph_rag and graph_rag_cov
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Custom causal_correctness metric")
print("=" * 70)

causal_rows = []
for tech in ["graph_rag", "graph_rag_cov"]:
    print(f"\n--- {tech} ---")
    data = all_ans[tech]
    answers = data["answers"]
    contexts_list = data.get("contexts", [[] for _ in answers])

    for trial in range(1, N_TRIALS + 1):
        t0 = time.perf_counter()
        agg = causal_correctness_batch(
            judge, BENCHMARK_QUERIES, answers, contexts_list,
        )
        wall = time.perf_counter() - t0
        m = agg["mean_score"]
        print(f"  trial {trial}: mean causal_correctness="
              f"{(m if m is not None else float('nan')):.3f}  ({wall:.0f}s)")
        # Per-query breakdown
        for qi, pq in enumerate(agg["per_query"], 1):
            if pq["score"] is None:
                print(f"    Q{qi}: N/A")
            else:
                print(f"    Q{qi}: {pq['score']:.3f}  "
                      f"({pq['n_supported']}/{pq['n_inferred']}/{pq['n_unsupported']} s/i/u)")
        causal_rows.append({
            "technique": tech, "trial": trial,
            "mean_causal_correctness": agg["mean_score"],
            "n_scored_queries": agg["n_scored_queries"],
            "wall_seconds": round(wall, 1),
        })

causal_df = pd.DataFrame(causal_rows)
causal_df.to_csv("causal_correctness_raw.csv", index=False)

# Aggregate
print("\n=== AGGREGATE: causal_correctness ===")
for tech in causal_df["technique"].unique():
    sub = causal_df[causal_df["technique"] == tech]
    m = sub["mean_causal_correctness"].mean()
    s = sub["mean_causal_correctness"].std(ddof=1)
    print(f"  {tech:<24s} {m:.3f} ± {s:.3f}  (n={len(sub)} trials)")

print("\n=== AGGREGATE: standard RAGAS (gpt-4.1 judge) ===")
for tech in ["graph_rag", "graph_rag_cov"]:
    sub = combined[(combined["technique"] == tech)
                   & combined["answer_relevancy"].notna()]
    if len(sub) == 0:
        print(f"  {tech}: no data")
        continue
    print(f"\n  {tech} (n={len(sub)}):")
    for m in ["answer_relevancy", "faithfulness", "context_precision"]:
        if m in sub.columns and sub[m].notna().any():
            print(f"    {m:<22s} {sub[m].mean():.3f} ± {sub[m].std(ddof=1):.3f}")
