"""Quantify run-to-run RAGAS variance for 3 retrieval configs.

5 independent evaluations per config = 15 total RAGAS runs.
Reports mean +/- stdev for faithfulness, answer_relevancy, context_precision.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from rag_harness import (
    HarnessConfig, RetrievalConfig, run_model, BENCHMARK_QUERIES,
)

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
    generator_provider="openai", generator_model="gpt-4o-mini",
    judge_provider="openai", judge_model="gpt-4o-mini",
)

CONFIGS = [
    ("mmr_baseline",
     RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5)),
    ("mmr_with_rerank",
     RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5,
                     rerank=True, rerank_pool=20, rerank_top_n=6)),
    ("hybrid_rerank_pool30",
     RetrievalConfig(k=6, strategy="similarity",
                     hybrid=True, hybrid_pool=30,
                     rerank=True, rerank_pool=30, rerank_top_n=6)),
]
N_TRIALS = 5

raw_rows = []
for label, rcfg in CONFIGS:
    print(f"\n========== {label}  ({rcfg.label()}) ==========")
    for trial in range(1, N_TRIALS + 1):
        t0 = time.perf_counter()
        try:
            r = run_model(
                "BAAI/bge-base-en-v1.5", cfg,
                queries=BENCHMARK_QUERIES, retrieval=rcfg,
            )
            row = {
                "config": label, "trial": trial,
                "faithfulness": r["faithfulness"],
                "answer_relevancy": r["answer_relevancy"],
                "context_precision": r["context_precision"],
                "wall_seconds": round(time.perf_counter() - t0, 1),
            }
            print(f"  trial {trial}: faith={row['faithfulness']:.3f} "
                  f"ar={row['answer_relevancy']:.3f} cp={row['context_precision']:.3f} "
                  f"({row['wall_seconds']:.0f}s)")
        except Exception as e:
            print(f"  trial {trial}: ERROR {e}")
            row = {"config": label, "trial": trial, "error": str(e)}
        raw_rows.append(row)

raw_df = pd.DataFrame(raw_rows)
raw_df.to_csv("variance_raw.csv", index=False)

# Aggregate
agg = (
    raw_df.dropna(subset=["faithfulness"])
    .groupby("config")
    .agg(
        faith_mean=("faithfulness", "mean"),
        faith_std=("faithfulness", "std"),
        ar_mean=("answer_relevancy", "mean"),
        ar_std=("answer_relevancy", "std"),
        cp_mean=("context_precision", "mean"),
        cp_std=("context_precision", "std"),
        n=("trial", "count"),
    )
    .round(4)
)
# Composite (simple mean of three)
agg["composite_mean"] = (agg["faith_mean"] + agg["ar_mean"] + agg["cp_mean"]) / 3
agg["composite_std"] = np.sqrt(agg["faith_std"]**2 + agg["ar_std"]**2 + agg["cp_std"]**2) / 3
agg = agg.round(4)
agg.to_csv("variance_summary.csv")

print("\n========== AGGREGATE ==========")
print(agg.to_string())
