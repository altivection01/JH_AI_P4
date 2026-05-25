"""Experiment 1: try alternative rerankers for the hybrid retrieval pipeline.

Compares three configurations, all on top of bge-base + token chunking + hybrid retrieval:
  - bge-reranker-v2-m3      (current default; established baseline)
  - jina-reranker-v2-base-multilingual  (different architecture, local)
  - no reranker             (hybrid + RRF only — let BM25 contribution survive)

Uses gpt-4o-mini judge. n=3 trials per config to get stdev.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from rag_harness import (
    HarnessConfig, RetrievalConfig, run_model, BENCHMARK_QUERIES,
)

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"], openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
    generator_provider="openai", generator_model="gpt-4o-mini",
    judge_provider="openai", judge_model="gpt-4o-mini",
)

CONFIGS = [
    ("hybrid_no_rerank", RetrievalConfig(
        k=6, strategy="similarity",
        hybrid=True, hybrid_pool=30,
    )),
    ("hybrid_bge_rerank", RetrievalConfig(
        k=6, strategy="similarity",
        hybrid=True, hybrid_pool=30,
        rerank=True, rerank_pool=30, rerank_top_n=6,
        rerank_model="BAAI/bge-reranker-v2-m3",
    )),
    ("hybrid_jina_rerank", RetrievalConfig(
        k=6, strategy="similarity",
        hybrid=True, hybrid_pool=30,
        rerank=True, rerank_pool=30, rerank_top_n=6,
        rerank_model="jinaai/jina-reranker-v2-base-multilingual",
    )),
]

N_TRIALS = 3
rows = []

for label, rcfg in CONFIGS:
    print(f"\n========== {label}  ({rcfg.label()}) ==========")
    for trial in range(1, N_TRIALS + 1):
        t0 = time.perf_counter()
        try:
            r = run_model("BAAI/bge-base-en-v1.5", cfg,
                          queries=BENCHMARK_QUERIES, retrieval=rcfg)
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
        rows.append(row)

raw = pd.DataFrame(rows)
raw.to_csv("experiment_rerankers_raw.csv", index=False)

# Aggregate mean ± stdev
import numpy as np
agg = (raw.dropna(subset=["faithfulness"])
       .groupby("config")
       .agg(faith_mean=("faithfulness", "mean"),
            faith_std=("faithfulness", "std"),
            ar_mean=("answer_relevancy", "mean"),
            ar_std=("answer_relevancy", "std"),
            cp_mean=("context_precision", "mean"),
            cp_std=("context_precision", "std"),
            n=("trial", "count"))
       .round(4))
agg.to_csv("experiment_rerankers_summary.csv")

print("\n========== AGGREGATE ==========")
print(agg.to_string())
