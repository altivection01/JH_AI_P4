"""Sweep: baseline MMR vs +rerank vs +hybrid vs +hybrid+rerank.

Embedder fixed at bge-base; chunks fixed at 380 tokens / 60 overlap (the new
tuned baseline). Judge = gpt-4o-mini.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from rag_harness import (
    HarnessConfig, RetrievalConfig, sweep_retrieval, BENCHMARK_QUERIES,
)

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
    generator_provider="openai", generator_model="gpt-4o-mini",
    judge_provider="openai", judge_model="gpt-4o-mini",
)

grid = [
    # Baseline
    RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5),

    # + reranker (cross-encoder over top-20 candidates)
    RetrievalConfig(
        k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5,
        rerank=True, rerank_pool=20, rerank_top_n=6,
    ),

    # Hybrid (BM25 + vector RRF)
    RetrievalConfig(
        k=6, strategy="similarity",
        hybrid=True, hybrid_pool=20,
    ),

    # Hybrid + rerank — the full pipeline
    RetrievalConfig(
        k=6, strategy="similarity",
        hybrid=True, hybrid_pool=30,
        rerank=True, rerank_pool=30, rerank_top_n=6,
    ),

    # Hybrid + rerank with a larger candidate pool
    RetrievalConfig(
        k=6, strategy="similarity",
        hybrid=True, hybrid_pool=50,
        rerank=True, rerank_pool=50, rerank_top_n=6,
    ),
]

t0 = time.perf_counter()
df = sweep_retrieval("BAAI/bge-base-en-v1.5", cfg, grid, queries=BENCHMARK_QUERIES)
print(f"\n=== Hybrid + rerank sweep done in {time.perf_counter()-t0:.0f}s ===\n")
print(df.to_string(index=False))
df.to_csv("hybrid_rerank_sweep.csv", index=False)
