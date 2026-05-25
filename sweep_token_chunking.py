"""Sweep token-based chunking against the char baseline.

Holds embedder=bge-base, retrieval=MMR(k=6, λ=0.5), judge=gpt-4o-mini fixed.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

from rag_harness import (
    HarnessConfig, RetrievalConfig, sweep_chunking, BENCHMARK_QUERIES,
)

keys = json.load(open("config.json"))
base = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    generator_provider="openai", generator_model="gpt-4o-mini",
    judge_provider="openai", judge_model="gpt-4o-mini",
)
rcfg = RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5)

# (chunk_size, overlap, unit)
grid = [
    (900, 150, "char"),    # current baseline (reusing cached store)
    (180,  30, "token"),
    (230,  40, "token"),   # ~equivalent budget to 900 chars
    (280,  50, "token"),
    (380,  60, "token"),   # larger semantic units, still <512 token bge-base limit
]

t0 = time.perf_counter()
df = sweep_chunking("BAAI/bge-base-en-v1.5", base, grid, rcfg, queries=BENCHMARK_QUERIES)
print(f"\n=== Token-vs-char sweep done in {time.perf_counter()-t0:.0f}s ===\n")
print(df.to_string(index=False))
df.to_csv("chunk_sweep_token_vs_char.csv", index=False)
