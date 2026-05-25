"""Re-run e5-mistral with the official MTEB instruction template.

Same retrieval config (MMR k=6, λ=0.5) and same judge (gpt-4o-mini) as the
main 5-model comparison. Result goes into a separate CSV so we can compare
the two e5-mistral runs side by side.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from rag_harness import HarnessConfig, RetrievalConfig, run_model, BENCHMARK_QUERIES

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    generator_provider="openai", generator_model="gpt-4o-mini",
    judge_provider="openai", judge_model="gpt-4o-mini",
)
retrieval = RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5)

MODEL = "intfloat/e5-mistral-7b-instruct"

t0 = time.perf_counter()
r = run_model(MODEL, cfg, queries=BENCHMARK_QUERIES, retrieval=retrieval)
wall = time.perf_counter() - t0

row = {
    "model": MODEL,
    "prompt_variant": "official_mteb",
    "faithfulness": round(r["faithfulness"], 4),
    "answer_relevancy": round(r["answer_relevancy"], 4),
    "context_precision": round(r["context_precision"], 4),
    "ingest_seconds": r["ingest_seconds"],
    "avg_query_seconds": r["avg_query_seconds"],
    "wall_seconds": round(wall, 1),
}
print(f"\n=== Result ===")
for k, v in row.items():
    print(f"  {k}: {v}")

pd.DataFrame([row]).to_csv("e5_mistral_official.csv", index=False)

# Also save answers/contexts for inspection
with open("e5_mistral_official_detail.json", "w") as f:
    json.dump({
        "answers": r["answers"],
        "per_query": r["per_query"],
    }, f, indent=2, default=str)
print("\nSaved: e5_mistral_official.csv + e5_mistral_official_detail.json")
