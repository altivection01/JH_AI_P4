"""Full 5-model embedding comparison under the tuned retrieval config.

Checkpoints after each model so an interruption doesn't lose progress.
"""
import json
import os
import time
import traceback
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from rag_harness import HarnessConfig, RetrievalConfig, run_model, BENCHMARK_QUERIES

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    generator_provider="openai", generator_model="gpt-4o-mini",
    judge_provider="openai", judge_model="gpt-4o-mini",
    trust_remote_code=True,   # required by nomic-v1.5; safe for the others
)

retrieval = RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5)

MODELS = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "nomic-ai/nomic-embed-text-v1.5",
    "intfloat/e5-mistral-7b-instruct",   # GPU-bound on MPS; slow but works
]

OUT_CSV = "model_comparison_full.csv"
OUT_JSON = "model_comparison_detail.json"

summary_rows: list[dict] = []
detail: dict = {}

# Resume support: load existing results if present
if os.path.exists(OUT_CSV):
    print(f"[resume] found existing {OUT_CSV}, loading prior rows")
    summary_rows = pd.read_csv(OUT_CSV).to_dict(orient="records")
if os.path.exists(OUT_JSON):
    detail = json.load(open(OUT_JSON))

completed = {r["model"] for r in summary_rows if "model" in r and "error" not in r}
print(f"[resume] completed models so far: {sorted(completed) or 'none'}")

run_started = time.perf_counter()

for model_name in MODELS:
    if model_name in completed:
        print(f"\n=== SKIP {model_name} (already done) ===")
        continue

    print(f"\n=== {model_name} ===")
    t0 = time.perf_counter()
    try:
        r = run_model(model_name, cfg, queries=BENCHMARK_QUERIES, retrieval=retrieval)
        wall = time.perf_counter() - t0

        summary_rows.append({
            "model": model_name,
            "faithfulness": round(r["faithfulness"], 4),
            "answer_relevancy": round(r["answer_relevancy"], 4),
            "context_precision": round(r["context_precision"], 4),
            "ingest_seconds": r["ingest_seconds"],
            "avg_query_seconds": r["avg_query_seconds"],
            "wall_seconds": round(wall, 1),
        })
        detail[model_name] = {
            "per_query": r["per_query"],
            "answers": r["answers"],
        }
        print(f"[done] {model_name} in {wall:.0f}s -> "
              f"faith={r['faithfulness']:.3f} ar={r['answer_relevancy']:.3f} "
              f"cp={r['context_precision']:.3f}")
    except Exception as e:
        print(f"[error] {model_name}: {e}")
        traceback.print_exc()
        summary_rows.append({"model": model_name, "error": str(e)})

    # Checkpoint after every model
    pd.DataFrame(summary_rows).to_csv(OUT_CSV, index=False)
    json.dump(detail, open(OUT_JSON, "w"), indent=2)
    print(f"[checkpoint] wrote {OUT_CSV} ({len(summary_rows)} rows)")

total_wall = time.perf_counter() - run_started
print(f"\n=== ALL DONE in {total_wall/60:.1f} min ===\n")
df = pd.DataFrame(summary_rows)
if "context_precision" in df:
    df = df.sort_values(
        ["context_precision", "faithfulness", "answer_relevancy"], ascending=False
    )
print(df.to_string(index=False))
