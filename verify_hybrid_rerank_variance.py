"""Run 10 more trials of hybrid+rerank(pool=30) to verify whether the original
trial 4 catastrophic AR=0.508 was a real instability or a one-off RAGAS judge fluke.

Combine with the original 5 trials -> 15 total trials. If the heavy left tail
disappears, the technique was unfairly maligned; if it recurs, our verdict holds.
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
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
    generator_provider="openai", generator_model="gpt-4o-mini",
    judge_provider="openai", judge_model="gpt-4o-mini",
)

rcfg = RetrievalConfig(
    k=6, strategy="similarity",
    hybrid=True, hybrid_pool=30,
    rerank=True, rerank_pool=30, rerank_top_n=6,
)

N_NEW_TRIALS = 10

new_rows = []
for trial in range(1, N_NEW_TRIALS + 1):
    t0 = time.perf_counter()
    try:
        r = run_model("BAAI/bge-base-en-v1.5", cfg,
                      queries=BENCHMARK_QUERIES, retrieval=rcfg)
        row = {
            "config": "hybrid_rerank_pool30",
            "trial_set": "new",
            "new_trial": trial,
            "faithfulness": r["faithfulness"],
            "answer_relevancy": r["answer_relevancy"],
            "context_precision": r["context_precision"],
            "wall_seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"  new_trial {trial}: faith={row['faithfulness']:.3f} "
              f"ar={row['answer_relevancy']:.3f} cp={row['context_precision']:.3f} "
              f"({row['wall_seconds']:.0f}s)")
    except Exception as e:
        print(f"  new_trial {trial}: ERROR {e}")
        row = {"config": "hybrid_rerank_pool30", "trial_set": "new",
               "new_trial": trial, "error": str(e)}
    new_rows.append(row)

new_df = pd.DataFrame(new_rows)
new_df.to_csv("hybrid_rerank_variance_extended.csv", index=False)

# Combine with the original 5-trial variance study
orig = pd.read_csv("variance_raw.csv")
orig_hybrid = orig[orig["config"] == "hybrid_rerank_pool30"].copy()
orig_hybrid["trial_set"] = "original"
orig_hybrid = orig_hybrid.rename(columns={"trial": "new_trial"})

combined = pd.concat(
    [orig_hybrid[["config", "trial_set", "new_trial",
                  "faithfulness", "answer_relevancy", "context_precision"]],
     new_df[["config", "trial_set", "new_trial",
             "faithfulness", "answer_relevancy", "context_precision"]]],
    ignore_index=True,
)

print("\n=== ALL TRIALS (original 5 + new 10 = 15 total) ===")
print(combined.to_string(index=False))

print("\n=== SUMMARY ===")
for col in ["faithfulness", "answer_relevancy", "context_precision"]:
    vals = combined[col].dropna()
    print(f"  {col:<22s} mean={vals.mean():.3f}  std={vals.std(ddof=1):.3f}  "
          f"min={vals.min():.3f}  max={vals.max():.3f}  n={len(vals)}")

# Specifically count "catastrophic" AR outliers (< 0.6)
ar_outliers = combined[combined["answer_relevancy"] < 0.6]
print(f"\n  AR < 0.6 outliers: {len(ar_outliers)} / {len(combined)} trials "
      f"({100*len(ar_outliers)/len(combined):.0f}%)")
print(f"  AR catastrophic-collapse rate (<0.55): "
      f"{len(combined[combined['answer_relevancy'] < 0.55])} / {len(combined)}")
