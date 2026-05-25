"""Full re-evaluation of all 6 techniques with gpt-4.1 as RAGAS judge, n=5 trials.

Replaces the gpt-4o-mini judge used in earlier runs. gpt-4.1 has materially
stronger multi-step reasoning, which directly improves the quality of:
  - claim extraction (Faithfulness step 1)
  - NLI-style entailment checks (Faithfulness step 2)
  - hypothetical question regeneration (Answer Relevancy step 1)

Checkpoints after each technique into ragas_gpt41_raw.csv so an interruption
mid-run doesn't lose work. Also writes ragas_gpt41_summary.csv aggregate.

Estimated cost: ~$3-5 total (gpt-4.1 at $2/M in, $8/M out).
"""
import json
import os
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

from rag_harness import HarnessConfig, BENCHMARK_QUERIES, build_embeddings

JUDGE_MODEL = "gpt-4.1"
N_TRIALS = 5

TECHNIQUES = [
    ("base_llm",           False),
    ("prompt_engineered",  False),
    ("base_rag",           True),
    ("tuned_rag",          True),
    ("graph_rag",          True),
    ("opus_tuned_rag",     True),
]

OUT_RAW = "ragas_gpt41_raw.csv"

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
)
judge = ChatOpenAI(model=JUDGE_MODEL, api_key=cfg.openai_api_key, temperature=0)
emb = build_embeddings("BAAI/bge-base-en-v1.5", cfg)
answers = json.load(open("all_answers.json"))

# Resume support
done_keys = set()
all_rows = []
if os.path.exists(OUT_RAW):
    prev = pd.read_csv(OUT_RAW)
    all_rows = prev.to_dict(orient="records")
    done_keys = {(r["technique"], r["trial"]) for r in all_rows
                 if pd.notna(r.get("answer_relevancy"))}
    print(f"[resume] found {len(done_keys)} prior completed (technique, trial) pairs")


def evaluate_one(tech: str, has_context: bool, trial: int) -> dict:
    metrics = [AnswerRelevancy()]
    if has_context:
        metrics += [Faithfulness(), LLMContextPrecisionWithoutReference()]

    ds_dict = {
        "question": BENCHMARK_QUERIES,
        "answer": answers[tech]["answers"],
    }
    if has_context:
        ds_dict["contexts"] = answers[tech]["contexts"]
    else:
        ds_dict["contexts"] = [[] for _ in BENCHMARK_QUERIES]
    ds = Dataset.from_dict(ds_dict)

    t0 = time.perf_counter()
    result = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb)
    df = result.to_pandas()
    wall = time.perf_counter() - t0

    row = {
        "technique": tech, "trial": trial,
        "judge": JUDGE_MODEL, "wall_seconds": round(wall, 1),
        "answer_relevancy": float(df["answer_relevancy"].mean()),
    }
    if has_context:
        row["faithfulness"] = float(df["faithfulness"].mean())
        row["context_precision"] = float(
            df["llm_context_precision_without_reference"].mean()
        )
    return row


# ============================================================================
# Main loop
# ============================================================================
overall_start = time.perf_counter()
print(f"\nJudge: {JUDGE_MODEL} | trials per technique: {N_TRIALS}")
print(f"Techniques: {[t for t, _ in TECHNIQUES]}\n")

for tech, has_ctx in TECHNIQUES:
    print(f"========== {tech} ({'RAG' if has_ctx else 'no retrieval'}) ==========")
    for trial in range(1, N_TRIALS + 1):
        if (tech, trial) in done_keys:
            print(f"  trial {trial}: SKIP (already done)")
            continue
        try:
            row = evaluate_one(tech, has_ctx, trial)
            log = f"  trial {trial}: AR={row['answer_relevancy']:.3f}"
            if has_ctx:
                log += f"  faith={row['faithfulness']:.3f}  cp={row['context_precision']:.3f}"
            log += f"  ({row['wall_seconds']:.0f}s)"
            print(log)
            all_rows.append(row)
        except Exception as e:
            print(f"  trial {trial}: ERROR {e}")
            all_rows.append({
                "technique": tech, "trial": trial, "judge": JUDGE_MODEL,
                "error": str(e),
            })
        # Checkpoint after every trial
        pd.DataFrame(all_rows).to_csv(OUT_RAW, index=False)

# ============================================================================
# Build aggregate summary
# ============================================================================
df = pd.DataFrame(all_rows)
df_good = df[df["answer_relevancy"].notna()].copy()

summary_rows = []
for tech, _ in TECHNIQUES:
    sub = df_good[df_good["technique"] == tech]
    if len(sub) == 0:
        continue
    rec = {"technique": tech, "n_trials": int(len(sub)), "judge": JUDGE_MODEL}
    for m in ["answer_relevancy", "faithfulness", "context_precision"]:
        if m in sub.columns and sub[m].notna().any():
            rec[f"{m}_mean"] = round(sub[m].mean(), 4)
            rec[f"{m}_std"] = round(sub[m].std(ddof=1), 4) if len(sub) > 1 else 0.0
        else:
            rec[f"{m}_mean"] = None
            rec[f"{m}_std"] = None
    summary_rows.append(rec)

summary = pd.DataFrame(summary_rows)
summary.to_csv("ragas_gpt41_summary.csv", index=False)

wall_total = time.perf_counter() - overall_start
print(f"\n========== COMPLETE ==========")
print(f"Total wall: {wall_total/60:.1f} min")
print("\n--- Summary (mean ± stdev) ---")
for _, r in summary.iterrows():
    print(f"\n{r['technique']:<22s} (n={r['n_trials']} trials)")
    for m in ["answer_relevancy", "faithfulness", "context_precision"]:
        mean = r.get(f"{m}_mean"); std = r.get(f"{m}_std")
        if mean is not None and not pd.isna(mean):
            print(f"  {m:<22s} {mean:.3f} ± {std:.3f}")
        else:
            print(f"  {m:<22s} N/A")
