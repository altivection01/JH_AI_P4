"""Score all four techniques in `all_answers.json` with RAGAS, N trials each.

For the two non-RAG techniques (base_llm, prompt_engineered), only Answer
Relevancy is applicable since Faithfulness and Context Precision require
retrieved context.

Output:
    four_technique_eval_raw.csv    — every trial row
    four_technique_eval_summary.csv — mean ± stdev per technique × metric
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
)

from rag_harness import build_embeddings, HarnessConfig, BENCHMARK_QUERIES

N_TRIALS = 3
TECHNIQUES = ["base_llm", "prompt_engineered", "base_rag", "tuned_rag"]

keys = json.load(open("config.json"))
cfg = HarnessConfig(groq_api_key=keys["GROQ_API_KEY"],
                    openai_api_key=keys["OPENAI_API_KEY"])
judge = ChatOpenAI(model="gpt-4o-mini", api_key=keys["OPENAI_API_KEY"], temperature=0)
# Embeddings model used only for AnswerRelevancy's question-answer similarity step.
emb = build_embeddings("BAAI/bge-base-en-v1.5", cfg)

answers = json.load(open("all_answers.json"))

rows = []
for tech in TECHNIQUES:
    has_context = tech in ("base_rag", "tuned_rag")
    metrics = [AnswerRelevancy()]
    if has_context:
        metrics += [Faithfulness(), LLMContextPrecisionWithoutReference()]

    print(f"\n=== {tech} ({'RAG' if has_context else 'no retrieval'}) ===")

    for trial in range(1, N_TRIALS + 1):
        ds_dict = {
            "question": BENCHMARK_QUERIES,
            "answer": answers[tech]["answers"],
        }
        if has_context:
            # Non-empty contexts required even for non-context metrics; we pass
            # the retrieved chunks the technique actually used.
            ds_dict["contexts"] = answers[tech]["contexts"]
        else:
            # AnswerRelevancy doesn't need contexts, but the Dataset still needs
            # a column. Pass an empty list per row.
            ds_dict["contexts"] = [[] for _ in BENCHMARK_QUERIES]
        ds = Dataset.from_dict(ds_dict)

        t0 = time.perf_counter()
        try:
            result = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb)
            df = result.to_pandas()
            wall = time.perf_counter() - t0
            row = {"technique": tech, "trial": trial,
                   "wall_seconds": round(wall, 1)}
            row["answer_relevancy"] = float(df["answer_relevancy"].mean())
            if has_context:
                row["faithfulness"] = float(df["faithfulness"].mean())
                row["context_precision"] = float(
                    df["llm_context_precision_without_reference"].mean()
                )
            print(f"  trial {trial}: AR={row['answer_relevancy']:.3f}"
                  + (f"  faith={row['faithfulness']:.3f}"
                     f"  cp={row['context_precision']:.3f}" if has_context else "")
                  + f"  ({wall:.0f}s)")
            rows.append(row)
        except Exception as e:
            print(f"  trial {trial}: ERROR {e}")
            rows.append({"technique": tech, "trial": trial, "error": str(e)})

raw = pd.DataFrame(rows)
raw.to_csv("four_technique_eval_raw.csv", index=False)

# Summary
metric_cols = ["answer_relevancy", "faithfulness", "context_precision"]
summary_rows = []
for tech in TECHNIQUES:
    sub = raw[raw["technique"] == tech].dropna(subset=["answer_relevancy"])
    if len(sub) == 0:
        continue
    rec = {"technique": tech, "n_trials": len(sub)}
    for m in metric_cols:
        if m in sub.columns and sub[m].notna().any():
            rec[f"{m}_mean"] = round(sub[m].mean(), 4)
            rec[f"{m}_std"]  = round(sub[m].std(ddof=1), 4) if len(sub) > 1 else 0.0
        else:
            rec[f"{m}_mean"] = None
            rec[f"{m}_std"] = None
    summary_rows.append(rec)

summary = pd.DataFrame(summary_rows)
summary.to_csv("four_technique_eval_summary.csv", index=False)

print("\n\n========== SUMMARY (mean ± stdev) ==========")
for _, r in summary.iterrows():
    print(f"\n{r['technique']:<22s} (n={r['n_trials']} trials)")
    for m in metric_cols:
        mean = r.get(f"{m}_mean")
        std = r.get(f"{m}_std")
        if mean is not None and not pd.isna(mean):
            print(f"  {m:<22s} {mean:.3f} ± {std:.3f}")
        else:
            print(f"  {m:<22s} N/A (no retrieval)")
