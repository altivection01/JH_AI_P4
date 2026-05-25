"""Run citation_accuracy across all RAG techniques that cite sources.

Two passes:
  1. page_offset=0 (chunk-metadata convention, LLM's view) — 3 trials
     This measures whether the LLM is correctly citing the chunks it saw.
  2. page_offset=-1 (printed-PDF-page convention, analyst's view) — 1 trial
     This demonstrates the off-by-one bug an analyst would experience.

Both runs use gpt-4.1 as judge, heavily throttled to stay under tier-1
30K TPM cap.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from langchain_openai import ChatOpenAI

from citation_verifier import citation_accuracy_batch, extract_citations

# Same benchmark queries (inline to avoid the ragas import chain)
BENCHMARK_QUERIES = [
    "Q1", "Q2", "Q3", "Q4", "Q5",   # placeholders; we only need indexing
]

# Techniques to evaluate. Skip base_llm and prompt_engineered (no citations).
TECHNIQUES_TO_VERIFY = [
    "base_rag", "tuned_rag", "graph_rag", "opus_tuned_rag", "graph_rag_cov",
]

keys = json.load(open("config.json"))
llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=keys["OPENAI_API_KEY"],
    temperature=0,
    max_retries=10,
    timeout=180,
)
all_ans = json.load(open("all_answers.json"))


def evaluate_pass(page_offset: int, n_trials: int, label: str) -> list[dict]:
    rows = []
    print(f"\n{'='*70}\n  PASS: {label} (page_offset={page_offset}, n_trials={n_trials})\n{'='*70}")
    for tech in TECHNIQUES_TO_VERIFY:
        if tech not in all_ans:
            print(f"  [skip] {tech}: not in all_answers.json")
            continue
        answers = all_ans[tech]["answers"]
        # Pre-count citations across answers
        cit_counts = [len(extract_citations(a)) for a in answers]
        if sum(cit_counts) == 0:
            print(f"  [skip] {tech}: no citations found in any answer")
            continue
        print(f"\n--- {tech} ({sum(cit_counts)} total citations across {len(answers)} answers) ---")

        for trial in range(1, n_trials + 1):
            t0 = time.perf_counter()
            agg = citation_accuracy_batch(
                llm, answers,
                pause_between_queries=20.0,
                page_offset=page_offset,
            )
            wall = time.perf_counter() - t0
            m = agg["mean_score"]
            print(f"  trial {trial}: mean citation_accuracy="
                  f"{(m if m is not None else float('nan')):.3f}  ({wall:.0f}s)")
            for qi, pq in enumerate(agg["per_query"], 1):
                if pq["score"] is None:
                    print(f"    Q{qi}: no citations")
                else:
                    print(f"    Q{qi}: {pq['score']:.3f}  "
                          f"({pq['n_supported']}/{pq['n_partial']}/{pq['n_unsupported']} sup/par/uns, "
                          f"n={pq['n_citations']})")
            rows.append({
                "technique": tech, "trial": trial, "pass": label,
                "page_offset": page_offset,
                "mean_citation_accuracy": agg["mean_score"],
                "n_scored_queries": agg["n_scored_queries"],
                "wall_seconds": round(wall, 1),
            })
            # Long pause between trials of the same technique
            if trial < n_trials:
                print(f"  (pausing 45s before next trial)")
                time.sleep(45)
        # Pause between techniques
        print(f"  (pausing 60s before next technique)")
        time.sleep(60)
    return rows


# ----- Pass 1: chunk-metadata convention (3 trials) -----
pass1_rows = evaluate_pass(page_offset=0, n_trials=3, label="chunk_meta")

# Save intermediate
pd.DataFrame(pass1_rows).to_csv("citation_accuracy_raw.csv", index=False)
print("\n[checkpoint] saved citation_accuracy_raw.csv after pass 1")

# Long pause before pass 2
print("\n(pausing 90s before analyst-convention pass)")
time.sleep(90)

# ----- Pass 2: analyst convention (1 trial) -----
pass2_rows = evaluate_pass(page_offset=-1, n_trials=1, label="analyst_view")

all_rows = pass1_rows + pass2_rows
df = pd.DataFrame(all_rows)
df.to_csv("citation_accuracy_raw.csv", index=False)

print("\n========== AGGREGATE ==========")
for pass_label in ["chunk_meta", "analyst_view"]:
    sub = df[df["pass"] == pass_label]
    if len(sub) == 0:
        continue
    print(f"\n--- {pass_label} ---")
    for tech in sub["technique"].unique():
        s = sub[sub["technique"] == tech]
        m = s["mean_citation_accuracy"].mean()
        sd = s["mean_citation_accuracy"].std(ddof=1) if len(s) > 1 else 0.0
        print(f"  {tech:<22s} {m:.3f} ± {sd:.3f}  (n={len(s)} trials)")
