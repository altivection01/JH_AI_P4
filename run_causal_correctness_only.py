"""Step 2 only: causal_correctness for graph_rag vs graph_rag_cov.

Standalone runner so it doesn't share TPM window with RAGAS calls.
Heavy throttling (6s between calls, 30s between queries, 60s between trials)
to stay well under the OpenAI tier-1 gpt-4.1 cap (30K TPM).
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from langchain_openai import ChatOpenAI

from causal_correctness import causal_correctness_batch

# Inline copy of BENCHMARK_QUERIES from rag_harness to avoid pulling in
# RAGAS (which currently has a broken langchain_community.vertexai import)
BENCHMARK_QUERIES = [
    "How is the rapid global expansion of artificial intelligence data centres "
    "impacting overall electricity demand and straining existing power grid infrastructure?",
    "How is the unprecedented wave of new US liquefied natural gas (LNG) export "
    "capacity expected to impact natural gas affordability and spur additional "
    "demand in price-sensitive Asian markets by 2030?",
    "How are the surge in US electricity demand and the 2025 federal emergency "
    "policy interventions collectively affecting the retirement schedules, "
    "capacity planning, and generation output of domestic coal-fired power plants?",
    "How are the increasing frequency of negative wholesale electricity prices "
    "and the regulatory shift towards two-sided Contracts for Difference (CfDs) "
    "in Europe altering the revenue expectations and financial agility of "
    "developers investing in utility-scale solar PV?",
    'How do the tax credit modifications under the US "One Big Beautiful Bill Act" '
    "(OBBBA) affect the investment economics of using domestic versus imported "
    "feedstocks for Sustainable Aviation Fuel (SAF), and what cascading impact "
    "will this biofuel transition have on the capacity rationalisation of "
    "traditional US West Coast refineries?",
]

N_TRIALS = 5
keys = json.load(open("config.json"))
llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=keys["OPENAI_API_KEY"],
    temperature=0,
    max_retries=10,   # tier-1 occasionally bursts; long backoff is fine
    timeout=180,
)

all_ans = json.load(open("all_answers.json"))

rows = []
detail_rows = []   # per-query, per-trial details for richer notebook narration

print("Starting Step 2: causal_correctness on graph_rag and graph_rag_cov")
print("Throttle: 6s between claim checks, 30s between queries, 60s between trials")

for tech in ["graph_rag", "graph_rag_cov"]:
    print(f"\n========== {tech} ==========")
    data = all_ans[tech]
    answers = data["answers"]
    contexts_list = data.get("contexts", [[] for _ in answers])

    for trial in range(1, N_TRIALS + 1):
        t0 = time.perf_counter()
        agg = causal_correctness_batch(
            llm, BENCHMARK_QUERIES, answers, contexts_list,
        )
        wall = time.perf_counter() - t0
        m = agg["mean_score"]
        print(f"\n  trial {trial}: mean causal_correctness="
              f"{(m if m is not None else float('nan')):.3f}  ({wall:.0f}s)")
        for qi, pq in enumerate(agg["per_query"], 1):
            if pq["score"] is None:
                print(f"    Q{qi}: N/A (no causal claims)")
            else:
                print(f"    Q{qi}: {pq['score']:.3f}  "
                      f"({pq['n_supported']}/{pq['n_inferred']}/{pq['n_unsupported']} s/i/u)")
                detail_rows.append({
                    "technique": tech, "trial": trial, "query_idx": qi,
                    "score": pq["score"],
                    "n_supported": pq["n_supported"],
                    "n_inferred": pq["n_inferred"],
                    "n_unsupported": pq["n_unsupported"],
                    "n_causal": pq["n_causal"],
                })
        rows.append({
            "technique": tech, "trial": trial,
            "mean_causal_correctness": agg["mean_score"],
            "n_scored_queries": agg["n_scored_queries"],
            "wall_seconds": round(wall, 1),
        })
        # Checkpoint after every trial
        pd.DataFrame(rows).to_csv("causal_correctness_raw.csv", index=False)
        pd.DataFrame(detail_rows).to_csv("causal_correctness_per_query.csv", index=False)

        # Long pause between trials to give TPM window full recovery
        if trial < N_TRIALS:
            print(f"  (pausing 60s before next trial)")
            time.sleep(60)
    # And between techniques
    if tech == "graph_rag":
        print("\n  (pausing 90s before switching technique)")
        time.sleep(90)

# Final aggregate
print("\n========== AGGREGATE causal_correctness ==========")
df = pd.read_csv("causal_correctness_raw.csv")
for tech in df["technique"].unique():
    sub = df[df["technique"] == tech]
    m = sub["mean_causal_correctness"].mean()
    s = sub["mean_causal_correctness"].std(ddof=1)
    print(f"  {tech:<24s} {m:.3f} ± {s:.3f}  (n={len(sub)} trials)")
