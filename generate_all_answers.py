"""Generate all 20 answers (4 strategies × 5 benchmark queries) and save to JSON.

Uses Groq llama-3.3-70b-versatile as generator across all four strategies so
differences are attributable to retrieval + prompting, not the model.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

from langchain_groq import ChatGroq

from rag_harness import HarnessConfig, BENCHMARK_QUERIES
import answer_runners as ar

keys = json.load(open("config.json"))

cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    # Generator: Groq 70B (the notebook's prescribed model)
    generator_provider="groq",
    generator_model="llama-3.3-70b-versatile",
    # Judge: 8b-instant (used later for RAGAS eval)
    judge_provider="groq",
    judge_model="llama-3.1-8b-instant",
)

llm = ChatGroq(
    model=cfg.generator_model,
    groq_api_key=cfg.groq_api_key,
    temperature=0,
)

print("Building vector stores...")
ar.build_strategies(cfg)

STRATEGIES = [
    ("base_llm", ar.base_llm_answer),
    ("prompt_engineered", ar.prompt_engineered_answer),
    ("base_rag", ar.base_rag_answer),
    ("tuned_rag", ar.tuned_rag_answer),
]

results: dict = {name: {"answers": [], "contexts": [], "latencies": []} for name, _ in STRATEGIES}

for strat_name, strat_fn in STRATEGIES:
    print(f"\n=== {strat_name} ===")
    for i, q in enumerate(BENCHMARK_QUERIES, 1):
        t0 = time.perf_counter()
        try:
            answer, contexts = strat_fn(q, llm)
            wall = time.perf_counter() - t0
            print(f"  Q{i}: {wall:.1f}s, {len(answer)} chars")
        except Exception as e:
            print(f"  Q{i}: ERROR {e}")
            answer, contexts, wall = f"[ERROR: {e}]", [], 0.0
        results[strat_name]["answers"].append(answer)
        results[strat_name]["contexts"].append(contexts)
        results[strat_name]["latencies"].append(round(wall, 2))

results["_meta"] = {
    "queries": BENCHMARK_QUERIES,
    "generator_model": cfg.generator_model,
    "generator_provider": cfg.generator_provider,
}

with open("all_answers.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Saved all_answers.json ===")
for strat_name, _ in STRATEGIES:
    avg = sum(results[strat_name]["latencies"]) / len(results[strat_name]["latencies"])
    avg_chars = sum(len(a) for a in results[strat_name]["answers"]) / len(results[strat_name]["answers"])
    print(f"  {strat_name:<20s} avg {avg:.1f}s, avg {avg_chars:.0f} chars")
