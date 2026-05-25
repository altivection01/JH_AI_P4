"""Generate graph_rag_docling answers + 3-trial RAGAS eval.

Uses the freshly-extracted Docling-based knowledge graph + bge-base vector
index on Document nodes. Same retrieval architecture (4 vector + 4 graph),
same generator (llama-3.3-70b), same judge (gpt-4.1).
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference,
)
from ragas.run_config import RunConfig

from graph_rag import (
    attach_neo4j, setup_vector_index, neo4j_graph, graph_rag_answer,
)
from rag_harness import HarnessConfig, BENCHMARK_QUERIES, build_embeddings


N_TRIALS = 3
RAGAS_CFG = RunConfig(max_workers=2, timeout=900)

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
)
attach_neo4j(cfg, keys)

# Step 1: Build vector index on Document nodes
print("Building Neo4j vector index on Document nodes…")
t0 = time.perf_counter()
vec = setup_vector_index(cfg)
graph = neo4j_graph(cfg)
print(f"  done in {time.perf_counter()-t0:.0f}s "
      f"({vec.node_label} nodes indexed)")

# Step 2: Generate answers
print("\n=== Generating graph_rag_docling answers ===")
gen = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=cfg.groq_api_key,
    temperature=0,
    max_tokens=1024,
)
answers, contexts, walls = [], [], []
for i, q in enumerate(BENCHMARK_QUERIES, 1):
    t0 = time.perf_counter()
    ans, ctxs = graph_rag_answer(q, vec, graph, gen, k_seed=4, k_expansion=4)
    walls.append(round(time.perf_counter() - t0, 2))
    answers.append(ans)
    contexts.append(ctxs)
    print(f"  Q{i}: {walls[-1]}s, {len(ans)} chars, {len(ctxs)} contexts")

all_ans = json.load(open("all_answers.json"))
all_ans["graph_rag_docling"] = {
    "answers": answers, "contexts": contexts, "wall_seconds": walls,
}
json.dump(all_ans, open("all_answers.json", "w"), indent=2)
print("Saved graph_rag_docling to all_answers.json")

# Step 3: RAGAS eval (3 trials with gpt-4.1)
print(f"\n=== Running {N_TRIALS}-trial RAGAS eval ===")
judge = ChatOpenAI(
    model="gpt-4.1", api_key=cfg.openai_api_key,
    temperature=0, max_retries=10, timeout=600,
)
emb_for_judge = build_embeddings("BAAI/bge-base-en-v1.5", cfg)
metrics = [AnswerRelevancy(), Faithfulness(), LLMContextPrecisionWithoutReference()]

rows = []
for trial in range(1, N_TRIALS + 1):
    t0 = time.perf_counter()
    ds = Dataset.from_dict({
        "question": BENCHMARK_QUERIES,
        "answer":   answers,
        "contexts": contexts,
    })
    try:
        r = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb_for_judge,
                     run_config=RAGAS_CFG).to_pandas()
        row = {
            "technique": "graph_rag_docling", "trial": trial, "judge": "gpt-4.1",
            "answer_relevancy": float(r["answer_relevancy"].mean()),
            "faithfulness": float(r["faithfulness"].mean()),
            "context_precision": float(r["llm_context_precision_without_reference"].mean()),
            "wall_seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"  trial {trial}: AR={row['answer_relevancy']:.3f}  "
              f"faith={row['faithfulness']:.3f}  cp={row['context_precision']:.3f}  "
              f"({row['wall_seconds']:.0f}s)")
        rows.append(row)
    except Exception as e:
        print(f"  trial {trial}: ERROR {e}")

# Append to ragas_gpt41_raw.csv
new_df = pd.DataFrame(rows)
try:
    existing = pd.read_csv("ragas_gpt41_raw.csv")
    existing = existing[existing["technique"] != "graph_rag_docling"]
    combined = pd.concat([existing, new_df], ignore_index=True)
except FileNotFoundError:
    combined = new_df
combined.to_csv("ragas_gpt41_raw.csv", index=False)

# Final summary
print(f"\n=== FINAL: graph_rag_docling (n={N_TRIALS}) ===")
for m in ["answer_relevancy", "faithfulness", "context_precision"]:
    print(f"  {m:<22s} {new_df[m].mean():.3f} ± {new_df[m].std(ddof=1):.3f}")
