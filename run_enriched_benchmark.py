"""Generate Tuned RAG answers against the Docling and Docling+Charts stores,
then evaluate with the same 5-trial gpt-4.1 RAGAS methodology.

Comparison:
  - tuned_rag                    : existing PyMuPDF baseline (in all_answers.json)
  - tuned_rag_docling            : same retrieval, Docling text only
  - tuned_rag_docling_charts     : same retrieval, Docling text + VLM chart extracts
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference,
)
from ragas.run_config import RunConfig


BENCHMARK_QUERIES = [
    "How is the rapid global expansion of artificial intelligence data centres impacting overall electricity demand and straining existing power grid infrastructure?",
    "How is the unprecedented wave of new US liquefied natural gas (LNG) export capacity expected to impact natural gas affordability and spur additional demand in price-sensitive Asian markets by 2030?",
    "How are the surge in US electricity demand and the 2025 federal emergency policy interventions collectively affecting the retirement schedules, capacity planning, and generation output of domestic coal-fired power plants?",
    "How are the increasing frequency of negative wholesale electricity prices and the regulatory shift towards two-sided Contracts for Difference (CfDs) in Europe altering the revenue expectations and financial agility of developers investing in utility-scale solar PV?",
    'How do the tax credit modifications under the US "One Big Beautiful Bill Act" (OBBBA) affect the investment economics of using domestic versus imported feedstocks for Sustainable Aviation Fuel (SAF), and what cascading impact will this biofuel transition have on the capacity rationalisation of traditional US West Coast refineries?',
]

SYSTEM = (
    "You are a senior energy markets analyst at Lumina Energy Partners. "
    "Answer ONLY from the provided IEA report excerpts. "
    "For every key quantitative claim, cite the source report and page in square "
    "brackets, e.g. [Gas2025 p.42]. "
    "If the excerpts do not contain the answer, say so explicitly. "
    "Structure as an investment-committee briefing: "
    "(1) Headline finding, (2) supporting data with citations, "
    "(3) cross-sector linkages, (4) countervailing forces. "
    "Be concise; 200-350 words. Avoid speculation beyond the sources."
)

STORES = [
    ("tuned_rag_docling",
     "chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling",
     "iea_docling"),
    ("tuned_rag_docling_charts",
     "chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling_charts",
     "iea_docling_charts"),
]
N_TRIALS = 3
keys = json.load(open("config.json"))

# Embedder
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "mps" if torch.backends.mps.is_available() else "cpu",
                  "model_kwargs": {"torch_dtype": torch.float16}
                  if torch.backends.mps.is_available() else {}},
    encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
)

# Generator (same as baseline)
generator = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=keys["GROQ_API_KEY"],
    temperature=0,
    max_tokens=1024,
)

# RAGAS judge
judge = ChatOpenAI(
    model="gpt-4.1", api_key=keys["OPENAI_API_KEY"],
    temperature=0, max_retries=10, timeout=600,
)
RAGAS_CFG = RunConfig(max_workers=2, timeout=900)

all_ans = json.load(open("all_answers.json"))


def format_context(docs) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", 0)
        is_chart = d.metadata.get("is_chart_extract", False)
        tag = " — CHART DATA" if is_chart else ""
        parts.append(f"[{src} p.{page}{tag}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def generate_for_store(label: str, persist_dir: str, collection: str) -> dict:
    print(f"\n========== Generating {label} ==========")
    store = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    print(f"  store has {store._collection.count()} chunks")

    answers, contexts = [], []
    for i, q in enumerate(BENCHMARK_QUERIES, 1):
        t0 = time.perf_counter()
        docs = store.max_marginal_relevance_search(q, k=6, fetch_k=20,
                                                    lambda_mult=0.5)
        ctx = format_context(docs)
        resp = generator.invoke([
            SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Question:\n{q}\n\nIEA excerpts:\n{ctx}"),
        ])
        answers.append(resp.content)
        contexts.append([d.page_content for d in docs])
        # Count how many of the 6 retrieved chunks were chart extracts
        n_chart = sum(1 for d in docs if d.metadata.get("is_chart_extract"))
        print(f"  Q{i}: {time.perf_counter()-t0:.1f}s, "
              f"{len(resp.content)} chars, {len(docs)} chunks "
              f"({n_chart} chart-data)")

    return {"answers": answers, "contexts": contexts}


def evaluate_technique(tech: str, data: dict) -> list[dict]:
    metrics = [AnswerRelevancy(), Faithfulness(), LLMContextPrecisionWithoutReference()]
    rows = []
    print(f"\n--- Evaluating {tech} (n={N_TRIALS} trials) ---")
    for trial in range(1, N_TRIALS + 1):
        t0 = time.perf_counter()
        ds = Dataset.from_dict({
            "question": BENCHMARK_QUERIES,
            "answer":   data["answers"],
            "contexts": data["contexts"],
        })
        try:
            r = evaluate(ds, metrics=metrics, llm=judge, embeddings=embeddings,
                         run_config=RAGAS_CFG).to_pandas()
            row = {
                "technique": tech, "trial": trial, "judge": "gpt-4.1",
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
    return rows


# ----- Generate answers -----
for tech, persist, collection in STORES:
    result = generate_for_store(tech, persist, collection)
    all_ans[tech] = result

json.dump(all_ans, open("all_answers.json", "w"), indent=2)
print(f"\nSaved {[t for t,_,_ in STORES]} answers to all_answers.json")

# ----- Evaluate -----
all_eval_rows = []
for tech, _, _ in STORES:
    rows = evaluate_technique(tech, all_ans[tech])
    all_eval_rows.extend(rows)

eval_df = pd.DataFrame(all_eval_rows)
# Append to ragas_gpt41_raw.csv
try:
    existing = pd.read_csv("ragas_gpt41_raw.csv")
    # Remove any prior rows for these techniques (in case of re-run)
    existing = existing[~existing["technique"].isin(
        [t for t, _, _ in STORES])]
    combined = pd.concat([existing, eval_df], ignore_index=True)
except FileNotFoundError:
    combined = eval_df
combined.to_csv("ragas_gpt41_raw.csv", index=False)

print(f"\n=== FINAL: enriched-corpus comparison ===")
for tech in [t for t, _, _ in STORES] + ["tuned_rag"]:
    sub = combined[(combined["technique"] == tech)
                   & combined["answer_relevancy"].notna()]
    if len(sub) == 0:
        continue
    print(f"\n  {tech} (n={len(sub)} trials):")
    for m in ["answer_relevancy", "faithfulness", "context_precision"]:
        if m in sub.columns and sub[m].notna().any():
            print(f"    {m:<22s} {sub[m].mean():.3f} ± {sub[m].std(ddof=1):.3f}")
