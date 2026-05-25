"""Generate Opus 4.5 answers using the Tuned RAG retrieval pipeline.

Same retrieval as Tuned RAG (bge-base + MMR k=6 + token chunking + cited prompt).
Only the generator changes from llama-3.3-70b to claude-opus-4-5.

Save after generation, before RAGAS. Then run 3-trial RAGAS evaluation.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy, Faithfulness, LLMContextPrecisionWithoutReference,
)

from rag_harness import (
    HarnessConfig, RetrievalConfig, BENCHMARK_QUERIES,
    retrieve, get_or_build_store, _format_context, build_embeddings,
)

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    anthropic_api_key=keys["ANTHROPIC_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
)
retrieval_cfg = RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5)

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

# =====================================================================
# Generation
# =====================================================================
print("=== Generating Opus 4.5 answers (Tuned RAG retrieval) ===")
store, _ = get_or_build_store("BAAI/bge-base-en-v1.5", cfg)
opus_llm = ChatAnthropic(
    model="claude-opus-4-5",
    anthropic_api_key=cfg.anthropic_api_key,
    temperature=0,
    max_tokens=1024,
)

answers, contexts, walls = [], [], []
for i, q in enumerate(BENCHMARK_QUERIES, 1):
    t0 = time.perf_counter()
    docs = retrieve(q, store, retrieval_cfg, cfg=cfg)
    context_str = _format_context(docs)
    resp = opus_llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Question:\n{q}\n\nIEA excerpts:\n{context_str}"),
    ])
    wall = time.perf_counter() - t0
    answers.append(resp.content)
    contexts.append([d.page_content for d in docs])
    walls.append(round(wall, 2))
    print(f"  Q{i}: {wall:.1f}s, {len(resp.content)} chars, {len(docs)} chunks")

# Save IMMEDIATELY so the answers aren't lost if RAGAS crashes
all_ans = json.load(open("all_answers.json"))
all_ans["opus_tuned_rag"] = {
    "answers": answers,
    "contexts": contexts,
    "wall_seconds": walls,
}
json.dump(all_ans, open("all_answers.json", "w"), indent=2)
print(f"  Saved opus_tuned_rag to all_answers.json (mean {sum(walls)/len(walls):.1f}s/query)")

# =====================================================================
# RAGAS evaluation: 3 trials
# =====================================================================
print("\n=== RAGAS evaluation: 3 trials ===")
judge = ChatOpenAI(model="gpt-4o-mini", api_key=cfg.openai_api_key, temperature=0)
emb = build_embeddings("BAAI/bge-base-en-v1.5", cfg)
metrics = [AnswerRelevancy(), Faithfulness(), LLMContextPrecisionWithoutReference()]

rows = []
for trial in range(1, 4):
    ds = Dataset.from_dict({
        "question": BENCHMARK_QUERIES,
        "answer":   answers,
        "contexts": contexts,
    })
    r = evaluate(ds, metrics=metrics, llm=judge, embeddings=emb).to_pandas()
    row = {
        "trial": trial,
        "answer_relevancy":  float(r["answer_relevancy"].mean()),
        "faithfulness":      float(r["faithfulness"].mean()),
        "context_precision": float(r["llm_context_precision_without_reference"].mean()),
    }
    print(f"  trial {trial}: AR={row['answer_relevancy']:.3f}  "
          f"faith={row['faithfulness']:.3f}  cp={row['context_precision']:.3f}")
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("opus_tuned_rag_eval_raw.csv", index=False)
print(f"\n=== Opus 4.5 + Tuned RAG (n=3) ===")
for m in ["answer_relevancy", "faithfulness", "context_precision"]:
    print(f"  {m:<22s} {df[m].mean():.3f} ± {df[m].std(ddof=1):.3f}")
