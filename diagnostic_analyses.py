"""Two diagnostic analyses on the IEA corpus / graph / retrieval pipeline.

Analysis 1: Term-frequency / TF-IDF / long-tail structure of the entity graph.
Analysis 2: BM25 vs dense retrieval overlap on the 30-query benchmark.

Outputs:
  diagnostic_analysis1_entity_stats.csv
  diagnostic_analysis1_relation_stats.csv
  diagnostic_analysis1_entity_type_stats.csv
  diagnostic_analysis1_tail.png
  diagnostic_analysis2_overlap_raw.csv
  diagnostic_analysis2_overlap_summary.csv
  diagnostic_analysis2_overlap.png
"""
import json
import math
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rag_harness import (
    HarnessConfig, _get_bm25, _tokenize_bm25, BENCHMARK_QUERIES, build_embeddings,
)
from graph_rag import attach_neo4j, neo4j_graph, setup_vector_index

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
)
attach_neo4j(cfg, keys)


# ===========================================================================
# ANALYSIS 1 — Graph term-frequency / TF-IDF / long-tail
# ===========================================================================
print("=" * 70)
print("ANALYSIS 1: Entity and relation distribution analysis")
print("=" * 70)

g = neo4j_graph(cfg)

# 1a. Entity-mention frequency: how many chunks does each entity appear in?
print("\n[1a] Computing entity document-frequency…")
entity_rows = g.query("""
    MATCH (d:Document)-[:MENTIONS]->(e:`__Entity__`)
    WITH e, count(DISTINCT d) AS doc_freq
    RETURN e.id AS entity, doc_freq,
           [l IN labels(e) WHERE l <> '__Entity__'] AS types
    ORDER BY doc_freq DESC
""")
ent_df = pd.DataFrame(entity_rows)
ent_df["primary_type"] = ent_df["types"].apply(lambda t: t[0] if t else "Untyped")
N_chunks = g.query("MATCH (d:Document) RETURN count(d) AS n")[0]["n"]
ent_df["idf"] = ent_df["doc_freq"].apply(lambda df: math.log(N_chunks / df))
ent_df["tfidf_proxy"] = ent_df["doc_freq"] * ent_df["idf"]  # rough relevance score
ent_df.to_csv("diagnostic_analysis1_entity_stats.csv", index=False)

n_singletons = (ent_df["doc_freq"] == 1).sum()
n_2_5        = ((ent_df["doc_freq"] >= 2) & (ent_df["doc_freq"] <= 5)).sum()
n_6_20       = ((ent_df["doc_freq"] >= 6) & (ent_df["doc_freq"] <= 20)).sum()
n_21_plus    = (ent_df["doc_freq"] >= 21).sum()
total = len(ent_df)
print(f"  Total entities: {total}")
print(f"  Singletons (df=1):       {n_singletons:>5d}  ({100*n_singletons/total:.1f}%)")
print(f"  Rare    (df=2-5):        {n_2_5:>5d}  ({100*n_2_5/total:.1f}%)")
print(f"  Common  (df=6-20):       {n_6_20:>5d}  ({100*n_6_20/total:.1f}%)")
print(f"  Frequent (df=21+):       {n_21_plus:>5d}  ({100*n_21_plus/total:.1f}%)")
print(f"  Max doc_freq:            {ent_df['doc_freq'].max()}")
print(f"  Median doc_freq:         {ent_df['doc_freq'].median():.1f}")

print("\n  Top 10 most-mentioned entities:")
for _, r in ent_df.head(10).iterrows():
    print(f"    {r['primary_type']:<16s} {r['entity']:<40s} {r['doc_freq']:>4d} chunks")

# 1b. Relation type distribution
print("\n[1b] Computing relation type distribution…")
rel_rows = g.query("""
    MATCH ()-[r]->()
    RETURN type(r) AS rel_type, count(*) AS n
    ORDER BY n DESC
""")
rel_df = pd.DataFrame(rel_rows)
total_edges = rel_df["n"].sum()
rel_df["pct"] = 100 * rel_df["n"] / total_edges
rel_df.to_csv("diagnostic_analysis1_relation_stats.csv", index=False)
print(f"  Total edges: {total_edges}")
print(f"  Distinct relation types: {len(rel_df)}")
print(f"\n  Distribution:")
for _, r in rel_df.iterrows():
    print(f"    {r['rel_type']:<18s} {r['n']:>6d}  ({r['pct']:.1f}%)")
top3_share = rel_df.head(3)["pct"].sum()
print(f"\n  Top-3 relation types account for {top3_share:.1f}% of all edges")

# 1c. Entity-type composition
print("\n[1c] Entity-type composition…")
type_counts = ent_df["primary_type"].value_counts()
type_df = type_counts.reset_index()
type_df.columns = ["entity_type", "n"]
type_df["pct"] = 100 * type_df["n"] / total
type_df.to_csv("diagnostic_analysis1_entity_type_stats.csv", index=False)
for _, r in type_df.iterrows():
    print(f"    {r['entity_type']:<18s} {r['n']:>6d}  ({r['pct']:.1f}%)")

# 1d. Plot the long-tail distribution
print("\n[1d] Generating long-tail visualization…")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rank-frequency log-log (Zipf check)
ranks = np.arange(1, len(ent_df) + 1)
axes[0].loglog(ranks, ent_df["doc_freq"].values, marker=".", linestyle="none",
                alpha=0.4, markersize=2)
axes[0].set_xlabel("Entity rank (log)")
axes[0].set_ylabel("Document frequency (log)")
axes[0].set_title(f"Entity mention frequency vs rank (n={total})\n"
                   f"Power-law/Zipfian check")
axes[0].grid(True, alpha=0.3)

# Histogram of doc_freq buckets
buckets = ["1", "2-5", "6-20", "21-50", "51+"]
counts = [n_singletons, n_2_5, n_6_20,
          ((ent_df["doc_freq"] >= 21) & (ent_df["doc_freq"] <= 50)).sum(),
          (ent_df["doc_freq"] > 50).sum()]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
axes[1].bar(buckets, counts, color=colors)
axes[1].set_xlabel("Document frequency bucket")
axes[1].set_ylabel("Number of entities")
axes[1].set_title(f"Entity coverage distribution\n"
                   f"{100*n_singletons/total:.1f}% are singletons (df=1)")
for i, c in enumerate(counts):
    axes[1].text(i, c + total*0.005, f"{c}\n({100*c/total:.1f}%)",
                  ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("diagnostic_analysis1_tail.png", dpi=120, bbox_inches="tight")
print(f"  Saved diagnostic_analysis1_tail.png")


# ===========================================================================
# ANALYSIS 2 — BM25 vs Dense retrieval overlap
# ===========================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: BM25 vs dense retrieval overlap")
print("=" * 70)

# Combine benchmark + synthetic queries
all_queries = list(BENCHMARK_QUERIES)
synthetic = json.load(open("synthetic_queries.json"))
all_queries.extend(synthetic)
print(f"\nQuery set: {len(BENCHMARK_QUERIES)} benchmark + {len(synthetic)} synthetic "
      f"= {len(all_queries)} total")

# Build/load BM25 (uses cached chunks from rag_harness)
print("Building BM25 index…")
bm25, chunks = _get_bm25(cfg)

def _normalize_content(text: str) -> str:
    """Strip Neo4j's `\\ntext: ` prefix and normalize whitespace."""
    if text.startswith("\ntext: "):
        text = text[7:]
    elif text.startswith("text: "):
        text = text[6:]
    return " ".join(text.split())[:200]

def chunk_key(d):
    return (d.metadata.get("source"),
            d.metadata.get("page"),
            _normalize_content(d.page_content))

# Build a key for BM25-side chunks too
bm25_keys = [chunk_key(c) for c in chunks]

# Build dense retriever
print("Loading dense retriever (bge-base from Neo4j vector index)…")
vec = setup_vector_index(cfg)

# RBO (rank-biased overlap) implementation
def rbo(list1, list2, p=0.9):
    """Rank-Biased Overlap — gives partial credit for near-rank agreement."""
    s1, s2 = set(), set()
    overlap = 0.0
    for d in range(1, max(len(list1), len(list2)) + 1):
        if d <= len(list1):
            s1.add(list1[d - 1])
        if d <= len(list2):
            s2.add(list2[d - 1])
        intersect_at_d = len(s1 & s2)
        agreement = intersect_at_d / d
        overlap += p ** (d - 1) * agreement
    return (1 - p) * overlap

# Per-query overlap at k=5 and k=10
print("\nComputing per-query overlap…")
rows = []
K_values = [5, 10]

for qi, q in enumerate(all_queries, 1):
    # Dense top-K
    dense_docs = vec.similarity_search(q, k=max(K_values))
    dense_keys = [chunk_key(d) for d in dense_docs]

    # BM25 top-K
    bm25_scores = bm25.get_scores(_tokenize_bm25(q))
    top_idx = np.argsort(bm25_scores)[::-1][:max(K_values)]
    bm25_top_keys = [bm25_keys[i] for i in top_idx]

    row = {"query_id": qi, "is_benchmark": qi <= len(BENCHMARK_QUERIES),
           "query_preview": q[:80]}
    for k in K_values:
        d_set = set(dense_keys[:k])
        b_set = set(bm25_top_keys[:k])
        inter = d_set & b_set
        union = d_set | b_set
        row[f"overlap@{k}"]  = len(inter) / k
        row[f"jaccard@{k}"]  = len(inter) / len(union) if union else 0.0
    row["rbo_p0.9"] = rbo(dense_keys, bm25_top_keys, p=0.9)
    rows.append(row)

overlap_df = pd.DataFrame(rows)
overlap_df.to_csv("diagnostic_analysis2_overlap_raw.csv", index=False)

# Aggregate
summary = {
    "n_queries": len(all_queries),
    "overlap@5_mean":    round(overlap_df["overlap@5"].mean(), 3),
    "overlap@5_std":     round(overlap_df["overlap@5"].std(ddof=1), 3),
    "overlap@5_min":     round(overlap_df["overlap@5"].min(), 3),
    "overlap@5_max":     round(overlap_df["overlap@5"].max(), 3),
    "overlap@10_mean":   round(overlap_df["overlap@10"].mean(), 3),
    "overlap@10_std":    round(overlap_df["overlap@10"].std(ddof=1), 3),
    "jaccard@10_mean":   round(overlap_df["jaccard@10"].mean(), 3),
    "rbo_p0.9_mean":     round(overlap_df["rbo_p0.9"].mean(), 3),
}
# Split by benchmark vs synthetic
bench = overlap_df[overlap_df["is_benchmark"]]
synth = overlap_df[~overlap_df["is_benchmark"]]
summary["bench_overlap@10_mean"]  = round(bench["overlap@10"].mean(), 3)
summary["synth_overlap@10_mean"]  = round(synth["overlap@10"].mean(), 3)

pd.DataFrame([summary]).to_csv("diagnostic_analysis2_overlap_summary.csv", index=False)

print(f"\n--- Aggregate overlap (n={len(all_queries)} queries) ---")
for k, v in summary.items():
    print(f"  {k:<26s} {v}")

print(f"\n--- Per-query overlap (sample, first 8) ---")
print(overlap_df.head(8)[["query_id","is_benchmark","overlap@5","overlap@10","rbo_p0.9","query_preview"]].to_string(index=False))

# Plot the overlap distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Histogram of overlap@10
axes[0].hist(overlap_df["overlap@10"], bins=11, range=(0, 1.05),
              color="#1f77b4", edgecolor="black", alpha=0.85)
axes[0].axvline(overlap_df["overlap@10"].mean(), color="red", linestyle="--",
                 label=f"mean={overlap_df['overlap@10'].mean():.2f}")
axes[0].set_xlabel("Overlap@10  (|dense ∩ bm25| / 10)")
axes[0].set_ylabel("Query count")
axes[0].set_title(f"Distribution of BM25 vs dense overlap@10\n"
                   f"n={len(all_queries)} queries")
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis="y")

# Bench vs synth comparison
axes[1].boxplot([bench["overlap@10"].values, synth["overlap@10"].values],
                tick_labels=["Benchmark (n=5)", "Synthetic (n=25)"],
                patch_artist=True,
                boxprops={"facecolor":"#aec7e8"}, medianprops={"color":"red"})
axes[1].set_ylabel("Overlap@10")
axes[1].set_title("Overlap@10: benchmark vs synthetic queries")
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("diagnostic_analysis2_overlap.png", dpi=120, bbox_inches="tight")
print(f"\n  Saved diagnostic_analysis2_overlap.png")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
