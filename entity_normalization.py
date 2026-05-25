"""Entity normalization: merge surface variants of the same concept in the Neo4j graph.

Approach:
  1. Pull all entity ids and their primary type from Neo4j.
  2. Compute bge-base embeddings for every entity name.
  3. For each entity, find candidates (same primary type, cosine > THRESHOLD).
  4. Build canonical clusters via union-find.
  5. Pick canonical id per cluster (highest doc_freq; ties broken by shortest name).
  6. Redirect all edges and MENTIONS from aliases to canonical, delete aliases.
  7. Report compression statistics.

Outputs:
  entity_normalization_clusters.csv  - one row per (canonical, alias)
  entity_normalization_stats.csv     - summary metrics
"""
import json
import time
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from rag_harness import HarnessConfig, build_embeddings
from graph_rag import attach_neo4j, neo4j_graph

THRESHOLD = 0.85
RESTRICT_TO_SAME_TYPE = True   # only merge entities sharing the same primary type

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"], openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
)
attach_neo4j(cfg, keys)
g = neo4j_graph(cfg)

# 1. Pull entities + types + document frequency
print("[1] Loading entity list from Neo4j…")
rows = g.query("""
    MATCH (e:`__Entity__`)
    OPTIONAL MATCH (e)<-[:MENTIONS]-(d:Document)
    RETURN e.id AS id, [l IN labels(e) WHERE l <> '__Entity__'] AS types,
           count(DISTINCT d) AS doc_freq
""")
ent_df = pd.DataFrame(rows)
ent_df["primary_type"] = ent_df["types"].apply(lambda t: t[0] if t else "Untyped")
print(f"  {len(ent_df):,} entities loaded")

# 2. Embed each entity name (we already have an embedder for bge-base)
print("[2] Embedding entity names with bge-base…")
embeddings_fn = build_embeddings("BAAI/bge-base-en-v1.5", cfg)
# Underlying sentence-transformers model lives at embeddings_fn.inner.client
# (when PrefixedEmbeddings) or embeddings_fn.client (when plain HuggingFaceEmbeddings)
client = (embeddings_fn.inner.client
          if hasattr(embeddings_fn, "inner") else embeddings_fn.client)

names = ent_df["id"].tolist()
t0 = time.perf_counter()
vecs = client.encode(names, batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False)
print(f"  {len(vecs)} embeddings in {time.perf_counter()-t0:.1f}s, "
      f"dim={vecs.shape[1]}")

# 3. Compute pairwise similarity — within-type only (the typical case;
#    cross-type matches like Region/Country are intentionally suppressed).
print(f"[3] Finding similarity pairs (threshold={THRESHOLD}, "
      f"within_type={RESTRICT_TO_SAME_TYPE})…")
pairs = []  # (i, j, sim)
if RESTRICT_TO_SAME_TYPE:
    type_to_indices = defaultdict(list)
    for i, t in enumerate(ent_df["primary_type"].tolist()):
        type_to_indices[t].append(i)
    print(f"  {len(type_to_indices)} primary types — biggest cluster: "
          f"{max(len(v) for v in type_to_indices.values())} entities")
    for ptype, idxs in type_to_indices.items():
        if len(idxs) < 2:
            continue
        sub = vecs[idxs]                       # (n, d)
        sim = sub @ sub.T                      # (n, n)
        np.fill_diagonal(sim, 0.0)             # ignore self-matches
        hits_i, hits_j = np.where(sim >= THRESHOLD)
        for i, j in zip(hits_i, hits_j):
            if i < j:                          # de-duplicate (i,j) vs (j,i)
                pairs.append((idxs[i], idxs[j], float(sim[i, j])))
else:
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, 0.0)
    i_arr, j_arr = np.where(sim >= THRESHOLD)
    for i, j in zip(i_arr, j_arr):
        if i < j:
            pairs.append((i, j, float(sim[i, j])))

print(f"  {len(pairs):,} similar pairs (>= {THRESHOLD})")

# 4. Union-find to build clusters
print("[4] Building clusters via union-find…")
parent = list(range(len(ent_df)))

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb

for i, j, _ in pairs:
    union(i, j)

clusters = defaultdict(list)
for i in range(len(ent_df)):
    clusters[find(i)].append(i)
multi_clusters = {root: members for root, members in clusters.items()
                  if len(members) > 1}
print(f"  {len(multi_clusters):,} multi-entity clusters covering "
      f"{sum(len(m) for m in multi_clusters.values()):,} entities")

# 5. Pick canonical per cluster
print("[5] Picking canonical id per cluster…")
canonical_mapping: dict = {}   # alias_id -> canonical_id
cluster_rows = []
for root, members in multi_clusters.items():
    sub = ent_df.iloc[members].copy()
    # Highest doc_freq wins; ties broken by shortest name (most likely canonical form)
    sub = sub.sort_values(by=["doc_freq"], ascending=False)
    sub["name_len"] = sub["id"].str.len()
    sub = sub.sort_values(by=["doc_freq", "name_len"], ascending=[False, True])
    canonical_id = sub.iloc[0]["id"]
    canonical_type = sub.iloc[0]["primary_type"]
    canonical_freq = sub.iloc[0]["doc_freq"]
    for _, row in sub.iterrows():
        if row["id"] != canonical_id:
            canonical_mapping[row["id"]] = canonical_id
            cluster_rows.append({
                "cluster_root": canonical_id,
                "cluster_type": canonical_type,
                "alias": row["id"],
                "alias_freq": row["doc_freq"],
                "canonical_freq": canonical_freq,
            })

clusters_df = pd.DataFrame(cluster_rows)
clusters_df.to_csv("entity_normalization_clusters.csv", index=False)

print(f"  {len(canonical_mapping):,} aliases will be merged into "
      f"{len(multi_clusters):,} canonical entities")
print(f"\n  Sample clusters (top 15 by size):")
size_per_cluster = clusters_df.groupby("cluster_root").size().reset_index(name="n_aliases")
size_per_cluster = size_per_cluster.sort_values("n_aliases", ascending=False)
for _, r in size_per_cluster.head(15).iterrows():
    aliases = clusters_df[clusters_df["cluster_root"] == r["cluster_root"]]
    sample_aliases = ", ".join(aliases["alias"].head(4).tolist())
    if len(aliases) > 4:
        sample_aliases += f", + {len(aliases)-4} more"
    print(f"    [{r['n_aliases']+1:>2d}] {r['cluster_root']:<32s} <- {sample_aliases}")

# 6. Apply merges in Neo4j
print("\n[6] Applying merges in Neo4j…")
t0 = time.perf_counter()
n_merged = 0
for alias_id, canonical_id in canonical_mapping.items():
    # Redirect outgoing edges
    g.query("""
        MATCH (alias:`__Entity__` {id: $alias_id})-[r]->(other)
        WHERE NOT other:Document
        MATCH (canon:`__Entity__` {id: $canon_id})
        MERGE (canon)-[new_r:RELATED_TO {type: type(r)}]->(other)
        DELETE r
    """, params={"alias_id": alias_id, "canon_id": canonical_id})
    # Redirect incoming edges (excluding MENTIONS handled separately)
    g.query("""
        MATCH (other)-[r]->(alias:`__Entity__` {id: $alias_id})
        WHERE NOT other:Document
        MATCH (canon:`__Entity__` {id: $canon_id})
        MERGE (other)-[new_r:RELATED_TO {type: type(r)}]->(canon)
        DELETE r
    """, params={"alias_id": alias_id, "canon_id": canonical_id})
    # Redirect MENTIONS from Documents
    g.query("""
        MATCH (d:Document)-[r:MENTIONS]->(alias:`__Entity__` {id: $alias_id})
        MATCH (canon:`__Entity__` {id: $canon_id})
        MERGE (d)-[:MENTIONS]->(canon)
        DELETE r
    """, params={"alias_id": alias_id, "canon_id": canonical_id})
    # Delete alias node
    g.query("MATCH (a:`__Entity__` {id: $alias_id}) DETACH DELETE a",
            params={"alias_id": alias_id})
    n_merged += 1
    if n_merged % 200 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {n_merged}/{len(canonical_mapping)} merged "
              f"({elapsed:.0f}s, {n_merged/elapsed:.1f}/s)")

print(f"  Merge complete in {time.perf_counter()-t0:.1f}s")

# 7. Report new graph stats
print("\n[7] Post-merge graph statistics:")
ent_after = g.query("MATCH (e:`__Entity__`) RETURN count(e) AS n")[0]["n"]
edges_after = g.query("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
singles_after = g.query("""
    MATCH (e:`__Entity__`)
    WITH e, count{ (e)<-[:MENTIONS]-(:Document) } AS df
    WHERE df = 1
    RETURN count(e) AS n
""")[0]["n"]
print(f"  Entities:   {len(ent_df):,} → {ent_after:,}  (compression: {(1-ent_after/len(ent_df))*100:.1f}%)")
print(f"  Edges:      51,227 → {edges_after:,}")
print(f"  Singletons: 9,795 → {singles_after:,}  (reduction: {(1-singles_after/9795)*100:.1f}%)")

stats = {
    "threshold": THRESHOLD,
    "n_pairs": len(pairs),
    "n_clusters": len(multi_clusters),
    "n_aliases_merged": len(canonical_mapping),
    "entities_before": len(ent_df),
    "entities_after": ent_after,
    "compression_pct": round((1 - ent_after/len(ent_df))*100, 2),
    "edges_after": edges_after,
    "singletons_after": singles_after,
    "singleton_reduction_pct": round((1 - singles_after/9795)*100, 2),
}
pd.DataFrame([stats]).to_csv("entity_normalization_stats.csv", index=False)
print(f"\nSaved entity_normalization_stats.csv")
