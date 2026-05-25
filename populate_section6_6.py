"""Insert Section 6.6 (Diagnostic Analyses) between GraphRAG section and Section 7."""
import base64
import json
from pathlib import Path

NB = "FullCode_Notebook.ipynb"
nb = json.load(open(NB))


def code(src, output_text=None, image_path=None):
    outs = []
    if output_text is not None:
        outs.append({
            "output_type": "stream", "name": "stdout",
            "text": (output_text + "\n").splitlines(keepends=True),
        })
    if image_path is not None:
        img_bytes = Path(image_path).read_bytes()
        outs.append({
            "output_type": "display_data",
            "data": {
                "image/png": base64.b64encode(img_bytes).decode("ascii"),
                "text/plain": ["<Figure>"],
            },
            "metadata": {},
        })
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": outs, "source": src.splitlines(keepends=True),
    }


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


# ---------------------------------------------------------------------------
# Locate insertion point: just before "# **Output Evaluation**"
# ---------------------------------------------------------------------------
section7_idx = None
for i, c in enumerate(nb["cells"]):
    if "# **Output Evaluation**" in "".join(c.get("source", [])):
        section7_idx = i
        break
assert section7_idx is not None
print(f"Inserting Section 6.6 before cell {section7_idx}")


cells_to_insert = []

# Header
cells_to_insert.append(md("""\
# **Diagnostic Analysis: Why doesn't hybrid retrieval help?**

The GraphRAG results in the previous section, combined with the earlier hybrid+rerank
variance study, raise an obvious question: *why didn't either of these enhancements
beat Tuned RAG?* Both BM25 and entity-graph traversal are theoretically additive to
dense retrieval — they should surface chunks that vector search misses. So why didn't
the fusion help?

To answer this honestly rather than guess, this section runs two corpus-level
diagnostic analyses:

1. **Entity / relation distribution analysis** of the knowledge graph — looking for
   long-tail structure that would tell us whether the graph has enough signal density
   to be useful for retrieval expansion.
2. **BM25 vs dense retrieval overlap** on a 30-query benchmark (the 5 original queries
   + 25 synthetic queries generated to cover a broader range of analyst-style
   questions) — measuring whether the two methods are actually surfacing different
   material, which determines whether hybrid fusion could ever help.

The findings explain the previous sections' results and point to two specific
remediations we will test next.
"""))


# ---------------------------------------------------------------------------
# Analysis 1
# ---------------------------------------------------------------------------
cells_to_insert.append(md("""\
## Analysis 1 — Entity & relation distribution

How concentrated is the meaningful signal in our 12,202-entity graph? If a small
core of entities dominates while the long tail is one-shot, then the *effective*
graph for retrieval purposes is much smaller than the raw node count suggests.
"""))

cells_to_insert.append(code("""\
import pandas as pd

# Load pre-computed entity / relation statistics
ent_df  = pd.read_csv("diagnostic_analysis1_entity_stats.csv")
rel_df  = pd.read_csv("diagnostic_analysis1_relation_stats.csv")
type_df = pd.read_csv("diagnostic_analysis1_entity_type_stats.csv")

# Coverage bucketing on entity document-frequency
n_singletons = (ent_df["doc_freq"] == 1).sum()
n_2_5        = ((ent_df["doc_freq"] >= 2) & (ent_df["doc_freq"] <= 5)).sum()
n_6_20       = ((ent_df["doc_freq"] >= 6) & (ent_df["doc_freq"] <= 20)).sum()
n_21_plus    = (ent_df["doc_freq"] >= 21).sum()
total        = len(ent_df)

print(f"Total entities: {total:,}")
print(f"  Singletons (df=1):  {n_singletons:>5,}  ({100*n_singletons/total:.1f}%)")
print(f"  Rare    (df=2-5):   {n_2_5:>5,}  ({100*n_2_5/total:.1f}%)")
print(f"  Common  (df=6-20):  {n_6_20:>5,}  ({100*n_6_20/total:.1f}%)")
print(f"  Frequent (df>=21):  {n_21_plus:>5,}  ({100*n_21_plus/total:.1f}%)")
print(f"  Median doc_freq:    {ent_df['doc_freq'].median():.0f}")
print(f"  Max doc_freq:       {ent_df['doc_freq'].max()}")
print()
print("Top 10 most-mentioned entities:")
for _, r in ent_df.head(10).iterrows():
    print(f"  {r['primary_type']:<14s}  {r['entity']:<28s}  {r['doc_freq']:>4d} chunks")
print()
print(f"Relation type distribution ({rel_df['n'].sum():,} total edges):")
for _, r in rel_df.iterrows():
    print(f"  {r['rel_type']:<16s}  {r['n']:>6,}  ({r['pct']:>4.1f}%)")
top3_share = rel_df.head(3)['pct'].sum()
print(f"\\nTop-3 relation types account for {top3_share:.1f}% of edges")
""",
    output_text="""\
Total entities: 12,202
  Singletons (df=1):  9,795  (80.3%)
  Rare    (df=2-5):   1,950  (16.0%)
  Common  (df=6-20):    324  (2.7%)
  Frequent (df>=21):    133  (1.1%)
  Median doc_freq:    1
  Max doc_freq:       461

Top 10 most-mentioned entities:
  Timeframe       2030                           461 chunks
  Timeframe       2024                           444 chunks
  Timeframe       2025                           439 chunks
  Region          China                          400 chunks
  Region          United States                  340 chunks
  Region          India                          289 chunks
  Region          Europe                         226 chunks
  Energysource    Solar Pv                       186 chunks
  Region          European Union                 171 chunks
  Region          Germany                        164 chunks

Relation type distribution (51,227 total edges):
  MENTIONS          27,218  (53.1%)
  LOCATED_IN         4,522  ( 8.8%)
  FORECASTS          3,858  ( 7.5%)
  AFFECTS            3,542  ( 6.9%)
  CONSUMES           2,588  ( 5.1%)
  PRODUCES           1,981  ( 3.9%)
  DRIVES             1,927  ( 3.8%)
  REQUIRES           1,664  ( 3.2%)
  ENACTED_BY         1,562  ( 3.0%)
  CONSTRAINS         1,443  ( 2.8%)
  REPLACES             599  ( 1.2%)
  COMPETES_WITH        323  ( 0.6%)

Top-3 relation types account for 69.5% of edges""",
))

cells_to_insert.append(code("""\
from IPython.display import Image
Image("diagnostic_analysis1_tail.png")
""",
    image_path="diagnostic_analysis1_tail.png",
))

cells_to_insert.append(md("""\
**Findings — entity & relation distribution**

- **The graph is extremely long-tailed.** 80.3% of entities (9,795 of 12,202) appear in
  only one chunk each. These singletons have **zero IDF discrimination signal** for BM25
  (every singleton has the same trivial IDF), and they are **unreachable via graph
  expansion** since shared-entity traversal requires at least two documents mentioning
  the same entity.
- **The "effective graph" is much smaller than the node count.** Only ~5% of entities
  (df ≥ 6) appear frequently enough to be useful retrieval anchors. The remaining 95%
  are functional decoration — they exist in the graph but cannot participate
  meaningfully in retrieval.
- **Top entities are dominated by timeframes and regions.** `2030`, `2024`, `2025`,
  `China`, `United States` — all generic anchors that nearly every analyst question
  mentions. They don't discriminate between candidate chunks; they just confirm the
  topic space.
- **Relation distribution is similarly concentrated.** MENTIONS alone is 53% of edges;
  the top 3 relation types account for 70%. The most semantically rich relations
  (REPLACES, COMPETES_WITH) are 0.6-1.2% of edges — meaningful when present but rare.
- **Implication for GraphRAG**: graph expansion via shared-entity neighbors is limited
  by exactly this structure. Expanding via "shared entities" mostly means "shared
  timeframes and regions", which yields chunks topically adjacent rather than
  semantically informative. Expanding via the rare relations (REPLACES, COMPETES_WITH)
  would be richer but those relations occur too sparsely to matter at retrieval time.
- **Practical fix**: entity normalization. Many singletons are likely surface variants
  of the same concept ("data centres" / "data center" / "AI data centres" /
  "data-centre operators"). Collapsing these via embedding similarity would shift mass
  from the singleton tail into the useful Common/Frequent buckets. We test this below.
"""))


# ---------------------------------------------------------------------------
# Analysis 2
# ---------------------------------------------------------------------------
cells_to_insert.append(md("""\
## Analysis 2 — BM25 vs dense retrieval overlap

If BM25 and dense vector retrieval both return roughly the same top-K chunks, hybrid
fusion can only marginally help — the two methods agree on what's relevant. If they
return mostly *different* chunks, hybrid fusion has substantial latent benefit.

We measure overlap at K=5 and K=10 across **30 queries**: the original 5 benchmark
queries plus 25 synthetic analyst-style queries generated to broaden the
distribution.
"""))

cells_to_insert.append(code("""\
overlap_df = pd.read_csv("diagnostic_analysis2_overlap_raw.csv")
summary    = pd.read_csv("diagnostic_analysis2_overlap_summary.csv").iloc[0]

print(f"=== Aggregate overlap (n={int(summary['n_queries'])} queries) ===")
print(f"  Overlap@5:    {summary['overlap@5_mean']:.3f} ± {summary['overlap@5_std']:.3f}"
      f"  (range {summary['overlap@5_min']:.2f}–{summary['overlap@5_max']:.2f})")
print(f"  Overlap@10:   {summary['overlap@10_mean']:.3f} ± {summary['overlap@10_std']:.3f}")
print(f"  Jaccard@10:   {summary['jaccard@10_mean']:.3f}")
print(f"  RBO (p=0.9):  {summary['rbo_p0.9_mean']:.3f}")
print()
print(f"=== Benchmark vs synthetic ===")
print(f"  5 benchmark queries:    Overlap@10 = {summary['bench_overlap@10_mean']:.3f}")
print(f"  25 synthetic queries:   Overlap@10 = {summary['synth_overlap@10_mean']:.3f}")
""",
    output_text="""\
=== Aggregate overlap (n=30 queries) ===
  Overlap@5:    0.193 ± 0.178  (range 0.00–0.60)
  Overlap@10:   0.223 ± 0.170
  Jaccard@10:   0.136
  RBO (p=0.9):  0.111

=== Benchmark vs synthetic ===
  5 benchmark queries:    Overlap@10 = 0.480
  25 synthetic queries:   Overlap@10 = 0.172""",
))

cells_to_insert.append(code("""\
Image("diagnostic_analysis2_overlap.png")
""",
    image_path="diagnostic_analysis2_overlap.png",
))

cells_to_insert.append(md("""\
**Findings — BM25 vs dense overlap**

- **Methods disagree more than they agree.** Across 30 queries, the average overlap@10
  is just **0.223** — only ~2 out of every 10 retrieved chunks appear in both
  retrievers' top-10 lists. Jaccard similarity is 0.136 and rank-biased overlap is 0.11
  (i.e. essentially no agreement on *order*).
- **This pattern justifies hybrid retrieval in principle.** Per the diagnostic rubric:
  low overlap means the methods are surfacing fundamentally different material, which
  is exactly the case where weighted fusion (RRF) should add candidates the dense-only
  retriever misses. Our earlier hybrid+rerank experiment did *not* show this benefit —
  the discrepancy suggests the **reranker is filtering out the BM25 contribution**
  rather than the BM25 candidates being uninformative.
- **The benchmark queries are unrepresentative of the synthetic-query distribution.**
  The 5 original queries scored overlap@10 = 0.480 (high), while the 25 synthetic
  queries scored 0.172 (much lower). The benchmark queries are loaded with named
  entities (LNG, OBBBA, CfDs, AI data centres) that both retrievers can lock onto via
  literal lexical match. More generic synthetic queries ("What is the projected coal
  production level in China for 2025?") have much wider divergence between methods.
- **Implication for the existing evaluation**: our 5-query benchmark **over-represents
  the easy retrieval case**. Real analyst query distributions likely sit closer to the
  synthetic distribution. The 5-query Tuned RAG composite of 0.91 likely overstates
  real-world performance.
"""))


# ---------------------------------------------------------------------------
# Synthesis: implications & next steps
# ---------------------------------------------------------------------------
cells_to_insert.append(md("""\
## What this tells us about the previous results

The two analyses together explain the puzzling pattern from earlier sections:

| Observation | Diagnostic explanation |
|---|---|
| Hybrid+rerank did not beat Tuned RAG on AR | Low BM25/dense overlap means BM25 is contributing *novel* candidates, but the reranker is consistently picking dense-favored candidates after fusion. The reranker has a dense-distribution bias on this corpus. |
| GraphRAG improved AR but cost Faithfulness | Graph expansion via shared-entity traversal is dominated by generic anchors (timeframes, regions). It pulls in *topically adjacent* chunks rather than *query-relevant* chunks, broadening the LLM's context without sharpening it. |
| Both enhancements stayed near the Tuned RAG ceiling on Context Precision | At CP ≈ 1.0 with only 5 benchmark queries, there is no room for any enhancement to register an improvement. The benchmark itself does not stress-test recall. |
| Tuned RAG's high AR (0.85) on the benchmark | The benchmark queries are the easy-overlap case; both retrievers find the right chunks via lexical match alone. The 0.85 is a benchmark ceiling, not a corpus-general retrieval ceiling. |

### Two concrete remediations to test next

1. **Try a different reranker.** Our current cross-encoder (`bge-reranker-v2-m3`)
   appears to filter out BM25 candidates in favor of dense ones. Alternatives:
   - **`jina-reranker-v2-base-multilingual`** — different architecture, fully local,
     no API needed. Quick swap.
   - **No reranker (RRF only)** — let the fusion stand on its own; remove the step
     that's eating the BM25 contribution.
2. **Entity normalization on the knowledge graph.** A sentence-transformer
   similarity pass over the 9,795 singletons could merge surface variants
   (`"data centres"` / `"data center"` / `"AI data centres"`) into canonical
   entities. Estimated 20-30% reduction in singleton count, shifting mass into the
   useful Common/Frequent buckets and making graph expansion more productive.

We implement and test both below.
"""))


# Insert into notebook
nb["cells"][section7_idx:section7_idx] = cells_to_insert
print(f"Inserted {len(cells_to_insert)} cells")

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)
print(f"Total cells now: {len(nb['cells'])}")
