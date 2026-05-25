"""GraphRAG implementation for the Lumina / IEA project.

Architecture:
  PDF chunks (existing, token-380/60) ──► LLMGraphTransformer (GPT-4.1)
                                                 │
                                                 ▼
                                         Neo4j entity graph
                                         + Document nodes
                                                 │
                                                 ▼
                              ┌─────── Neo4j vector index ───────┐
                              ▼                                   ▼
                       Vector seed search                Graph 1-hop expansion
                              │                                   │
                              └──────────────┬────────────────────┘
                                             ▼
                                    Deduplicated context
                                             │
                                             ▼
                                       LLM synthesis
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_openai import ChatOpenAI

from rag_harness import (
    BENCHMARK_QUERIES,
    HarnessConfig,
    _format_context,
    build_embeddings,
    load_and_chunk,
)


# ---------------------------------------------------------------------------
# Domain schema — constrains entity extraction quality
# ---------------------------------------------------------------------------
ALLOWED_NODES = [
    "EnergySource",     # coal, oil, gas, solar, wind, nuclear, hydro
    "Region",           # US, EU, China, India, Asia, OECD
    "Policy",           # OBBBA, IRA, CfD, AI Growth Zones, BRICS, REPowerEU
    "Technology",       # LNG, SAF, EV, battery storage, hydrogen, CCS
    "Infrastructure",   # LNG terminal, refinery, data centre, grid, pipeline
    "Sector",           # transport, heat, power, industrial, residential
    "Metric",           # quantitative claims with units (e.g., "20 bcm/yr")
    "TimeFrame",        # 2025, 2030, "by 2030", "medium-term"
]

ALLOWED_RELATIONSHIPS = [
    "AFFECTS",          # generic causation
    "REQUIRES",
    "PRODUCES",
    "CONSUMES",
    "LOCATED_IN",
    "ENACTED_BY",
    "FORECASTS",
    "REPLACES",
    "COMPETES_WITH",
    "DRIVES",           # MarketDriver DRIVES growth/decline
    "CONSTRAINS",       # grid bottlenecks CONSTRAINS solar deployment
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
GRAPH_RAG_SYSTEM = (
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


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
def neo4j_graph(cfg: HarnessConfig) -> Neo4jGraph:
    g = Neo4jGraph(
        url=cfg.neo4j_uri,
        username=cfg.neo4j_user,
        password=cfg.neo4j_password,
        refresh_schema=False,   # APOC not installed
    )
    # add_graph_documents calls refresh_schema() which also needs APOC; stub it.
    g.refresh_schema = lambda: None
    return g


# ---------------------------------------------------------------------------
# Build the graph (one-time ingest)
# ---------------------------------------------------------------------------
async def _extract_concurrent(
    transformer: LLMGraphTransformer,
    docs: list[Document],
    concurrency: int,
    progress_every: int = 25,
) -> list:
    sem = asyncio.Semaphore(concurrency)
    results: list = [None] * len(docs)
    start = time.perf_counter()
    completed = {"n": 0}

    async def worker(i: int, d: Document):
        async with sem:
            try:
                out = await transformer.aconvert_to_graph_documents([d])
                results[i] = out[0] if out else None
            except Exception as e:
                print(f"  [chunk {i}] error: {e}")
                results[i] = None
            completed["n"] += 1
            if completed["n"] % progress_every == 0:
                elapsed = time.perf_counter() - start
                rate = completed["n"] / elapsed
                remaining = (len(docs) - completed["n"]) / rate
                print(f"  [{completed['n']}/{len(docs)}] "
                      f"{rate:.1f}/s  est. remaining {remaining/60:.1f} min")

    await asyncio.gather(*(worker(i, d) for i, d in enumerate(docs)))
    return [r for r in results if r is not None]


def build_graph(
    cfg: HarnessConfig,
    extraction_model: str = "gpt-4.1-2025-04-14",
    concurrency: int = 10,
    limit: int | None = None,
) -> Neo4jGraph:
    """Build the entity-relationship graph from the same chunks used for vector RAG.

    Runs in parallel across `concurrency` workers; ~15-25 min for 2,786 chunks.
    """
    chunks = load_and_chunk(cfg)
    if limit:
        chunks = chunks[:limit]
    print(f"[build_graph] extracting from {len(chunks)} chunks "
          f"using {extraction_model} with concurrency={concurrency}")

    extractor_llm = ChatOpenAI(
        model=extraction_model,
        api_key=cfg.openai_api_key,
        temperature=0,
        max_retries=10,        # tier-1 occasionally bursts; long backoff is fine
        timeout=120,
    )
    transformer = LLMGraphTransformer(
        llm=extractor_llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=False,
        relationship_properties=False,
    )

    t0 = time.perf_counter()
    graph_docs = asyncio.run(_extract_concurrent(transformer, chunks, concurrency))
    extract_seconds = time.perf_counter() - t0

    graph = neo4j_graph(cfg)
    # Wipe any previous run to guarantee a clean re-build
    graph.query("MATCH (n) DETACH DELETE n")
    print(f"[build_graph] writing {len(graph_docs)} graph docs to Neo4j…")
    _write_graph_documents_no_apoc(graph, graph_docs, ALLOWED_NODES)

    # Stats
    n_nodes = graph.query("MATCH (n) RETURN count(n) AS n")[0]["n"]
    n_rels = graph.query("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
    n_docs = graph.query("MATCH (d:Document) RETURN count(d) AS n")[0]["n"]
    by_label = graph.query(
        "MATCH (n) WHERE NOT n:Document AND NOT n:__Entity__ "
        "WITH labels(n) AS labs UNWIND labs AS l WITH l WHERE l <> '__Entity__' "
        "RETURN l AS label, count(*) AS n ORDER BY n DESC"
    )

    print(f"\n[build_graph] complete in {extract_seconds/60:.1f} min")
    print(f"  Documents: {n_docs}")
    print(f"  Entities:  {n_nodes - n_docs}")
    print(f"  Edges:     {n_rels}")
    print(f"  Top labels:")
    for r in by_label[:15]:
        print(f"    {r['label']:<22s} {r['n']}")
    return graph


# ---------------------------------------------------------------------------
# APOC-free graph ingest. langchain-neo4j's add_graph_documents() relies on
# apoc.create.addLabels and apoc.meta.data for dynamic typed labels and
# schema introspection. We replicate the essentials with parameterized Cypher.
# Safe because entity/edge type names come from our ALLOWED_NODES /
# ALLOWED_RELATIONSHIPS whitelists, not from user input.
# ---------------------------------------------------------------------------
def _safe_label(name: str, allowed: set[str]) -> str:
    """Sanitize a label/relationship name. Falls back to a generic label."""
    if name in allowed:
        return name
    cleaned = "".join(c if c.isalnum() else "_" for c in name)
    if cleaned and cleaned[0].isalpha():
        return cleaned
    return "Unknown"


def _write_graph_documents_no_apoc(graph, graph_docs, allowed_nodes):
    allowed = set(allowed_nodes)
    # Constraint on the entity id for fast MERGE — one-time, idempotent
    try:
        graph.query(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:`__Entity__`) REQUIRE e.id IS UNIQUE"
        )
        graph.query(
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS "
            "FOR (d:Document) REQUIRE d.id IS UNIQUE"
        )
    except Exception:
        pass

    n_nodes_written = 0
    n_rels_written = 0
    n_docs_written = 0

    for gd in graph_docs:
        if gd is None:
            continue

        # 1. Write the source Document node (carries the chunk text + metadata)
        src = gd.source
        src_meta = src.metadata or {}
        doc_id = f"{src_meta.get('source', 'unknown')}::{src_meta.get('page', 0)}::{hash(src.page_content) % 10**10}"
        graph.query(
            "MERGE (d:Document {id: $id}) "
            "SET d.text = $text, d.source = $source, d.page = $page",
            params={
                "id": doc_id,
                "text": src.page_content,
                "source": src_meta.get("source"),
                "page": src_meta.get("page"),
            },
        )
        n_docs_written += 1

        # 2. Write entity nodes, attaching the typed label
        for node in gd.nodes:
            typed_label = _safe_label(node.type, allowed)
            graph.query(
                f"MERGE (e:`__Entity__` {{id: $id}}) "
                f"SET e:`{typed_label}`",
                params={"id": node.id},
            )
            # MENTIONS edge from Document -> Entity
            graph.query(
                "MATCH (d:Document {id: $doc_id}), (e:`__Entity__` {id: $eid}) "
                "MERGE (d)-[:MENTIONS]->(e)",
                params={"doc_id": doc_id, "eid": node.id},
            )
            n_nodes_written += 1

        # 3. Write entity -> entity relationships
        for rel in gd.relationships:
            rel_type = _safe_label(rel.type, set(ALLOWED_RELATIONSHIPS) | {"RELATED_TO"})
            graph.query(
                f"MATCH (a:`__Entity__` {{id: $src_id}}), "
                f"(b:`__Entity__` {{id: $dst_id}}) "
                f"MERGE (a)-[r:`{rel_type}`]->(b)",
                params={"src_id": rel.source.id, "dst_id": rel.target.id},
            )
            n_rels_written += 1

    print(f"  Wrote {n_docs_written} Documents, {n_nodes_written} entity-mentions, "
          f"{n_rels_written} entity-edges")


# ---------------------------------------------------------------------------
# Vector index on Document nodes
# ---------------------------------------------------------------------------
def setup_vector_index(cfg: HarnessConfig) -> Neo4jVector:
    """Build (or attach to) the Neo4j vector index over Document nodes.

    Document nodes were created during graph build by include_source=True.
    """
    embeddings = build_embeddings("BAAI/bge-base-en-v1.5", cfg)
    return Neo4jVector.from_existing_graph(
        embedding=embeddings,
        url=cfg.neo4j_uri,
        username=cfg.neo4j_user,
        password=cfg.neo4j_password,
        index_name="iea_documents",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )


# ---------------------------------------------------------------------------
# Retrieval — vector seed + graph expansion
# ---------------------------------------------------------------------------
GRAPH_EXPANSION_CYPHER = """
// Identify seed documents by (source, page) pairs returned from vector search,
// then expand to other Documents that mention shared entities.
UNWIND $seed_keys AS sk
MATCH (seed:Document {source: sk.source, page: sk.page})
WITH collect(DISTINCT seed) AS seeds
UNWIND seeds AS seed
MATCH (seed)-[:MENTIONS]->(e)<-[:MENTIONS]-(neighbor:Document)
WHERE NOT neighbor IN seeds
WITH neighbor, count(DISTINCT e) AS shared_entities
ORDER BY shared_entities DESC
LIMIT $limit
RETURN neighbor.text AS text, neighbor.source AS source, neighbor.page AS page,
       shared_entities
"""


def graph_rag_retrieve(
    query: str,
    vector_index: Neo4jVector,
    graph: Neo4jGraph,
    k_seed: int = 4,
    k_expansion: int = 4,
) -> list[Document]:
    """Hybrid retrieval: vector seed chunks + 1-hop entity-shared neighbors.

    Returns up to k_seed + k_expansion chunks, deduped, with rich metadata.
    """
    # Stage 1: vector seed
    seeds = vector_index.similarity_search(query, k=k_seed)
    seed_keys = [
        {"source": s.metadata.get("source"), "page": s.metadata.get("page")}
        for s in seeds
        if s.metadata.get("source") is not None
    ]

    # Stage 2: graph expansion via shared-entity traversal
    if seed_keys:
        rows = graph.query(GRAPH_EXPANSION_CYPHER, params={
            "seed_keys": seed_keys,
            "limit": k_expansion,
        })
        expansion = [
            Document(
                page_content=r["text"],
                metadata={"source": r["source"], "page": r["page"],
                          "shared_entities": r["shared_entities"],
                          "via": "graph"},
            )
            for r in rows
        ]
    else:
        expansion = []

    # Tag seeds
    for s in seeds:
        s.metadata["via"] = "vector"

    # Dedup by (source, page, content prefix)
    seen = set()
    out = []
    for d in seeds + expansion:
        key = (d.metadata.get("source"), d.metadata.get("page"),
               d.page_content[:60])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def graph_rag_answer(
    query: str,
    vector_index: Neo4jVector,
    graph: Neo4jGraph,
    llm,
    k_seed: int = 4,
    k_expansion: int = 4,
) -> tuple[str, list[str]]:
    docs = graph_rag_retrieve(query, vector_index, graph,
                              k_seed=k_seed, k_expansion=k_expansion)
    context = _format_context(docs)
    user = f"Question:\n{query}\n\nIEA excerpts:\n{context}"
    resp = llm.invoke([
        SystemMessage(content=GRAPH_RAG_SYSTEM),
        HumanMessage(content=user),
    ])
    return resp.content, [d.page_content for d in docs]


# ---------------------------------------------------------------------------
# Patch HarnessConfig to carry Neo4j credentials
# ---------------------------------------------------------------------------
def attach_neo4j(cfg: HarnessConfig, keys: dict) -> HarnessConfig:
    cfg.neo4j_uri = keys["NEO4J_URI"]
    cfg.neo4j_user = keys["NEO4J_USER"]
    cfg.neo4j_password = keys["NEO4J_PASSWORD"]
    return cfg


# ---------------------------------------------------------------------------
# End-to-end run helpers
# ---------------------------------------------------------------------------
def generate_all_answers(cfg: HarnessConfig, generator) -> dict:
    """Run all 5 benchmark queries through GraphRAG, return answers + contexts."""
    graph = neo4j_graph(cfg)
    vector_index = setup_vector_index(cfg)
    results: dict = {"answers": [], "contexts": [], "latencies": []}
    for q in BENCHMARK_QUERIES:
        t0 = time.perf_counter()
        ans, ctxs = graph_rag_answer(q, vector_index, graph, generator)
        results["answers"].append(ans)
        results["contexts"].append(ctxs)
        results["latencies"].append(round(time.perf_counter() - t0, 2))
        print(f"  Q ({results['latencies'][-1]}s): {len(ans)} chars, "
              f"{len(ctxs)} contexts")
    return results


if __name__ == "__main__":
    # Quick smoke test: connectivity only
    keys = json.load(open("config.json"))
    cfg = HarnessConfig(groq_api_key=keys["GROQ_API_KEY"],
                        openai_api_key=keys["OPENAI_API_KEY"],
                        chunk_size=380, chunk_overlap=60, chunk_unit="token")
    attach_neo4j(cfg, keys)
    g = neo4j_graph(cfg)
    print("Connected. Existing node count:",
          g.query("MATCH (n) RETURN count(n) AS n")[0]["n"])
