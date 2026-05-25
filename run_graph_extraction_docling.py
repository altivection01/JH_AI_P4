"""Re-extract the knowledge graph from Docling chunks (not PyMuPDF).

Same gpt-4.1-mini extractor + same allowed-nodes / allowed-relationships
schema as the original run. Only the source chunks change.
"""
import asyncio
import json
import time
import warnings

warnings.filterwarnings("ignore")

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from graph_rag import (
    ALLOWED_NODES, ALLOWED_RELATIONSHIPS,
    _extract_concurrent, _write_graph_documents_no_apoc,
    attach_neo4j, neo4j_graph,
)
from rag_harness import HarnessConfig


SRC_STORE = "chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling"
SRC_COLLECTION = "iea_docling"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    keys = json.load(open("config.json"))
    cfg = HarnessConfig(
        groq_api_key=keys["GROQ_API_KEY"],
        openai_api_key=keys["OPENAI_API_KEY"],
        chunk_size=380, chunk_overlap=60, chunk_unit="token",
    )
    attach_neo4j(cfg, keys)

    # Load Docling chunks from the existing Chroma collection
    device = pick_device()
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device,
                      "model_kwargs": {"torch_dtype": torch.float16}
                      if device in ("mps", "cuda") else {}},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    store = Chroma(
        collection_name=SRC_COLLECTION,
        embedding_function=emb,
        persist_directory=SRC_STORE,
    )
    data = store.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(data["documents"], data["metadatas"])
    ]
    print(f"Loaded {len(docs)} Docling chunks")

    # Wipe Neo4j (we're replacing the entire graph)
    graph = neo4j_graph(cfg)
    print("Wiping existing Neo4j graph…")
    graph.query("MATCH (n) DETACH DELETE n")

    # Re-extract
    extractor_llm = ChatOpenAI(
        model="gpt-4.1-mini-2025-04-14",
        api_key=cfg.openai_api_key,
        temperature=0,
        max_retries=10,
        timeout=120,
    )
    transformer = LLMGraphTransformer(
        llm=extractor_llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=False,
        relationship_properties=False,
    )

    print(f"Extracting entities from {len(docs)} chunks at concurrency=15…")
    t0 = time.perf_counter()
    graph_docs = asyncio.run(_extract_concurrent(transformer, docs, concurrency=15))
    extract_sec = time.perf_counter() - t0

    print(f"Writing {len(graph_docs)} graph docs to Neo4j (APOC-free)…")
    _write_graph_documents_no_apoc(graph, graph_docs, ALLOWED_NODES)

    # Stats
    n_docs = graph.query("MATCH (d:Document) RETURN count(d) AS n")[0]["n"]
    n_ent = graph.query("MATCH (e:`__Entity__`) RETURN count(e) AS n")[0]["n"]
    n_rel = graph.query("MATCH ()-[r]->() RETURN count(r) AS n")[0]["n"]
    print(f"\n=== Extraction complete in {extract_sec/60:.1f} min ===")
    print(f"  Documents: {n_docs}")
    print(f"  Entities:  {n_ent}")
    print(f"  Edges:     {n_rel}")


if __name__ == "__main__":
    main()
