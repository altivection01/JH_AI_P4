"""Full entity-relationship extraction over all 1,643 chunks using GPT-4.1.

Runs concurrently with rate-limit safe parallelism. Checkpoint-friendly: clears
the Neo4j DB at start, writes all extractions in batch at the end.
"""
import json
import time
import warnings

warnings.filterwarnings("ignore")

from rag_harness import HarnessConfig
from graph_rag import attach_neo4j, build_graph

keys = json.load(open("config.json"))
cfg = HarnessConfig(
    groq_api_key=keys["GROQ_API_KEY"],
    openai_api_key=keys["OPENAI_API_KEY"],
    chunk_size=380, chunk_overlap=60, chunk_unit="token",
)
attach_neo4j(cfg, keys)

t0 = time.perf_counter()
build_graph(cfg, extraction_model="gpt-4.1-mini-2025-04-14", concurrency=15)
wall = time.perf_counter() - t0

print(f"\n=== TOTAL WALL TIME: {wall/60:.1f} min ===")
