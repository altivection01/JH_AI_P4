"""Four answer-generation strategies for the Lumina RAG project.

Each strategy is a function that takes a query and returns (answer, contexts).
Contexts are lists of strings (or [] for non-RAG strategies). The same generator
LLM (Groq llama-3.3-70b-versatile) is used across all four so differences are
attributable to retrieval + prompting, not the model itself.

Strategies:
    1) base_llm_answer        — LLM only, minimal system prompt
    2) prompt_engineered_answer — LLM only, analyst system prompt
    3) base_rag_answer        — naive RAG: MiniLM embedder, similarity k=4, simple prompt
    4) tuned_rag_answer       — winning config: bge-base, MMR k=6 λ=0.5, 900/150 chunks,
                                analyst+citation system prompt
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.messages import HumanMessage, SystemMessage

from rag_harness import (
    HarnessConfig,
    RetrievalConfig,
    _format_context,
    build_embeddings,
    get_or_build_store,
    retrieve,
)


# ---------------------------------------------------------------------------
# Prompt definitions — these are what makes each strategy distinct.
# ---------------------------------------------------------------------------

MINIMAL_SYSTEM = "You are a helpful assistant. Answer the user's question."

ANALYST_SYSTEM = (
    "You are a senior energy markets analyst at Lumina Energy Partners. "
    "When answering, structure your response for an investment-committee briefing: "
    "(1) Lead with the headline finding. "
    "(2) Support with specific quantitative data points wherever possible. "
    "(3) Identify cross-sector linkages — how developments in one energy market "
    "(coal, oil, gas, electricity, renewables) affect the others. "
    "(4) Flag countervailing forces and key uncertainties. "
    "(5) Be concise; aim for 200-350 words. "
    "Do not speculate beyond what's reasonable from publicly available data."
)

NAIVE_RAG_SYSTEM = (
    "Use the following context to answer the question. "
    "If the context doesn't contain the answer, say you don't know."
)

CITED_ANALYST_SYSTEM = (
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
# Strategy 1: Base LLM (no retrieval, minimal prompt)
# ---------------------------------------------------------------------------

def base_llm_answer(query: str, llm) -> tuple[str, list[str]]:
    resp = llm.invoke([
        SystemMessage(content=MINIMAL_SYSTEM),
        HumanMessage(content=query),
    ])
    return resp.content, []


# ---------------------------------------------------------------------------
# Strategy 2: Prompt-engineered LLM (no retrieval, analyst prompt)
# ---------------------------------------------------------------------------

def prompt_engineered_answer(query: str, llm) -> tuple[str, list[str]]:
    resp = llm.invoke([
        SystemMessage(content=ANALYST_SYSTEM),
        HumanMessage(content=query),
    ])
    return resp.content, []


# ---------------------------------------------------------------------------
# Strategy 3: Base RAG (default retrieval, simple prompt)
# Uses sentence-transformers' default model (MiniLM, 384-dim), default chunking,
# default similarity retrieval. Represents a "wired it up but didn't tune" baseline.
# ---------------------------------------------------------------------------

BASE_RAG_CFG = None   # set by build_strategies()
BASE_RAG_STORE = None
BASE_RAG_RETR = RetrievalConfig(k=4, strategy="similarity")


def base_rag_answer(query: str, llm) -> tuple[str, list[str]]:
    docs = retrieve(query, BASE_RAG_STORE, BASE_RAG_RETR)
    context = _format_context(docs)
    user = f"Context:\n{context}\n\nQuestion: {query}"
    resp = llm.invoke([
        SystemMessage(content=NAIVE_RAG_SYSTEM),
        HumanMessage(content=user),
    ])
    return resp.content, [d.page_content for d in docs]


# ---------------------------------------------------------------------------
# Strategy 4: Tuned RAG (winning config from sweeps)
# ---------------------------------------------------------------------------

TUNED_RAG_CFG = None
TUNED_RAG_STORE = None
TUNED_RAG_RETR = RetrievalConfig(k=6, strategy="mmr", fetch_k=20, lambda_mult=0.5)


def tuned_rag_answer(query: str, llm) -> tuple[str, list[str]]:
    docs = retrieve(query, TUNED_RAG_STORE, TUNED_RAG_RETR)
    context = _format_context(docs)
    user = f"Question:\n{query}\n\nIEA excerpts:\n{context}"
    resp = llm.invoke([
        SystemMessage(content=CITED_ANALYST_SYSTEM),
        HumanMessage(content=user),
    ])
    return resp.content, [d.page_content for d in docs]


# ---------------------------------------------------------------------------
# One-time builder — call after cfg is constructed, before invoking strategies.
# ---------------------------------------------------------------------------

def build_strategies(cfg: HarnessConfig) -> None:
    """Build both vector stores (base RAG: MiniLM, default chunks; tuned: bge-base)."""
    global BASE_RAG_STORE, TUNED_RAG_STORE
    from dataclasses import replace

    # Base RAG: out-of-the-box defaults
    base_cfg = replace(cfg, chunk_size=1000, chunk_overlap=100)
    BASE_RAG_STORE, _ = get_or_build_store(
        "sentence-transformers/all-MiniLM-L6-v2", base_cfg
    )

    # Tuned RAG: our winning config
    tuned_cfg = replace(cfg, chunk_size=900, chunk_overlap=150)
    TUNED_RAG_STORE, _ = get_or_build_store(
        "BAAI/bge-base-en-v1.5", tuned_cfg
    )

    print(f"[strategies] base RAG store:  {BASE_RAG_STORE._collection.count()} vectors (MiniLM)")
    print(f"[strategies] tuned RAG store: {TUNED_RAG_STORE._collection.count()} vectors (bge-base)")
