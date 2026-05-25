"""
Embedding-model comparison harness for the Lumina / IEA RAG project.

Holds chunking, retriever, and generator FIXED across runs; varies only the
embedding model. One Chroma collection per model so runs are cached on disk.

Usage:
    from rag_harness import HarnessConfig, run_model, BENCHMARK_QUERIES

    cfg = HarnessConfig(groq_api_key=GROQ_API_KEY)
    row = run_model("BAAI/bge-base-en-v1.5", cfg)
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from datasets import Dataset
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
)


BENCHMARK_QUERIES: list[str] = [
    "How is the rapid global expansion of artificial intelligence data centres "
    "impacting overall electricity demand and straining existing power grid infrastructure?",
    "How is the unprecedented wave of new US liquefied natural gas (LNG) export "
    "capacity expected to impact natural gas affordability and spur additional "
    "demand in price-sensitive Asian markets by 2030?",
    "How are the surge in US electricity demand and the 2025 federal emergency "
    "policy interventions collectively affecting the retirement schedules, "
    "capacity planning, and generation output of domestic coal-fired power plants?",
    "How are the increasing frequency of negative wholesale electricity prices "
    "and the regulatory shift towards two-sided Contracts for Difference (CfDs) "
    "in Europe altering the revenue expectations and financial agility of "
    "developers investing in utility-scale solar PV?",
    'How do the tax credit modifications under the US "One Big Beautiful Bill Act" '
    "(OBBBA) affect the investment economics of using domestic versus imported "
    "feedstocks for Sustainable Aviation Fuel (SAF), and what cascading impact "
    "will this biofuel transition have on the capacity rationalisation of "
    "traditional US West Coast refineries?",
]


# Per-model prompt formatting. Some models REQUIRE these prefixes.
# (query_prefix, passage_prefix)
MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/e5-large-v2": ("query: ", "passage: "),
    "intfloat/e5-base-v2": ("query: ", "passage: "),
    # Exact instruction template from the HF model card / MTEB benchmark setup.
    "intfloat/e5-mistral-7b-instruct": (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "",
    ),
    "Salesforce/SFR-Embedding-Mistral": (
        "Instruct: Given a question, retrieve passages that answer it\nQuery: ",
        "",
    ),
    "BAAI/bge-large-en-v1.5": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "BAAI/bge-base-en-v1.5": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "mixedbread-ai/mxbai-embed-large-v1": (
        "Represent this sentence for searching relevant passages: ",
        "",
    ),
    "nomic-ai/nomic-embed-text-v1.5": (
        "search_query: ",
        "search_document: ",
    ),
}


def _slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def pick_device() -> str:
    # CUDA wins when present (H100/A100 etc. — we're explicitly renting one).
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    """bf16 on modern CUDA (Ampere+), fp16 on MPS, fp32 on CPU."""
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def pick_batch_size(device: str) -> int:
    return {"cuda": 128, "mps": 32, "cpu": 8}.get(device, 16)


def describe_environment(cfg: "HarnessConfig | None" = None) -> dict:
    """Print and return a snapshot of the runtime — handy as a notebook cell."""
    info = {
        "torch": torch.__version__,
        "device": cfg.device if cfg else pick_device(),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
        info["gpu_mem_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
    if torch.backends.mps.is_available():
        info["mps_available"] = True
    info["dtype"] = str(pick_dtype(info["device"]))
    info["batch_size"] = pick_batch_size(info["device"])
    for k, v in info.items():
        print(f"  {k:<18} {v}")
    return info


@dataclass
class HarnessConfig:
    groq_api_key: str
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    doc_folder: str = "IEAReports/"
    persist_root: str = "chroma_stores"
    chunk_size: int = 900
    chunk_overlap: int = 150
    chunk_unit: str = "char"   # "char" or "token"
    # When chunk_unit == "token", chunks are sized by the embedder's tokenizer.
    chunk_tokenizer: str = "BAAI/bge-base-en-v1.5"
    k: int = 6
    # Provider can be "groq" or "openai" for each role.
    generator_provider: str = "groq"
    generator_model: str = "llama-3.3-70b-versatile"
    judge_provider: str = "groq"
    judge_model: str = "llama-3.1-8b-instant"
    device: str = field(default_factory=pick_device)
    dtype: object = None  # resolved from device if left None
    batch_size: int = 0   # resolved from device if left 0
    low_precision: bool = True  # use fp16/bf16 on GPU; ignored on CPU
    trust_remote_code: bool = False  # set True for nomic-v1.5, jina-v3, gte-Qwen2

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = pick_dtype(self.device)
        if not self.batch_size:
            self.batch_size = pick_batch_size(self.device)

    def _chunk_key(self) -> str:
        # Distinguish char- and token-based chunking on disk so they don't collide.
        unit = "" if self.chunk_unit == "char" else f"_{self.chunk_unit}"
        return f"c{self.chunk_size}_o{self.chunk_overlap}{unit}"

    def collection_for(self, model_name: str) -> str:
        return f"iea_{_slug(model_name)}_{self._chunk_key()}"

    def persist_dir_for(self, model_name: str) -> str:
        return os.path.join(
            self.persist_root,
            f"{_slug(model_name)}_{self._chunk_key()}",
        )


@dataclass
class RetrievalConfig:
    """Retrieval-only knobs — varies without re-embedding."""
    k: int = 6
    strategy: str = "similarity"   # "similarity" | "mmr"
    fetch_k: int = 20              # MMR only: pool size before re-ranking
    lambda_mult: float = 0.5       # MMR only: 1.0=relevance, 0.0=diversity
    # Hybrid retrieval (BM25 + vector via Reciprocal Rank Fusion)
    hybrid: bool = False
    hybrid_pool: int = 30          # candidates from each retriever before fusion
    rrf_k: int = 60                # RRF damping constant (60 is the published default)
    # Cross-encoder reranking, applied LAST after vector/hybrid retrieval
    rerank: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_n: int = 6          # final docs after rerank (replaces .k for the LLM)
    rerank_pool: int = 20          # candidates before rerank (overrides .k upstream)

    def label(self) -> str:
        parts = []
        if self.hybrid:
            parts.append(f"hybrid(pool={self.hybrid_pool})")
        else:
            parts.append(f"mmr(k={self.k},fetch={self.fetch_k},λ={self.lambda_mult})"
                         if self.strategy == "mmr" else f"sim(k={self.k})")
        if self.rerank:
            parts.append(f"rerank({self.rerank_top_n}/{self.rerank_pool})")
        return " + ".join(parts)


# ---------------------------------------------------------------------------
# Ingest (cached across runs by chunk_size/overlap — chunks don't depend on
# the embedding model, so we compute them once per Python session).
# ---------------------------------------------------------------------------

_CHUNK_CACHE: dict[tuple, list] = {}
_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(name: str):
    if name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer
        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name, use_fast=True)
    return _TOKENIZER_CACHE[name]


def load_and_chunk(cfg: HarnessConfig):
    key = (cfg.doc_folder, cfg.chunk_size, cfg.chunk_overlap, cfg.chunk_unit, cfg.chunk_tokenizer)
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]

    pdf_files = sorted(glob(os.path.join(cfg.doc_folder, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {cfg.doc_folder}")

    docs = []
    for path in pdf_files:
        pages = PyMuPDFLoader(path).load()
        source = Path(path).stem
        for p in pages:
            p.metadata["source"] = source
            p.metadata["sector"] = source.replace("2025", "").replace("2026", "")
        docs.extend(pages)

    splitter_kwargs = dict(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    if cfg.chunk_unit == "token":
        tok = _get_tokenizer(cfg.chunk_tokenizer)
        splitter_kwargs["length_function"] = (
            lambda t: len(tok.encode(t, add_special_tokens=False))
        )

    splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)
    chunks = splitter.split_documents(docs)
    _CHUNK_CACHE[key] = chunks
    unit_label = "tokens" if cfg.chunk_unit == "token" else "chars"
    print(f"[ingest] {len(pdf_files)} PDFs -> {len(docs)} pages -> {len(chunks)} chunks "
          f"({cfg.chunk_size} {unit_label}, {cfg.chunk_overlap} overlap)")
    return chunks


# ---------------------------------------------------------------------------
# Embeddings + vector store (one collection per model, persisted on disk).
# ---------------------------------------------------------------------------

class PrefixedEmbeddings(Embeddings):
    """Wraps any Embeddings with per-call query/passage prefixes.

    E5/BGE/nomic require asymmetric prefixes (e.g. 'query: ' vs 'passage: ').
    The community HuggingFaceEmbeddings dropped its built-in prefix kwargs, so
    we apply them here instead.
    """

    def __init__(self, inner: Embeddings, query_prefix: str, passage_prefix: str):
        self.inner = inner
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts):
        return self.inner.embed_documents([self.passage_prefix + t for t in texts])

    def embed_query(self, text):
        return self.inner.embed_query(self.query_prefix + text)


def build_embeddings(model_name: str, cfg: HarnessConfig) -> Embeddings:
    query_prefix, passage_prefix = MODEL_PREFIXES.get(model_name, ("", ""))

    model_kwargs: dict = {
        "device": cfg.device,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.low_precision and cfg.device in ("mps", "cuda"):
        model_kwargs["model_kwargs"] = {"torch_dtype": cfg.dtype}

    inner = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": cfg.batch_size},
    )
    if query_prefix or passage_prefix:
        return PrefixedEmbeddings(inner, query_prefix, passage_prefix)
    return inner


def get_or_build_store(model_name: str, cfg: HarnessConfig) -> tuple[Chroma, float]:
    persist_dir = cfg.persist_dir_for(model_name)
    collection = cfg.collection_for(model_name)
    embeddings = build_embeddings(model_name, cfg)

    already_built = os.path.isdir(persist_dir) and os.listdir(persist_dir)

    t0 = time.perf_counter()
    if already_built:
        store = Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        n = store._collection.count()
        if n == 0:
            print(f"[store] cached dir at {persist_dir} is EMPTY — rebuilding")
            import shutil
            shutil.rmtree(persist_dir)
            already_built = False
        else:
            print(f"[store] reusing cached collection at {persist_dir} ({n} vectors)")

    if not already_built:
        chunks = load_and_chunk(cfg)
        os.makedirs(persist_dir, exist_ok=True)
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection,
            persist_directory=persist_dir,
        )
        store.persist()
    ingest_seconds = time.perf_counter() - t0
    return store, ingest_seconds


# ---------------------------------------------------------------------------
# RAG generation.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an energy markets analyst assistant for Lumina Energy Partners. "
    "Answer ONLY from the provided IEA report excerpts. For every key claim, "
    "cite the source report and page in square brackets, e.g. [Gas2025 p.42]. "
    "If the excerpts do not contain the answer, say so explicitly. "
    "Be concise, quantitative where the source is, and avoid speculation."
)


def _format_context(docs) -> str:
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[{src} p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def _make_llm(provider: str, model: str, cfg: HarnessConfig):
    if provider == "groq":
        return ChatGroq(model=model, groq_api_key=cfg.groq_api_key, temperature=0)
    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain_openai not installed")
        if not cfg.openai_api_key:
            raise RuntimeError("cfg.openai_api_key not set")
        return ChatOpenAI(model=model, api_key=cfg.openai_api_key, temperature=0)
    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain_anthropic not installed")
        if not cfg.anthropic_api_key:
            raise RuntimeError("cfg.anthropic_api_key not set")
        return ChatAnthropic(
            model=model,
            anthropic_api_key=cfg.anthropic_api_key,
            temperature=0,
            max_tokens=2048,
        )
    raise ValueError(f"unknown provider: {provider}")


# BM25 and reranker caches — keyed on (corpus, chunk params) and model name.
_BM25_CACHE: dict = {}
_RERANKER_CACHE: dict = {}


def _tokenize_bm25(text: str) -> list[str]:
    """Lowercase whitespace tokenizer for BM25. Keeps acronyms (LNG, OBBBA)
    intact after lowercasing and strips simple punctuation."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_bm25(cfg: HarnessConfig):
    """Build (or fetch cached) BM25 index from the same chunks the vector store uses."""
    key = (cfg.doc_folder, cfg.chunk_size, cfg.chunk_overlap, cfg.chunk_unit)
    if key in _BM25_CACHE:
        return _BM25_CACHE[key]
    from rank_bm25 import BM25Okapi
    chunks = load_and_chunk(cfg)
    tokenised = [_tokenize_bm25(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenised)
    _BM25_CACHE[key] = (bm25, chunks)
    return bm25, chunks


def _get_reranker(model_name: str, cfg: HarnessConfig):
    if model_name in _RERANKER_CACHE:
        return _RERANKER_CACHE[model_name]
    from sentence_transformers import CrossEncoder
    kwargs = {"device": cfg.device, "trust_remote_code": True}
    ce = CrossEncoder(model_name, **kwargs)
    _RERANKER_CACHE[model_name] = ce
    return ce


def _rrf_fuse(ranked_lists: list[list], k: int = 60):
    """Reciprocal Rank Fusion. Each list is documents in descending relevance."""
    scores: dict = {}
    keyed: dict = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            # Identify docs by (source, page, content-hash) — Chroma doesn't expose ids
            key = (doc.metadata.get("source"), doc.metadata.get("page"), hash(doc.page_content))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            keyed[key] = doc
    return [keyed[k] for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def retrieve(query: str, store: Chroma, rcfg: RetrievalConfig, cfg: HarnessConfig | None = None):
    # Stage 1: how big a candidate pool do we need?
    pool = rcfg.rerank_pool if rcfg.rerank else (rcfg.hybrid_pool if rcfg.hybrid else rcfg.k)

    # Stage 2: dense retrieval
    if rcfg.strategy == "mmr" and not rcfg.hybrid:
        dense = store.max_marginal_relevance_search(
            query, k=pool, fetch_k=max(rcfg.fetch_k, pool * 3), lambda_mult=rcfg.lambda_mult,
        )
    else:
        dense = store.similarity_search(query, k=pool)

    # Stage 3: optional hybrid fusion with BM25
    if rcfg.hybrid:
        if cfg is None:
            raise ValueError("hybrid retrieval requires cfg (for chunk key)")
        bm25, chunks = _get_bm25(cfg)
        scores = bm25.get_scores(_tokenize_bm25(query))
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
        sparse = [chunks[i] for i in top_idx]
        candidates = _rrf_fuse([dense, sparse], k=rcfg.rrf_k)
    else:
        candidates = dense

    # Stage 4: optional cross-encoder rerank
    if rcfg.rerank:
        if cfg is None:
            raise ValueError("rerank requires cfg (for device)")
        ce = _get_reranker(rcfg.rerank_model, cfg)
        pairs = [(query, d.page_content) for d in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
        ranked = [d for _, d in sorted(zip(scores, candidates), key=lambda kv: -kv[0])]
        return ranked[:rcfg.rerank_top_n]

    # No rerank — trim hybrid pool to k for final return
    return candidates[:rcfg.k]


def answer_query(query: str, store: Chroma, llm: ChatGroq, rcfg: RetrievalConfig,
                 cfg: HarnessConfig | None = None):
    docs = retrieve(query, store, rcfg, cfg=cfg)
    context = _format_context(docs)
    user = f"Question:\n{query}\n\nIEA excerpts:\n{context}"
    resp = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)])
    return resp.content, docs


# ---------------------------------------------------------------------------
# Evaluation (RAGAS) — one row per query, aggregated to one row per model.
# ---------------------------------------------------------------------------

def run_model(
    model_name: str,
    cfg: HarnessConfig,
    queries: list[str] | None = None,
    retrieval: RetrievalConfig | None = None,
) -> dict:
    """Build (or reuse) the index for `model_name`, answer the benchmark
    queries, score with RAGAS, return a summary row."""
    queries = queries or BENCHMARK_QUERIES
    rcfg = retrieval or RetrievalConfig(k=cfg.k)

    store, ingest_seconds = get_or_build_store(model_name, cfg)
    llm = _make_llm(cfg.generator_provider, cfg.generator_model, cfg)

    answers, contexts, latencies = [], [], []
    for q in queries:
        t0 = time.perf_counter()
        ans, docs = answer_query(q, store, llm, rcfg, cfg=cfg)
        latencies.append(time.perf_counter() - t0)
        answers.append(ans)
        contexts.append([d.page_content for d in docs])

    ds = Dataset.from_dict({
        "question": queries,
        "answer": answers,
        "contexts": contexts,
    })

    judge = _make_llm(cfg.judge_provider, cfg.judge_model, cfg)
    scores = evaluate(
        ds,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            LLMContextPrecisionWithoutReference(),
        ],
        llm=judge,
        embeddings=build_embeddings(model_name, cfg),
    )
    scores_df = scores.to_pandas()

    return {
        "model": model_name,
        "retrieval": rcfg.label(),
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "ingest_seconds": round(ingest_seconds, 2),
        "avg_query_seconds": round(sum(latencies) / len(latencies), 2),
        "faithfulness": float(scores_df["faithfulness"].mean()),
        "answer_relevancy": float(scores_df["answer_relevancy"].mean()),
        "context_precision": float(
            scores_df["llm_context_precision_without_reference"].mean()
        ),
        "per_query": scores_df.to_dict(orient="records"),
        "answers": answers,
    }


def sweep_retrieval(
    model_name: str,
    cfg: HarnessConfig,
    retrieval_grid: list[RetrievalConfig],
    queries: list[str] | None = None,
) -> pd.DataFrame:
    """Hold model + chunking fixed, vary retrieval. Cheap because the index
    is built once and reused across all retrieval configs."""
    rows = []
    for rcfg in retrieval_grid:
        print(f"\n--- {rcfg.label()} ---")
        try:
            r = run_model(model_name, cfg, queries=queries, retrieval=rcfg)
        except Exception as e:
            print(f"[error] {rcfg.label()}: {e}")
            rows.append({"retrieval": rcfg.label(), "error": str(e)})
            continue
        rows.append({
            "retrieval": r["retrieval"],
            "faithfulness": round(r["faithfulness"], 3),
            "answer_relevancy": round(r["answer_relevancy"], 3),
            "context_precision": round(r["context_precision"], 3),
            "avg_query_seconds": r["avg_query_seconds"],
        })
    return pd.DataFrame(rows)


def sweep_chunking(
    model_name: str,
    base_cfg: HarnessConfig,
    chunk_grid: list[tuple],   # tuples of (size, overlap) or (size, overlap, unit)
    rcfg: RetrievalConfig,
    queries: list[str] | None = None,
) -> pd.DataFrame:
    """For each (chunk_size, overlap[, unit]), build a fresh index and evaluate.
    Each row is one full ingest + eval — slower but necessary."""
    from dataclasses import replace
    rows = []
    for entry in chunk_grid:
        if len(entry) == 2:
            size, overlap = entry
            unit = base_cfg.chunk_unit
        else:
            size, overlap, unit = entry
        cfg = replace(base_cfg, chunk_size=size, chunk_overlap=overlap, chunk_unit=unit)
        print(f"\n--- chunk_size={size} overlap={overlap} unit={unit} ---")
        try:
            r = run_model(model_name, cfg, queries=queries, retrieval=rcfg)
        except Exception as e:
            print(f"[error] c={size}/o={overlap}: {e}")
            rows.append({"chunk_size": size, "chunk_overlap": overlap, "error": str(e)})
            continue
        rows.append({
            "chunk_size": size,
            "chunk_overlap": overlap,
            "unit": unit,
            "ingest_seconds": r["ingest_seconds"],
            "faithfulness": round(r["faithfulness"], 3),
            "answer_relevancy": round(r["answer_relevancy"], 3),
            "context_precision": round(r["context_precision"], 3),
        })
    return pd.DataFrame(rows)


def run_comparison(models: list[str], cfg: HarnessConfig) -> pd.DataFrame:
    rows = []
    for m in models:
        print(f"\n=== {m} ===")
        try:
            rows.append(run_model(m, cfg))
        except Exception as e:
            print(f"[error] {m}: {e}")
            rows.append({"model": m, "error": str(e)})
    df = pd.DataFrame(rows)
    cols = [
        "model", "faithfulness", "answer_relevancy", "context_precision",
        "ingest_seconds", "avg_query_seconds",
    ]
    return df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]


# ---------------------------------------------------------------------------
# Portable index export / import — build on H100, evaluate on Mac (or vice versa).
# ---------------------------------------------------------------------------

def export_index(model_name: str, cfg: HarnessConfig, out_dir: str = "exports") -> str:
    """Tar a persisted Chroma collection so it can be scp'd between machines."""
    import tarfile
    src = cfg.persist_dir_for(model_name)
    if not os.path.isdir(src):
        raise FileNotFoundError(f"No persisted store at {src} — run ingest first")
    os.makedirs(out_dir, exist_ok=True)
    archive = os.path.join(out_dir, f"{_slug(model_name)}.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(src, arcname=_slug(model_name))
    size_mb = os.path.getsize(archive) / 1e6
    print(f"[export] {archive}  ({size_mb:.1f} MB)")
    return archive


def import_index(archive_path: str, cfg: HarnessConfig) -> str:
    """Unpack a tarred Chroma collection into cfg.persist_root."""
    import tarfile
    os.makedirs(cfg.persist_root, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(cfg.persist_root)
    print(f"[import] unpacked into {cfg.persist_root}")
    return cfg.persist_root


DEFAULT_MODEL_SET = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "intfloat/e5-mistral-7b-instruct",
    "nomic-ai/nomic-embed-text-v1.5",   # requires trust_remote_code=True
]
