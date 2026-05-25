"""Build a vector store combining Docling text chunks + VLM-extracted chart data.

Reads:
  - chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling   (existing Docling text)
  - chart_extracts/<source>.json                                   (VLM chart tables)

Writes:
  - chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling_charts
    (Docling text + synthetic chart-data chunks)
"""
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


SOURCE_STORE = "chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling"
TARGET_STORE = "chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling_charts"
COLLECTION_SRC = "iea_docling"
COLLECTION_TGT = "iea_docling_charts"
CHART_DIR = Path("chart_extracts")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_chart_chunks() -> list[Document]:
    """Create one Document per non-empty chart extract."""
    docs: list[Document] = []
    for f in sorted(CHART_DIR.glob("*.json")):
        source = f.stem
        for entry in json.load(open(f)):
            if "extracted" not in entry:
                continue
            extracted = entry["extracted"].strip()
            if extracted.upper() == "NONE" or len(extracted) < 30:
                continue
            page = entry["page"]
            # PyMuPDF rendered page N (1-indexed) — convert to chunk-meta 0-indexed
            # to match how text chunks store their page metadata
            metadata_page = page - 1
            docs.append(Document(
                page_content=(f"[CHART DATA from {source} p.{metadata_page}]\n\n"
                              + extracted),
                metadata={
                    "source": source,
                    "page": metadata_page,
                    "extractor": "vlm_chart",
                    "is_chart_extract": True,
                },
            ))
    return docs


def main():
    device = pick_device()
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device,
                      "model_kwargs": {"torch_dtype": torch.float16}
                      if device in ("mps", "cuda") else {}},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )

    # Load existing Docling-text store
    print(f"Loading existing Docling text store…")
    src_store = Chroma(
        collection_name=COLLECTION_SRC,
        embedding_function=embeddings,
        persist_directory=SOURCE_STORE,
    )
    n_text = src_store._collection.count()
    print(f"  Docling text chunks: {n_text}")

    # Pull all existing chunks (without embeddings) and re-add to target
    print("Fetching existing chunks for re-ingest…")
    all_data = src_store.get(include=["documents", "metadatas"])
    text_docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(all_data["documents"], all_data["metadatas"])
    ]
    print(f"  {len(text_docs)} text docs loaded")

    # Load VLM chart chunks
    chart_docs = load_chart_chunks()
    print(f"  {len(chart_docs)} chart-data chunks loaded")

    # Build new store with both
    all_docs = text_docs + chart_docs
    print(f"\nEmbedding {len(all_docs)} total chunks via {device}…")
    t0 = time.perf_counter()
    target = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=COLLECTION_TGT,
        persist_directory=TARGET_STORE,
    )
    print(f"  done in {time.perf_counter()-t0:.0f}s")
    print(f"  enriched store has {target._collection.count()} embeddings")

    # Summary by source/extractor
    print("\nSummary:")
    from collections import Counter
    by_extractor = Counter(
        d.metadata.get("extractor", "unknown") for d in all_docs
    )
    for ext, n in by_extractor.most_common():
        print(f"  {ext:<20s} {n} chunks")


if __name__ == "__main__":
    main()
