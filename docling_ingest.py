"""Stage 1: Re-ingest the IEA corpus with Docling, then chunk + embed.

Output structure:
  docling_md/<source>.md           — full markdown per PDF
  docling_chunks/<source>.json     — list of chunks with source/page metadata
  chroma_stores/<embedder_slug>_docling/  — separate vector store

The chunks use our existing token-380/60 splitter so we can A/B fairly against
the PyMuPDF baseline.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from docling.document_converter import DocumentConverter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer


PDF_DIR = Path("IEAReports")
MD_DIR = Path("docling_md")
CHUNKS_DIR = Path("docling_chunks")
MD_DIR.mkdir(exist_ok=True)
CHUNKS_DIR.mkdir(exist_ok=True)


def ingest_pdf(pdf_path: Path) -> str:
    """Convert one PDF to Docling markdown (cached on disk)."""
    md_path = MD_DIR / f"{pdf_path.stem}.md"
    if md_path.exists():
        print(f"  [{pdf_path.stem}] cached")
        return md_path.read_text()
    print(f"  [{pdf_path.stem}] converting…")
    t0 = time.perf_counter()
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    md = result.document.export_to_markdown()
    md_path.write_text(md)
    print(f"  [{pdf_path.stem}] {len(md):,} chars, {time.perf_counter()-t0:.0f}s")
    return md


def estimate_page_from_offset(md_text: str, offset: int) -> int:
    """Approximate page number for a character offset in the Docling markdown.

    Docling preserves a 'PAGE | N' header in the IEA reports — we use those
    landmarks to interpolate.
    """
    page_anchors = []
    for m in re.finditer(r"PAGE\s*\|\s*(\d+)", md_text):
        try:
            page_anchors.append((m.start(), int(m.group(1))))
        except ValueError:
            continue
    if not page_anchors:
        return 0
    last = page_anchors[0][1]
    for anchor_off, page_n in page_anchors:
        if anchor_off > offset:
            break
        last = page_n
    return last


def chunk_md(md_text: str, source: str, tok, chunk_size: int = 380,
             chunk_overlap: int = 60) -> list[Document]:
    """Chunk Docling markdown with the same token splitter as PyMuPDF baseline.

    Each chunk inherits a `source` and an *estimated* `page` from the nearest
    'PAGE | N' anchor that Docling preserved from the original layout.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda t: len(tok.encode(t, add_special_tokens=False)),
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(md_text)
    docs = []
    cursor = 0
    for txt in raw_chunks:
        idx = md_text.find(txt[:100], cursor)
        if idx == -1:
            idx = cursor
        page = estimate_page_from_offset(md_text, idx)
        cursor = idx + len(txt)
        docs.append(Document(
            page_content=txt,
            metadata={
                "source": source,
                "page": page,
                "extractor": "docling",
            },
        ))
    return docs


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Re-ingesting {len(pdfs)} PDFs with Docling…\n")

    tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", use_fast=True)
    all_chunks: list[Document] = []
    stats = []

    for pdf in pdfs:
        md = ingest_pdf(pdf)
        chunks = chunk_md(md, source=pdf.stem, tok=tok)
        all_chunks.extend(chunks)
        stats.append({"source": pdf.stem,
                      "md_chars": len(md),
                      "n_chunks": len(chunks)})
        # Save chunks for inspection
        with open(CHUNKS_DIR / f"{pdf.stem}.json", "w") as f:
            json.dump(
                [{"text": c.page_content, **c.metadata} for c in chunks],
                f, indent=2,
            )

    print(f"\nTotal: {sum(s['n_chunks'] for s in stats)} chunks across "
          f"{len(pdfs)} reports")
    for s in stats:
        print(f"  {s['source']:<18s} {s['md_chars']:>8,} chars  "
              f"{s['n_chunks']:>4d} chunks")

    # Embed + persist
    device = pick_device()
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": device,
                      "model_kwargs": {"torch_dtype": torch.float16}
                      if device in ("mps", "cuda") else {}},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )
    persist_dir = "chroma_stores/baai_bge_base_en_v1_5_c380_o60_token_docling"
    collection = "iea_docling"
    print(f"\nEmbedding {len(all_chunks)} chunks via {device}…")
    t0 = time.perf_counter()
    store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=collection,
        persist_directory=persist_dir,
    )
    print(f"  done in {time.perf_counter()-t0:.0f}s; persisted at {persist_dir}")
    print(f"  store has {store._collection.count()} embeddings")


if __name__ == "__main__":
    main()
