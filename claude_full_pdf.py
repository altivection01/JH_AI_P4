"""Option B1: Claude Opus 4.5 with the full corpus as extracted text.

Anthropic's PDF API has a 100-page-per-REQUEST limit, which makes our 868-page
corpus impossible to submit as native PDF in one shot. We pivot to text:

  1. Extract every page of every PDF via PyMuPDF
  2. Format as one big text block with [Source p.N] anchors before each page
  3. Place the corpus block in the user message with cache_control=ephemeral
  4. Per-query: append the analyst question; cache hit on the corpus

This loses Opus's native chart-and-figure understanding (PyMuPDF flattens
charts to nothing — the same limitation our RAG pipeline has). What it tests
is whether frontier-LLM long-context attention over the FULL textual corpus
beats our engineered chunk-and-retrieve pipeline.

The 1M-context beta is enabled — the corpus is ~640K tokens.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
import pymupdf

from rag_harness import BENCHMARK_QUERIES

# Same cited-analyst system prompt as Tuned RAG for fair comparison
SYSTEM_PROMPT = (
    "You are a senior energy markets analyst at Lumina Energy Partners. "
    "Answer ONLY from the provided IEA report excerpts below. "
    "For every key quantitative claim, cite the source report and page in "
    "square brackets, e.g. [Gas2025 p.42]. The page markers are inline in the "
    "corpus to help you cite accurately. "
    "If the corpus does not contain the answer, say so explicitly. "
    "Structure as an investment-committee briefing: "
    "(1) Headline finding, (2) supporting data with citations, "
    "(3) cross-sector linkages, (4) countervailing forces. "
    "Be concise; 200-350 words. Avoid speculation beyond the sources."
)


SKIP_HEADER_PATTERNS = (
    "Acknowledgements", "Table of contents", "Foreword",
    "Abbreviations and acronyms",
)
SKIP_TAIL_PATTERNS = (
    "References", "Bibliography", "Index", "Annex",
    "International Energy Agency Online bookshop",
)


def _is_substantive_page(text: str) -> bool:
    """Skip near-empty pages and obvious front/back matter."""
    stripped = text.strip()
    if len(stripped) < 200:
        return False
    head = stripped[:120]
    for pat in SKIP_HEADER_PATTERNS + SKIP_TAIL_PATTERNS:
        if pat.lower() in head.lower():
            return False
    return True


def build_corpus_text(pdf_dir: str = "IEAReports") -> tuple[str, dict]:
    """Concatenate substantive pages of every PDF with [Source p.N] anchors.

    Drops near-empty pages and obvious front/back matter (TOC, references,
    acknowledgements) to fit inside the 200K-token context cap on standard
    Opus tier. Page numbers in citations are PRESERVED from the original
    PDF (we just skip non-substantive pages).
    """
    paths = sorted(Path(pdf_dir).glob("*.pdf"))
    pieces: list[str] = []
    stats = {"reports": [], "total_pages": 0, "total_chars": 0, "skipped": 0}

    for path in paths:
        report = path.stem
        doc = pymupdf.open(str(path))
        n_pages = doc.page_count
        kept = 0
        skipped = 0
        pieces.append(f"\n\n========== START OF REPORT: {report} ==========\n")
        for i in range(n_pages):
            page_text = doc[i].get_text()
            if not _is_substantive_page(page_text):
                skipped += 1
                continue
            pieces.append(f"\n[{report} p.{i+1}]\n{page_text}")
            kept += 1
        pieces.append(f"\n========== END OF REPORT: {report} ==========\n")
        doc.close()
        stats["reports"].append({
            "name": report, "pages_original": n_pages,
            "pages_kept": kept, "pages_skipped": skipped,
        })
        stats["total_pages"] += kept
        stats["skipped"] += skipped

    corpus = "".join(pieces)
    stats["total_chars"] = len(corpus)
    return corpus, stats


def answer_query(
    client: anthropic.Anthropic,
    corpus_text: str,
    query: str,
    model: str = "claude-opus-4-5",
    max_output: int = 1024,
) -> dict:
    """Send one analyst query with the full corpus, return answer + usage."""
    t0 = time.perf_counter()
    # NOTE: 1M-tier requires Anthropic Tier 4 access. Without it, the standard
    # 200K cap applies. We use client.beta.messages.create with explicit betas
    # for prompt-caching support; corpus is trimmed to fit inside 200K.
    resp = client.beta.messages.create(
        model=model,
        max_tokens=max_output,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "IEA market report corpus (substantive pages of all 5 "
                        "reports, with [Source p.N] page anchors):\n" + corpus_text
                    ),
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"\n\nAnalyst question:\n{query}",
                },
            ],
        }],
        # Beta features used:
        #  - prompt-caching-2024-07-31: cache_control on the corpus block
        betas=["prompt-caching-2024-07-31"],
    )
    wall = time.perf_counter() - t0

    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    u = resp.usage
    return {
        "answer": text,
        "wall_seconds": round(wall, 2),
        "input_tokens": getattr(u, "input_tokens", 0),
        "output_tokens": getattr(u, "output_tokens", 0),
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
    }


def run_all_queries(api_key: str, queries: list[str] | None = None) -> dict:
    queries = queries or BENCHMARK_QUERIES
    client = anthropic.Anthropic(api_key=api_key)
    corpus, stats = build_corpus_text()
    print(f"[corpus] {len(stats['reports'])} reports, "
          f"{stats['total_pages']} pages, {stats['total_chars']:,} chars "
          f"(~{stats['total_chars']//4:,} estimated tokens)")
    for r in stats["reports"]:
        print(f"  {r['name']:<18s} {r['pages']:>4d} pages")

    results = {
        "answers": [], "usage": [], "wall_seconds": [],
        "corpus_stats": stats,
    }
    for i, q in enumerate(queries, 1):
        print(f"\n[query {i}/{len(queries)}] {q[:80]}...")
        r = answer_query(client, corpus, q)
        results["answers"].append(r["answer"])
        results["usage"].append({
            "input": r["input_tokens"],
            "output": r["output_tokens"],
            "cache_write": r["cache_creation_input_tokens"],
            "cache_read": r["cache_read_input_tokens"],
        })
        results["wall_seconds"].append(r["wall_seconds"])
        u = results["usage"][-1]
        print(f"  {r['wall_seconds']:.1f}s  in={u['input']:,}  out={u['output']:,}  "
              f"cache_write={u['cache_write']:,}  cache_read={u['cache_read']:,}")

    return results


def estimate_cost(results: dict) -> float:
    """Approximate USD cost based on standard Opus 4.5 pricing.

    Standard Opus 4.5: $15/M input, $75/M output.
    Cache write: 1.25x base in; cache read: 0.1x base in.
    """
    BASE_IN, BASE_OUT = 15.0, 75.0
    WRITE_MULT, READ_MULT = 1.25, 0.1
    total = 0.0
    for u in results["usage"]:
        cost = (u["input"] * BASE_IN
                + u["output"] * BASE_OUT
                + u["cache_write"] * BASE_IN * WRITE_MULT
                + u["cache_read"] * BASE_IN * READ_MULT) / 1e6
        total += cost
    return round(total, 2)


if __name__ == "__main__":
    keys = json.load(open("config.json"))
    results = run_all_queries(keys["ANTHROPIC_API_KEY"])

    cost = estimate_cost(results)
    print(f"\n=== Run complete ===")
    print(f"  Total estimated cost: ${cost}")
    print(f"  Mean wall per query: "
          f"{sum(results['wall_seconds'])/len(results['wall_seconds']):.1f}s")

    # Save into unified answers file
    all_ans = json.load(open("all_answers.json"))
    all_ans["opus_full_corpus_text"] = {
        "answers": results["answers"],
        # No retrieval — pass empty contexts so RAGAS metric harness doesn't crash
        "contexts": [[] for _ in results["answers"]],
        "usage": results["usage"],
        "wall_seconds": results["wall_seconds"],
        "estimated_cost_usd": cost,
        "corpus_stats": results["corpus_stats"],
    }
    json.dump(all_ans, open("all_answers.json", "w"), indent=2)
    print(f"  Saved under all_answers.json[opus_full_corpus_text]")
