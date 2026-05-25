"""Section 6.8 Option B: Frontier LLMs with the full corpus loaded into context.

Two variants:
  - opus_full_corpus:  Claude Opus 4.5  + trimmed corpus  (200K-tier cap, ~818 pages)
  - gpt41_full_corpus: OpenAI GPT-4.1   + full corpus     (1M context, all 868 pages)

Same cited-analyst system prompt as Tuned RAG for fair comparison.

Anthropic explicit prompt caching (`cache_control: ephemeral`) on the corpus block.
OpenAI prompt caching is automatic for prompts >1024 tokens — no parameter needed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
import pymupdf
from openai import OpenAI

from rag_harness import BENCHMARK_QUERIES

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

# Pages we exclude when trimming for the 200K-token Anthropic tier
SKIP_HEADER_PATTERNS = (
    "Acknowledgements", "Table of contents", "Foreword",
    "Abbreviations and acronyms",
)
SKIP_TAIL_PATTERNS = (
    "References", "Bibliography", "Index", "Annex",
    "International Energy Agency Online bookshop",
)


def _is_substantive(text: str, min_chars: int = 400) -> bool:
    """Filter pages by content density.

    Pages with <400 chars of text are typically: blank pages, front matter
    (covers, dividers, acknowledgements one-liners), or figure-only pages
    where PyMuPDF extracts essentially nothing because the page is a chart.
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return False
    head = stripped[:120].lower()
    for pat in SKIP_HEADER_PATTERNS + SKIP_TAIL_PATTERNS:
        if pat.lower() in head:
            return False
    return True


def build_corpus_text(pdf_dir: str = "IEAReports",
                      trim_for_200k: bool = False) -> tuple[str, dict]:
    """Concatenate every PDF page with [Source p.N] anchors.

    `trim_for_200k=True` drops front/back matter to fit inside Anthropic's
    standard-tier 200K-token cap. `False` includes every page.
    """
    paths = sorted(Path(pdf_dir).glob("*.pdf"))
    pieces: list[str] = []
    stats = {"reports": [], "total_pages": 0, "total_chars": 0,
             "skipped": 0, "trimmed": trim_for_200k}

    for path in paths:
        report = path.stem
        doc = pymupdf.open(str(path))
        n_pages = doc.page_count
        kept = skipped = 0
        pieces.append(f"\n\n========== START OF REPORT: {report} ==========\n")
        for i in range(n_pages):
            page_text = doc[i].get_text()
            if trim_for_200k and not _is_substantive(page_text):
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


# ---------------------------------------------------------------------------
# Provider-specific query functions
# ---------------------------------------------------------------------------

def answer_opus(client: anthropic.Anthropic, corpus: str, query: str,
                model: str = "claude-opus-4-5", max_output: int = 1024) -> dict:
    t0 = time.perf_counter()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=max_output,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": ("IEA market report corpus (substantive pages of all "
                             "5 reports, with [Source p.N] anchors):\n" + corpus),
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": f"\n\nAnalyst question:\n{query}"},
            ],
        }],
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


def answer_gpt(client: OpenAI, corpus: str, query: str,
               model: str = "gpt-4.1", max_output: int = 1024) -> dict:
    """OpenAI Responses API call. Prompt caching is automatic for prompts >1024 tokens.

    Cache hits are reported in usage.prompt_tokens_details.cached_tokens.
    """
    t0 = time.perf_counter()
    # Use Chat Completions for backward compat (Responses API is newer)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_output,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text",
                 "text": ("IEA market report corpus (full text, all 5 reports, "
                          "with [Source p.N] anchors):\n" + corpus)},
                {"type": "text",
                 "text": f"\n\nAnalyst question:\n{query}"},
            ]},
        ],
    )
    wall = time.perf_counter() - t0
    text = resp.choices[0].message.content
    u = resp.usage
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0)
    return {
        "answer": text,
        "wall_seconds": round(wall, 2),
        "input_tokens": getattr(u, "prompt_tokens", 0),
        "output_tokens": getattr(u, "completion_tokens", 0),
        "cached_tokens": cached,
    }


# ---------------------------------------------------------------------------
# Cost estimators
# ---------------------------------------------------------------------------

def cost_opus(usage_records: list[dict]) -> float:
    """Opus 4.5 standard pricing: $15/M in, $75/M out.
    Cache write: 1.25x base in; cache read: 0.1x base in."""
    BASE_IN, BASE_OUT = 15.0, 75.0
    WRITE_MULT, READ_MULT = 1.25, 0.1
    total = 0.0
    for u in usage_records:
        c = (u["input"] * BASE_IN
             + u["output"] * BASE_OUT
             + u["cache_write"] * BASE_IN * WRITE_MULT
             + u["cache_read"] * BASE_IN * READ_MULT) / 1e6
        total += c
    return round(total, 2)


def cost_gpt41(usage_records: list[dict]) -> float:
    """GPT-4.1 pricing: $2/M in (uncached), $0.50/M in (cached), $8/M out."""
    UNCACHED_IN, CACHED_IN, OUT = 2.0, 0.5, 8.0
    total = 0.0
    for u in usage_records:
        uncached = u["input"] - u["cached"]
        c = (uncached * UNCACHED_IN
             + u["cached"] * CACHED_IN
             + u["output"] * OUT) / 1e6
        total += c
    return round(total, 2)


# ---------------------------------------------------------------------------
# End-to-end runners
# ---------------------------------------------------------------------------

def run_opus_full_corpus(api_key: str, queries: list[str] | None = None) -> dict:
    queries = queries or BENCHMARK_QUERIES
    client = anthropic.Anthropic(api_key=api_key)
    corpus, stats = build_corpus_text(trim_for_200k=True)
    print(f"[corpus] {stats['total_pages']} pages kept, "
          f"{stats['skipped']} skipped, {stats['total_chars']:,} chars")

    answers, usage, walls = [], [], []
    for i, q in enumerate(queries, 1):
        print(f"  [Q{i}/{len(queries)}] {q[:70]}…")
        r = answer_opus(client, corpus, q)
        answers.append(r["answer"])
        usage.append({
            "input": r["input_tokens"], "output": r["output_tokens"],
            "cache_write": r["cache_creation_input_tokens"],
            "cache_read": r["cache_read_input_tokens"],
        })
        walls.append(r["wall_seconds"])
        u = usage[-1]
        print(f"    {r['wall_seconds']:.1f}s  in={u['input']:,} out={u['output']:,} "
              f"cw={u['cache_write']:,} cr={u['cache_read']:,}")

    cost = cost_opus(usage)
    return {
        "answers": answers, "usage": usage, "wall_seconds": walls,
        "estimated_cost_usd": cost, "corpus_stats": stats,
    }


def run_gpt41_full_corpus(api_key: str, queries: list[str] | None = None) -> dict:
    queries = queries or BENCHMARK_QUERIES
    client = OpenAI(api_key=api_key)
    corpus, stats = build_corpus_text(trim_for_200k=False)
    print(f"[corpus] {stats['total_pages']} pages (untrimmed), "
          f"{stats['total_chars']:,} chars")

    answers, usage, walls = [], [], []
    for i, q in enumerate(queries, 1):
        print(f"  [Q{i}/{len(queries)}] {q[:70]}…")
        r = answer_gpt(client, corpus, q)
        answers.append(r["answer"])
        usage.append({
            "input": r["input_tokens"], "output": r["output_tokens"],
            "cached": r["cached_tokens"],
        })
        walls.append(r["wall_seconds"])
        u = usage[-1]
        print(f"    {r['wall_seconds']:.1f}s  in={u['input']:,} out={u['output']:,} "
              f"cached={u['cached']:,}")

    cost = cost_gpt41(usage)
    return {
        "answers": answers, "usage": usage, "wall_seconds": walls,
        "estimated_cost_usd": cost, "corpus_stats": stats,
    }


if __name__ == "__main__":
    keys = json.load(open("config.json"))
    all_ans = json.load(open("all_answers.json"))

    print("\n=== Running Opus 4.5 + trimmed corpus ===")
    opus = run_opus_full_corpus(keys["ANTHROPIC_API_KEY"])
    all_ans["opus_full_corpus"] = {
        "answers": opus["answers"],
        "contexts": [[] for _ in opus["answers"]],
        "usage": opus["usage"], "wall_seconds": opus["wall_seconds"],
        "estimated_cost_usd": opus["estimated_cost_usd"],
        "corpus_stats": opus["corpus_stats"],
    }
    print(f"  Opus cost: ${opus['estimated_cost_usd']}")

    print("\n=== Running GPT-4.1 + full corpus ===")
    gpt = run_gpt41_full_corpus(keys["OPENAI_API_KEY"])
    all_ans["gpt41_full_corpus"] = {
        "answers": gpt["answers"],
        "contexts": [[] for _ in gpt["answers"]],
        "usage": gpt["usage"], "wall_seconds": gpt["wall_seconds"],
        "estimated_cost_usd": gpt["estimated_cost_usd"],
        "corpus_stats": gpt["corpus_stats"],
    }
    print(f"  GPT-4.1 cost: ${gpt['estimated_cost_usd']}")

    json.dump(all_ans, open("all_answers.json", "w"), indent=2)
    print(f"\nSaved opus_full_corpus + gpt41_full_corpus to all_answers.json")
    print(f"Combined cost: ${opus['estimated_cost_usd'] + gpt['estimated_cost_usd']}")
