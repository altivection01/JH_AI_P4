"""Citation verification — does each [Source p.N] reference actually support
the claim it's attached to?

Pipeline:
  1. Parse [Source p.N]-style citations from an answer using regex
  2. For each citation, identify the *cited claim* (the sentence containing it)
  3. Load the cited page directly from the original PDF
  4. Ask gpt-4.1 to judge whether the page supports the claim
       - supported   : page contains evidence directly supporting the claim
       - partial     : page mentions the topic but doesn't fully support the
                       specific claim (e.g., adjacent fact, different number)
       - unsupported : page doesn't support the claim at all (citation is
                       likely the wrong page or fabricated)

  5. Aggregate to a per-answer citation_accuracy score:
       (supported + 0.5 * partial) / total_citations
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# Matches [Electricity2026 p.59], [Gas2025, p. 42], [Coal2025 p.7], etc.
CITATION_PATTERN = re.compile(
    r"\[(?P<source>[A-Za-z][A-Za-z0-9_]*\d{4})\s*[,]?\s*p\.?\s*(?P<page>\d+)\]",
    re.IGNORECASE,
)

PDF_DIR = Path("IEAReports")
_PDF_CACHE: dict[str, pymupdf.Document] = {}


def _open_pdf(source: str) -> pymupdf.Document:
    """Load a source PDF, caching the open document."""
    if source not in _PDF_CACHE:
        candidates = list(PDF_DIR.glob(f"{source}.pdf"))
        if not candidates:
            # Tolerant match: case-insensitive
            candidates = [p for p in PDF_DIR.glob("*.pdf")
                          if p.stem.lower() == source.lower()]
        if not candidates:
            raise FileNotFoundError(f"No PDF found for source {source!r} in {PDF_DIR}")
        _PDF_CACHE[source] = pymupdf.open(str(candidates[0]))
    return _PDF_CACHE[source]


def get_page_text(source: str, page_num: int, page_offset: int = 0) -> str | None:
    """Fetch the text of a cited page from the named source PDF.

    `page_offset` is the adjustment from the cited number to the PyMuPDF
    0-indexed page. Our chunking pipeline uses PyMuPDFLoader which returns
    0-indexed page metadata; the LLM cites those numbers verbatim. So the
    correct lookup for citation `p.N` is `doc[N]` (offset=0). An analyst
    reading the PDF expects 1-indexed pages where the printed page numbers
    appear; for that view, use offset=-1 (so `p.N` → `doc[N-1]`).

    This pipeline asymmetry is documented in Section 6.10 as a UX bug worth
    fixing in production by adjusting the chunk metadata to be 1-indexed
    before the LLM sees it.
    """
    try:
        doc = _open_pdf(source)
    except FileNotFoundError:
        return None
    idx = page_num + page_offset
    if idx < 0 or idx >= doc.page_count:
        return None
    return doc[idx].get_text()


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return [(start, end, sentence_text)] for sentence-like chunks."""
    # Split on sentence enders, retain positions
    out = []
    cursor = 0
    for piece in re.split(r"(?<=[.!?])\s+", text):
        start = text.find(piece, cursor)
        if start == -1:
            continue
        end = start + len(piece)
        cursor = end
        out.append((start, end, piece))
    return out


@dataclass
class Citation:
    source: str
    page: int
    raw: str
    claim_text: str
    span: tuple[int, int]
    verdict: str = ""        # "supported" | "partial" | "unsupported"
    score: float = 0.0
    reason: str = ""
    page_text: str = ""


def extract_citations(answer: str) -> list[Citation]:
    """Find every [Source p.N] reference and the sentence it appears in."""
    sentences = _split_sentences(answer)
    citations: list[Citation] = []
    for m in CITATION_PATTERN.finditer(answer):
        # Find the sentence containing this citation
        claim = ""
        for start, end, sent in sentences:
            if start <= m.start() < end:
                claim = sent.strip()
                break
        if not claim:
            # Fall back: take 200 chars around the citation
            claim = answer[max(0, m.start() - 150): m.end() + 10].strip()
        citations.append(Citation(
            source=m.group("source"),
            page=int(m.group("page")),
            raw=m.group(0),
            claim_text=claim,
            span=(m.start(), m.end()),
        ))
    return citations


VERIFIER_SYS = """\
You are auditing a single citation in an investment briefing.

You receive:
  - the claim sentence containing the citation
  - the source name and page number cited
  - the actual text of that page

Decide whether the cited page supports the claim. Use a strict standard:

- supported  : the page contains evidence that directly supports the claim
  (or a close paraphrase). The specific number, relationship, or fact in
  the claim should appear on this page.
- partial    : the page is on-topic — it mentions the entities/subject —
  but does NOT support the specific quantitative or relational assertion
  in the claim. (E.g., page discusses LNG capacity but the claim's number
  is on a different page.)
- unsupported: the page does not address the topic, or contradicts it.
  This is the case for fabricated/wrong-page citations.

Respond JSON only: {"verdict": "supported"|"partial"|"unsupported",
"reason": "<one short sentence>"}
"""


SCORE_MAP = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def verify_citation(llm: ChatOpenAI, c: Citation,
                    page_offset: int = 0) -> Citation:
    """Judge whether the cited page supports the claim.

    page_offset=0 follows the chunk-metadata convention (LLM's view).
    page_offset=-1 follows the printed-PDF-page convention (analyst's view).
    """
    page_text = get_page_text(c.source, c.page, page_offset=page_offset)
    if page_text is None:
        c.verdict = "unsupported"
        c.score = 0.0
        c.reason = f"page {c.page} does not exist in {c.source}.pdf"
        return c
    c.page_text = page_text[:6000]   # cap to control tokens

    user = (
        f"Claim sentence: {c.claim_text}\n"
        f"Citation: [{c.source} p.{c.page}]\n\n"
        f"Page {c.page} of {c.source}:\n{c.page_text}"
    )
    resp = llm.invoke([
        SystemMessage(content=VERIFIER_SYS),
        HumanMessage(content=user),
    ])
    try:
        parsed = json.loads(_strip_fence(resp.content))
        c.verdict = parsed.get("verdict", "unsupported")
        c.reason = parsed.get("reason", "")
    except json.JSONDecodeError:
        c.verdict = "unsupported"
        c.reason = "judge returned malformed JSON"
    c.score = SCORE_MAP.get(c.verdict, 0.0)
    return c


def citation_accuracy_one_answer(
    llm: ChatOpenAI,
    answer: str,
    throttle_seconds: float = 5.0,
    page_offset: int = 0,
) -> dict:
    """Compute citation_accuracy for one answer.

    page_offset: see get_page_text(). 0 = chunk-metadata convention
    (matches the LLM's citations as written), -1 = printed-PDF-page
    convention (matches what an analyst sees opening the PDF).
    """
    citations = extract_citations(answer)
    if not citations:
        return {"score": None, "n_citations": 0, "details": []}

    n_supported = n_partial = n_unsupported = 0
    for i, c in enumerate(citations):
        verify_citation(llm, c, page_offset=page_offset)
        if c.verdict == "supported":
            n_supported += 1
        elif c.verdict == "partial":
            n_partial += 1
        else:
            n_unsupported += 1
        if i < len(citations) - 1:
            time.sleep(throttle_seconds)

    total = len(citations)
    score = sum(c.score for c in citations) / total
    return {
        "score": score,
        "n_citations": total,
        "n_supported": n_supported,
        "n_partial": n_partial,
        "n_unsupported": n_unsupported,
        "details": [
            {"raw": c.raw, "source": c.source, "page": c.page,
             "claim": c.claim_text[:200],
             "verdict": c.verdict, "reason": c.reason, "score": c.score}
            for c in citations
        ],
    }


def citation_accuracy_batch(
    llm: ChatOpenAI,
    answers: list[str],
    pause_between_queries: float = 30.0,
    page_offset: int = 0,
) -> dict:
    """Compute citation_accuracy across a list of answers; return aggregate."""
    per_query = []
    for i, ans in enumerate(answers):
        per_query.append(citation_accuracy_one_answer(
            llm, ans, page_offset=page_offset))
        if i < len(answers) - 1:
            time.sleep(pause_between_queries)
    scored = [r["score"] for r in per_query if r["score"] is not None]
    return {
        "per_query": per_query,
        "mean_score": (sum(scored) / len(scored)) if scored else None,
        "n_scored_queries": len(scored),
        "n_no_citations": sum(1 for r in per_query if r["score"] is None),
    }
