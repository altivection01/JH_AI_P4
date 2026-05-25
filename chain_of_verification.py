"""Chain-of-Verification (CoV) on top of GraphRAG.

For each query:
  1. Take the existing GraphRAG draft answer
  2. Extract atomic claims, classifying each as quantitative / causal / procedural
  3. For every causal claim ("A leads to B" / "A drives B" / etc.), retrieve
     targeted evidence using the entities in the claim, then ask a strong-reasoning
     LLM whether the SPECIFIC causal link is supported (vs. just having A and B
     individually mentioned somewhere)
  4. For every quantitative claim, verify the specific number against retrieved
     evidence
  5. Rewrite the final answer keeping only verified claims; explicitly hedge
     inferred-but-not-stated relationships

The "draft" generator stays `llama-3.3-70b-versatile` (free) so we isolate the
verification step as the only new component vs. baseline GraphRAG. Claim
extraction, verification, and final rewrite all use `gpt-4.1` — they are the
reasoning-heavy steps that benefit most from a strong model.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from graph_rag import (
    GRAPH_RAG_SYSTEM,
    attach_neo4j,
    graph_rag_retrieve,
    neo4j_graph,
    setup_vector_index,
)
from rag_harness import BENCHMARK_QUERIES, HarnessConfig, _format_context


ClaimType = Literal["quantitative", "causal", "procedural"]


@dataclass
class Claim:
    text: str
    type: ClaimType
    entities: list[str] = field(default_factory=list)
    verdict: str = ""          # "supported", "inferred", "unsupported"
    evidence_chunks: list[str] = field(default_factory=list)
    reasoning: str = ""


CLAIM_EXTRACTION_SYS = """\
You are a careful analyst reviewing an investment briefing for sourcing rigor.
Extract every atomic factual claim. Classify each:

- quantitative: a specific number, percentage, year, or magnitude
  (e.g., "data centres account for ~20% of US demand growth")
- causal:       a cause-effect or driver relationship between two named things
  (e.g., "OBBBA tax credits accelerate West Coast refinery closures")
- procedural:   a description of a mechanism, framework, or process without
  asserting a specific number or causal link
  (e.g., "Two-sided CfDs expose generators to bidirectional price risk")

For each claim list the key named entities (companies, policies, regions,
energy sources, technologies, time periods).

Return a JSON list. Each element: {claim, type, entities}. No prose around it.
"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_claims(llm: ChatOpenAI, answer: str) -> list[Claim]:
    """Decompose a draft answer into atomic claims."""
    resp = llm.invoke([
        SystemMessage(content=CLAIM_EXTRACTION_SYS),
        HumanMessage(content=f"Briefing:\n\n{answer}"),
    ])
    try:
        raw = json.loads(_strip_code_fence(resp.content))
    except json.JSONDecodeError as e:
        print(f"  [warn] claim extraction returned malformed JSON: {e}")
        return []
    claims: list[Claim] = []
    for item in raw:
        try:
            claims.append(Claim(
                text=item["claim"],
                type=item["type"],
                entities=item.get("entities", []),
            ))
        except KeyError:
            continue
    return claims


VERIFICATION_SYS = """\
You are verifying ONE claim against retrieved evidence excerpts. Apply a
STRICT entailment standard — do NOT credit "the entities are individually
mentioned somewhere" as support for a CAUSAL link. A claim is:

- supported   : the evidence states the relationship/number explicitly, or
                a direct paraphrase
- inferred    : both entities/numbers are present in the evidence but the
                relationship/exact value is implied or assembled across
                multiple chunks; the inference is plausible but the source
                does not commit to it directly
- unsupported : the evidence does not contain either the relationship or
                contradicts it

Respond in JSON only: {"verdict": "supported"|"inferred"|"unsupported",
"reasoning": "<one sentence>"}
"""


def verify_claim(
    llm: ChatOpenAI,
    claim: Claim,
    vector_index,
    graph,
    k_seed: int = 4,
    k_expansion: int = 4,
) -> Claim:
    """Retrieve entity-targeted evidence and judge support level."""
    # Build a retrieval query from the claim + its key entities
    retrieval_query = claim.text + " " + " ".join(claim.entities)
    docs = graph_rag_retrieve(
        retrieval_query, vector_index, graph,
        k_seed=k_seed, k_expansion=k_expansion,
    )
    context = _format_context(docs)
    claim.evidence_chunks = [
        f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}] "
        f"{d.page_content[:200]}…"
        for d in docs[:4]
    ]

    user_msg = (
        f"Claim type: {claim.type}\n"
        f"Claim: {claim.text}\n\n"
        f"Evidence:\n{context}"
    )
    resp = llm.invoke([
        SystemMessage(content=VERIFICATION_SYS),
        HumanMessage(content=user_msg),
    ])
    try:
        parsed = json.loads(_strip_code_fence(resp.content))
        claim.verdict = parsed.get("verdict", "")
        claim.reasoning = parsed.get("reasoning", "")
    except json.JSONDecodeError:
        claim.verdict = "unsupported"
        claim.reasoning = "verifier returned malformed JSON"
    return claim


REWRITE_SYS = """\
You are rewriting an investment briefing to be auditable. You receive:
  1. The original analyst question
  2. The draft briefing
  3. A list of every claim in the draft, each tagged supported / inferred /
     unsupported, with verifier reasoning

Rewrite the briefing with these rules:
  - Keep all supported claims verbatim or near-verbatim; preserve their
    citations
  - For inferred claims, retain the substance but mark the inference
    explicitly with a phrase like "the report does not state this directly,
    but it implies…" — do NOT delete inferred-but-plausible context, but be
    transparent about the inference
  - REMOVE unsupported claims entirely (they failed verification against the
    sources)
  - Preserve the structured analyst-briefing format: (1) headline finding,
    (2) supporting data with citations, (3) cross-sector linkages,
    (4) countervailing forces
  - Be concise; 200-350 words
  - Use the same citation format as the draft: [Source p.N]

If MORE THAN HALF of the original claims are unsupported, the briefing
should explicitly state that the IEA reports do not adequately address the
question, and explain what they DO say.
"""


def rewrite_with_verifications(
    llm: ChatOpenAI,
    query: str,
    draft: str,
    claims: list[Claim],
) -> str:
    if not claims:
        return draft + "\n\n[Note: claim extraction failed; returning original.]"

    claims_blob_parts = []
    for i, c in enumerate(claims, 1):
        claims_blob_parts.append(
            f"\n[{i}] type={c.type}  verdict={c.verdict}\n"
            f"    claim: {c.text}\n"
            f"    verifier reasoning: {c.reasoning}"
        )
    claims_blob = "".join(claims_blob_parts)

    user_msg = (
        f"Original question:\n{query}\n\n"
        f"Draft briefing:\n{draft}\n\n"
        f"Per-claim verifications:{claims_blob}\n\n"
        f"Now rewrite the briefing with the rules above."
    )
    resp = llm.invoke([
        SystemMessage(content=REWRITE_SYS),
        HumanMessage(content=user_msg),
    ])
    return resp.content


def run_cov_pipeline(
    query: str,
    draft_answer: str,
    draft_contexts: list[str],
    cfg: HarnessConfig,
    llm_reasoning: ChatOpenAI,
) -> dict:
    """Full CoV pipeline for one query.

    Returns: {answer, claims, verified_contexts, wall_seconds}
    """
    t0 = time.perf_counter()
    vec = setup_vector_index(cfg)
    graph = neo4j_graph(cfg)

    # Step 1: extract claims from the draft
    claims = extract_claims(llm_reasoning, draft_answer)
    print(f"  extracted {len(claims)} claims "
          f"({sum(1 for c in claims if c.type == 'causal')} causal, "
          f"{sum(1 for c in claims if c.type == 'quantitative')} quantitative, "
          f"{sum(1 for c in claims if c.type == 'procedural')} procedural)")

    # Step 2: verify each claim
    for c in claims:
        verify_claim(llm_reasoning, c, vec, graph)
    n_sup = sum(1 for c in claims if c.verdict == "supported")
    n_inf = sum(1 for c in claims if c.verdict == "inferred")
    n_uns = sum(1 for c in claims if c.verdict == "unsupported")
    print(f"  verdicts: {n_sup} supported, {n_inf} inferred, {n_uns} unsupported")

    # Step 3: rewrite using verifications
    final = rewrite_with_verifications(llm_reasoning, query, draft_answer, claims)

    wall = time.perf_counter() - t0

    # Build a flattened context list that combines the original draft contexts
    # with all evidence chunks pulled for verification — used by RAGAS later
    combined_contexts = list(draft_contexts)
    for c in claims:
        combined_contexts.extend(c.evidence_chunks)
    # Dedupe by prefix
    seen = set()
    unique_contexts = []
    for ctx in combined_contexts:
        k = ctx[:80]
        if k in seen:
            continue
        seen.add(k)
        unique_contexts.append(ctx)

    return {
        "answer": final,
        "claims": [
            {"text": c.text, "type": c.type, "entities": c.entities,
             "verdict": c.verdict, "reasoning": c.reasoning}
            for c in claims
        ],
        "contexts": unique_contexts,
        "wall_seconds": round(wall, 2),
        "n_supported": n_sup,
        "n_inferred": n_inf,
        "n_unsupported": n_uns,
    }


if __name__ == "__main__":
    keys = json.load(open("config.json"))
    cfg = HarnessConfig(
        groq_api_key=keys["GROQ_API_KEY"],
        openai_api_key=keys["OPENAI_API_KEY"],
        anthropic_api_key=keys.get("ANTHROPIC_API_KEY"),
        chunk_size=380, chunk_overlap=60, chunk_unit="token",
    )
    attach_neo4j(cfg, keys)

    # Reasoning LLM for claim extraction + verification + rewrite
    reasoner = ChatOpenAI(
        model="gpt-4.1", api_key=cfg.openai_api_key,
        temperature=0, max_retries=4,
    )

    # Load existing GraphRAG drafts
    all_ans = json.load(open("all_answers.json"))
    drafts = all_ans["graph_rag"]
    draft_answers = drafts["answers"]
    draft_contexts = drafts["contexts"]

    cov_answers: list[str] = []
    cov_contexts: list[list[str]] = []
    cov_artifacts: list[dict] = []
    walls: list[float] = []

    for i, q in enumerate(BENCHMARK_QUERIES):
        print(f"\n=== Q{i+1}/{len(BENCHMARK_QUERIES)}: {q[:80]}… ===")
        result = run_cov_pipeline(
            q, draft_answers[i], draft_contexts[i],
            cfg, reasoner,
        )
        cov_answers.append(result["answer"])
        cov_contexts.append(result["contexts"])
        cov_artifacts.append({
            "query": q,
            "claims": result["claims"],
            "n_supported": result["n_supported"],
            "n_inferred": result["n_inferred"],
            "n_unsupported": result["n_unsupported"],
            "wall_seconds": result["wall_seconds"],
        })
        walls.append(result["wall_seconds"])
        print(f"  wall: {result['wall_seconds']:.1f}s, "
              f"final answer {len(result['answer'])} chars")

    all_ans["graph_rag_cov"] = {
        "answers": cov_answers,
        "contexts": cov_contexts,
        "wall_seconds": walls,
    }
    json.dump(all_ans, open("all_answers.json", "w"), indent=2)
    json.dump(cov_artifacts, open("cov_verification_artifacts.json", "w"), indent=2)
    print(f"\nSaved graph_rag_cov to all_answers.json")
    print(f"Saved per-claim verification artifacts to cov_verification_artifacts.json")
