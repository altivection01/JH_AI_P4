"""Custom 'causal correctness' metric to complement RAGAS Faithfulness.

RAGAS Faithfulness asks: 'Is every claim in the answer supported by SOME
retrieved context chunk?' That gives partial credit when an answer asserts
'A causes B' as long as both A and B appear in context — even if no chunk
actually states the causal relationship.

For multi-hop analyst questions (Q3, Q5 in our benchmark) this is the
exact failure mode CoV is designed to catch. We need a metric that scores
the relationship grounding, not just the entity grounding.

Definition: for each *causal* claim in the answer (statements of the form
'X drives Y', 'A leads to B', 'P accelerates Q', etc.):
  - SUPPORTED  : the cause→effect relationship is stated or directly
    paraphrased in some retrieved chunk
  - INFERRED   : the cause and effect appear in context but the
    relationship between them is the LLM's inference, not asserted by
    the source
  - UNSUPPORTED: the relationship contradicts or is absent from context

causal_correctness = (#supported + 0.5 * #inferred) / #causal_claims

(Inferred gets half credit because a plausible inference based on the source
is better than a fabricated claim but worse than a stated relationship.)

If an answer contains no causal claims, the score is reported as None
(N/A — the metric is only defined for causal-bearing answers).
"""
from __future__ import annotations

import json
import re
import time
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


EXTRACT_CAUSAL_SYS = """\
Extract every CAUSAL claim from the analyst briefing. A causal claim is a
statement asserting that one thing leads to, drives, causes, accelerates,
constrains, increases, decreases, enables, or otherwise produces a change
in another thing.

For each causal claim, identify the cause (X) and effect (Y) as short
noun phrases.

Return JSON only: a list of objects {claim, cause, effect}. No prose.
"""

JUDGE_CAUSAL_SYS = """\
You are judging whether ONE causal claim is supported by retrieved evidence.

The claim is of the form: <cause> → <effect>. Be STRICT:

- supported : the evidence explicitly states this causal relationship,
  or a direct paraphrase of it. Both the cause AND the effect appear, AND
  the relationship between them is stated.
- inferred  : the cause and effect both appear in the evidence (possibly
  in separate chunks) and the relationship is plausible from the source,
  but the source does not explicitly assert the link — it's the analyst's
  inference.
- unsupported : the relationship is not in the evidence, OR the evidence
  contradicts it.

Respond JSON only: {"verdict": "supported"|"inferred"|"unsupported",
"reason": "<one short sentence>"}
"""


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def extract_causal_claims(llm: ChatOpenAI, answer: str) -> list[dict]:
    resp = llm.invoke([
        SystemMessage(content=EXTRACT_CAUSAL_SYS),
        HumanMessage(content=f"Briefing:\n\n{answer}"),
    ])
    try:
        return json.loads(_strip_fence(resp.content))
    except json.JSONDecodeError:
        return []


def judge_causal_claim(
    llm: ChatOpenAI,
    cause: str,
    effect: str,
    claim_text: str,
    context_text: str,
) -> dict:
    user = (
        f"Claim: {claim_text}\n"
        f"Cause: {cause}\n"
        f"Effect: {effect}\n\n"
        f"Evidence:\n{context_text}"
    )
    resp = llm.invoke([
        SystemMessage(content=JUDGE_CAUSAL_SYS),
        HumanMessage(content=user),
    ])
    try:
        parsed = json.loads(_strip_fence(resp.content))
        return {
            "verdict": parsed.get("verdict", "unsupported"),
            "reason": parsed.get("reason", ""),
        }
    except json.JSONDecodeError:
        return {"verdict": "unsupported", "reason": "judge returned malformed JSON"}


def causal_correctness_one_query(
    llm: ChatOpenAI,
    answer: str,
    contexts: list[str],
    throttle_seconds: float = 6.0,   # ~10 calls/min, safely under tier-1 30K TPM
) -> dict:
    """Compute causal_correctness for one (answer, contexts) pair."""
    causal_claims = extract_causal_claims(llm, answer)
    if not causal_claims:
        return {"score": None, "n_causal": 0, "details": []}
    time.sleep(throttle_seconds)

    context_blob = "\n\n---\n\n".join(contexts)
    details = []
    n_supported = n_inferred = n_unsupported = 0
    for i, c in enumerate(causal_claims):
        cause = c.get("cause", "")
        effect = c.get("effect", "")
        claim_text = c.get("claim", "")
        if not (cause and effect):
            continue
        verdict_obj = judge_causal_claim(llm, cause, effect, claim_text, context_blob)
        details.append({
            "claim": claim_text, "cause": cause, "effect": effect,
            **verdict_obj,
        })
        v = verdict_obj["verdict"]
        if v == "supported":
            n_supported += 1
        elif v == "inferred":
            n_inferred += 1
        else:
            n_unsupported += 1
        # Throttle between calls to stay under TPM cap
        if i < len(causal_claims) - 1:
            time.sleep(throttle_seconds)

    total = n_supported + n_inferred + n_unsupported
    if total == 0:
        return {"score": None, "n_causal": 0, "details": details}
    score = (n_supported + 0.5 * n_inferred) / total
    return {
        "score": score,
        "n_causal": total,
        "n_supported": n_supported,
        "n_inferred": n_inferred,
        "n_unsupported": n_unsupported,
        "details": details,
    }


def causal_correctness_batch(
    llm: ChatOpenAI,
    questions: list[str],
    answers: list[str],
    contexts_list: list[list[str]],
) -> dict:
    """Compute causal_correctness across multiple queries; return aggregate."""
    per_query = []
    for q, ans, ctxs in zip(questions, answers, contexts_list):
        per_query.append(causal_correctness_one_query(llm, ans, ctxs))
    # Aggregate: mean of per-query scores (excluding None / no-causal queries)
    scored = [r["score"] for r in per_query if r["score"] is not None]
    return {
        "per_query": per_query,
        "mean_score": (sum(scored) / len(scored)) if scored else None,
        "n_scored_queries": len(scored),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--techniques", nargs="+",
                        default=["graph_rag", "graph_rag_cov"])
    args = parser.parse_args()

    keys = json.load(open("config.json"))
    llm = ChatOpenAI(model="gpt-4.1", api_key=keys["OPENAI_API_KEY"],
                     temperature=0, max_retries=4)
    all_ans = json.load(open("all_answers.json"))
    from rag_harness import BENCHMARK_QUERIES

    import pandas as pd
    rows = []
    for tech in args.techniques:
        if tech not in all_ans:
            print(f"[skip] technique {tech!r} not in all_answers.json")
            continue
        data = all_ans[tech]
        answers = data["answers"]
        contexts_list = data.get("contexts", [[] for _ in answers])

        print(f"\n=== {tech} ===")
        for trial in range(1, args.trials + 1):
            t0 = time.perf_counter()
            agg = causal_correctness_batch(
                llm, BENCHMARK_QUERIES, answers, contexts_list,
            )
            wall = time.perf_counter() - t0
            print(f"  trial {trial}: mean causal_correctness="
                  f"{(agg['mean_score'] if agg['mean_score'] is not None else float('nan')):.3f} "
                  f"on {agg['n_scored_queries']}/{len(answers)} queries  ({wall:.0f}s)")
            for qi, pq in enumerate(agg["per_query"], 1):
                if pq["score"] is None:
                    sc = "N/A (no causal claims)"
                else:
                    sc = f"{pq['score']:.3f} ({pq['n_supported']}/{pq['n_inferred']}/{pq['n_unsupported']} s/i/u)"
                print(f"    Q{qi}: {sc}")
            rows.append({
                "technique": tech, "trial": trial,
                "mean_causal_correctness": agg["mean_score"],
                "n_scored_queries": agg["n_scored_queries"],
                "wall_seconds": round(wall, 1),
            })

    df = pd.DataFrame(rows)
    df.to_csv("causal_correctness_raw.csv", index=False)

    print("\n=== AGGREGATE ===")
    for tech in args.techniques:
        sub = df[df["technique"] == tech]
        if len(sub) == 0:
            continue
        m = sub["mean_causal_correctness"].mean()
        s = sub["mean_causal_correctness"].std(ddof=1) if len(sub) > 1 else 0
        print(f"  {tech:<24s} {m:.3f} ± {s:.3f}  (n={len(sub)} trials)")
