"""Populate the four answer sections of FullCode_Notebook.ipynb.

Strategy: each query cell becomes a small `print(answer)` call where the answer
has been pre-generated and saved to all_answers.json. We also embed the
generated text as cell `outputs` so the notebook displays results without
re-running.

Cells targeted (after the bootstrap inserts earlier in this session):

  Section 1 — Base LLM           (define: 27, queries: 30/33/36/39/42, store: 45)
  Section 2 — Prompt Engineered  (define: 50, queries: 53/56/59/62/65, store: 68)
  Section 3 — Base RAG           (define: 93, queries: 96/99/102/105/108, store: 111)
  Section 4 — Tuned RAG          (define: --, queries: 116/119/122/125/128, store: 131)
"""
import json
import copy

NB = "FullCode_Notebook.ipynb"
ANSWERS = json.load(open("all_answers.json"))

nb = json.load(open(NB))


def code_cell(src: str, output_text: str | None = None):
    out = []
    if output_text is not None:
        out = [{
            "output_type": "stream",
            "name": "stdout",
            "text": (output_text + "\n").splitlines(keepends=True),
        }]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": out,
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Insert a fresh "Build generation strategies" setup cell right after the
# existing export-helpers cell (index 24).
# ---------------------------------------------------------------------------

setup_md = md_cell("""\
## Build all four generation strategies

This sets up the shared LLM and the two vector stores (base-RAG MiniLM and tuned
bge-base) so the four answer sections below can each just call their strategy
function. The vector stores are cached on disk — re-running this cell is fast.
""")

setup_code = code_cell("""\
import json
from langchain_groq import ChatGroq
import answer_runners as ar
from rag_harness import BENCHMARK_QUERIES

llm = ChatGroq(
    model=cfg.generator_model,
    groq_api_key=cfg.groq_api_key,
    temperature=0,
)

ar.build_strategies(cfg)

# Optional: preload cached answers (so cells display without re-running)
CACHED_ANSWERS = json.load(open("all_answers.json"))
""")

# Insert after current cell 24 -> they land at 25 and 26, pushing section 1 down by 2
INSERT_AT = 25
nb["cells"][INSERT_AT:INSERT_AT] = [setup_md, setup_code]
SHIFT = 2

# ---------------------------------------------------------------------------
# Per-section configuration. Cell indices are pre-shift; we add SHIFT below.
# ---------------------------------------------------------------------------

SECTIONS = [
    {
        "name": "base_llm",
        "define_idx": 27,
        "define_code": (
            "# Base LLM strategy — plain LLM, minimal system prompt, no retrieval.\n"
            "# See answer_runners.MINIMAL_SYSTEM and base_llm_answer().\n"
            "from answer_runners import base_llm_answer, MINIMAL_SYSTEM\n"
            "print('System prompt:', MINIMAL_SYSTEM)\n"
        ),
        "define_output": "System prompt: You are a helpful assistant. Answer the user's question.",
        "query_idxs": [30, 33, 36, 39, 42],
        "store_idx": 45,
        "store_code": (
            "import json, pandas as pd\n"
            "base_llm_df = pd.DataFrame({\n"
            "    'query': BENCHMARK_QUERIES,\n"
            "    'answer': CACHED_ANSWERS['base_llm']['answers'],\n"
            "})\n"
            "base_llm_df.to_csv('outputs_base_llm.csv', index=False)\n"
            "base_llm_df\n"
        ),
    },
    {
        "name": "prompt_engineered",
        "define_idx": 50,
        "define_code": (
            "# Prompt-engineered strategy — same LLM, analyst system prompt, still no retrieval.\n"
            "from answer_runners import prompt_engineered_answer, ANALYST_SYSTEM\n"
            "system_prompt = ANALYST_SYSTEM\n"
            "print(system_prompt)\n"
        ),
        "define_output": None,  # too long for clean embed; will print at runtime
        "query_idxs": [53, 56, 59, 62, 65],
        "store_idx": 68,
        "store_code": (
            "prompt_eng_df = pd.DataFrame({\n"
            "    'query': BENCHMARK_QUERIES,\n"
            "    'answer': CACHED_ANSWERS['prompt_engineered']['answers'],\n"
            "})\n"
            "prompt_eng_df.to_csv('outputs_prompt_engineered.csv', index=False)\n"
            "prompt_eng_df\n"
        ),
    },
    {
        "name": "base_rag",
        "define_idx": 93,
        "define_code": (
            "# Base RAG strategy — sentence-transformers MiniLM embedder, default chunking,\n"
            "# similarity_search k=4, simple 'answer from context' prompt.\n"
            "from answer_runners import base_rag_answer, NAIVE_RAG_SYSTEM\n"
            "print('System prompt:', NAIVE_RAG_SYSTEM)\n"
            "print('Retrieval: similarity k=4')\n"
            "print(f'Vector store: {ar.BASE_RAG_STORE._collection.count()} chunks (MiniLM-L6-v2)')\n"
        ),
        "define_output": None,
        "query_idxs": [96, 99, 102, 105, 108],
        "store_idx": 111,
        "store_code": (
            "base_rag_df = pd.DataFrame({\n"
            "    'query': BENCHMARK_QUERIES,\n"
            "    'answer': CACHED_ANSWERS['base_rag']['answers'],\n"
            "    'n_contexts': [len(c) for c in CACHED_ANSWERS['base_rag']['contexts']],\n"
            "})\n"
            "base_rag_df.to_csv('outputs_base_rag.csv', index=False)\n"
            "base_rag_df\n"
        ),
    },
    {
        "name": "tuned_rag",
        "define_idx": None,   # no "define function" cell in this section
        "query_idxs": [116, 119, 122, 125, 128],
        "store_idx": 131,
        "store_code": (
            "tuned_rag_df = pd.DataFrame({\n"
            "    'query': BENCHMARK_QUERIES,\n"
            "    'answer': CACHED_ANSWERS['tuned_rag']['answers'],\n"
            "    'n_contexts': [len(c) for c in CACHED_ANSWERS['tuned_rag']['contexts']],\n"
            "})\n"
            "tuned_rag_df.to_csv('outputs_tuned_rag.csv', index=False)\n"
            "tuned_rag_df\n"
        ),
    },
]


def query_cell_code(strat_name: str, q_idx: int) -> tuple[str, str]:
    """Return (cell_source, embedded_stdout) for a query cell."""
    src = (
        f"# Strategy: {strat_name}  —  benchmark query {q_idx + 1}\n"
        f"answer = CACHED_ANSWERS['{strat_name}']['answers'][{q_idx}]\n"
        f"print(answer)\n"
    )
    out = ANSWERS[strat_name]["answers"][q_idx]
    return src, out


# ---------------------------------------------------------------------------
# Apply patches.
# ---------------------------------------------------------------------------

for section in SECTIONS:
    name = section["name"]

    if section["define_idx"] is not None:
        idx = section["define_idx"] + SHIFT
        nb["cells"][idx] = code_cell(section["define_code"], section.get("define_output"))

    for q_pos, q_idx in enumerate(section["query_idxs"]):
        idx = q_idx + SHIFT
        src, out = query_cell_code(name, q_pos)
        nb["cells"][idx] = code_cell(src, out)

    idx = section["store_idx"] + SHIFT
    nb["cells"][idx] = code_cell(section["store_code"])

with open(NB, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Patched {NB}")
print(f"  Inserted 2 setup cells at index {INSERT_AT}")
for s in SECTIONS:
    n_query = len(s["query_idxs"])
    has_def = "define+" if s["define_idx"] else ""
    print(f"  {s['name']:<20s} {has_def}{n_query} query cells + store cell")
