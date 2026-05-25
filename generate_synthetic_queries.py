"""Generate 25 additional analyst-style queries to augment the 5 benchmarks.

Mix of difficulty levels:
  - Single-hop factual ("What is the projected LNG capacity in 2030?")
  - Cross-sector synthesis ("How do EV adoption and grid investment interact?")
  - Multi-document comparative ("How do US and EU coal retirement schedules differ?")

Uses gpt-4o-mini at temperature=0.2 for variety.
"""
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

keys = json.load(open("config.json"))
llm = ChatOpenAI(model="gpt-4o-mini", api_key=keys["OPENAI_API_KEY"],
                 temperature=0.2, max_retries=4)

SYSTEM = """You write realistic analyst questions for Lumina Energy Partners,
an investment firm with positions across coal, oil, gas, electricity, and
renewables. Questions will be answered against the IEA's 2025/2026 market
reports: Coal 2025, Oil 2025, Gas 2025, Renewables 2025, Electricity 2026.

Write questions an actual analyst would ask before an investment-committee
meeting. Mix difficulty:
  - 8 single-source factual ("What is X?", "How big is Y?")
  - 8 cross-sector synthesis ("How does A affect B?")
  - 9 multi-hop or comparative (involve 3+ entities, geographies, or time periods)

Each question must be:
- Specific to actual content of IEA medium-term outlooks (avoid generic ML/AI/finance questions)
- Anchored to specific timeframes, regions, technologies, or policies where appropriate
- 1-3 sentences long
- Phrased the way a real analyst writes, not academic

Return JSON: a list of 25 question strings. No commentary, just the array."""

USER = """Examples of the existing 5 benchmark questions:
1. How is the rapid global expansion of artificial intelligence data centres impacting overall electricity demand and straining existing power grid infrastructure?
2. How is the unprecedented wave of new US liquefied natural gas (LNG) export capacity expected to impact natural gas affordability and spur additional demand in price-sensitive Asian markets by 2030?
3. How are the surge in US electricity demand and the 2025 federal emergency policy interventions collectively affecting the retirement schedules, capacity planning, and generation output of domestic coal-fired power plants?
4. How are the increasing frequency of negative wholesale electricity prices and the regulatory shift towards two-sided Contracts for Difference (CfDs) in Europe altering the revenue expectations and financial agility of developers investing in utility-scale solar PV?
5. How do the tax credit modifications under the US "One Big Beautiful Bill Act" (OBBBA) affect the investment economics of using domestic versus imported feedstocks for Sustainable Aviation Fuel (SAF), and what cascading impact will this biofuel transition have on the capacity rationalisation of traditional US West Coast refineries?

Generate 25 additional questions in the same domain, covering different sectors,
regions, technologies, and policy mechanisms. Avoid duplicating the topics above."""

print("Generating 25 synthetic queries via gpt-4o-mini…")
resp = llm.invoke([SystemMessage(content=SYSTEM), HumanMessage(content=USER)])
content = resp.content.strip()
# Strip markdown fences if present
if content.startswith("```"):
    content = content.split("```", 2)[1]
    if content.startswith("json"):
        content = content[4:]
queries = json.loads(content.strip())

print(f"Generated {len(queries)} queries")
for i, q in enumerate(queries[:5], 1):
    print(f"  [{i}] {q[:100]}{'…' if len(q)>100 else ''}")
print(f"  …")
for i, q in enumerate(queries[-2:], len(queries)-1):
    print(f"  [{i+1}] {q[:100]}{'…' if len(q)>100 else ''}")

with open("synthetic_queries.json", "w") as f:
    json.dump(queries, f, indent=2)
print(f"\nSaved to synthetic_queries.json")
