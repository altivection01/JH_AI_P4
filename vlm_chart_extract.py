"""Stage 2: VLM-based chart data extraction from chart-bearing pages.

After Docling re-ingest, each markdown file has `<!-- image -->` placeholders
where the original PDF had a figure (chart, table image, photo). We:

  1. Find every <!-- image --> marker
  2. Map back to the nearest PDF page via the 'PAGE | N' anchors Docling preserves
  3. Deduplicate to unique pages
  4. Render each unique page as a PNG (via PyMuPDF)
  5. Send each page image to GPT-4o vision with: "Extract any chart data on
     this page as a structured markdown table; if there are no data charts,
     return 'NONE'"
  6. Store non-empty extractions as synthetic chunks tagged
     [Source p.N — chart data extract]

These synthetic chunks then get added to the same Chroma collection so the
retriever can surface chart data alongside the surrounding prose.

Cost estimate: ~100-200 unique chart pages × ~$0.01-0.02 per vision call
= ~$1-3 total across all 5 PDFs.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import pymupdf
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


MD_DIR = Path("docling_md")
CHARTS_DIR = Path("chart_extracts")
CHARTS_DIR.mkdir(exist_ok=True)
PAGE_IMAGES_DIR = Path("page_images")
PAGE_IMAGES_DIR.mkdir(exist_ok=True)


def page_anchors_from_md(md_text: str) -> list[tuple[int, int]]:
    """Return [(char_offset, page_num)] from Docling-preserved 'PAGE | N'."""
    anchors = []
    for m in re.finditer(r"PAGE\s*\|\s*(\d+)", md_text):
        try:
            anchors.append((m.start(), int(m.group(1))))
        except ValueError:
            pass
    return anchors


def page_for_offset(anchors: list[tuple[int, int]], offset: int) -> int:
    page = 0
    for o, p in anchors:
        if o > offset:
            break
        page = p
    return page


def find_chart_pages(source: str) -> list[int]:
    """Return the deduped list of PDF pages that contain charts/figures.

    Reads from chart_pages.json which is built by build_chart_manifest.py
    using Docling's structured DocumentConverter result (which preserves
    per-picture page metadata that the markdown export strips).
    """
    manifest_path = Path("chart_pages.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        return manifest.get(source, [])
    # Fallback: parse markdown (will fail to find pages since Docling strips
    # the IEA-style PAGE | N footers as noise).
    return []


def render_page_png(pdf_path: Path, page_num_pyMuPDF_index: int,
                    dpi: int = 144) -> bytes:
    """Render one PDF page at given DPI; return PNG bytes."""
    doc = pymupdf.open(str(pdf_path))
    page = doc[page_num_pyMuPDF_index]
    mat = pymupdf.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png = pix.tobytes("png")
    doc.close()
    return png


CHART_EXTRACT_SYS = """\
You are an analyst extracting numerical data from PDF page images.

The user will send you one page from an IEA energy market report. Your job:

  - If the page contains a DATA CHART (line chart, bar chart, stacked column,
    pie chart, time-series plot), extract the visible data into a structured
    markdown table. Include axis labels, units, time periods, regional or
    sectoral categories, and as many numerical data points as you can read.
  - If the page has multiple charts, output one table per chart, separated
    by chart titles.
  - Always include the chart's title/caption if visible.
  - If the page has NO data chart (just text, logos, decorative images,
    photos, or non-quantitative figures), respond exactly:
        NONE

Return ONLY the markdown tables (or NONE). No explanatory prose around them.
"""


def extract_chart_data(llm: ChatOpenAI, png_bytes: bytes,
                        source: str, page_num: int) -> str:
    """Send one page image to GPT-4o vision; return extracted markdown."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    resp = llm.invoke([
        SystemMessage(content=CHART_EXTRACT_SYS),
        HumanMessage(content=[
            {"type": "text",
             "text": f"This is page {page_num} of {source}.pdf. "
                     "Extract any chart data as markdown tables, or reply NONE."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]),
    ])
    return resp.content.strip()


def run_extraction(pdf_dir: Path, llm: ChatOpenAI,
                    throttle_seconds: float = 4.0,
                    limit: int | None = None) -> dict:
    """Process all chart-bearing pages across all PDFs."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    all_extracts: list[dict] = []

    for pdf in pdfs:
        source = pdf.stem
        chart_pages = find_chart_pages(source)
        if limit is not None:
            chart_pages = chart_pages[:limit]
        print(f"\n[{source}] {len(chart_pages)} chart-bearing pages")

        cache_path = CHARTS_DIR / f"{source}.json"
        existing = json.load(open(cache_path)) if cache_path.exists() else []
        done_pages = {e["page"] for e in existing}

        for i, page_n in enumerate(chart_pages):
            if page_n in done_pages:
                continue
            try:
                png = render_page_png(pdf, page_n - 1)  # PyMuPDF 0-indexed
            except Exception as e:
                print(f"  page {page_n}: render error {e}")
                continue
            # Save PNG for inspection / future use
            img_path = PAGE_IMAGES_DIR / f"{source}_p{page_n}.png"
            if not img_path.exists():
                img_path.write_bytes(png)

            try:
                t0 = time.perf_counter()
                md_table = extract_chart_data(llm, png, source, page_n)
                wall = time.perf_counter() - t0
                kind = "NONE" if md_table.strip().upper() == "NONE" else (
                    f"{md_table.count('|---')} table(s)"
                    if "|---" in md_table else f"{len(md_table)} chars")
                print(f"  p.{page_n}: {kind}  ({wall:.1f}s)")
                existing.append({
                    "page": page_n,
                    "extracted": md_table,
                    "wall_seconds": round(wall, 2),
                })
            except Exception as e:
                print(f"  p.{page_n}: extraction error {e}")
                existing.append({
                    "page": page_n,
                    "error": str(e),
                })
            # Persist after every page (checkpoint)
            with open(cache_path, "w") as f:
                json.dump(existing, f, indent=2)
            time.sleep(throttle_seconds)

        all_extracts.extend(existing)

    return {
        "total_pages_processed": len(all_extracts),
        "pages_with_data": sum(
            1 for e in all_extracts
            if "extracted" in e and e["extracted"].strip().upper() != "NONE"
        ),
        "pages_no_data": sum(
            1 for e in all_extracts
            if "extracted" in e and e["extracted"].strip().upper() == "NONE"
        ),
        "pages_errored": sum(1 for e in all_extracts if "error" in e),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit pages per PDF (for smoke testing)")
    args = parser.parse_args()

    keys = json.load(open("config.json"))
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=keys["OPENAI_API_KEY"],
        temperature=0,
        max_retries=10,
        timeout=180,
    )

    pdf_dir = Path("IEAReports")
    summary = run_extraction(pdf_dir, llm, throttle_seconds=4.0,
                              limit=args.limit)
    print(f"\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
