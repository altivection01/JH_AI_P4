"""Re-parse each PDF with Docling to get per-page chart locations.

Output: chart_pages.json with {source: [page_num, ...]}
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from docling.document_converter import DocumentConverter


pdfs = sorted(Path("IEAReports").glob("*.pdf"))
manifest: dict[str, list[int]] = {}
converter = DocumentConverter()

for pdf in pdfs:
    t0 = time.perf_counter()
    print(f"[{pdf.stem}] parsing…", flush=True)
    result = converter.convert(str(pdf))
    doc = result.document
    pages = sorted({
        p.prov[0].page_no for p in doc.pictures if p.prov
    }) if hasattr(doc, "pictures") else []
    manifest[pdf.stem] = pages
    print(f"  {len(pages)} unique chart-bearing pages  ({time.perf_counter()-t0:.0f}s)",
          flush=True)

with open("chart_pages.json", "w") as f:
    json.dump(manifest, f, indent=2)

total = sum(len(v) for v in manifest.values())
print(f"\nTotal chart-bearing pages across corpus: {total}")
print(f"Estimated VLM cost (~$0.01/page): ~${total*0.01:.2f}")
print(f"Estimated time (~9s/page): ~{total*9/60:.0f} min")
