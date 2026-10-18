"""Download and inventory the official USGS Kootenai River core appendix."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_kootenai_river_2011"
URL = "https://pubs.usgs.gov/sir/2011/5006/pdf/sir20115006_appendixA.pdf"
NAME = "sir20115006_appendixA.pdf"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / NAME
    if not path.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "CoreAI-source-research/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-") or len(raw) < 100_000:
        raise ValueError("Downloaded file is not the expected PDF")
    result = {
        "source_report": "https://pubs.usgs.gov/publication/sir20115006",
        "source_pdf_url": URL,
        "doi": "10.3133/sir20115006",
        "publication": "USGS Scientific Investigations Report 2011-5006, Appendix A: Lithologic Strip Logs With Sediment Sampling Depth Interval for Chemical Analysis and Photographs of Sediment Cores",
        "local_pdf": NAME,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "training_approved": False,
        "warning": "Do not infer exact photo-interval labels from page-level co-location; inspect image resolution, depth scale and source captions first.",
    }
    (ROOT / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"bytes": result["bytes"], "sha256": result["sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
