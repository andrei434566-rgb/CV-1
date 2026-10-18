"""Download and inventory the USGS Kootenai River core photographs.

The Appendix A PDF links each photograph from a lithologic strip log.  The
links establish the core, but do not by themselves establish the exact depth
or lithologic description of an individual photograph.  This script therefore
stages source files for review and never emits training-approved labels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data" / "external_raw" / "usgs_2011_kootenai_river"
PDF = SOURCE_DIR / "sir20115006_appendixA.pdf"
INVENTORY = SOURCE_DIR / "photo_inventory.json"
SOURCE_PAGE = "https://pubs.usgs.gov/sir/2011/5006/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def photo_links() -> list[dict]:
    reader = PdfReader(PDF)
    found: dict[str, dict] = {}
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        well = re.search(r"\bTB-\d+\b", text)
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            uri = action.get("/URI")
            if not uri:
                continue
            parsed = urlparse(str(uri))
            if (parsed.hostname != "pubs.usgs.gov"
                    or not parsed.path.startswith("/sir/2011/5006/appendixes/")
                    or not parsed.path.endswith(".jpg")):
                continue
            url = f"https://pubs.usgs.gov{parsed.path}"
            if url not in found:
                found[url] = {
                    "url": url,
                    "well_id": well.group(0) if well else None,
                    "pdf_page": page_index + 1,
                    "source_pdf": str(PDF),
                    "source_page": SOURCE_PAGE,
                    "training_approved": False,
                    "interval_status": "not_aligned_to_photo",
                }
    return list(found.values())


def download(row: dict) -> dict:
    parsed = urlparse(row["url"])
    relative = Path(*parsed.path.split("/")[5:])
    path = SOURCE_DIR / "photos" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        request = Request(row["url"], headers={"User-Agent": "CoreSedTrainer/0.1 (USGS research data inventory)"})
        with urlopen(request, timeout=45) as response:
            content_type = response.headers.get("Content-Type", "")
            if "image/jpeg" not in content_type:
                raise ValueError(f"Unexpected content type for {row['url']}: {content_type}")
            data = response.read()
        temporary = path.with_suffix(".jpg.part")
        temporary.write_bytes(data)
        temporary.replace(path)
    row = dict(row)
    row["path"] = str(path)
    row["bytes"] = path.stat().st_size
    row["sha256"] = sha256_file(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    rows = photo_links()
    completed: list[dict] = []
    errors: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, row): row for row in rows}
        for future in as_completed(futures):
            try:
                completed.append(future.result())
            except Exception as exc:
                errors.append({"url": futures[future]["url"], "error": str(exc)})
    completed.sort(key=lambda row: (row["pdf_page"], row["url"]))
    INVENTORY.write_text(
        json.dumps({"source_page": SOURCE_PAGE, "source_pdf_sha256": sha256_file(PDF), "photos": completed, "errors": errors},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"linked_photos": len(rows), "downloaded": len(completed), "errors": len(errors),
                      "bytes": sum(row["bytes"] for row in completed), "inventory": str(INVENTORY)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
