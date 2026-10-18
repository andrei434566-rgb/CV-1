"""Fetch photo URLs embedded in the USGS Kootenai River core-log PDF.

The PDF has selectable lithology text and links to separate JPEG photographs.
Joining on PDF page/core ID is not an exact interval-to-pixel annotation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_kootenai_river_2011"
PDF = ROOT / "sir20115006_appendixA.pdf"
PHOTO_DIR = ROOT / "photos"
PAGE_INDEX = ROOT / "photo_link_index.json"
DOWNLOAD_INDEX = ROOT / "photo_download_inventory.json"
CORE_ID = re.compile(r"^\s*(TB-\d+(?:-\d+)?)\s*$", re.MULTILINE)
PRINT_LOCK = threading.Lock()
BASE = "https://pubs.usgs.gov/sir/2011/5006/appendixes"
# Six PDF links contain a wrong year/path or lack the JPEG extension. Each
# replacement was checked against the official USGS host before use.
CORRECTIONS = {
    **{f"https://pubs.usgs.gov/sir/2010/5006/appendixes/TB2-A2{suffix}.jpg":
       f"{BASE}/sir2011-5006_FigureA2_TB19/TB19-A2{suffix}.jpg"
       for suffix in ("d1", "d2", "e1", "e2")},
    f"{BASE}/sir2011-5006_FigureA18_TB3/TB3-A18i2":
        f"{BASE}/sir2011-5006_FigureA18_TB3/TB3-A18i2.jpg",
    f"{BASE}/sir2011-5006_FigureA4_TB17/TB17-A4e2":
        f"{BASE}/sir2011-5006_FigureA4_TB17/TB17-A4e2.jpg",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_index() -> dict:
    source = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    if sha256_file(PDF) != source["sha256"]:
        raise ValueError("Source PDF differs from download inventory")
    pages = []
    references: dict[str, dict] = {}
    for idx, page in enumerate(PdfReader(PDF).pages, 1):
        text = page.extract_text() or ""
        match = CORE_ID.search(text)
        core_id = match.group(1) if match else None
        urls = []
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A", {})
            raw_url = action.get("/URI")
            if not raw_url:
                continue
            url = str(raw_url).replace("http://pubs.usgs.gov/", "https://pubs.usgs.gov/", 1)
            parsed = urllib.parse.urlparse(url)
            if parsed.hostname != "pubs.usgs.gov" or parsed.scheme != "https":
                raise ValueError(f"Unexpected linked host: {url}")
            urls.append(url)
            if url not in references:
                references[url] = {"url": url, "pdf_pages": [], "core_ids": []}
            references[url]["pdf_pages"].append(idx)
            if core_id and core_id not in references[url]["core_ids"]:
                references[url]["core_ids"].append(core_id)
        pages.append({"pdf_page": idx, "core_id": core_id,
                      "source_text_en": text, "photo_links": urls,
                      "photo_interval_aligned": False, "training_approved": False})
    result = {"source_pdf": PDF.name, "source_pdf_sha256": source["sha256"],
              "pages": pages, "unique_photo_links": list(references.values()),
              "summary": {"pages": len(pages), "pages_with_core_id": sum(p["core_id"] is not None for p in pages),
                          "links": sum(len(p["photo_links"]) for p in pages),
                          "unique_links": len(references)},
              "warning": "The link and text share a page, but individual image intervals and visible labels need separate verification."}
    PAGE_INDEX.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def local_name(url: str) -> str:
    stem = Path(urllib.parse.urlparse(url).path).name
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)
    if not stem.lower().endswith((".jpg", ".jpeg")):
        stem += ".jpg"
    return f"{hashlib.sha256(url.encode()).hexdigest()[:12]}_{stem}"


def fetch(record: dict) -> dict:
    url = record["url"]
    target = PHOTO_DIR / local_name(url)
    result = {**record, "local_file": str(target.relative_to(ROOT)).replace("\\", "/"),
              "photo_interval_aligned": False, "training_approved": False}
    last_error = None
    for candidate_url in (url, *([CORRECTIONS[url]] if url in CORRECTIONS else [])):
        for attempt in range(3):
            try:
                if target.is_file():
                    raw = target.read_bytes()
                else:
                    request = urllib.request.Request(candidate_url, headers={"User-Agent": "CoreAI-source-research/1.0"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if urllib.parse.urlparse(response.geturl()).hostname != "pubs.usgs.gov":
                            raise ValueError("Redirected outside official USGS host")
                        raw = response.read(15_000_001)
                    if len(raw) > 15_000_000:
                        raise ValueError("Unexpectedly large individual photo")
                if not raw.startswith(b"\xff\xd8"):
                    raise ValueError("Response is not JPEG")
                with Image.open(io.BytesIO(raw)) as opened:
                    opened.verify()
                with Image.open(io.BytesIO(raw)) as opened:
                    width, height = opened.size
                if not target.is_file():
                    part = target.with_suffix(target.suffix + ".part")
                    part.write_bytes(raw)
                    part.replace(target)
                result.update({"status": "downloaded", "retrieved_url": candidate_url,
                               "source_link_corrected": candidate_url != url,
                               "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                               "size_px": [width, height]})
                return result
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
                last_error = error
                if attempt < 2:
                    time.sleep(1 + attempt)
    result.update({"status": "failed", "error": f"{type(last_error).__name__}: {last_error}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    index = extract_index()
    print(json.dumps(index["summary"], ensure_ascii=False), flush=True)
    if args.index_only:
        return
    PHOTO_DIR.mkdir(exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(fetch, record) for record in index["unique_photo_links"]]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if i % 25 == 0 or i == len(futures):
                with PRINT_LOCK:
                    print(f"{i}/{len(futures)} links; failed={sum(r['status'] == 'failed' for r in results)}", flush=True)
    results.sort(key=lambda result: result["url"])
    inventory = {"source_pdf_sha256": index["source_pdf_sha256"], "photos": results,
                 "summary": {"unique_links": len(results),
                             "downloaded": sum(r["status"] == "downloaded" for r in results),
                             "failed": sum(r["status"] == "failed" for r in results),
                             "downloaded_bytes": sum(r.get("bytes", 0) for r in results)},
                 "approved_training_pairs": 0}
    DOWNLOAD_INDEX.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
