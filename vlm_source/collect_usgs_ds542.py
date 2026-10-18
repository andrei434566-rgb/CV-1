"""Inventory/download exact-ID USGS DS542 core source files.

No interval photo labels are created. Every downloaded source remains in
quarantine until page depth, physical observations, and Russian wording are
checked against the scanned geologist description.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data/external_raw/usgs_chandeleur_ds542"
TABLE = DEST / "Data_Table.html"
BASE = "https://pubs.usgs.gov/ds/542/HTML/Data_Table.html"
KINDS = {
    "photo_pdf": ("/Table_Data/Core_Photographs_PDF/", ".pdf"),
    "description_jpg": ("/Table_Data/Core_Description_Sheets/", ".jpg"),
    "digitized_log_txt": ("/Table_Data/Digitized_Core_Logs/", ".txt"),
}


def catalog() -> list[dict]:
    source = TABLE.read_text(encoding="latin-1")
    records = []
    for row_html in re.findall(r"<tr\b[^>]*>.*?</tr>", source, re.I | re.S):
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", row_html, re.I | re.S)
        if len(cells) < 5:
            continue
        first = html.unescape(re.sub(r"<[^>]+>", "", cells[0]))
        well_match = re.search(r"07\s*SCC\s*[-_]?\s*(\d{1,3})", first, re.I)
        if not well_match:
            continue
        core_number = f"07SCC{int(well_match.group(1)):02d}"
        assets = {}
        for kind, (fragment, extension) in KINDS.items():
            matches = []
            for raw_url in re.findall(r'href\s*=\s*["\']([^"\']+)["\']', row_html, re.I):
                url = urllib.parse.urljoin(BASE, html.unescape(raw_url))
                parsed = urllib.parse.urlparse(url)
                if parsed.netloc == "pubs.usgs.gov" and fragment in parsed.path and parsed.path.lower().endswith(extension):
                    matches.append(url)
            if matches:
                assets[kind] = sorted(set(matches))[0]
        if all(key in assets for key in KINDS):
            description_path = urllib.parse.urlparse(assets["description_jpg"]).path.upper()
            if "07SCC_04CI" in description_path:
                site = "CI"
            elif "07SCC_05GH" in description_path:
                site = "GH"
            else:
                raise ValueError(f"Unknown site in {description_path}")
            well_id = core_number + site
            records.append({"well_id": well_id, "core_number": core_number,
                            "site": site, "assets": assets,
                            "training_approved": False, "pixel_depth_verified": False,
                            "russian_description_geologist_approved": False})
    ids = [row["well_id"] for row in records]
    if len(ids) != len(set(ids)):
        first_duplicate = next(key for key in ids if ids.count(key) > 1)
        raise ValueError(f"Duplicate core ID {first_duplicate}")
    return sorted(records, key=lambda row: row["well_id"])


def download(url: str, output: Path) -> dict:
    if output.is_file() and output.stat().st_size:
        status = "existing"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        part = output.with_suffix(output.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "CoreResearchInventory/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response, part.open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                if part.stat().st_size < 100:
                    raise ValueError("Unexpectedly short source file")
                part.replace(output)
                status = "downloaded"
                break
            except (OSError, urllib.error.URLError, ValueError) as exc:
                last_error = exc
                time.sleep(attempt + 1)
        else:
            return {"url": url, "status": "error", "error": str(last_error)}
    sha = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return {"url": url, "path": str(output.relative_to(ROOT)).replace("\\", "/"),
            "bytes": output.stat().st_size, "sha256": sha.hexdigest(), "status": status}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    records = catalog()
    if args.limit:
        records = records[:args.limit]
    if args.download:
        if not 1 <= args.workers <= 8:
            raise ValueError("workers must be between 1 and 8")

        def collect(row: dict) -> dict:
            local = DEST / row["well_id"]
            return {
                "photo_pdf": download(row["assets"]["photo_pdf"], local / f'{row["well_id"]}_photo.pdf'),
                "description_jpg": download(row["assets"]["description_jpg"], local / f'{row["well_id"]}_description.jpg'),
                "digitized_log_txt": download(row["assets"]["digitized_log_txt"], local / f'{row["well_id"]}_log.txt'),
            }
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(collect, row): row for row in records}
            for index, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                row["local_assets"] = future.result()
                print(json.dumps({"processed": index, "well_id": row["well_id"],
                                  "status": [item["status"] for item in row["local_assets"].values()]},
                                 ensure_ascii=False), flush=True)
    manifest = {"source_id": "usgs_chandeleur_ds542", "official_page": "https://pubs.usgs.gov/ds/542/",
                "source_table": BASE, "training_approved": False,
                "reason": "Core IDs match, but individual PDF pages have not all been mapped to depth/description intervals",
                "core_count": len(records), "records": records}
    path = DEST / ("download_inventory.json" if args.download else "catalog.json")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"core_count": len(records), "inventory": str(path.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
