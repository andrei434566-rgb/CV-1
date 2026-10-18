"""Stream the official VC4 photo+sketch PDF with size and SHA-256 audit."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_conconully_2025"
NAME = "VC4_sediment_core_photo_sketch_description.pdf"


def main() -> None:
    inventory_path = ROOT / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    matches = [row for row in inventory["larger_photo_assets_not_downloaded"] if row["name"] == NAME]
    if len(matches) != 1:
        raise ValueError("VC4 PDF not found in official metadata inventory")
    item = matches[0]
    expected = int(item["bytes"])
    final = ROOT / NAME
    part = ROOT / (NAME + ".part")
    if final.is_file() and final.stat().st_size == expected:
        print("Already downloaded; verifying checksum")
    else:
        offset = part.stat().st_size if part.is_file() else 0
        if offset >= expected:
            raise ValueError("Partial file is at least as large as expected")
        headers = {"User-Agent": "CoreSed research downloader/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(item["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and response.status != 206:
                offset = 0
                mode = "wb"
            else:
                mode = "ab" if offset else "wb"
            with part.open(mode) as destination:
                downloaded = offset
                next_report = ((downloaded // 50_000_000) + 1) * 50_000_000
                while True:
                    block = response.read(4 * 1024 * 1024)
                    if not block:
                        break
                    destination.write(block)
                    downloaded += len(block)
                    if downloaded >= next_report:
                        print(f"Downloaded {downloaded}/{expected} bytes", flush=True)
                        next_report += 50_000_000
        if part.stat().st_size != expected:
            raise ValueError(f"Incomplete PDF: {part.stat().st_size}/{expected} bytes; partial kept for resume")
        part.replace(final)
    digest = hashlib.sha256()
    with final.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    inventory["downloaded_photo"] = {"name": NAME, "bytes": expected,
                                     "sha256": digest.hexdigest(), "source_url": item["url"]}
    inventory["larger_photo_assets_not_downloaded"] = [
        row for row in inventory["larger_photo_assets_not_downloaded"] if row["name"] != NAME]
    inventory_path.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory["downloaded_photo"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
