"""Collect small official Conconully metadata before large photo downloads."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_conconully_2025"
ITEM_URL = "https://www.sciencebase.gov/catalog/item/682bc184d4be024992ebdd66?format=json"
OFFICIAL_PAGE = ("https://www.usgs.gov/data/sediment-computed-tomography-ct-scans-"
                 "photographs-and-core-log-descriptions-conconully")
SMALL = {"Conconully_sediment_core_field_data.csv",
         "Conconully_sediment_cores_initial_descriptions.csv",
         "CRSE2_core_photos.pdf",
         "Conconully_sediment_cores_metadata.xml"}


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CoreSed research downloader/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(get(ITEM_URL))
    files = []
    for item in metadata["files"]:
        name = item["name"]
        if name not in SMALL:
            continue
        if item.get("size", 0) > 1_000_000:
            raise ValueError(f"Unexpectedly large small-file candidate: {name}")
        raw = get(item["downloadUri"])
        if len(raw) != item["size"]:
            raise ValueError(f"Size mismatch for {name}: {len(raw)} != {item['size']}")
        (ROOT / name).write_bytes(raw)
        files.append({"name": name, "url": item["downloadUri"],
                      "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    if {r["name"] for r in files} != SMALL:
        raise ValueError("Not all four small assets were retrieved")
    result = {"official_page": OFFICIAL_PAGE, "doi": "10.5066/P14DXYOD",
              "license": "CC0 1.0 Universal on official USGS data release",
              "sciencebase_item": ITEM_URL, "files": files,
              "larger_photo_assets_not_downloaded": [
                  {"name": f["name"], "bytes": f["size"], "url": f["downloadUri"]}
                  for f in metadata["files"] if f["name"] in {
                      "VC1_sediment_core_photo.pdf", "VC4_sediment_core_photo_sketch_description.pdf"}],
              "training_approved": False}
    (ROOT / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"downloaded": len(files), "bytes": sum(r["bytes"] for r in files),
                      "large_assets": result["larger_photo_assets_not_downloaded"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
