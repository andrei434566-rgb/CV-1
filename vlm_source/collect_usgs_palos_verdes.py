"""Download the official CC0 Palos Verdes core photos and graphic-log PDF."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


BASE = "https://www.sciencebase.gov/catalog/item/"
ROOT_ID = "5547a069e4b0a658d794d1f1"
CHILD_IDS = (
    "5efe1bb782ce3fd7e8a81283",  # photographs
    "5efe1b7f82ce3fd7e8a811f1",  # graphically described logs
)
WANTED = {"PalosVerdesCores_Photos.zip", "PalosVerdesCores_Figures.pdf"}
OUT = Path(__file__).parent / "data" / "external_raw" / "usgs_palos_verdes"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent = get_json(f"{BASE}{ROOT_ID}?format=json")
    inventory = {
        "source": parent["title"],
        "source_url": f"https://www.usgs.gov/data/sediment-core-data-offshore-palos-verdes-california",
        "sciencebase_parent_url": f"{BASE}{ROOT_ID}",
        "doi": "10.5066/F7G44NB6",
        "rights": "CC0 1.0 Universal stated on official USGS data release",
        "training_approved": False,
        "reason": "Photo/log depth and target text need exact alignment and expert review.",
        "files": [],
    }
    for child_id in CHILD_IDS:
        child = get_json(f"{BASE}{child_id}?format=json")
        for item in child.get("files", []):
            name = item["name"]
            if name not in WANTED:
                continue
            path = OUT / name
            if not path.exists() or path.stat().st_size != item["size"]:
                request = urllib.request.Request(item["downloadUri"], headers={"User-Agent": "CoreSedTrainer/0.1 public source collection"})
                with urllib.request.urlopen(request, timeout=300) as response, path.open("wb") as target:
                    while chunk := response.read(4 * 1024 * 1024):
                        target.write(chunk)
            if path.stat().st_size != item["size"]:
                raise RuntimeError(f"Download size mismatch for {name}")
            entry = {
                "name": name,
                "source_child_url": f"{BASE}{child_id}",
                "url": item["downloadUri"],
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            if name.endswith(".zip"):
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad:
                        raise RuntimeError(f"Corrupt ZIP member {bad}")
                    entry["archive_members"] = len(archive.infolist())
                    entry["photo_members"] = [m.filename for m in archive.infolist() if m.filename.lower().endswith(".bmp")]
            else:
                if path.open("rb").read(5) != b"%PDF-":
                    raise RuntimeError(f"Invalid PDF signature {name}")
            inventory["files"].append(entry)
            print(f"{name}: {path.stat().st_size:,} bytes")
    if {item["name"] for item in inventory["files"]} != WANTED:
        raise RuntimeError("Official release did not provide all desired files")
    (OUT / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "inventory.json")


if __name__ == "__main__":
    main()
