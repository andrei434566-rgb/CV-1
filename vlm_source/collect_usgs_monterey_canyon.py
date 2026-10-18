"""Download the official CC0 Monterey Canyon core photographs and graphic logs."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

from collect_usgs_palos_verdes import BASE, get_json, sha256


ROOT_ID = "572a72cfe4b0b13d391a48b2"
CHILD_IDS = ("5f00c3f782ce0afb2445d6c3", "5f00c50d82ce0afb2445d6d1")
WANTED = {"NorthernFlankMontereyCanyonCores_Photos.zip", "NorthernFlankMontereyCanyonCores_Figures.pdf"}
OUT = Path(__file__).parent / "data" / "external_raw" / "usgs_monterey_canyon"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = {
        "source_url": "https://www.usgs.gov/data/sediment-core-data-northern-flank-monterey-canyon-offshore-california",
        "sciencebase_parent_url": f"{BASE}{ROOT_ID}",
        "doi": "10.5066/F7J67F2K",
        "rights": "CC0 1.0 Universal on official USGS release",
        "training_approved": False,
        "reason": "Graphic-log intervals must be aligned to each photo section and checked in Russian.",
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
                raise RuntimeError(f"Size mismatch for {name}")
            entry = {"name": name, "source_child_url": f"{BASE}{child_id}", "url": item["downloadUri"], "bytes": path.stat().st_size, "sha256": sha256(path)}
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    broken = archive.testzip()
                    if broken:
                        raise RuntimeError(f"Broken ZIP member: {broken}")
                    entry["archive_members"] = len(archive.infolist())
                    entry["photo_members"] = [m.filename for m in archive.infolist() if m.filename.lower().endswith(".bmp") and not m.filename.startswith("__MACOSX/")]
            elif path.open("rb").read(5) != b"%PDF-":
                raise RuntimeError(f"Invalid PDF signature: {name}")
            inventory["files"].append(entry)
            print(f"{name}: {path.stat().st_size:,} bytes", flush=True)
    if {item["name"] for item in inventory["files"]} != WANTED:
        raise RuntimeError("A required official source file was missing")
    (OUT / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
