"""Collect a small licensed IODP Expedition 359 photo pilot for LILY alignment."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path


RECORD_ID = "14015049"
API = f"https://zenodo.org/api/records/{RECORD_ID}"
NAMES = ("COREPHOTO-README.txt", "data_by_hole.zip", "U1465B.zip", "U1466B.zip", "U1469B.zip")
OUT = Path(__file__).parent / "data" / "external_raw" / "iodp_359_photo_pilot"


def digest(path: Path, name: str) -> str:
    h = hashlib.new(name)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as response:
        record = json.load(response)
    license_id = record.get("metadata", {}).get("license", {}).get("id")
    if license_id != "cc-by-4.0":
        raise RuntimeError(f"Unexpected license {license_id!r}")
    available = {row["key"]: row for row in record["files"]}
    result = {
        "source": "IODP Expedition 359 Core composite images",
        "source_url": f"https://zenodo.org/records/{RECORD_ID}",
        "doi": record.get("doi"),
        "license": license_id,
        "candidate_description_source": "https://zenodo.org/records/10425539 (LILY v2 CleanLITH, expedition 359)",
        "training_approved": False,
        "files": [],
    }
    for name in NAMES:
        row = available[name]
        target = OUT / name
        if not target.exists() or target.stat().st_size != row["size"]:
            request = urllib.request.Request(row["links"]["self"], headers={"User-Agent": "CoreSedTrainer/0.1 public source collection"})
            with urllib.request.urlopen(request, timeout=180) as response, target.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        expected = row["checksum"].split(":", 1)[1]
        if target.stat().st_size != row["size"] or digest(target, "md5") != expected:
            raise RuntimeError(f"Source size or MD5 mismatch: {name}")
        entry = {"name": name, "bytes": row["size"], "url": row["links"]["self"], "md5": expected, "sha256": digest(target, "sha256")}
        if name.endswith(".zip"):
            with zipfile.ZipFile(target) as archive:
                broken = archive.testzip()
                if broken:
                    raise RuntimeError(f"Broken ZIP member {broken}")
                entry["archive_members"] = archive.namelist()
        result["files"].append(entry)
        print(name, row["size"], flush=True)
    (OUT / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
