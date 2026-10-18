"""Collect the small, openly licensed lithology/description parts of IODP LILY v2.

Raw files are source material, not automatically approved photo-description pairs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


RECORD_ID = "10425539"
API = f"https://zenodo.org/api/records/{RECORD_ID}"
FILES = ("CleanLITH.zip", "RawLITH.zip", "RawDESC.zip")
OUT = Path(__file__).parent / "data" / "external_raw" / "iodp_lily_v2"


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(API, timeout=60) as response:
        record = json.load(response)
    license_id = record.get("metadata", {}).get("license", {}).get("id")
    if license_id != "cc-by-4.0":
        raise RuntimeError(f"Unexpected license {license_id!r}; inspect source before download")
    files_by_name = {item["key"]: item for item in record["files"]}
    inventory = {
        "source": "IODP LILY v2",
        "record_url": f"https://zenodo.org/records/{RECORD_ID}",
        "api_url": API,
        "doi": record.get("doi"),
        "license": license_id,
        "training_approved": False,
        "reason": "Lithology text is not aligned to photo pixels or reviewed in Russian.",
        "files": [],
    }
    for name in FILES:
        source = files_by_name[name]
        target = OUT / name
        if not target.exists() or target.stat().st_size != source["size"]:
            request = urllib.request.Request(source["links"]["self"], headers={"User-Agent": "CoreSedTrainer/0.1 public research collection"})
            with urllib.request.urlopen(request, timeout=180) as response, target.open("wb") as dest:
                while chunk := response.read(1024 * 1024):
                    dest.write(chunk)
        expected_md5 = source["checksum"].split(":", 1)[1]
        actual_md5 = digest(target, "md5")
        if actual_md5 != expected_md5:
            raise RuntimeError(f"Checksum mismatch for {name}")
        with zipfile.ZipFile(target) as archive:
            broken = archive.testzip()
            if broken:
                raise RuntimeError(f"Broken ZIP member: {broken}")
            members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
            headers = []
            for member in members[:3]:
                with archive.open(member) as raw:
                    first = raw.readline().decode("utf-8-sig", errors="replace")
                headers.append({"member": member.filename, "header": next(csv.reader([first]))})
        inventory["files"].append({
            "name": name,
            "url": source["links"]["self"],
            "bytes": target.stat().st_size,
            "md5": actual_md5,
            "sha256": digest(target, "sha256"),
            "zip_csv_members": len(members),
            "sample_headers": headers,
        })
        print(f"{name}: {target.stat().st_size:,} bytes, {len(members)} CSV members")
    (OUT / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT / "inventory.json")


if __name__ == "__main__":
    main()
