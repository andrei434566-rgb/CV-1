"""Fetch a small, checksum-verified IODP 366 pilot from Zenodo.

The files are kept as primary-source archives. This script creates no
training labels or image crops.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "external_raw" / "iodp_366_core_photos"
RECORD = "https://zenodo.org/records/19614224"
API = "https://zenodo.org/api/records/19614224/files"
FILES = {
    "COREPHOTO-README.txt": "ca08eb151f420965fb5b0838eece16bf",
    "data_by_hole.zip": "edb1cc36885abd197ab64a49b2d711ab",
    "U1491A.zip": "b9058d8e70e9724b38ea0f94068d4e14",
    "U1493A.zip": "3ce0387453f430bc812001484fc521d1",
    "U1495A.zip": "858fdc8084b0b589d27a310f2bac1fc0",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    entries = []
    for name, expected_md5 in FILES.items():
        target = DEST / name
        if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != expected_md5:
            temp = DEST / (name + ".partial")
            digest = hashlib.md5()
            with session.get(f"{API}/{name}/content", stream=True, timeout=120) as response:
                response.raise_for_status()
                with temp.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            out.write(chunk)
                            digest.update(chunk)
            if digest.hexdigest() != expected_md5:
                temp.unlink(missing_ok=True)
                raise ValueError(f"MD5 mismatch for {name}: {digest.hexdigest()}")
            temp.replace(target)
        entries.append({
            "name": name,
            "url": f"{API}/{name}/content",
            "bytes": target.stat().st_size,
            "md5": expected_md5,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        })
        print(f"{name}: {target.stat().st_size:,} bytes, MD5 OK", flush=True)
    inventory = {
        "source": RECORD,
        "doi": "10.5281/zenodo.19614224",
        "download_date": date.today().isoformat(),
        "rights": "IODP JRSO expedition photos: CC BY 4.0 except where otherwise noted; attribution required (https://www.iodp.tamu.edu/about/copyright.html). Zenodo record displays no per-record license; verify before downstream release.",
        "training_approved": False,
        "files": entries,
    }
    (DEST / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
