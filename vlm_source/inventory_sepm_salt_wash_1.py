"""Inventory the official SEPM Salt Wash 1 research download.

These files are research references; no training examples are created here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data/external_raw/sepm_exxon_salt_wash_1_research"
SOURCES = {
    "source_core_description.png": "https://sepm.org/exxon/morrison/files/1643257442303.cu0cgg18luhb73xr.png",
    "SW1_section1.jpg": "https://sepm.org/files/SW1section1.inx5auio6uc31rxm.jpg",
}


def main() -> None:
    files = []
    for name, url in SOURCES.items():
        path = DEST / name
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        with Image.open(path) as im:
            width, height = im.size
        files.append({
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "source_url": url,
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "sha256": digest.hexdigest(),
            "training_approved": False,
        })
    inventory = {
        "source_id": "sepm_exxon_salt_wash_1",
        "page": "https://sepm.org/exxon/morrison/salt-wash-1",
        "rights_page": "https://www.sepm.org/exxon/core-data-permissions",
        "scope": "research reference only; no interval image-description pairs verified",
        "poster_pdf_status": "Official poster link returned HTTP 404 on 2026-10-13",
        "training_approved": False,
        "files": files,
    }
    (DEST / "inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"files": len(files), "total_bytes": sum(f["bytes"] for f in files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
