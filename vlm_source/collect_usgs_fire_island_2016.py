"""Collect official Fire Island 2016 core photos and logs for review.

This collector only establishes source provenance and archive integrity.
Matching core IDs do not constitute approved photo-description training pairs.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_fire_island_2016"
SOURCE = "https://www.usgs.gov/data/archive-sediment-data-vibracores-collected-2016-fire-island-new-york"
BASE = "https://coastal.er.usgs.gov/data-release/doi-F7FN15GX/data/"
NAMES = ("2016-322-FA_CorePhotos.zip", "2016-322-FA_CoreLogs.zip")
BLOCK = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for piece in iter(lambda: stream.read(4 * BLOCK), b""):
            digest.update(piece)
    return digest.hexdigest()


def download(name: str) -> dict:
    url = BASE + name
    destination = ROOT / name
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-15"})
    with urllib.request.urlopen(request, timeout=35) as response:
        match = re.fullmatch(r"bytes 0-15/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match or len(response.read()) != 16:
            raise RuntimeError(f"Range probe failed: {url}")
        expected = int(match.group(1))
    if not destination.exists() or destination.stat().st_size != expected:
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists() and partial.stat().st_size % BLOCK:
            partial.unlink()
        start = partial.stat().st_size if partial.exists() else 0
        if start > expected:
            partial.unlink()
            start = 0
        with partial.open("ab") as output:
            while start < expected:
                end = min(start + BLOCK, expected) - 1
                request = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    chunk = response.read()
                    if (response.status != 206
                            or response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected}"
                            or len(chunk) != end - start + 1):
                        raise RuntimeError(f"Range response mismatch: {name} {start}-{end}")
                    output.write(chunk)
                start = end + 1
                print(f"{name}: {start}/{expected}", flush=True)
        partial.replace(destination)
    with zipfile.ZipFile(destination) as archive:
        broken = archive.testzip()
        if broken:
            raise RuntimeError(f"ZIP CRC failure: {name}/{broken}")
        members = [item.filename for item in archive.infolist() if not item.is_dir()]
    return {"url": url, "bytes": expected, "sha256": sha256(destination),
            "zip_crc_ok": True, "members": members}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    files = {name: download(name) for name in NAMES}
    inventory = {"source_url": SOURCE, "doi": "10.5066/F7FN15GX",
                 "license": "CC0 1.0 Universal on USGS data release",
                 "files": files, "training_approved": False,
                 "reason": "No photo-to-description interval alignment or visual QA yet."}
    (ROOT / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {"bytes": data["bytes"], "members": len(data["members"])}
                      for name, data in files.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
