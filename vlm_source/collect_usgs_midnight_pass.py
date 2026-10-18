"""Collect the official CC0 Midnight Pass photo/log archives with checksums."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_midnight_pass"
SOURCE = "https://www.usgs.gov/data/sediment-data-cores-collected-2025-midnight-pass-florida"
BASE = "https://coastal.er.usgs.gov/data-release/doi-P1CQ6QRR/data/"
NAMES = ("2024-330-FA_MP_CorePhotos.zip", "2024-330-FA_MP_CoreLogs.zip")


def download(name: str) -> dict:
    destination = ROOT / name
    url = BASE + name
    probe = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-15"})
    with urllib.request.urlopen(probe, timeout=30) as response:
        match = re.fullmatch(r"bytes 0-15/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match or len(response.read()) != 16:
            raise RuntimeError(f"Range probe failed: {name}")
        expected_bytes = int(match.group(1))
    if not destination.exists() or destination.stat().st_size != expected_bytes:
        part = destination.with_suffix(destination.suffix + ".part")
        if part.exists() and part.stat().st_size % (1024 * 1024) != 0:
            part.unlink()
        start = part.stat().st_size if part.exists() else 0
        if start > expected_bytes:
            part.unlink()
            start = 0
        with part.open("ab") as out:
            while start < expected_bytes:
                end = min(start + 1024 * 1024, expected_bytes) - 1
                request = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    expected_range = f"bytes {start}-{end}/{expected_bytes}"
                    chunk = response.read()
                    if (response.status != 206 or response.headers.get("Content-Range") != expected_range
                            or len(chunk) != end - start + 1):
                        raise RuntimeError(f"Range response mismatch: {name} {start}-{end}")
                    out.write(chunk)
                start = end + 1
        part.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as inp:
        for chunk in iter(lambda: inp.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    with zipfile.ZipFile(destination) as archive:
        broken = archive.testzip()
        if broken:
            raise RuntimeError(f"Invalid ZIP member: {name}/{broken}")
        members = [row.filename for row in archive.infolist() if not row.is_dir()]
    return {"url": BASE + name, "bytes": destination.stat().st_size,
            "sha256": digest.hexdigest(), "zip_crc_ok": True, "members": members}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    files = {name: download(name) for name in NAMES}
    result = {"source_url": SOURCE, "doi": "10.5066/P1CQ6QRR",
              "license": "CC0 1.0 Universal on USGS data release",
              "files": files, "training_approved": False,
              "reason": "Photo/log depth and visible features have not been verified for each interval."}
    (ROOT / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {"bytes": row["bytes"], "member_count": len(row["members"])}
                      for name, row in files.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
