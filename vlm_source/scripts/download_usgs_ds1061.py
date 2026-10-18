"""Download the official USGS DS1061 report for read-only source audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests


URL = "https://pubs.usgs.gov/ds/1061/ds1061.pdf"
DEST = Path(__file__).resolve().parents[1] / "data" / "external_raw" / "usgs_cache_creek_ds1061"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / "ds1061.pdf"
    temp = DEST / "ds1061.pdf.partial"
    if target.exists() and target.stat().st_size:
        print(f"Already downloaded: {target.stat().st_size:,} bytes")
        return
    sha = hashlib.sha256()
    size = 0
    with requests.get(URL, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        print(f"Content-Length: {response.headers.get('Content-Length', 'unknown')}", flush=True)
        with temp.open("wb") as out:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                out.write(chunk)
                sha.update(chunk)
                size += len(chunk)
                if size // (20 * 1024 * 1024) != (size - len(chunk)) // (20 * 1024 * 1024):
                    print(f"Downloaded {size / 1024 / 1024:.1f} MiB", flush=True)
    if temp.read_bytes()[:5] != b"%PDF-":
        raise ValueError("Response was not a PDF")
    temp.replace(target)
    print(f"Done: {size:,} bytes, SHA256 {sha.hexdigest()}")


if __name__ == "__main__":
    main()
