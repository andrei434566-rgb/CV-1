"""Download or prepare the same public datasets as the application library."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.infrastructure.open_dataset_library import available_files, prepare_source, register_files
from app.infrastructure.open_dataset_sources import download_artifact, list_sources, resolve_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "download", "prepare"))
    parser.add_argument("--source", choices=[source.id for source in list_sources()], action="append")
    parser.add_argument("--storage", type=Path, default=ROOT / "open_datasets")
    args = parser.parse_args()
    sources = args.source or [source.id for source in list_sources()]
    last_progress = {}

    def progress(event):
        if "downloaded_bytes" in event:
            key = event.get("filename", "")
            current = int(event.get("downloaded_bytes") or 0)
            if current - last_progress.get(key, -32 * 1024**2) >= 32 * 1024**2:
                print(f"{key}: {current / 1024**2:.1f} MB", flush=True)
                last_progress[key] = current
        else:
            print(json.dumps(event, ensure_ascii=False, default=str), flush=True)

    for source_id in sources:
        local = available_files(args.storage, source_id, ROOT)
        if args.action == "status":
            print(json.dumps({"source_id": source_id, "files": [str(path) for path in local]}, ensure_ascii=False), flush=True)
        elif args.action == "download":
            for artifact in resolve_artifacts(source_id):
                found = None
                for path in local:
                    if path.name != artifact.filename:
                        continue
                    if artifact.expected_bytes and path.stat().st_size != artifact.expected_bytes:
                        continue
                    if artifact.checksum:
                        digest = hashlib.new(artifact.checksum_algorithm)
                        with path.open("rb") as stream:
                            for block in iter(lambda: stream.read(1024**2), b""):
                                digest.update(block)
                        if digest.hexdigest().casefold() != artifact.checksum.casefold():
                            continue
                    found = path
                    break
                if found is None:
                    found = download_artifact(artifact, args.storage / source_id / "archives", progress=progress)
                register_files(args.storage, source_id, [found])
                print("Ready: " + str(found), flush=True)
        else:
            result = prepare_source(args.storage, source_id, local, progress=progress)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
