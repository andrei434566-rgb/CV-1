"""Inventory the public Q204 archive without extracting multi-gigabyte images."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "external_raw" / "ukcs_q204_q205_coremdlr" / "Q204_data.zip"
REPORT = ARCHIVE.parent / "inventory.json"
DOCUMENT_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt", ".md", ".pdf", ".las", ".xml", ".zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir()]
        documents = []
        for item in entries:
            path = PurePosixPath(item.filename)
            if path.suffix.lower() in DOCUMENT_SUFFIXES:
                documents.append({"path": item.filename, "bytes": item.file_size})
        report = {
            "source": "https://figshare.com/articles/dataset/UKCS_Q204_Q205_Subsurface_Data_products_for_Machine_Learning_Study/14265689",
            "archive": str(ARCHIVE),
            "archive_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": sha256(ARCHIVE),
            "file_count": len(entries),
            "uncompressed_bytes": sum(item.file_size for item in entries),
            "top_level": dict(Counter(PurePosixPath(item.filename).parts[0] for item in entries)),
            "suffixes": dict(Counter(PurePosixPath(item.filename).suffix.lower() or "(none)" for item in entries)),
            "documents": documents,
            "image_path_examples": [item.filename for item in entries if PurePosixPath(item.filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".jp2", ".tif", ".tiff"}][:30],
            "training_status": "inventory_only; no records accepted until depth registration, lithology labels and rights are checked",
        }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("archive_bytes", "file_count", "uncompressed_bytes", "top_level", "suffixes", "documents", "image_path_examples")}, ensure_ascii=True, indent=2)[:15000])


if __name__ == "__main__":
    main()
