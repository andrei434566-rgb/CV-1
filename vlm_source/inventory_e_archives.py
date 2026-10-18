"""Read ZIP central directories on E: without extracting archive payloads."""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path


SOURCE = Path(r"E:\Керн фото")
OUTPUT = Path(__file__).resolve().parent / "data" / "e_drive_archive_inventory_20261008.json"
IMAGE = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DOCUMENT = {".xls", ".xlsx", ".doc", ".docx", ".pdf", ".txt", ".csv"}


def main() -> None:
    archives = []
    for path in sorted(SOURCE.rglob("*.zip")):
        item = {"path": str(path), "archive_bytes": path.stat().st_size}
        try:
            with zipfile.ZipFile(path, metadata_encoding="cp866") as archive:
                files = [member for member in archive.infolist() if not member.is_dir()]
            kinds = Counter(Path(member.filename).suffix.lower() for member in files)
            descriptions = [member.filename for member in files if any(
                word in member.filename.lower() for word in ("описан", "литолог", "фото керна", "corelog", "corephoto")
            )]
            likely_core_photos = [member.filename for member in files if Path(member.filename).suffix.lower() in IMAGE
                                  and any(word in member.filename.lower() for word in ("керн", "core", "фото ящик"))
                                  and not any(word in member.filename.lower() for word in ("скан-образ", "сканобраз", "scan0"))]
            likely_core_descriptions = [member.filename for member in files if Path(member.filename).suffix.lower() in DOCUMENT
                                        and ("описан" in member.filename.lower() and "керн" in member.filename.lower())]
            item.update({
                "file_count": len(files),
                "image_count": sum(kinds[extension] for extension in IMAGE),
                "document_count": sum(kinds[extension] for extension in DOCUMENT),
                "uncompressed_bytes": sum(member.file_size for member in files),
                "description_name_examples": descriptions[:25],
                "likely_core_photo_count": len(likely_core_photos),
                "likely_core_description_count": len(likely_core_descriptions),
                "likely_core_photo_examples": likely_core_photos[:8],
                "likely_core_description_examples": likely_core_descriptions[:8],
            })
        except (OSError, zipfile.BadZipFile, RuntimeError) as error:
            item["error"] = str(error)
        archives.append(item)
    result = {
        "root": str(SOURCE),
        "archives": archives,
        "archive_count": len(archives),
        "archives_with_images_and_documents": sum(a.get("image_count", 0) > 0 and a.get("document_count", 0) > 0 for a in archives),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive_count": result["archive_count"],
                      "archives_with_images_and_documents": result["archives_with_images_and_documents"],
                      "errors": sum("error" in a for a in archives)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
