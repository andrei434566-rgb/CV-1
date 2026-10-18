"""Inspect R-281 DOCX files inside a source archive without extracting them."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ARCHIVE = Path(r"E:\Керн фото\электронная копия для проверки (16).zip")
OUT = Path(__file__).resolve().parent / "data" / "r281_docx_inventory.json"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def main() -> None:
    rows = []
    with zipfile.ZipFile(ARCHIVE, metadata_encoding="cp866") as source:
        for member in source.infolist():
            if "Описание Керна" not in member.filename or not member.filename.endswith(".docx"):
                continue
            with zipfile.ZipFile(io.BytesIO(source.read(member))) as document:
                xml = ET.fromstring(document.read("word/document.xml"))
                paragraphs = ["".join(t.text or "" for t in p.iter(W + "t")) for p in xml.iter(W + "p")]
                media = [info for info in document.infolist() if info.filename.startswith("word/media/")]
            rows.append({"member": member.filename, "docx_bytes": member.file_size,
                         "paragraph_count": len(paragraphs), "text_characters": sum(map(len, paragraphs)),
                         "media_count": len(media), "media_bytes": sum(m.file_size for m in media),
                         "text_sample": "\n".join(p for p in paragraphs if p.strip())[:1800]})
    OUT.write_text(json.dumps({"source_archive": str(ARCHIVE), "documents": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"documents": len(rows), "with_media": sum(r["media_count"] > 0 for r in rows),
                      "media_total": sum(r["media_count"] for r in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
