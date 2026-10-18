"""Record paragraph and embedded-image order in an R-281 core description."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ARCHIVE = Path(r"E:\Керн фото\электронная копия для проверки (16).zip")
OUT = Path(__file__).resolve().parent / "data" / "r281_docx_structure_sample.json"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def main() -> None:
    with zipfile.ZipFile(ARCHIVE, metadata_encoding="cp866") as source:
        member = next(m for m in source.namelist() if "Описание Керна №09" in m and m.endswith(".docx"))
        with zipfile.ZipFile(io.BytesIO(source.read(member))) as document:
            xml = ET.fromstring(document.read("word/document.xml"))
            rels = ET.fromstring(document.read("word/_rels/document.xml.rels"))
            targets = {item.get("Id"): item.get("Target") for item in rels}
            sequence = []
            for p in xml.iter(W + "p"):
                text = "".join(t.text or "" for t in p.iter(W + "t")).strip()
                images = [targets.get(blip.get(R + "embed")) for blip in p.iter(A + "blip")]
                if text or images:
                    sequence.append({"text": text, "images": images})
    OUT.write_text(json.dumps({"member": member, "sequence": sequence}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"blocks": len(sequence), "images": sum(len(x["images"]) for x in sequence)}))


if __name__ == "__main__":
    main()
