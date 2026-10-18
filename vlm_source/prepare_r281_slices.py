"""Extract meter-cut photos and adjacent Russian descriptions from R-281 DOCX reports.

The pair is document-adjacent; it is a candidate, not an approved visual target.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


ROOT = Path(__file__).resolve().parent
ARCHIVE = Path(r"E:\Керн фото\электронная копия для проверки (16).zip")
OUT = ROOT / "data" / "r281_slices"
IMAGES = OUT / "images"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
CUT = re.compile(r"^Срез\s*(\d+)\s*$", re.IGNORECASE)
DEPTH = re.compile(r"\b(\d{4})(?:[,.]0)?\s*м\b", re.IGNORECASE)
ROCK = re.compile(r"^\s*(песчаник|алевролит|аргиллит|глин[а-я]*|угол[ья]|известняк|мергель|доломит)[а-я]*\b", re.IGNORECASE)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []
    with zipfile.ZipFile(ARCHIVE, metadata_encoding="cp866") as source:
        for member in source.infolist():
            if "Описание Керна" not in member.filename or not member.filename.endswith(".docx"):
                continue
            match = re.search(r"№\s*0?(\d+)", Path(member.filename).name)
            if not match:
                errors.append({"member": member.filename, "error": "core number missing"})
                continue
            core_number = int(match.group(1))
            with zipfile.ZipFile(io.BytesIO(source.read(member))) as document:
                xml = ET.fromstring(document.read("word/document.xml"))
                rels = ET.fromstring(document.read("word/_rels/document.xml.rels"))
                targets = {item.get("Id"): item.get("Target") for item in rels}
                blocks = []
                for paragraph in xml.iter(W + "p"):
                    text = "".join(t.text or "" for t in paragraph.iter(W + "t")).strip()
                    images = [targets.get(blip.get(R + "embed")) for blip in paragraph.iter(A + "blip")]
                    if text or images:
                        blocks.append((text, images))
                cut_positions = [(i, int(CUT.fullmatch(text).group(1))) for i, (text, _) in enumerate(blocks) if CUT.fullmatch(text)]
                for ordinal, (start, cut_number) in enumerate(cut_positions):
                    end = cut_positions[ordinal + 1][0] if ordinal + 1 < len(cut_positions) else len(blocks)
                    group = blocks[start:end]
                    depths = [int(m.group(1)) for text, _ in group if (m := DEPTH.search(text))]
                    images = [image for _, paragraph_images in group for image in paragraph_images if image]
                    descriptions = [text for text, _ in group if ROCK.match(text) or "переслаивание" in text.lower()[:70]]
                    if len(depths) != 1 or len(images) != 1 or len(descriptions) != 1:
                        errors.append({"member": member.filename, "cut": cut_number,
                                       "depths": depths, "images": images, "descriptions": descriptions,
                                       "block_text": [text for text, _ in group if text][:8]})
                        continue
                    depth = depths[0]
                    image_member = "word/" + images[0].lstrip("/")
                    image_bytes = document.read(image_member)
                    suffix = Path(image_member).suffix.lower()
                    image_path = IMAGES / f"r281_core{core_number:02d}_cut{cut_number:02d}_{depth}m{suffix}"
                    image_path.write_bytes(image_bytes)
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        width, height = image.size
                    description = descriptions[0]
                    matched_rock = ROCK.match(description)
                    rock = matched_rock.group(1).lower() if matched_rock else "переслаивание"
                    rock = {"глины": "глина", "глинист": "глина", "уголь": "уголь"}.get(rock, rock)
                    records.append({
                        "record_id": f"r281_core{core_number:02d}_cut{cut_number:02d}",
                        "well_id": "281Р", "core_number": core_number, "cut_number": cut_number,
                        "depth_m": depth, "image": str(image_path), "image_size_px": [width, height],
                        "description": description, "rock_first_word": rock,
                        "source_archive": str(ARCHIVE), "source_document_member": member.filename,
                        "source_image_member": image_member,
                        "pairing_method": "image embedded beside matching meter-cut depth and description in DOCX",
                        "photo_view": "core_cross_section", "source_pair_approved_by_user": False,
                        "training_approved": False,
                        "visual_target_reviewed_by_geologist": False,
                        "limitations": "Photo is a low-resolution, blue-tinted cut-face image. Report description may include non-visible composition and cement.",
                    })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    summary = {"source_archive": str(ARCHIVE), "records": len(records),
               "core_numbers": sorted({r["core_number"] for r in records}),
               "rock_first_word_counts": dict(Counter(r["rock_first_word"] for r in records)),
               "unparsed_cuts": len(errors), "errors": errors[:30]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
