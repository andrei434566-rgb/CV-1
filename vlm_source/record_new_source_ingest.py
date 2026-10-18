"""Record provenance and review state of the newly downloaded USGS material."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external_raw"
OUTPUT = ROOT / "data" / "new_external_ingest_20261011.json"


def file_record(path: Path, url: str) -> dict:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "source_url": url, "bytes": path.stat().st_size,
            "sha256": sha.hexdigest()}


def main() -> None:
    kootenai = RAW / "usgs_2011_kootenai_river"
    cat = RAW / "usgs_2010_cat_island"
    photo_index = json.loads((kootenai / "photo_inventory.json").read_text(encoding="utf-8"))
    cat_index = json.loads((ROOT / "data" / "cat_island_candidates" / "summary.json").read_text(encoding="utf-8"))
    pilot_index = json.loads((ROOT / "data" / "cat_island_lithology_pilot" / "summary.json").read_text(encoding="utf-8"))
    forced_index = json.loads((ROOT / "data" / "cat_island_four_class_pilot" / "summary.json").read_text(encoding="utf-8"))
    forced_eval = json.loads((ROOT / "outputs" / "cat_island_four_class_evaluation.json").read_text(encoding="utf-8"))["summary"]
    payload = {
        "created_date": "2026-10-11",
        "purpose": "Source provenance for experimental core VLM research; no automatic approval of photo labels",
        "sources": [
            {
                "id": "usgs_2011_kootenai_river", "doi": "10.3133/sir20115006",
                "source_page": "https://pubs.usgs.gov/sir/2011/5006/",
                "material": "riverbed core lithologic strip logs and linked photographs",
                "setting": "modern riverbed", "source_photos_downloaded": len(photo_index["photos"]),
                "broken_source_photo_links": photo_index["errors"],
                "files": [file_record(kootenai / "sir20115006_appendixA.pdf",
                                      "https://pubs.usgs.gov/sir/2011/5006/pdf/sir20115006_appendixA.pdf")],
                "photo_inventory": str(kootenai / "photo_inventory.json"),
                "training_status": "downloaded_for_depth_and_quality_review; excluded_from_current_training",
            },
            {
                "id": "usgs_2010_cat_island", "doi": "10.3133/ds834",
                "source_page": "https://pubs.usgs.gov/ds/0834/ds834_data.html",
                "material": "terrestrial beach-ridge and marine vibracore PDFs with embedded photos and lithologic descriptions",
                "source_core_pdfs": len(cat_index["cores"]),
                "staged_photo_candidates": cat_index["photo_candidates"],
                "staged_nonempty_descriptions": cat_index["nonempty_candidates"],
                "pilot_train_crops": pilot_index["train_images"],
                "pilot_holdout_crops": pilot_index["heldout_images"],
                "forced_four_class_train_crops": forced_index["counts"]["train"],
                "forced_four_class_holdout_crops": forced_index["counts"]["heldout"],
                "forced_four_class_adapter_correct": forced_eval["adapter_correct"],
                "forced_four_class_base_correct": forced_eval["base_correct"],
                "files": [
                    file_record(cat / name, f"https://pubs.usgs.gov/ds/0834/data/{name}")
                    for name in ("10BIM03_Terrestrial_core_logs_pdfs.zip",
                                 "10BIM06_Marine_core_logs_pdfs.zip")
                ],
                "candidate_manifest": str(ROOT / "data" / "cat_island_candidates" / "source_candidates.jsonl"),
                "pilot_manifest": str(ROOT / "data" / "cat_island_lithology_pilot" / "train.jsonl"),
                "training_status": "experimental_lithology_only; not_approved_for_production",
            },
        ],
        "rights_note": "USGS-authored data are generally public domain; individual third-party figures must be checked in source metadata. Preserve source attribution.",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
