"""Extract and index the official USGS core data archives in external_raw.

This keeps downloaded source archives intact, extracts each archive into its
own folder, records SHA-256 checksums and builds candidate photo/log links.
Candidate links are reference material only: a log image/PDF is not converted
to a verified text target here and the resulting rows are not training-ready.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath


PROJECT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT / "data" / "external_raw"
PAIR_FILE = PROJECT / "data" / "external_candidate_pairs.jsonl"
MANIFEST_FILE = PROJECT / "data" / "external_manifest.json"
INVENTORY_FILE = PROJECT / "data" / "external_inventory.csv"

SOURCES = {
    "usgs_2009_apalachicola_bay": {
        "title": "Vibracore, radiocarbon, microfossil, seismic-reflection, and grain-size data from Apalachicola Bay, Florida",
        "doi": "10.3133/ofr20091031",
        "metadata_url": "https://pubs.usgs.gov/of/2009/1031/html/catalog.html",
        "rights": "USGS Open-File Report data catalog; metadata lists no use constraints for the core-photo archive. Preserve USGS attribution and source metadata.",
        "setting": "coastal bay / estuarine; modern analogue, not a confirmed fluvial paleochannel",
    },
    "usgs_2021_308_central_florida": {
        "title": "Core descriptions and sedimentologic data from vibracores collected in 2021 from Central Florida Gulf Coast Barrier Islands",
        "doi": "10.5066/P14L5SVG",
        "metadata_url": "https://cmgds.marine.usgs.gov/catalog/spcmsc/2021-308-FA_Sediment_metadata.faq.html",
        "rights": "USGS data release; imagery headers identify USGS credit and Public Domain. Retain attribution and metadata.",
        "setting": "barrier-island / coastal; modern analogue, not a confirmed fluvial paleochannel",
    },
    "usgs_2021_312_fire_island": {
        "title": "Core descriptions and sedimentologic data from vibracores and sand augers collected in 2021 and 2022 from Fire Island, New York",
        "doi": "10.5066/P91P1T88",
        "metadata_url": "https://cmgds.marine.usgs.gov/catalog/spcmsc/2021-312-FA_Sediment_metadata.faq.html",
        "rights": "USGS data release; retain USGS source attribution and embedded image metadata.",
        "setting": "back-barrier / beach / dune / washover; modern coastal analogue",
    },
    "usgs_2023_309_st_andrew_bay": {
        "title": "Sedimentologic data from vibracores collected in 2023 from St. Andrew Bay, Florida",
        "doi": "10.5066/P1AFJDNE",
        "metadata_url": "https://cmgds.marine.usgs.gov/catalog/spcmsc/2023-309-FA_Sediment_metadata.faq.html",
        "rights": "Metadata states USGS public-domain data are freely redistributable with proper source attribution.",
        "setting": "bay / estuarine; modern analogue, not a confirmed fluvial paleochannel",
    },
    "usgs_2024_330_midnight_pass": {
        "title": "Sediment data from cores collected in 2025 at Midnight Pass, Florida",
        "doi": "10.5066/P1CQ6QRR",
        "metadata_url": "https://cmgds.marine.usgs.gov/catalog/spcmsc/2024-330-FA_MP_core_metadata.faq.html",
        "rights": "Metadata marks this USGS data release CC0 1.0 Universal; attribution is retained as scientific provenance.",
        "setting": "tidal inlet / estuary; modern analogue, not a confirmed fluvial paleochannel",
    },
}

ARCHIVE_URLS = {
    "usgs_2009_apalachicola_bay": {
        "CorePhotos.zip": "https://pubs.usgs.gov/of/2009/1031/data_catalog/Core_Photos/App-07_CorePhotos.zip",
        "FieldLogs.zip": "https://pubs.usgs.gov/of/2009/1031/data_catalog/Core_Field_Descriptions/App-07_FieldLogs.zip",
        "CoreSummaries.zip": "https://pubs.usgs.gov/of/2009/1031/data_catalog/Core_Summary_Descriptions/App-07_Summaries.zip",
        "CoreLocations.zip": "https://pubs.usgs.gov/of/2009/1031/data_catalog/Core_Locations/App-07_Vibracores.zip",
        "GrainSize.zip": "https://pubs.usgs.gov/of/2009/1031/data_catalog/Grain_Size/App-07_GrainSize.zip",
        "AgeDates.zip": "https://pubs.usgs.gov/of/2009/1031/data_catalog/Age_Dates/App-07_AgeDates.zip",
    },
    "usgs_2021_308_central_florida": {
        "CorePhotos.zip": "https://coastal.er.usgs.gov/data-release/doi-P14L5SVG/data/2021-308-FA_CorePhotos.zip",
        "CoreLogs.zip": "https://coastal.er.usgs.gov/data-release/doi-P14L5SVG/data/2021-308-FA_CoreLogs.zip",
        "CoreSites.zip": "https://coastal.er.usgs.gov/data-release/doi-P14L5SVG/data/2021-308-FA_CoreSites.zip",
        "GrainSizeData.zip": "https://coastal.er.usgs.gov/data-release/doi-P14L5SVG/data/2021-308-FA_GrainSizeData.zip",
    },
    "usgs_2021_312_fire_island": {
        "CorePhotos.zip": "https://coastal.er.usgs.gov/data-release/doi-P91P1T88/data/2021-312-FA_CorePhotos.zip",
        "CoreLogs.zip": "https://coastal.er.usgs.gov/data-release/doi-P91P1T88/data/2021-312-FA_CoreLogs.zip",
        "CoreSites.zip": "https://coastal.er.usgs.gov/data-release/doi-P91P1T88/data/2021-312-FA_CoreSites.zip",
        "GrainSizeData.zip": "https://coastal.er.usgs.gov/data-release/doi-P91P1T88/data/2021-312-FA_GrainSizeData.zip",
        "AgeControl.zip": "https://coastal.er.usgs.gov/data-release/doi-P91P1T88/data/2021-312-FA_AgeControl.zip",
    },
    "usgs_2023_309_st_andrew_bay": {
        "CorePhotos.zip": "https://coastal.er.usgs.gov/data-release/doi-P1AFJDNE/data/2023-309-FA_CorePhotos.zip",
        "CoreLogs.zip": "https://coastal.er.usgs.gov/data-release/doi-P1AFJDNE/data/2023-309-FA_CoreLogs.zip",
        "CoreSites.zip": "https://coastal.er.usgs.gov/data-release/doi-P1AFJDNE/data/2023-309-FA_CoreSites.zip",
        "GrainSize.zip": "https://coastal.er.usgs.gov/data-release/doi-P1AFJDNE/data/2023-309-FA_GrainSize.zip",
        "AgeControl.zip": "https://coastal.er.usgs.gov/data-release/doi-P1AFJDNE/data/2023-309-FA_AgeControl.zip",
        "LossOnIgnition.zip": "https://coastal.er.usgs.gov/data-release/doi-P1AFJDNE/data/2023-309-FA_LossOnIgnition.zip",
    },
    "usgs_2024_330_midnight_pass": {
        "CorePhotos.zip": "https://coastal.er.usgs.gov/data-release/doi-P1CQ6QRR/data/2024-330-FA_MP_CorePhotos.zip",
        "CoreLogs.zip": "https://coastal.er.usgs.gov/data-release/doi-P1CQ6QRR/data/2024-330-FA_MP_CoreLogs.zip",
        "CoreSites.zip": "https://coastal.er.usgs.gov/data-release/doi-P1CQ6QRR/data/2024-330-FA_MP_CoreSites.zip",
        "GrainSizeData.zip": "https://coastal.er.usgs.gov/data-release/doi-P1CQ6QRR/data/2024-330-FA_MP_GrainSizeData.zip",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member_path(root: Path, name: str) -> Path:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if member.is_absolute() or any(part in ("", ".", "..") for part in member.parts):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    destination = root.joinpath(*member.parts).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"ZIP member escapes extraction folder: {name!r}")
    return destination


def extract_archive(archive_path: Path, destination: Path) -> tuple[list[dict], list[str]]:
    inventory: list[dict] = []
    nested_archives: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            path = safe_member_path(destination, entry.filename)
            if path.suffix.lower() == ".zip":
                # One USGS site package embeds the already downloaded photo ZIP.
                # Preserve the embedding archive, but avoid a duplicate expansion.
                nested_archives.append(entry.filename)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as source, path.open("wb") as target:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            inventory.append({
                "path": str(path.relative_to(PROJECT)),
                "archive": str(archive_path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
            })
    return inventory, nested_archives


def core_key(path: Path, dataset: str) -> str | None:
    stem = path.stem.upper()
    if dataset == "usgs_2009_apalachicola_bay":
        match = re.search(r"(?:APP|APP-07_|07_APP_|07008-)(\d{1,2})", stem)
        if match:
            return f"APP{int(match.group(1)):02d}"
        return None

    stem = re.sub(r"^202[134]-\d{3}-FA[-_]", "", stem)
    stem = re.sub(r"_CORELOG$", "", stem)
    stem = re.sub(r"_FULLSTITCH$", "", stem)
    stem = re.sub(r"_\d+(?:-\d+)(?:CM)?$", "", stem)
    stem = re.sub(r"_\d+(?:-\d+)_FULLSTITCH$", "", stem)
    return stem


def candidates(dataset: str) -> list[dict]:
    extracted = RAW_ROOT / dataset / "extracted"
    photo_dir = extracted / "CorePhotos"
    log_dir = extracted / "CoreLogs"
    if not photo_dir.exists():
        return []

    photos = [p for p in photo_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    logs = [p for p in log_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".pdf"}] if log_dir.exists() else []
    if dataset == "usgs_2009_apalachicola_bay":
        logs.extend((extracted / "FieldLogs").rglob("*.pdf") if (extracted / "FieldLogs").exists() else [])
        logs.extend((extracted / "CoreSummaries").rglob("*.jpg") if (extracted / "CoreSummaries").exists() else [])

    by_key: dict[str, list[Path]] = {}
    for log in logs:
        key = core_key(log, dataset)
        if key:
            by_key.setdefault(key, []).append(log)

    rows = []
    for photo in photos:
        key = core_key(photo, dataset)
        matches = sorted(by_key.get(key or "", []), key=lambda p: (
            0 if p.suffix.lower() == ".pdf" else 1 if "CORELOG" in p.stem.upper() else 2,
            p.name.lower(),
        ))
        rows.append({
            "pair_id": f"{dataset}:{key or photo.stem}:{photo.name}",
            "dataset": dataset,
            "core_key": key,
            "photo_path": str(photo.relative_to(PROJECT)),
            "log_path": str(matches[0].relative_to(PROJECT)) if matches else None,
            "other_log_candidates": [str(p.relative_to(PROJECT)) for p in matches[1:]],
            "doi": SOURCES[dataset]["doi"],
            "metadata_url": SOURCES[dataset]["metadata_url"],
            "setting": SOURCES[dataset]["setting"],
            "target_description_text": None,
            "status": "paired_reference_needs_review" if matches else "photo_unpaired",
            "trainable": False,
        })
    return rows


def main() -> None:
    if not RAW_ROOT.exists():
        raise SystemExit(f"Missing data folder: {RAW_ROOT}")

    archive_records: list[dict] = []
    all_inventory: list[dict] = []
    for archive_path in sorted(RAW_ROOT.rglob("*.zip")):
        dataset = archive_path.parent.name
        if dataset not in SOURCES:
            continue
        rel_path = archive_path.relative_to(PROJECT)
        extract_name = archive_path.stem
        extract_root = RAW_ROOT / dataset / "extracted" / extract_name
        inventory, nested = extract_archive(archive_path, extract_root)
        all_inventory.extend(inventory)
        archive_records.append({
            "dataset": dataset,
            "archive": str(rel_path),
            "source_url": ARCHIVE_URLS.get(dataset, {}).get(archive_path.name),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256(archive_path),
            "extracted_file_count": len(inventory),
            "nested_zip_skipped": nested,
        })

    # Pair creation uses the stable extracted directory names.
    pairs: list[dict] = []
    for dataset in SOURCES:
        pairs.extend(candidates(dataset))

    manifest = {
        "created_date": date.today().isoformat(),
        "project": "Qwen geological core VLM external source corpus",
        "use_note": "Downloaded public source material is staged for review. Candidate pair rows are not verified labels and are intentionally not included in the existing training JSONL.",
        "datasets": [
            {"dataset": key, **value} for key, value in SOURCES.items()
        ],
        "archives": archive_records,
        "candidate_photo_log_pairs": len(pairs),
        "paired_photo_log_candidates": sum(bool(row["log_path"]) for row in pairs),
        "unpaired_photos": sum(not row["log_path"] for row in pairs),
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with PAIR_FILE.open("w", encoding="utf-8") as stream:
        for row in pairs:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with INVENTORY_FILE.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "archive", "bytes"])
        writer.writeheader()
        writer.writerows(all_inventory)

    readme = [
        "# External USGS core data staged for Qwen",
        "",
        "This folder contains official USGS source archives plus extracted copies. The source archives are retained unchanged. See `../external_manifest.json` for DOI, metadata links, stated use terms, byte counts, and SHA-256 checksums.",
        "",
        "## Important for model training",
        "",
        "`../external_candidate_pairs.jsonl` maps whole-core photographs to likely corresponding log files where the core identifiers match. The rows are review candidates only: they do not contain OCR-verified target descriptions, and every row has `trainable: false`. Coastal, back-barrier, inlet, and bay settings are modern analogues; they must not be relabeled as fluvial paleochannels without geological review.",
        "",
        "The downloaded labels and photos have not been mixed into the existing Qwen training JSONL. Review and validate image-to-log identity, extract the original log text, and check the geological setting before creating approved training examples.",
        "",
        "Extraction inventory: `../external_inventory.csv`.",
        "",
    ]
    (RAW_ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(json.dumps({
        "archives": len(archive_records),
        "extracted_files": len(all_inventory),
        "candidate_pairs": len(pairs),
        "paired": sum(bool(row["log_path"]) for row in pairs),
        "unpaired": sum(not row["log_path"] for row in pairs),
        "manifest": str(MANIFEST_FILE),
        "pairs": str(PAIR_FILE),
        "inventory": str(INVENTORY_FILE),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
