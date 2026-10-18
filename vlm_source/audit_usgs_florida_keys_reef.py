"""Inventory CC0 Florida Keys reef-core photos and typed graphic logs by core ID."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).parent / "data" / "external_raw" / "usgs_florida_keys_reef"
SOURCE = "https://www.usgs.gov/data/descriptive-core-logs-core-photographs-radiocarbon-ages-and-accretion-data-holocene-reef-cores"
FILES = ("FKRT_Core_Data.zip", "FKRT_Descriptive_Core_Logs.zip", "FKRT_Core_Photographs.zip")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def match_id(member: str, core_ids: list[str]) -> str | None:
    stem = Path(member).stem.upper()
    for core_id in core_ids:
        if stem == core_id or stem.startswith(core_id + "-"):
            return core_id
    return None


def main() -> None:
    sources = {}
    for name in FILES:
        path = ROOT / name
        with zipfile.ZipFile(path) as archive:
            broken = archive.testzip()
            if broken:
                raise RuntimeError(f"Broken archive member: {name}/{broken}")
            members = archive.namelist()
        sources[name] = {
            "url": f"https://coastal.er.usgs.gov/data-release/doi-F7NV9HJX/data/{name}",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "zip_members": len(members),
            "zip_crc_ok": True,
        }
    with zipfile.ZipFile(ROOT / FILES[0]) as archive:
        rows = list(csv.DictReader(io.TextIOWrapper(archive.open("FKRT_Core_Data.csv"), encoding="utf-8-sig")))
    core_ids = sorted({row["Core ID"].upper() for row in rows}, key=len, reverse=True)
    paired = {core_id: {"photo_members": [], "log_members": []} for core_id in core_ids}
    unmatched = {"photo": [], "log": []}
    for kind, name in (("photo", FILES[2]), ("log", FILES[1])):
        with zipfile.ZipFile(ROOT / name) as archive:
            for member in archive.namelist():
                if not member.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                core_id = match_id(member, core_ids)
                if core_id:
                    paired[core_id][kind + "_members"].append(member)
                else:
                    unmatched[kind].append(member)
    for row in rows:
        paired[row["Core ID"].upper()]["metadata"] = {
            "core_id": row["Core ID"],
            "site": row["Site Name"],
            "subregion": row["Subregion"],
            "core_length_m": row["Total core length (m)"],
        }
    report = {
        "source_url": SOURCE,
        "doi": "10.5066/F7NV9HJX",
        "license": "CC0 1.0 Universal on official USGS data release",
        "files": sources,
        "metadata_core_records": len(rows),
        "cores_with_photo": sum(bool(row["photo_members"]) for row in paired.values()),
        "cores_with_log": sum(bool(row["log_members"]) for row in paired.values()),
        "cores_with_both": sum(bool(row["photo_members"]) and bool(row["log_members"]) for row in paired.values()),
        "photo_files_matched": sum(len(row["photo_members"]) for row in paired.values()),
        "log_files_matched": sum(len(row["log_members"]) for row in paired.values()),
        "unmatched_members": unmatched,
        "core_records": paired,
        "training_approved": False,
        "reason": "Core ID matching does not prove that each photo segment and lithologic log interval have the same depth; exact alignment and Russian visual target review are required.",
    }
    (ROOT / "inventory_and_core_index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("metadata_core_records", "cores_with_photo", "cores_with_log", "cores_with_both", "photo_files_matched", "log_files_matched", "unmatched_members")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
