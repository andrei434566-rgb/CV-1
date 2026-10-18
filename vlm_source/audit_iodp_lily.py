"""Summarize LILY lithology, keeping it separate from photo-training labels."""

from __future__ import annotations

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).parent / "data" / "external_raw" / "iodp_lily_v2"


def audit_zip(path: Path) -> dict:
    by_lithology: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    expeditions: Counter[str] = Counter()
    total = 0
    depth_pairs = 0
    examples = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with archive.open(name) as source:
                lines = (line.decode("utf-8-sig", errors="replace") for line in source)
                reader = csv.DictReader(lines)
                for row in reader:
                    total += 1
                    expedition = (row.get("Exp") or "").strip()
                    if expedition:
                        expeditions[expedition] += 1
                    lith = (row.get("Simplified Lithology") or row.get("Full Lithology") or "").strip()
                    if lith:
                        by_lithology[lith] += 1
                    lith_type = (row.get("Lithology Type") or "").strip()
                    if lith_type:
                        by_type[lith_type] += 1
                    top = (row.get("Top depth CSF-A (m)") or "").strip()
                    bottom = (row.get("Bottom depth CSF-A (m)") or "").strip()
                    if top and bottom:
                        depth_pairs += 1
                    if len(examples) < 6 and lith and top and bottom:
                        examples.append({
                            "Exp": expedition,
                            "Site": row.get("Site"),
                            "Hole": row.get("Hole"),
                            "Core": row.get("Core"),
                            "Sect": row.get("Sect"),
                            "top_m": top,
                            "bottom_m": bottom,
                            "lithology": lith,
                        })
    return {
        "archive": path.name,
        "rows": total,
        "rows_with_depth_pair": depth_pairs,
        "expeditions": len(expeditions),
        "top_lithologies": by_lithology.most_common(30),
        "top_lithology_types": by_type.most_common(20),
        "examples": examples,
    }


def main() -> None:
    audits = [audit_zip(ROOT / filename) for filename in ("CleanLITH.zip", "RawLITH.zip")]
    output = {
        "source_url": "https://zenodo.org/records/10425539",
        "license": "CC BY 4.0 (Zenodo API)",
        "training_approved": False,
        "reason": "These are depth-linked descriptions without verified links to photo pixels or Russian expert review.",
        "audits": audits,
    }
    (ROOT / "content_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    for audit in audits:
        print(audit["archive"], audit["rows"], "rows;", audit["expeditions"], "expeditions;", audit["rows_with_depth_pair"], "depth pairs")
        print("Top lithologies:", audit["top_lithologies"][:8])


if __name__ == "__main__":
    main()
