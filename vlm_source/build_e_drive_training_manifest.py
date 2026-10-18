"""Build an auditable Russian Qwen manifest from the user-confirmed E: drive pairs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
VISUAL_INPUT = DATA / "qwen3_vl_candidates.jsonl"
PAIR_INPUT = DATA / "candidate_pairs.jsonl"
TEXT_INPUT = DATA / "knowledge" / "text_knowledge_training_v1.jsonl"
OUTPUT = DATA / "qwen3_v4_e_drive_training.jsonl"
APPROVALS = DATA / "e_drive_pair_approvals_20261007.jsonl"
SUMMARY = DATA / "knowledge" / "e_drive_training_summary.json"
APPROVAL_ID = "user_batch_confirmation_2026-10-07"
APPROVAL_BASIS = (
    "User confirmed all interval-linked photo-description pairs from E:\\Керн фото "
    "for inclusion in the training set. Source wording is retained unchanged."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    visual = read_jsonl(VISUAL_INPUT)
    pairs_by_id = {row["id"]: row for row in read_jsonl(PAIR_INPUT)}
    approved_visual: list[dict] = []
    approval_rows: list[dict] = []
    seen_ids: set[str] = set()

    for row in visual:
        metadata = row.setdefault("metadata", {})
        record_id = metadata.get("record_id")
        pair = pairs_by_id.get(record_id)
        if not pair or not record_id.startswith("urengoy_41_"):
            continue
        if record_id in seen_ids:
            raise ValueError(f"Duplicate visual record: {record_id}")
        image_path = Path(row["image"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Source image is missing: {image_path}")
        target = row["conversations"][1]["value"]
        target_hash = hashlib.sha256(target.encode("utf-8")).hexdigest()
        metadata.update(
            {
                "record_type": "user_confirmed_image_text",
                "target_language": "ru",
                "source_group": "E_drive_real_core_R41",
                "source_well": "R-41",
                "well_id": "R-41",
                "source_workbook": pair["provenance"]["description_workbook"],
                "source_archive": pair["image"]["archive"],
                "source_archive_member": pair["image"]["member"],
                "image_depth_interval_m": pair["image_depth_interval_m"],
                "source_description_segments": pair["source_description_segments"],
                "pair_type": pair["pair_type"],
                "pairing_approved_by_user": True,
                "approval_id": APPROVAL_ID,
                "approval_basis": APPROVAL_BASIS,
                "training_approved": True,
                "geology_approved": True,
                "review_status": "user_batch_confirmed_2026-10-07",
            }
        )
        approval_rows.append(
            {
                "record_id": record_id,
                "target_sha256": target_hash,
                "decision": "approved_by_user_batch_confirmation",
                "approval_id": APPROVAL_ID,
                "approved_at": "2026-10-07",
                "basis": APPROVAL_BASIS,
                "source_workbook": pair["provenance"]["description_workbook"],
                "source_image": pair["image"]["member"],
            }
        )
        approved_visual.append(row)
        seen_ids.add(record_id)

    if not approved_visual:
        raise ValueError("No E: drive image-description pairs were found.")

    text_rows = read_jsonl(TEXT_INPUT)
    for row in text_rows:
        if row.get("metadata", {}).get("training_approved") is not True:
            raise ValueError("The text-only knowledge manifest contains an unapproved row.")

    mixed = approved_visual + text_rows
    write_jsonl(OUTPUT, mixed)
    write_jsonl(APPROVALS, approval_rows)
    summary = {
        "approval_id": APPROVAL_ID,
        "approval_basis": APPROVAL_BASIS,
        "visual_pairs": len(approved_visual),
        "source_description_rows": len(
            {segment["source_row"] for row in approved_visual for segment in row["metadata"]["source_description_segments"]}
        ),
        "photo_count": len({row["metadata"]["source_archive_member"] for row in approved_visual}),
        "photo_pairs_by_type": dict(Counter(row["metadata"]["pair_type"] for row in approved_visual)),
        "text_only_knowledge_rows": len(text_rows),
        "total_training_rows": len(mixed),
        "training_rows_by_type": dict(
            Counter(row["metadata"].get("record_type", "text_only_knowledge") for row in mixed)
        ),
        "unique_wells_in_visual_pairs": sorted({row["metadata"]["well_id"] for row in approved_visual}),
        "independent_well_validation_available": False,
        "manifest": str(OUTPUT.relative_to(PROJECT)),
        "approval_ledger": str(APPROVALS.relative_to(PROJECT)),
        "warning": (
            "The user-approved visual pairs currently come from one well (R-41). "
            "This supports a pilot, not an independent-well generalization claim."
        ),
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
