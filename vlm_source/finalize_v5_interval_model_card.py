"""Write provenance, metrics, and a qualitative smoke-check record for v5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DATASET = PROJECT / "data" / "e_drive_interval_crops_v2"
MANIFEST = DATASET / "qwen3_vl_interval_crops.jsonl"
SUMMARY = DATASET / "summary.json"
OUTPUT = PROJECT / "outputs" / "core_vlm_v5_interval_crops_lora"
METRICS = OUTPUT / "training_metrics.json"
ADAPTER = OUTPUT / "adapter_model.safetensors"

RESPONSE = (
    "Керн имеет однородную структуру с небольшой зернистостью. Видны небольшие "
    "включения, возможно, минеральные, но их детали не различимы."
)
V4_BASELINE_RESPONSE = (
    "Керн имеет однородную структуру с небольшой зернистостью. Видны небольшие "
    "включения, возможно, минеральные, но их детали не различимы на данном уровне."
)
TRAINING_ALIGNED_RESPONSE = (
    "Слой с неравномерной текстурой, состоящий из мелкозернистых, гладких, "
    "слабо-средне-светлых песчаников. Видны небольшие включения, возможно, "
    "глинистые или солонцеватые. Слой имеет неровную поверхность, с небольшими "
    "пустотами и неравномерной плотностью."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    sample = next(row for row in records if row.get("image"))
    target = json.loads(sample["conversations"][1]["value"])
    smoke_check = {
        "status": "failed_in_sample_qualitative_check",
        "image_seen_during_training": True,
        "record_id": sample["metadata"]["record_id"],
        "image": str(Path(sample["image"]).relative_to(PROJECT)),
        "depth_interval_m": sample["metadata"]["photo_tile_depth_interval_m"],
        "reference_first_sentence": target["description_segments"][0]["description"],
        "v4_baseline_output_on_same_crop": V4_BASELINE_RESPONSE,
        "model_output": RESPONSE,
        "v5_output_with_training_aligned_prompt": TRAINING_ALIGNED_RESPONSE,
        "valid_json": True,
        "comparison": "The default inference prompt produced a generic response. Matching the training prompt elicited sandstone and grain-size terms, but also unsupported inclusions and voids and did not reproduce the reference color or silty texture.",
        "assessment": [
            "The model did not reproduce the reference lithology, color, or grain-size terms.",
            "The model added possible mineral inclusions that are not supported by the reference description.",
            "This is an in-sample qualitative check, not an independent validation result.",
        ],
    }
    smoke_path = OUTPUT / "qualitative_smoke_check.json"
    smoke_path.write_text(json.dumps(smoke_check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    spec = {
        "name": "core-description-vlm-v0.5-interval-crops",
        "status": "experimental_candidate_qualitative_check_failed",
        "task": "image_to_structured_core_description",
        "model_branch": "VLM_DESCRIPTION",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "base_model_local_path": "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203",
        "adapter_path": "outputs/core_vlm_v5_interval_crops_lora/adapter_model.safetensors",
        "adapter_sha256": sha256(ADAPTER),
        "adapter_type": "LoRA",
        "training_date": "2026-10-08",
        "initialized_from_adapter": "outputs/core_vlm_v4_e_drive_lora",
        "training_data": {
            "manifest": "data/e_drive_interval_crops_v2/qwen3_vl_interval_crops.jsonl",
            "visual_crop_records": summary["visual_records"],
            "unique_source_description_rows": summary["unique_source_description_rows"],
            "source_photos": summary["source_photos"],
            "text_only_knowledge_records": summary["text_only_knowledge_records"],
            "total_records": summary["total_training_records"],
            "visual_source_well": "R-41",
            "user_approved_original_image_pairs_in_parent_v4": 149,
            "visual_target_scope": "First sentence of the source interval description; full wording retained in record metadata.",
        },
        "training_parameters": {
            "epochs": metrics["epoch"],
            "optimizer_steps": round(metrics["train_steps_per_second"] * metrics["train_runtime"]),
            "learning_rate": 0.000005,
            "max_image_side_px": 448,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "trainable_parameters": 3211264,
            "total_parameters": 2130743296,
        },
        "training_metrics": {
            "train_loss": metrics["train_loss"],
            "train_runtime_seconds": metrics["train_runtime"],
            "validation_metrics": None,
        },
        "independent_validation": None,
        "qualitative_smoke_check": "outputs/core_vlm_v5_interval_crops_lora/qualitative_smoke_check.json",
        "limitations": [
            "Visual targets come from one well (R-41); no independent-well validation or test split was available.",
            "Repeated crops from one source interval share the same first-sentence target.",
            "The qualitative check was in-sample and failed to reproduce key reference details.",
            "A lower training loss is not a measure of geological accuracy.",
            "The adapter does not perform YOLO segmentation or validated depositional-environment classification.",
        ],
    }
    spec_path = PROJECT / "model_spec_v5_interval_crops.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_spec": str(spec_path),
        "smoke_check": str(smoke_path),
        "adapter_sha256": spec["adapter_sha256"],
        "train_loss": metrics["train_loss"],
        "training_rows": summary["total_training_records"],
        "visual_records": summary["visual_records"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
