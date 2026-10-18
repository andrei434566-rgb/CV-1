"""Write the v2 experiment metrics and model card from saved run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
OUTPUT = PROJECT / "outputs" / "core_vlm_v2_external_lora"
METRICS = OUTPUT / "training_metrics.json"
MANIFEST = DATA / "qwen3_vl_v2_external_experiment.jsonl"
EXTERNAL_SUMMARY = DATA / "external_training_candidate_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    external = [row for row in records if row.get("metadata", {}).get("dataset")]
    groups = sorted({
        row.get("metadata", {}).get("dataset") or "well_R-41"
        for row in records
    })
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    metrics.update({
        "candidate_rows_used": len(records),
        "independent_data_sources": len(groups),
        "source_groups": groups,
        "independent_wells": None,
        "independent_well_count_note": "The external USGS rows describe sampled cores, not all well identifiers; data sources are not counted as wells.",
        "training_approved_rows_used": 0,
        "external_candidate_rows_used": len(external),
        "validation_metrics": None,
        "training_status": "experimental_unreviewed_candidates",
    })
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    adapter = OUTPUT / "adapter_model.safetensors"
    spec = {
        "name": "core-description-vlm-v0.2-external",
        "status": "trained_experimental_lora_adapter_unreviewed",
        "task": "image_to_structured_core_description",
        "model_branch": "VLM_DESCRIPTION",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "base_model_local_path": "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203",
        "adapter_path": "outputs/core_vlm_v2_external_lora/adapter_model.safetensors",
        "adapter_sha256": sha256(adapter),
        "adapter_type": "LoRA",
        "training_date": "2026-10-06",
        "candidate_pairs_used": len(records),
        "source_candidate_pairs": len(records) - len(external),
        "external_candidate_pairs": len(external),
        "training_approved_pairs": 0,
        "independent_data_sources": len(groups),
        "source_groups": groups,
        "independent_wells": None,
        "train_validation_test_split": None,
        "training_parameters": {
            "epochs": metrics["epoch"],
            "optimizer_steps": 120,
            "learning_rate": 0.00002,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "max_image_side_px": 448,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "trainable_parameters": 3211264,
            "total_parameters": 2130743296,
        },
        "training_metrics": {
            "train_loss": metrics["train_loss"],
            "train_runtime_seconds": metrics["train_runtime"],
            "train_samples_per_second": metrics["train_samples_per_second"],
            "validation_metrics": None,
        },
        "external_corpus": {
            "raw_corpus_manifest": "data/external_manifest.json",
            "candidate_pair_manifest": "data/external_candidate_pairs.jsonl",
            "transcription_manifest": "data/external_log_transcriptions.jsonl",
            "qwen_candidate_manifest": "data/external_qwen3_vl_candidates.jsonl",
            "external_candidate_summary": "data/external_training_candidate_summary.json",
            "training_manifest": "data/qwen3_vl_v2_external_experiment.jsonl",
            "external_qwen_candidates": len(external),
            "external_usgs_datasets_used": sorted({row["metadata"]["dataset"] for row in external}),
            "description_transcription": "Qwen3-VL-2B zero-shot OCR; not geologist-reviewed",
            "legacy_2009_scanned_field_logs": "retained in raw corpus; not text-extracted into VLM targets",
        },
        "qualitative_inference_check": {
            "status": "failed_in_sample",
            "record_id": "usgs_2021_308_central_florida:AK2:2021-308-FA-AK2_0-200cm.jpg",
            "image_seen_during_training": True,
            "valid_json": False,
            "observations": [
                "The response repeated one generic Russian color phrase many times.",
                "The JSON was truncated and could not be parsed.",
                "The output did not reproduce the source log intervals or its lithologic descriptions.",
            ],
            "independent_validation": False,
        },
        "target_fields": {
            "description_segments": ["top_m", "base_m", "description"],
            "depositional_environment": None,
        },
        "limitations": [
            "All 238 examples are unapproved candidates; no examples were geologist-approved before this exploratory run.",
            "The 89 external targets were machine-transcribed from USGS visual core logs and may contain OCR errors.",
            "No independent validation or test set was used; train loss is not an accuracy metric.",
            "External datasets cover modern coastal, back-barrier, inlet, and bay settings; they are not labels for fluvial paleochannels.",
            "This adapter describes lithologic intervals; it does not perform YOLO segmentation or validated depositional-environment classification.",
            "The original v1 adapter remains available in outputs/core_vlm_v1_lora/.",
        ],
        "dataset_manifest": "data/qwen3_vl_v2_external_experiment.jsonl",
        "training_metrics_file": "outputs/core_vlm_v2_external_lora/training_metrics.json",
        "training_script": "train_qwen3vl.py",
        "inference_script": "predict_core_vlm.py",
    }
    out = PROJECT / "model_spec_v2_external.json"
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model_spec": str(out),
        "adapter": str(adapter),
        "adapter_sha256": spec["adapter_sha256"],
        "train_loss": metrics["train_loss"],
        "rows": len(records),
        "external_rows": len(external),
        "source_groups": groups,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
