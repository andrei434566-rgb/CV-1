"""Save the v6 training record and document its failed qualitative check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / "outputs" / "core_vlm_v6_finetune_lora"
ADAPTER = OUTPUT / "adapter_model.safetensors"
METRICS = OUTPUT / "training_metrics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    qualitative = {
        "status": "failed_in_sample_qualitative_check",
        "evaluation_scope": "one crop already present in training; not independent validation",
        "record_id": "urengoy_41_3837p87_3840p81_44cb1385_row78_tile1",
        "image": "data/e_drive_interval_crops_v2/images/crop_00001.jpg",
        "depth_interval_m": [3838.56, 3839.31],
        "reference_first_sentence": "Песчаники светло-серые, средне-мелкозернистые, алевритистые.",
        "v5_training_aligned_output": (
            "Слой с неравномерной текстурой, состоящий из мелкозернистых, гладких, "
            "слабо-средне-светлых песчаников. Видны небольшие включения, возможно, "
            "глинистые или солонцеватые. Слой имеет неровную поверхность, с небольшими "
            "пустотами и неравномерной плотностью."
        ),
        "v6_default_decode_output": (
            "Серый, слабо-серовато-серый, с небольшими включениями. Слой имеет "
            "неравномерную текстуру, с заметными зернистыми включениями. Слой имеет "
            "неравномерную текстуру, с заметными зернистыми включениями. Слой имеет "
            "неравномерную текстуру, с заметными зернистыми включениями."
        ),
        "v6_repetition_controlled_output": {
            "valid_json": True,
            "raw": (
                '{"description_segments": [{"top_m": 3842, "base_m": 50, "description": '
                '"Слой из глинистых песчаников, состоящих из мелких гранул, с просветами и '
                'пустотами. Песчаники имеют светло-серый цвет, с оттенками серо-зеленого. '
                'Видны крупные пористые залежи, которые могут быть связаны с водными структурами."}]}'
            ),
            "assessment": "JSON syntax is valid, but generated depths and unsupported lithologic details are wrong.",
        },
        "assessment": [
            "v6 repeats a sentence and adds details absent from the reference.",
            "Repetition controls did not make the geological content reliable.",
            "The lower training loss is not evidence of better geological descriptions.",
            "Further fine-tuning on the same records is not justified until geologists review or correct examples.",
        ],
    }
    (OUTPUT / "qualitative_check.json").write_text(
        json.dumps(qualitative, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    spec = {
        "name": "core-description-vlm-v0.6-finetune",
        "status": "failed_qualitative_check_do_not_use_for_geological_descriptions",
        "task": "image_to_structured_core_description",
        "model_branch": "VLM_DESCRIPTION",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
        "adapter_path": "outputs/core_vlm_v6_finetune_lora/adapter_model.safetensors",
        "adapter_sha256": sha256(ADAPTER),
        "adapter_type": "LoRA",
        "training_date": "2026-10-08",
        "initialized_from_adapter": "outputs/core_vlm_v5_interval_crops_lora",
        "training_data": {
            "manifest": "data/e_drive_interval_crops_v2/qwen3_vl_interval_crops.jsonl",
            "total_records": metrics["candidate_rows_used"],
            "visual_crop_records": metrics["record_type_counts"]["user_confirmed_interval_crop"],
            "text_only_knowledge_records": metrics["record_type_counts"]["text_only_knowledge"],
            "visual_source_well": "R-41",
            "new_training_records_vs_v5": 0,
        },
        "training_parameters": {
            "epochs": metrics["epoch"],
            "optimizer_steps": 124,
            "learning_rate": 0.000002,
            "max_image_side_px": 448,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 4,
        },
        "training_metrics": {
            "train_loss": metrics["train_loss"],
            "train_runtime_seconds": metrics["train_runtime"],
            "validation_metrics": None,
        },
        "independent_validation": None,
        "qualitative_check": "outputs/core_vlm_v6_finetune_lora/qualitative_check.json",
        "limitations": [
            "The run continued training on the same 246 records and added no new labels.",
            "Visual examples come from one well; no independent-well validation or test split is available.",
            "The in-sample qualitative check failed with repeated text and unsupported details.",
            "This adapter is not suitable for geologist-facing draft descriptions.",
        ],
    }
    path = PROJECT / "model_spec_v6_finetune.json"
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_spec": str(path), "qualitative_check": str(OUTPUT / "qualitative_check.json"),
                      "adapter_sha256": spec["adapter_sha256"], "train_loss": metrics["train_loss"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
