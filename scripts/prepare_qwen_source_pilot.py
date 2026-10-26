"""Prepare a bounded, source-labelled Qwen pilot; dry run unless --write.

Only the clean DCID-512-7 variant and Mendeley's four source classes are
eligible. Source labels are translated, never expanded into descriptions.
Existing whole-well Mendeley partitions and DCID validation partitions stay
unchanged. Prepared project test records are not decoded, selected, or exported.
Original Mendeley folder splits are provenance and were regrouped by well into
project partitions.
"""

from __future__ import annotations

import argparse
import ast
import copy
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
VLM_ROOT = APP_ROOT.parent / "vlm_core_v1"
LABELS = {
    "mendeley": {
        "sandstone": "Песчаник.",
        "limestone": "Известняк.",
        "shale": "Глинистая порода (shale).",
        "garbage": "Не керн: помехи или пустой лоток.",
    },
    "dcid": {
        "1.Red sandstone": "Красный песчаник.",
        "2.Light sandstone": "Светлый песчаник.",
        "3.Gray siltstone": "Серый алевролит.",
        "4.Mudstone": "Глинистая порода (mudstone).",
        "5.Granite": "Гранит.",
        "6.Basalt": "Базальт.",
        "7.Marble": "Мрамор.",
    },
}
LICENSES = {"mendeley": "CC BY 4.0", "dcid": "CC BY-NC 4.0"}
SOURCE_URLS = {
    "mendeley": "https://data.mendeley.com/datasets/b6rj29mrss/1",
    "dcid": "https://huggingface.co/datasets/168sir/drill-core-image-dataset",
}
PROMPT = (
    "<image>\nОпредели класс породы или помехи на этом фрагменте. "
    "Ответь кратко на русском языке только названием класса. "
    "Не добавляй фацию, условия осадконакопления, глубину или невидимые признаки."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def resolve_prepared(storage: Path, source_id: str) -> Path:
    """Use the library's latest valid run; legacy folders need no state file."""
    from app.infrastructure.open_dataset_library import last_preparation

    storage = Path(storage).resolve()
    state_path = storage / source_id / "library.json"
    if state_path.exists():
        prepared = last_preparation(storage, source_id)
        if not prepared:
            raise ValueError(f"No current valid library preparation for {source_id}; prepare this source again")
        if prepared.get("task") != "classification" or prepared.get("readiness") != "ready":
            raise ValueError(f"Library preparation for {source_id} is not ready classification data")
        return Path(prepared["output_dir"]).resolve()
    # Older preparations predate library.json. Never use this fallback when
    # an existing state invalidated or rejected its newer preparation.
    legacy = storage / source_id / "prepared" / f"{source_id}_v1"
    if not (legacy / "manifest.jsonl").is_file():
        raise FileNotFoundError(f"No prepared source manifest for {source_id}: {legacy}")
    return legacy.resolve()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_hash(path: Path) -> str:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    digest = hashlib.sha256(f"{image.width}x{image.height}:RGB:".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def application_system_prompt(path: Path) -> str:
    """Read the application's constant without importing Qt, torch or the app."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "system_text" for target in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                values.append(node.value.value)
    if len(values) != 1 or not values[0].strip():
        raise ValueError("Expected exactly one constant application system_text assignment")
    return values[0]


def friendly_rows() -> tuple[list[dict], list[dict]]:
    pairs = [
        ("Привет!", "Здравствуйте! Я на связи. Чем помочь?"),
        ("Ты работаешь?", "Да, я здесь и готов ответить."),
        ("Спасибо за помощь.", "Пожалуйста! Если появятся вопросы, обращайтесь."),
        ("Что ты умеешь?", "Могу обсудить фото керна, предложить черновик описания и объяснить геологические термины. Для поиска в документах укажите папки в приложении."),
        ("Чем поможешь?", "Помогу разобрать вопрос, обсудить выбранное фото или проверить черновик описания керна. Напишите, с чего начнём."),
        ("Расскажи о себе.", "Я локальный ИИ-помощник Kern Analyzer. Общаюсь по-русски и помогаю геологу разбирать фотографии и материалы проекта."),
        ("Алло, ты тут?", "Да, я на связи. Напишите ваш вопрос."),
        ("Можно задать простой вопрос?", "Конечно. Задавайте вопрос, постараюсь объяснить понятно."),
        ("Пока, до завтра.", "До свидания! Продолжим, когда будет удобно."),
        ("Мне нужен короткий ответ.", "Хорошо, отвечу кратко. Какой у вас вопрос?"),
        ("Не понял, объясни проще.", "Уточните, какая часть непонятна, и я объясню её проще."),
        ("Как показать тебе фотографию?", "Прикрепите фотографию в чате приложения и напишите, что нужно рассмотреть."),
    ]
    heldout = [
        ("Эй, отвечаешь?", "Да, я здесь. Чем помочь?"),
        ("Благодарю, разобрался!", "Пожалуйста! Обращайтесь, если понадобится помощь."),
        ("Представься в двух словах.", "Я локальный помощник Kern Analyzer."),
        ("Можешь говорить по-русски?", "Да, я отвечаю по-русски."),
    ]
    result = []
    for split, items in (("train", pairs), ("holdout", heldout)):
        rows = []
        for index, (question, answer) in enumerate(items, 1):
            rows.append({
                "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
                "metadata": {
                    "record_id": f"source_pilot_friendly_{split}_{index:02d}",
                    "record_type": "authored_conversation_not_geological_label",
                    "source_group": "project_authored_russian_conversation",
                    "training_approved": split == "train",
                    "geology_approved": False,
                    "target_language": "ru",
                    "usage_scope": "ordinary_russian_conversation_and_application_scope_only",
                },
            })
        result.append(rows)
    return result[0], result[1]


def _balanced_pick(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Round-robin known wells; hash-stable shuffled rows otherwise."""
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("well_group") or "unknown_well"].append(row)
    rng = random.Random(seed)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    queues = []
    for group in group_names:
        ordered = sorted(groups[group], key=lambda row: row["source_sha256"])
        rng.shuffle(ordered)
        queues.append(deque(ordered))
    result = []
    while len(result) < count and any(queues):
        for queue in queues:
            if queue and len(result) < count:
                result.append(queue.popleft())
    if len(result) != count:
        raise ValueError(f"Only {len(result)} candidates available; requested {count}")
    return result


def _theory_rows(train_path: Path, holdout_path: Path) -> tuple[list[dict], list[dict]]:
    train, holdout = read_jsonl(train_path), read_jsonl(holdout_path)
    if not train or not holdout:
        raise ValueError("Theory train and holdout must both be nonempty")
    for is_train, rows in ((True, train), (False, holdout)):
        for row in rows:
            meta = row.get("metadata", {})
            conversations = row.get("conversations", [])
            if row.get("image") or len(conversations) != 2:
                raise ValueError("Theory must consist of text-only question/answer pairs")
            if meta.get("record_type") != "text_only_knowledge" or not meta.get("sources"):
                raise ValueError("Theory row is missing its source-attributed provenance")
            if is_train and meta.get("training_approved") is not True:
                raise ValueError("Theory train includes a row not approved for training")
            if not is_train and meta.get("training_approved") is True:
                raise ValueError("Theory holdout is marked for training")
            if meta.get("geology_approved") is True:
                raise ValueError("This pilot expects source-checked, not falsely expert-approved theory")
            if not all(isinstance(item.get("value"), str) and item["value"].strip() for item in conversations):
                raise ValueError("Empty theory question or answer")
    train_questions = {row["conversations"][0]["value"].strip().casefold() for row in train}
    holdout_questions = {row["conversations"][0]["value"].strip().casefold() for row in holdout}
    if train_questions & holdout_questions:
        raise ValueError("An identical theory question occurs in train and holdout")
    return train, holdout


def build_plan(
    storage: Path,
    theory_train: Path,
    theory_holdout: Path,
    *,
    train_per_class: int = 60,
    val_per_class: int = 8,
    seed: int = 20261022,
) -> tuple[dict[str, list[dict]], dict]:
    if not 1 <= train_per_class <= 256 or not 1 <= val_per_class <= 64:
        raise ValueError("Pilot bounds: 1..256 train/class and 1..64 validation/class")
    storage = storage.resolve()
    source_rows = {}
    preparations = {}
    source_hash_splits = defaultdict(set)
    global_hash_labels = defaultdict(set)
    provenance = {}
    for source_id, labels in LABELS.items():
        prepared = resolve_prepared(storage, source_id)
        manifest = prepared / "manifest.jsonl"
        preparations[source_id] = prepared
        rows = read_jsonl(manifest)
        if source_id == "dcid":
            rows = [row for row in rows if row.get("dataset_subset") == "DCID-512-7"]
        else:
            rows = [row for row in rows if row.get("dataset_subset") == "mendeley"]
        if not rows or {row["class_name"] for row in rows} != set(labels):
            raise ValueError(f"Unexpected or incomplete source class list for {source_id}")
        source_rows[source_id] = rows
        for row in rows:
            if row.get("source_id") != source_id or row.get("split") not in {"train", "val", "test"}:
                raise ValueError("Source identity or split is missing")
            source_hash = row.get("source_sha256", "")
            if len(source_hash) != 64:
                raise ValueError("Missing canonical source-pixel SHA-256")
            source_hash_splits[source_hash].add(row["split"])
            global_hash_labels[source_hash].add((source_id, row["class_name"]))
        provenance[source_id] = {
            "manifest": str(manifest),
            "manifest_sha256": file_hash(manifest),
            "included_subset": "DCID-512-7" if source_id == "dcid" else "mendeley",
            "source_url": SOURCE_URLS[source_id],
            "license": LICENSES[source_id],
            "available_counts": dict(Counter(row["split"] for row in rows)),
        }
    source_overlaps = sum(len(splits) > 1 for splits in source_hash_splits.values())
    label_conflicts = sum(len(labels) > 1 for labels in global_hash_labels.values())
    if source_overlaps or label_conflicts:
        raise ValueError(
            f"Source-pixel leakage/conflicts across included datasets: "
            f"{source_overlaps} cross-split, {label_conflicts} conflicting labels"
        )

    # Verify all Mendeley well groups across all three parts, not only the sample.
    well_splits = defaultdict(set)
    for row in source_rows["mendeley"]:
        if not row.get("well_group"):
            raise ValueError("A Mendeley crop has no verified well identity")
        well_splits[row["well_group"]].add(row["split"])
    if any(len(splits) > 1 for splits in well_splits.values()):
        raise ValueError("A Mendeley well appears in more than one split")

    exports = {"train_visual": [], "val_visual": []}
    selected_pixels = defaultdict(set)
    selected_labels = defaultdict(set)
    selected_counts = Counter()
    selected_original_source_split_counts = Counter()
    selected_wells = defaultdict(set)
    for source_id, labels in LABELS.items():
        for class_index, source_label in enumerate(sorted(labels)):
            for split, count in (("train", train_per_class), ("val", val_per_class)):
                candidates = [row for row in source_rows[source_id]
                              if row["split"] == split and row["class_name"] == source_label]
                rows = _balanced_pick(candidates, count, seed + class_index)
                for row in rows:
                    image = (preparations[source_id] / row["image"]).resolve()
                    if not image.is_relative_to(preparations[source_id].resolve()) or not image.is_file():
                        raise ValueError(f"Prepared image is missing or escapes the preparation: {image}")
                    actual_pixel_hash = pixel_hash(image)
                    selected_pixels[actual_pixel_hash].add(split)
                    selected_labels[actual_pixel_hash].add((source_id, source_label))
                    meta = {
                        "record_id": f"source_pilot_{source_id}_{row['source_sha256'][:24]}",
                        "record_type": "source_direct_crop_classification",
                        "source_group": f"open_source_{source_id}",
                        "dataset": source_id,
                        "dataset_subset": row["dataset_subset"],
                        "source_label_en": source_label,
                        "target_label_ru": labels[source_label],
                        "source_split": row.get("source_split"),
                        "partition": split,
                        "well_id": row.get("well_group"),
                        "source_pixel_sha256": row["source_sha256"],
                        "prepared_pixel_sha256": actual_pixel_hash,
                        "source_record": row.get("source"),
                        "source_manifest": provenance[source_id]["manifest"],
                        "source_url": SOURCE_URLS[source_id],
                        "license": LICENSES[source_id],
                        "target_language": "ru",
                        "label_provenance": "source_direct_class_literal_russian_translation",
                        "training_approved": split == "train",
                        "geology_approved": False,
                        "approval_basis": (
                            "User requested local Qwen fine-tuning on prepared source data; "
                            "approval covers source classification labels, not expert geological descriptions."
                        ),
                        "usage_scope": "short_source_crop_class_only; no full description, depth, facies or depositional environment",
                    }
                    exports[f"{split}_visual"].append({
                        "image": str(image),
                        "conversations": [
                            {"from": "human", "value": PROMPT + "\nДопустимые классы: " + "; ".join(value.rstrip(".") for value in labels.values()) + "."},
                            {"from": "gpt", "value": labels[source_label]},
                        ],
                        "metadata": meta,
                    })
                    selected_counts[f"{source_id}/{split}/{source_label}"] += 1
                    original_split = row.get("source_split")
                    original_split = str(original_split).strip() if original_split is not None else ""
                    selected_original_source_split_counts[f"{source_id}/{split}/{original_split or 'unknown'}"] += 1
                    if row.get("well_group"):
                        selected_wells[f"{source_id}/{split}"].add(row["well_group"])
    pixel_overlaps = sum(len(splits) > 1 for splits in selected_pixels.values())
    pixel_conflicts = sum(len(labels) > 1 for labels in selected_labels.values())
    selected_total = sum(len(rows) for rows in exports.values())
    if pixel_overlaps or pixel_conflicts or len(selected_pixels) != selected_total:
        raise ValueError("Selected prepared images have duplicate pixels, label conflicts, or train/val leakage")

    train_theory, holdout_theory = _theory_rows(theory_train, theory_holdout)
    exports["train_theory"], exports["holdout_theory"] = train_theory, holdout_theory
    exports["train_conversation"], exports["holdout_conversation"] = friendly_rows()
    system_source = APP_ROOT / "app/infrastructure/local_vlm_runner.py"
    system_prompt = application_system_prompt(system_source)
    summary = {
        "schema": "kern-qwen-source-pilot-v1",
        "status": "prepared_candidate_only_not_trained",
        "seed": seed,
        "train_per_class": train_per_class,
        "val_per_class": val_per_class,
        "counts": {key: len(rows) for key, rows in exports.items()},
        "selected_counts": dict(sorted(selected_counts.items())),
        "selected_original_source_split_counts": dict(sorted(selected_original_source_split_counts.items())),
        "split_definition": {
            "project_partition_field": "prepared manifest split; exported metadata.partition",
            "original_source_split_field": "prepared manifest source_split; exported metadata.source_split",
            "selection_policy": "Select only prepared project train/val records; exclude prepared project test records.",
            "cross_split_guards_apply_to": "prepared project partitions, not original source folder names",
            "mendeley": (
                "Original source train/val/test folders were regrouped by filename-derived well identity into "
                "disjoint prepared project/well partitions. Original-source test records may occur in project train/val."
            ),
            "dcid": "Prepared project partitions are image-level; verified well identities are unavailable.",
            "unknown_original_split": "Original source_split was absent; it is not inferred from the project partition.",
            "selected_original_source_split_count_key": "source_id/project_partition/original_source_split",
        },
        "selected_wells": {key: sorted(values) for key, values in selected_wells.items()},
        "global_source_pixel_cross_split_duplicates": source_overlaps,
        "global_source_pixel_label_conflicts": label_conflicts,
        "selected_actual_pixel_cross_split_duplicates": pixel_overlaps,
        "selected_actual_pixel_label_conflicts": pixel_conflicts,
        "selected_unique_actual_pixel_hashes": len(selected_pixels),
        "source_provenance": provenance,
        "system_prompt": {
            "source": str(system_source),
            "text": system_prompt,
            "sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "dynamic_expert_examples_included": False,
        },
        "theory": {
            "train_manifest": str(theory_train.resolve()),
            "train_sha256": file_hash(theory_train),
            "holdout_manifest": str(theory_holdout.resolve()),
            "holdout_sha256": file_hash(theory_holdout),
            "geologist_reviewed": False,
        },
        "excluded": ["DCID-512-35", "noise-512-35", "noise-512-7", "UKCS", "IODP", "UniBo", "R-41 replay", "all prepared project/well test records"],
        "recommended_initialization": "fresh Qwen/Qwen3-VL-2B-Instruct base; do not inherit rejected theory or narrow previous adapters",
        "limitations": [
            "No new weights, activation, or performance claim is produced by preparation.",
            "Mendeley wells are disjoint across splits, but validation/test wells were evaluated in prior experiments.",
            "Original Mendeley folder splits were regrouped by well; excluding project test does not exclude original-source test folders.",
            "DCID has no verified well identities; its image validation is not independent-well validation.",
            "Shale and mudstone labels remain source-specific and imply no unobserved diagnostic properties.",
            "Source classifications do not supply expert full descriptions or map to the 92-facies application catalogue.",
            "DCID is noncommercial; preserve attribution and license restrictions for any resulting candidate.",
            "Theory paraphrases are source checked, not expert approved; prior theory adapter was rejected.",
        ],
    }
    return exports, summary


def write_plan(output: Path, exports: dict[str, list[dict]], summary: dict) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing pilot: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        for key, rows in exports.items():
            (stage / f"{key}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
            )
        mixed = []
        for name in ("train_visual", "train_theory", "train_conversation"):
            for original in exports[name]:
                if original.get("metadata", {}).get("training_approved") is not True:
                    raise ValueError("Combined training manifest contains an unapproved row")
                row = copy.deepcopy(original)
                row["metadata"]["split"] = "train"
                mixed.append(row)
        (stage / "mixed.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mixed), encoding="utf-8"
        )
        (stage / "system_prompt.txt").write_text(summary["system_prompt"]["text"], encoding="utf-8")
        (stage / "summary.json").write_text(json.dumps({**summary, "mixed_training_rows": len(mixed),
            "mixed_training_manifest": "mixed.jsonl"}, ensure_ascii=False, indent=2), encoding="utf-8")
        stage.rename(output)
    except BaseException:
        # This directory was created by this function under the explicit output parent.
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}_"):
            shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, default=APP_ROOT / "open_datasets")
    parser.add_argument("--theory-train", type=Path, default=VLM_ROOT / "data/knowledge/theory_training_v5.jsonl")
    parser.add_argument("--theory-holdout", type=Path, default=VLM_ROOT / "data/knowledge/theory_holdout_v5.jsonl")
    parser.add_argument("--train-per-class", type=int, default=60)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "open_datasets/qwen_source_pilot_v1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Write manifests after all checks; existing directories are never overwritten")
    mode.add_argument("--dry-run", action="store_true", help="Default: check and print JSON without writing")
    args = parser.parse_args()
    exports, summary = build_plan(
        args.storage, args.theory_train, args.theory_holdout,
        train_per_class=args.train_per_class, val_per_class=args.val_per_class, seed=args.seed,
    )
    summary["dry_run"] = not args.write
    summary["output_dir"] = str(args.output_dir.resolve())
    if args.write:
        write_plan(args.output_dir, exports, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
