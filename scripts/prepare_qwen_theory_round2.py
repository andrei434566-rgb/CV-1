"""Prepare an immutable text-only Qwen theory/conversation experiment.

Preparation does not train, certify facts, install an adapter, or use the GPU.
Run with --write to export; the default is a dry run with all input checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unicodedata
from urllib.parse import urlparse

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from scripts.prepare_qwen_source_pilot import application_system_prompt, friendly_rows

THEORY_ROOT = APP_ROOT / "vlm_source/theory_v5"
CONVERSATION_ROOT = APP_ROOT / "vlm_source/theory_round2"
SYSTEM_SOURCE = APP_ROOT / "app/infrastructure/local_vlm_runner.py"
OPEN_QUESTION = re.compile(
    r"^\s*(?:как(?:ой|ая|ое|ие|ую|ого|ому|им|их|ими|овы|ов|ова|ово)?|почему|что|когда|чем|в\s+ч[её]м)\b",
    flags=re.IGNORECASE,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_question(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return re.sub(r"[\W_]+", " ", value, flags=re.UNICODE).strip()


def record_id(row: dict) -> str:
    return str(row.get("id") or row.get("metadata", {}).get("record_id") or "").strip()


def sanitize_open_question_answers(rows: list[dict], *, stream: str) -> list[dict]:
    """Remove only a contradictory literal yes/no preface; add no new facts."""
    edits = []
    for row in rows:
        question = row["conversations"][0]["value"]
        before = row["conversations"][1]["value"]
        if not OPEN_QUESTION.match(question):
            continue
        prefix = next((value for value in ("Нет. ", "Да. ") if before.startswith(value)), None)
        if prefix is None or not before[len(prefix):].strip():
            continue
        after = before[len(prefix):]
        edit = {"kind": "remove_yes_no_preface_from_open_question", "removed_prefix": prefix,
                "old_answer_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
                "new_answer_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest()}
        existing = row["metadata"].setdefault("preparation_edits", [])
        if not isinstance(existing, list):
            raise ValueError(f"preparation_edits must be a list: {record_id(row)}")
        existing.append(copy.deepcopy(edit))
        row["conversations"][1]["value"] = after
        edits.append({"record_id": record_id(row), "stream": stream, **edit})
    return edits


def apply_answer_overrides(streams: dict[str, list[dict]], path: Path) -> list[dict]:
    """Apply explicitly source-checked corrections, preserving original evidence."""
    overrides = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("Answer overrides must be a nonempty record-ID mapping")
    records = {}
    for stream, rows in streams.items():
        for row in rows:
            identity = record_id(row)
            if identity in records:
                raise ValueError(f"Duplicate record ID before overrides: {identity}")
            records[identity] = (stream, row)
    unexpected = set(overrides) - set(records)
    if unexpected:
        raise ValueError(f"Unknown answer override record IDs: {sorted(unexpected)}")
    edits = []
    for identity, override in overrides.items():
        if (not isinstance(override, dict) or not isinstance(override.get("answer"), str)
                or not override["answer"].strip() or not isinstance(override.get("reason"), str)
                or not override["reason"].strip() or override.get("geology_approved") is not False
                or not str(override.get("review_status", "")).startswith("source_checked")):
            raise ValueError(f"Override is not explicitly source checked: {identity}")
        stream, row = records[identity]
        before = row["conversations"][1]["value"]
        after = override["answer"]
        if before == after:
            raise ValueError(f"Answer override does not change its answer: {identity}")
        candidate = copy.deepcopy(row)
        candidate["conversations"][1]["value"] = after
        candidate["metadata"].update(sources=copy.deepcopy(override.get("sources")),
                                      review_status=override["review_status"])
        validated_rows([candidate], training=row["metadata"]["training_approved"], theory=True)
        edit = {"kind": "source_checked_answer_override", "reason": override["reason"],
                "review_status": override["review_status"], "override_file": str(Path(path).resolve()),
                "old_answer": before, "old_sources": copy.deepcopy(row["metadata"]["sources"]),
                "new_sources": copy.deepcopy(override["sources"]),
                "old_answer_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
                "new_answer_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest()}
        existing = candidate["metadata"].setdefault("preparation_edits", [])
        if not isinstance(existing, list):
            raise ValueError(f"preparation_edits must be a list: {identity}")
        existing.append(copy.deepcopy(edit))
        row.clear()
        row.update(candidate)
        edits.append({"record_id": identity, "stream": stream, **edit})
    return edits


def validated_rows(rows: list[dict], *, training: bool, theory: bool,
                   allow_unsplit: bool = False, label: str = "manifest") -> list[dict]:
    if not rows:
        raise ValueError(f"Empty {label}")
    result = []
    for number, original in enumerate(rows, 1):
        if not isinstance(original, dict):
            raise ValueError(f"Expected JSON object: {label}:{number}")
        row = copy.deepcopy(original)
        meta = row.get("metadata")
        if not isinstance(meta, dict) or not record_id(row):
            raise ValueError(f"Missing metadata or record ID: {label}:{number}")
        if any(row.get(key) for key in ("image", "images", "image_path", "video", "videos")):
            raise ValueError(f"Only text-only rows are allowed: {label}:{number}")
        splits = [str(value).strip().casefold() for value in
                  (row.get("split"), row.get("partition"), meta.get("split"),
                   meta.get("partition"), meta.get("source_split")) if value is not None and str(value).strip()]
        if "test" in splits:
            raise ValueError(f"Sealed test row refused: {label}:{number}")
        allowed_splits = {"train"} if training else {"holdout", "val", "validation"}
        if (not splits and not allow_unsplit) or any(value not in allowed_splits for value in splits):
            raise ValueError(f"Unexpected split in {label}:{number}: {splits}")
        if meta.get("training_approved") is not training:
            raise ValueError(f"Incorrect training_approved in {label}:{number}")
        if meta.get("geology_approved") is not False:
            raise ValueError(f"Expected geology_approved=false, not expert certification: {label}:{number}")
        if meta.get("target_language") != "ru":
            raise ValueError(f"Expected Russian target language: {label}:{number}")
        turns = row.get("conversations")
        if (not isinstance(turns, list) or len(turns) != 2
                or not all(isinstance(turn, dict) for turn in turns)
                or [turn.get("from") for turn in turns] != ["human", "gpt"]):
            raise ValueError(f"Expected exactly one human/gpt pair: {label}:{number}")
        if not all(isinstance(turn.get("value"), str) and turn["value"].strip() for turn in turns):
            raise ValueError(f"Empty question or answer: {label}:{number}")
        if any("<image>" in turn["value"].casefold() for turn in turns):
            raise ValueError(f"Image placeholder in text-only row: {label}:{number}")
        if not normalized_question(turns[0]["value"]):
            raise ValueError(f"Question has no searchable content: {label}:{number}")
        expected_type = "text_only_knowledge" if theory else "authored_conversation_not_geological_label"
        if meta.get("record_type") != expected_type:
            raise ValueError(f"Unexpected record_type: {label}:{number}")
        if theory:
            if not str(meta.get("review_status", "")).startswith(("source_checked", "user_requested_source_checked")):
                raise ValueError(f"Theory is not marked source checked: {label}:{number}")
            sources = meta.get("sources")
            if not isinstance(sources, list) or not sources:
                raise ValueError(f"Theory lacks source provenance: {label}:{number}")
            for source in sources:
                if not isinstance(source, dict) or not source.get("source_id"):
                    raise ValueError(f"Invalid source provenance: {label}:{number}")
                url = urlparse(str(source.get("url", "")))
                if url.scheme not in {"http", "https"} or not url.netloc:
                    raise ValueError(f"Source URL is missing or invalid: {label}:{number}")
                if not (source.get("title") or source.get("locator")):
                    raise ValueError(f"Source needs a title or locator: {label}:{number}")
        meta["split"] = "train" if training else "holdout"
        if not training and not theory:
            meta["evaluation_kind"] = "casual_guard"
        result.append(row)
    return result


def read_rows(path: Path, **kwargs) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON: {path.name}:{number}") from error
    return validated_rows(rows, label=str(path), **kwargs)


def check_partitions(train: list[dict], holdout: list[dict]) -> dict:
    all_ids = [record_id(row) for row in [*train, *holdout]]
    repeated_ids = [value for value, count in Counter(all_ids).items() if count > 1]
    if repeated_ids:
        raise ValueError(f"Duplicate record IDs: {repeated_ids[:5]}")
    train_counts = Counter(normalized_question(row["conversations"][0]["value"]) for row in train)
    holdout_counts = Counter(normalized_question(row["conversations"][0]["value"]) for row in holdout)
    overlap = set(train_counts) & set(holdout_counts)
    if overlap:
        raise ValueError(f"Normalized question occurs in train and holdout: {sorted(overlap)[:5]}")
    if any(count > 1 for count in holdout_counts.values()):
        raise ValueError("Duplicate normalized questions within holdout")
    return {"train_unique_normalized_questions": len(train_counts),
            "holdout_unique_normalized_questions": len(holdout_counts),
            "train_duplicate_question_excess_rows": sum(count - 1 for count in train_counts.values()),
            "train_holdout_normalized_question_overlap": 0, "duplicate_record_ids": 0}


def deduplicate_training_questions(streams: dict[str, list[dict]]) -> tuple[dict, list[dict]]:
    """Keep the first input question; record each exclusion without merging facts."""
    used, exclusions, seen = {}, [], {}
    for stream, rows in streams.items():
        used[stream] = []
        for row in rows:
            question = row["conversations"][0]["value"]
            key = normalized_question(question)
            if key not in seen:
                seen[key] = (stream, row)
                used[stream].append(row)
                continue
            kept_stream, kept = seen[key]
            exclusions.append({
                "excluded_id": record_id(row), "kept_id": record_id(kept),
                "question": question, "normalized_question": key,
                "excluded_stream": stream, "kept_stream": kept_stream,
                "excluded_answer_sha256": hashlib.sha256(row["conversations"][1]["value"].encode("utf-8")).hexdigest(),
                "kept_answer_sha256": hashlib.sha256(kept["conversations"][1]["value"].encode("utf-8")).hexdigest(),
                "reason": "duplicate_question_keep_first",
            })
    return used, exclusions


def build_plan(legacy_theory_train: Path, legacy_theory_holdout: Path,
               new_theory_train: Path, new_theory_holdout: Path,
               conversation_train: Path, conversation_holdout: Path, *,
               system_file: Path | None = None, system_source: Path = SYSTEM_SOURCE,
               answer_overrides: Path | None = None,
               additional_theory_train: list[Path] | None = None,
               additional_theory_holdout: list[Path] | None = None) -> tuple[dict, dict]:
    inputs = {"legacy_theory_train": Path(legacy_theory_train),
              "legacy_theory_holdout": Path(legacy_theory_holdout),
              "new_theory_train": Path(new_theory_train), "new_theory_holdout": Path(new_theory_holdout),
              "conversation_train": Path(conversation_train), "conversation_holdout": Path(conversation_holdout)}
    old_train = read_rows(inputs["legacy_theory_train"], training=True, theory=True, allow_unsplit=True)
    old_holdout = read_rows(inputs["legacy_theory_holdout"], training=False, theory=True, allow_unsplit=True)
    new_train = read_rows(inputs["new_theory_train"], training=True, theory=True)
    new_holdout = read_rows(inputs["new_theory_holdout"], training=False, theory=True)
    additional_train, additional_holdout = {}, {}
    for training, paths, streams, prefix in (
            (True, additional_theory_train or [], additional_train, "additional_theory_train"),
            (False, additional_theory_holdout or [], additional_holdout, "additional_theory_holdout")):
        for number, path in enumerate(paths, 1):
            key = f"{prefix}_{number:02d}"
            inputs[key] = Path(path)
            streams[key] = read_rows(inputs[key], training=training, theory=True)
    theory_streams = {"legacy_theory_train": old_train, "legacy_theory_holdout": old_holdout,
                      "new_theory_train": new_train, "new_theory_holdout": new_holdout,
                      **additional_train, **additional_holdout}
    override_edits = []
    if answer_overrides is not None:
        inputs["answer_overrides"] = Path(answer_overrides)
        override_edits = apply_answer_overrides(theory_streams, inputs["answer_overrides"])
    preparation_edits = []
    for stream, rows in theory_streams.items():
        preparation_edits.extend(sanitize_open_question_answers(rows, stream=stream))
    conversation = read_rows(inputs["conversation_train"], training=True, theory=False)
    new_casual = read_rows(inputs["conversation_holdout"], training=False, theory=False)
    old_casual = validated_rows(friendly_rows()[1], training=False, theory=False,
                                allow_unsplit=True, label="legacy authored conversation holdout")
    extra_train = [row for rows in additional_train.values() for row in rows]
    extra_holdout = [row for rows in additional_holdout.values() for row in rows]
    input_counts = {"legacy_theory_train": len(old_train), "legacy_theory_holdout": len(old_holdout),
                    "new_theory_train": len(new_train), "new_theory_holdout": len(new_holdout),
                    "additional_theory_train": len(extra_train), "additional_theory_holdout": len(extra_holdout),
                    "train_theory": len(old_train) + len(new_train) + len(extra_train), "train_conversation": len(conversation),
                    "mixed": len(old_train) + len(new_train) + len(extra_train) + len(conversation),
                    "holdout_theory": len(old_holdout) + len(new_holdout) + len(extra_holdout),
                    "holdout_conversation": len(old_casual) + len(new_casual),
                    "legacy_conversation_holdout": len(old_casual), "new_conversation_holdout": len(new_casual)}
    # Check all original identities before deduplication, so it cannot hide an
    # ID collision or any train/holdout question overlap.
    input_checks = check_partitions([*old_train, *new_train, *extra_train, *conversation],
                                    [*old_holdout, *new_holdout, *extra_holdout, *old_casual, *new_casual])
    used, exclusions = deduplicate_training_questions({"legacy_theory_train": old_train,
                                                       "new_theory_train": new_train,
                                                       **additional_train,
                                                       "conversation_train": conversation})
    old_train, new_train = used["legacy_theory_train"], used["new_theory_train"]
    conversation = used["conversation_train"]
    extra_train = [row for name in additional_train for row in used[name]]
    theory = [*old_train, *new_train, *extra_train]
    theory_holdout = [*old_holdout, *new_holdout, *extra_holdout]
    casual_holdout = [*old_casual, *new_casual]
    mixed = [*theory, *conversation]
    checks = check_partitions(mixed, [*theory_holdout, *casual_holdout])
    system = (Path(system_file).read_text(encoding="utf-8-sig").strip() if system_file
              else application_system_prompt(Path(system_source)))
    if not system.strip():
        raise ValueError("System prompt is empty")
    exports = {"train_theory": theory, "train_conversation": conversation, "mixed": mixed,
               "holdout_theory": theory_holdout, "holdout_conversation": casual_holdout}
    counts = {name: len(rows) for name, rows in exports.items()}
    counts.update(legacy_theory_train=len(old_train), legacy_theory_holdout=len(old_holdout),
                  new_theory_train=len(new_train), new_theory_holdout=len(new_holdout),
                  additional_theory_train=len(extra_train), additional_theory_holdout=len(extra_holdout),
                  legacy_conversation_holdout=len(old_casual), new_conversation_holdout=len(new_casual))
    sources = sorted({str(source["source_id"]) for row in [*theory, *theory_holdout]
                      for source in row["metadata"]["sources"]})
    summary = {
        "schema": "qwen_theory_round2_v1", "status": "prepared_text_only_candidate_data",
        "counts": counts, "partition_checks": checks,
        "input_counts": input_counts, "used_counts": copy.deepcopy(counts),
        "input_partition_checks": input_checks, "training_question_exclusions": exclusions,
        "preparation_edits": preparation_edits,
        "answer_overrides_applied": override_edits,
        "inputs": {name: {"path": str(path.resolve()), "sha256": file_hash(path)} for name, path in inputs.items()},
        "system_prompt": {"text": system, "source": str(Path(system_file or system_source).resolve()),
                          "sha256": hashlib.sha256(system.encode("utf-8")).hexdigest()},
        "source_ids": sources, "geologist_reviewed_rows": 0, "test_split_used": False,
        "images_in_training": 0, "old_conversation_holdout_origin": "prepare_qwen_source_pilot.friendly_rows()[1]",
        "limitations": [
            "Source-checked paraphrases are not expert-certified geological facts.",
            "Theory holdout includes related concepts and paraphrases; it is a regression check, not independent knowledge accuracy.",
            "Source links are validated structurally; preparation does not re-read or certify web content.",
            "Authored conversation replay is application dialogue, not an external geological dataset.",
            "Text-only training does not provide photo descriptions, facies labels, or new visual ground truth.",
            "No model weights, activation, or performance claim is produced by preparation.",
        ],
    }
    return exports, summary


def write_plan(output: Path, exports: dict, summary: dict) -> None:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing round: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}_", dir=output.parent))
    try:
        for name, rows in exports.items():
            (stage / f"{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (stage / "system_prompt.txt").write_text(summary["system_prompt"]["text"], encoding="utf-8")
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    except BaseException:
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(f".{output.name}_"):
            shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-theory-train", type=Path, default=THEORY_ROOT / "theory_training_v5.jsonl")
    parser.add_argument("--legacy-theory-holdout", type=Path, default=THEORY_ROOT / "theory_holdout_v5.jsonl")
    parser.add_argument("--new-theory-train", type=Path, required=True)
    parser.add_argument("--new-theory-holdout", type=Path, required=True)
    parser.add_argument("--additional-theory-train", type=Path, action="append", default=[],
                        help="Additional source-checked train JSONL; repeat for multiple files")
    parser.add_argument("--additional-theory-holdout", type=Path, action="append", default=[],
                        help="Additional source-checked holdout JSONL; repeat for multiple files")
    parser.add_argument("--conversation-train", type=Path, default=CONVERSATION_ROOT / "conversation_train.jsonl")
    parser.add_argument("--conversation-holdout", type=Path, default=CONVERSATION_ROOT / "conversation_holdout.jsonl")
    parser.add_argument("--system", type=Path, help="Use this saved system prompt instead of the static application instruction")
    parser.add_argument("--answer-overrides", type=Path,
                        help="Source-checked correction mapping; old text, sources and hashes remain in provenance")
    parser.add_argument("--output-dir", type=Path, default=APP_ROOT / "open_datasets/qwen_theory_round2_20261025")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    exports, summary = build_plan(args.legacy_theory_train, args.legacy_theory_holdout,
                                  args.new_theory_train, args.new_theory_holdout,
                                  args.conversation_train, args.conversation_holdout, system_file=args.system,
                                  answer_overrides=args.answer_overrides,
                                  additional_theory_train=args.additional_theory_train,
                                  additional_theory_holdout=args.additional_theory_holdout)
    summary.update(dry_run=not args.write, output_dir=str(args.output_dir.resolve()))
    if args.write:
        write_plan(args.output_dir, exports, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
