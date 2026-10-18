"""Build Qwen-format review candidates from photo/log pairs and log transcripts."""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
DATA = PROJECT / "data"
PAIR_FILE = DATA / "external_candidate_pairs.jsonl"
TRANSCRIPT_FILE = DATA / "external_log_transcriptions.jsonl"
OUTPUT_FILE = DATA / "external_qwen3_vl_candidates.jsonl"
NUMBER = r"\d+(?:[.,]\d+)?"
INTERVAL_RE = re.compile(rf"(?im)^\s*({NUMBER})\s*[-–—]\s*({NUMBER})\s*(cm|m)\s*:\s*")
PHOTO_DEPTH_RE = re.compile(rf"_(\d+(?:[.,]\d+)?)-(\d+(?:[.,]\d+)?)(?:cm)?(?:_|\.|$)", re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def number(value: str) -> float:
    return float(value.replace(",", "."))


def parse_intervals(transcript: str) -> list[dict]:
    matches = list(INTERVAL_RE.finditer(transcript))
    result = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(transcript)
        description = re.sub(r"\s+", " ", transcript[start:end]).strip(" \t\r\n;•-")
        unit = match.group(3).lower()
        factor = 0.01 if unit == "cm" else 1.0
        top_m = number(match.group(1)) * factor
        base_m = number(match.group(2)) * factor
        if base_m > top_m and description:
            result.append({"top_m": top_m, "base_m": base_m, "description": description})
    return result


def photo_interval(photo_path: str) -> tuple[float, float] | None:
    match = PHOTO_DEPTH_RE.search(Path(photo_path).name)
    if not match:
        return None
    return number(match.group(1)) / 100.0, number(match.group(2)) / 100.0


def clip_layers(layers: list[dict], interval: tuple[float, float] | None) -> tuple[list[dict], bool]:
    if interval is None:
        return layers, False
    photo_top, photo_base = interval
    clipped = []
    did_clip = False
    for layer in layers:
        top = max(photo_top, layer["top_m"])
        base = min(photo_base, layer["base_m"])
        if base <= top:
            continue
        did_clip = did_clip or top != layer["top_m"] or base != layer["base_m"]
        clipped.append({**layer, "top_m": top, "base_m": base})
    return clipped, did_clip


def main() -> None:
    pairs = load_jsonl(PAIR_FILE)
    transcripts = {row.get("log_path"): row for row in load_jsonl(TRANSCRIPT_FILE)}
    records = []
    skipped = {"missing_transcript": 0, "non_interval_transcript": 0, "empty_segment_overlap": 0,
               "non_log_summary": 0}
    for pair in pairs:
        log_path = pair.get("log_path")
        transcript_row = transcripts.get(log_path)
        if not transcript_row or not transcript_row.get("transcript", "").strip():
            skipped["missing_transcript"] += 1
            continue
        # Apalachicola's legacy summary sheets contain a colored graphic/legend,
        # not the detailed text log. Do not turn that legend into a text target.
        if "CoreSummaries" in (log_path or ""):
            skipped["non_log_summary"] += 1
            continue
        layers = parse_intervals(transcript_row["transcript"])
        if not layers:
            skipped["non_interval_transcript"] += 1
            continue
        segment, clipped = clip_layers(layers, photo_interval(pair["photo_path"]))
        if not segment:
            skipped["empty_segment_overlap"] += 1
            continue
        answer = {"description_segments": segment}
        prompt = (
            "Опиши видимую часть керна на фото. Верни только JSON с ключом "
            "description_segments; для каждого интервала укажи top_m, base_m "
            "(метры от верха этого керна) и description. Сохраняй язык и "
            "терминологию описания источника; не назначай среду осадконакопления "
            "и не добавляй признаки, которых нет в источнике."
        )
        records.append({
            "image": str((PROJECT / pair["photo_path"]).resolve()),
            "conversations": [
                {"from": "human", "value": "<image>\n" + prompt},
                {"from": "gpt", "value": json.dumps(answer, ensure_ascii=False)},
            ],
            "metadata": {
                "record_id": pair["pair_id"],
                "dataset": pair["dataset"],
                "core_key": pair["core_key"],
                "log_path": log_path,
                "transcription_method": transcript_row["transcription_method"],
                "transcription_review_status": transcript_row["transcription_review_status"],
                "training_approved": False,
                "source_doi": pair["doi"],
                "source_metadata_url": pair["metadata_url"],
                "setting": pair["setting"],
                "photo_interval_m": list(photo_interval(pair["photo_path"])) if photo_interval(pair["photo_path"]) else None,
                "description_layers_from_source_log": len(layers),
                "description_layers_in_photo_target": len(segment),
                "target_interval_clipped_to_photo_extent": clipped,
                "target_language": "source language preserved",
                "review_status": "candidate_requires_geologist_review",
            },
        })

    with OUTPUT_FILE.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "candidate_records": len(records),
        "approved_records": 0,
        "skipped": skipped,
        "output": str(OUTPUT_FILE),
        "note": "OCR-derived, unreviewed candidates; not automatically included in the approved training set.",
    }
    (DATA / "external_training_candidate_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
