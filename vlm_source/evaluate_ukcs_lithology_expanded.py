"""Compare the UKCS pilot and expanded adapters on the same fixed whole-well split."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_barnegat_binary_pilot import MODEL, ROOT, predict
from inspect_ukcs_labels import LABELS as PILOT_NAMES
from prepare_ukcs_source_terminology import NAMES


DATA = ROOT / "data/ukcs_q204_q205_lithology_expanded"
OLD_ADAPTER = ROOT / "outputs/core_vlm_ukcs_lithology_pilot_lora"
NEW_ADAPTER = ROOT / "outputs/core_vlm_ukcs_lithology_expanded_lora"
OUTPUT = ROOT / "outputs/ukcs_lithology_expanded_evaluation.json"
MODES = ("pilot", "expanded")
CODE_BY_NAME = {russian: code for code, (_, russian) in NAMES.items()}
CODE_BY_NAME.update({russian: code for code, russian in PILOT_NAMES.items()})


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_code(answer: str) -> str | None:
    # Exact classes only: a verbose or ambiguous answer fails the specified format.
    cleaned = answer.casefold().strip().strip(' .,:;!"«»').strip()
    return CODE_BY_NAME.get(cleaned)


def score(rows: list[dict], majority: str) -> dict:
    result = {
        "count": len(rows),
        "expected_by_source_code": dict(Counter(row["expected_code"] for row in rows)),
        "majority_code": majority,
        "majority_correct": sum(row["expected_code"] == majority for row in rows),
    }
    for mode in MODES:
        key = f"{mode}_code"
        result[f"{mode}_correct"] = sum(row[key] == row["expected_code"] for row in rows)
        result[f"{mode}_predicted_by_source_code"] = dict(Counter(row[key] or "unparsed" for row in rows))
        f1s = []
        recalls = {}
        for code in NAMES:
            tp = sum(row["expected_code"] == code and row[key] == code for row in rows)
            fp = sum(row["expected_code"] != code and row[key] == code for row in rows)
            fn = sum(row["expected_code"] == code and row[key] != code for row in rows)
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
            recalls[code] = tp / (tp + fn) if tp + fn else None
        result[f"{mode}_macro_f1"] = sum(f1s) / len(f1s)
        result[f"{mode}_recall_by_source_code"] = recalls
    return result


def collapse_runs(rows: list[dict]) -> list[dict]:
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["source_run_id"]].append(row)
    collapsed = []
    for run_id, tiles in by_run.items():
        item = {"source_run_id": run_id, "well_id": tiles[0]["well_id"],
                "expected_code": tiles[0]["expected_code"], "tiles": len(tiles)}
        if any(tile["expected_code"] != item["expected_code"] for tile in tiles):
            raise ValueError(f"Mixed labels within source run: {run_id}")
        for mode in MODES:
            counts = Counter(tile[f"{mode}_code"] for tile in tiles)
            best = counts.most_common()
            item[f"{mode}_code"] = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None
        collapsed.append(item)
    return collapsed


def main() -> None:
    train = read_rows(DATA / "train_source_terminology.jsonl")
    holdout = read_rows(DATA / "holdout_fixed_source_terminology.jsonl")
    train_wells = {row["metadata"]["well_id"] for row in train}
    holdout_wells = {row["metadata"]["well_id"] for row in holdout}
    if train_wells & holdout_wells or len(train_wells) != 9 or len(holdout_wells) != 3:
        raise ValueError("Whole-well split failed")
    majority = Counter(row["metadata"]["source_label_code"] for row in train).most_common(1)[0][0]
    records = []
    for row in holdout:
        meta = row["metadata"]
        records.append({
            "record_id": meta["record_id"], "well_id": meta["well_id"],
            "source_run_id": meta["source_run_id"], "image": row["image"],
            "depth_top_m": meta["depth_top_m"], "depth_bottom_m": meta["depth_bottom_m"],
            "expected_code": meta["source_label_code"],
            "expected_ru_editorial": NAMES[meta["source_label_code"]][1],
        })
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, OLD_ADAPTER)
    model.eval()
    for i, (row, record) in enumerate(zip(holdout, records), 1):
        raw = predict(model, processor, row)
        record["pilot_raw"] = raw
        record["pilot_code"] = to_code(raw)
        if i % 10 == 0 or i == len(holdout):
            print(f"pilot {i}/{len(holdout)}", flush=True)
    model.load_adapter(NEW_ADAPTER, adapter_name="expanded")
    model.set_adapter("expanded")
    model.eval()
    for i, (row, record) in enumerate(zip(holdout, records), 1):
        raw = predict(model, processor, row)
        record["expanded_raw"] = raw
        record["expanded_code"] = to_code(raw)
        record["expanded_exact_ru_name"] = raw.strip() == record["expected_ru_editorial"]
        if i % 10 == 0 or i == len(holdout):
            print(f"expanded {i}/{len(holdout)}", flush=True)
    runs = collapse_runs(records)
    report = {
        "scope": "Experimental five-class UKCS lithology only, not full descriptions or depositional facies.",
        "source_class_mapping": {code: {"english_paper": en, "russian_editorial": ru}
                                 for code, (en, ru) in NAMES.items()},
        "training_crops": len(train), "train_wells": sorted(train_wells),
        "fixed_holdout_crops": len(holdout), "holdout_wells": sorted(holdout_wells),
        "same_new_russian_prompt_for_both_adapters": True,
        "tile_level": score(records, majority),
        "source_run_level": score(runs, majority),
        "per_well": {well: score([row for row in records if row["well_id"] == well], majority)
                     for well in sorted(holdout_wells)},
        "expanded_exact_expected_russian_name_count": sum(row["expanded_exact_ru_name"] for row in records),
        "limitations": [
            "Source label arrays and photos are public research data; crop labels have not been reviewed by a geologist.",
            "Russian names are conservative editorial translations of source classes, not geological validation.",
            "This is a fixed development holdout already evaluated in a previous pilot; not a new blind test.",
            "Multiple tiles may share a source run, so source-run results are reported separately.",
            "All wells come from a related UKCS area; external generalization remains unmeasured.",
        ],
        "source_runs": runs, "records": records,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tile_level": report["tile_level"], "source_run_level": report["source_run_level"]},
                     ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
