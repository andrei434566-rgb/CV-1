"""Post-hoc binary grouping of UKCS outputs and independent external transfer."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_barnegat_binary_pilot import MODEL, ROOT, macro_f1, predict
from evaluate_barnegat_transfer import source_rows
from evaluate_ukcs_lithology_pilot import ADAPTER, DATA, normalize, read_rows


OUTPUT = ROOT / "outputs" / "ukcs_lithology_binary_transfer_evaluation.json"
MAP = {
    "песчаник": "песчаный",
    "глинистый песчаник": "песчаный",
    "нефтенасыщенный песчаник": "песчаный",
    "песчанистый аргиллит": "тонкозернистый",
    "аргиллит": "тонкозернистый",
}


def score(rows: list[dict]) -> dict:
    majority = Counter(r["expected"] for r in rows).most_common(1)[0][0]
    result = {
        "count": len(rows),
        "expected_distribution": dict(Counter(r["expected"] for r in rows)),
        "majority_correct": sum(r["expected"] == majority for r in rows),
        "base_correct": sum(r["base_label"] == r["expected"] for r in rows),
        "adapter_correct": sum(r["adapter_label"] == r["expected"] for r in rows),
        "base_macro_f1": macro_f1(rows, "base_label"),
        "adapter_macro_f1": macro_f1(rows, "adapter_label"),
        "adapter_prediction_distribution": dict(Counter(r["adapter_label"] or "неопределено" for r in rows)),
    }
    return result


def collapse(rows: list[dict], key: str) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result = []
    for group, tiles in grouped.items():
        item = {"group": group, "expected": tiles[0]["expected"], "tiles": len(tiles)}
        for mode in ("base", "adapter"):
            counts = Counter(r[f"{mode}_label"] for r in tiles)
            best = counts.most_common()
            item[f"{mode}_label"] = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None
        result.append(item)
    return result


def in_domain() -> dict:
    report = json.loads((ROOT / "outputs" / "ukcs_lithology_pilot_evaluation.json").read_text(encoding="utf-8"))
    rows = []
    for row in report["records"]:
        rows.append({
            "group": row["source_run_id"],
            "expected": MAP[row["expected"]],
            "base_label": MAP.get(row["base_label"]),
            "adapter_label": MAP.get(row["adapter_label"]),
        })
    return {"tile_level": score(rows), "original_run_level": score(collapse(rows, "group"))}


def main() -> None:
    prompt = read_rows("train_pilot.jsonl")[0]["conversations"][0]["value"]
    rows = source_rows()
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    base.eval()
    for index, row in enumerate(rows, 1):
        answer = predict(base, processor, {"image": row["image"], "conversations": [{"from": "human", "value": prompt}]})
        row["base_raw"] = answer
        row["base_five_class"] = normalize(answer)
        row["base_label"] = MAP.get(row["base_five_class"])
        print(f"base {index}/{len(rows)}", flush=True)
    tuned = PeftModel.from_pretrained(base, ADAPTER)
    tuned.eval()
    for index, row in enumerate(rows, 1):
        answer = predict(tuned, processor, {"image": row["image"], "conversations": [{"from": "human", "value": prompt}]})
        row["adapter_raw"] = answer
        row["adapter_five_class"] = normalize(answer)
        row["adapter_label"] = MAP.get(row["adapter_five_class"])
        print(f"adapter {index}/{len(rows)}", flush=True)
    r41 = [row for row in rows if row["dataset"] == "r41_ancient_core"]
    cat = [row for row in rows if row["dataset"] == "cat_island_modern_core"]
    report = {
        "scope": "Exploratory post-hoc binary grouping of five UKCS lithologies and transfer to source labels from R-41 and Cat Island.",
        "mapping": MAP,
        "prompt": prompt,
        "ukcs_heldout": in_domain(),
        "r41_tiles": score(r41),
        "r41_original_intervals": score(collapse(r41, "group")),
        "cat_island_tiles": score(cat),
        "limitations": [
            "The binary grouping was chosen after seeing the primary five-class result; it is exploratory, not a pre-registered primary metric.",
            "External source labels and crop alignment were automatically mapped and are not geologist-approved.",
            "R-41 has only one well and highly imbalanced source intervals; Cat Island has only seven wells and provisional cropped labels.",
        ],
        "external_records": rows,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("ukcs_heldout", "r41_original_intervals", "cat_island_tiles")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
