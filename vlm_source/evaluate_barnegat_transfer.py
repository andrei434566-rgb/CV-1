"""Diagnose binary Barnegat adapter transfer on unseen R-41 and Cat Island data.

The external labels are provisional. This report is not a certification of the
model, and the two external datasets must not be included in training here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_barnegat_binary_pilot import ADAPTER, MODEL, ROOT, macro_f1, normalize, predict


OUTPUT = ROOT / "outputs" / "barnegat_binary_transfer_evaluation.json"
PROMPT = (
    "По этому фрагменту фотографии керна выбери преобладающий материал. "
    "Ответь ровно одним словом из списка: песчаный, тонкозернистый."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_rows() -> list[dict]:
    rows: list[dict] = []
    r41 = read_jsonl(ROOT / "data" / "v8_single_column" / "test_shared_v7_rows.jsonl")
    for row in r41:
        description = row["metadata"]["source_description_full"].lower().strip()
        if description.startswith("песчаник"):
            label = "песчаный"
        elif description.startswith(("алевролит", "аргиллит")):
            label = "тонкозернистый"
        else:
            continue
        rows.append({
            "dataset": "r41_ancient_core",
            "group": str(row["metadata"]["source_row"]),
            "record_id": Path(row["image"]).stem,
            "image": row["image"],
            "expected": label,
            "source_label": description.split(maxsplit=1)[0],
        })
    cat = read_jsonl(ROOT / "data" / "cat_island_four_class_pilot" / "heldout.jsonl")
    for row in cat:
        source_label = json.loads(row["conversations"][1]["value"])["lithology"].lower()
        if source_label == "песок":
            label = "песчаный"
        elif source_label in ("глина", "ил"):
            label = "тонкозернистый"
        else:
            # "Глинистый песок" lacks an unambiguous binary dominant class.
            continue
        rows.append({
            "dataset": "cat_island_modern_core",
            "group": row["metadata"]["source_well"],
            "record_id": row["metadata"]["record_id"],
            "image": row["image"],
            "expected": label,
            "source_label": source_label,
        })
    return rows


def score(rows: list[dict]) -> dict:
    majority_label = Counter(r["expected"] for r in rows).most_common(1)[0][0]
    return {
        "count": len(rows),
        "groups": len({r["group"] for r in rows}),
        "expected_distribution": dict(Counter(r["expected"] for r in rows)),
        "majority_label": majority_label,
        "majority_correct": sum(r["expected"] == majority_label for r in rows),
        "base_correct": sum(r["base_label"] == r["expected"] for r in rows),
        "adapter_correct": sum(r["adapter_label"] == r["expected"] for r in rows),
        "base_macro_f1": macro_f1(rows, "base_label"),
        "adapter_macro_f1": macro_f1(rows, "adapter_label"),
        "base_prediction_distribution": dict(Counter(r["base_label"] or "неопределено" for r in rows)),
        "adapter_prediction_distribution": dict(Counter(r["adapter_label"] or "неопределено" for r in rows)),
    }


def collapse_r41(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    collapsed = []
    for group, tiles in grouped.items():
        result = {"group": group, "expected": tiles[0]["expected"], "tile_count": len(tiles)}
        for mode in ("base", "adapter"):
            counts = Counter(tile[f"{mode}_label"] for tile in tiles)
            best = counts.most_common()
            result[f"{mode}_label"] = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None
        collapsed.append(result)
    return collapsed


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    rows = source_rows()
    processor = AutoProcessor.from_pretrained(MODEL)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    model.eval()
    for index, row in enumerate(rows, 1):
        answer = predict(model, processor, {
            "image": row["image"],
            "conversations": [{"from": "human", "value": f"<image>\n{PROMPT}"}],
        })
        row["base_raw"] = answer
        row["base_label"] = normalize(answer)
        print(f"base {index}/{len(rows)}", flush=True)
    tuned = PeftModel.from_pretrained(model, ADAPTER)
    tuned.eval()
    for index, row in enumerate(rows, 1):
        answer = predict(tuned, processor, {
            "image": row["image"],
            "conversations": [{"from": "human", "value": f"<image>\n{PROMPT}"}],
        })
        row["adapter_raw"] = answer
        row["adapter_label"] = normalize(answer)
        print(f"adapter {index}/{len(rows)}", flush=True)
    r41 = [r for r in rows if r["dataset"] == "r41_ancient_core"]
    cat = [r for r in rows if r["dataset"] == "cat_island_modern_core"]
    r41_intervals = collapse_r41(r41)
    report = {
        "scope": "Exploratory external transfer, binary sand versus fine only; no full descriptions or environments.",
        "prompt": PROMPT,
        "training_overlap": "No R-41 or Cat Island examples were used to train this Barnegat adapter.",
        "r41_tiles": score(r41),
        "r41_original_intervals": score(r41_intervals),
        "cat_island_tiles": score(cat),
        "cat_island_excluded_ambiguous_clayey_sand": 2,
        "limitations": [
            "External labels were mapped automatically from source descriptions and not reviewed crop by crop by a geologist.",
            "R-41 contains only one well, with 16 sandstone and 3 siltstone intervals; its majority baseline is high.",
            "Multiple R-41 tiles per original description are correlated and counted separately only in the tile diagnostic.",
            "Cat Island crop/photo-depth alignment and original four-class labels have not been independently verified.",
        ],
        "r41_interval_records": r41_intervals,
        "records": rows,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("r41_original_intervals", "cat_island_tiles")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
