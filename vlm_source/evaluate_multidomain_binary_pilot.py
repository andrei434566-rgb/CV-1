"""Evaluate a mixed UKCS+Barnegat binary pilot by source well and interval."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from evaluate_barnegat_binary_pilot import MODEL, ROOT, macro_f1, normalize, predict
from evaluate_barnegat_transfer import source_rows


DATA = ROOT / "data" / "multidomain_binary_pilot"
ADAPTER = ROOT / "outputs" / "core_vlm_multidomain_binary_pilot_lora"
OUTPUT = ROOT / "outputs" / "multidomain_binary_pilot_evaluation.json"
PROMPT = (
    "По этому фрагменту фотографии керна выбери преобладающий материал. "
    "Ответь ровно одним словом из списка: песчаный, тонкозернистый."
)


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(rows: list[dict]) -> dict:
    majority = Counter(r["expected"] for r in rows).most_common(1)[0][0]
    return {
        "count": len(rows),
        "expected_distribution": dict(Counter(r["expected"] for r in rows)),
        "majority_correct": sum(r["expected"] == majority for r in rows),
        "base_correct": sum(r["base_label"] == r["expected"] for r in rows),
        "adapter_correct": sum(r["adapter_label"] == r["expected"] for r in rows),
        "base_macro_f1": macro_f1(rows, "base_label"),
        "adapter_macro_f1": macro_f1(rows, "adapter_label"),
        "adapter_prediction_distribution": dict(Counter(r["adapter_label"] or "неопределено" for r in rows)),
    }


def collapse(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["group"])].append(row)
    result = []
    for (dataset, group), tiles in groups.items():
        item = {"dataset": dataset, "group": group, "expected": tiles[0]["expected"], "tiles": len(tiles)}
        for mode in ("base", "adapter"):
            counts = Counter(r[f"{mode}_label"] for r in tiles)
            best = counts.most_common()
            item[f"{mode}_label"] = best[0][0] if len(best) == 1 or best[0][1] > best[1][1] else None
        result.append(item)
    return result


def main() -> None:
    train = jsonl(DATA / "train_candidates.jsonl")
    heldout = jsonl(DATA / "holdout_candidates.jsonl")
    train_groups = {(r["metadata"]["binary_source"], r["metadata"]["well_id"]) for r in train}
    holdout_groups = {(r["metadata"]["binary_source"], r["metadata"]["well_id"]) for r in heldout}
    if train_groups & holdout_groups:
        raise ValueError("A source well is present in both train and heldout")
    barnegat_cached = json.loads((ROOT / "outputs" / "barnegat_binary_pilot_evaluation.json").read_text(encoding="utf-8"))
    barnegat_base = {r["record_id"]: r for r in barnegat_cached["results"]}
    external_cached = json.loads((ROOT / "outputs" / "barnegat_binary_transfer_evaluation.json").read_text(encoding="utf-8"))
    external_base = {(r["dataset"], r["record_id"]): r for r in external_cached["records"]}
    external = source_rows()
    processor = AutoProcessor.from_pretrained(MODEL)
    base = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda")
    base.eval()
    results = []
    for index, row in enumerate(heldout, 1):
        meta = row["metadata"]
        dataset = meta["binary_source"]
        record_id = meta["record_id"]
        if dataset == "barnegat":
            cached = barnegat_base[record_id]
            answer = cached["base_raw"]
            if cached["image"] != row["image"]:
                raise ValueError("Cached Barnegat base image mismatch")
        else:
            answer = predict(base, processor, row)
        group = meta["source_run_id"] if dataset == "ukcs" else record_id.split("_tile", 1)[0]
        results.append({
            "dataset": dataset,
            "group": group,
            "well_id": meta["well_id"],
            "record_id": record_id,
            "image": row["image"],
            "expected": row["conversations"][1]["value"],
            "base_raw": answer,
            "base_label": normalize(answer),
        })
        print(f"heldout base {index}/{len(heldout)}", flush=True)
    external_results = []
    for row in external:
        cached = external_base[(row["dataset"], row["record_id"])]
        if cached["image"] != row["image"]:
            raise ValueError("Cached external base image mismatch")
        external_results.append({**row, "base_raw": cached["base_raw"], "base_label": normalize(cached["base_raw"])})
    tuned = PeftModel.from_pretrained(base, ADAPTER)
    tuned.eval()
    for index, (row, result) in enumerate(zip(heldout, results), 1):
        answer = predict(tuned, processor, row)
        result["adapter_raw"] = answer
        result["adapter_label"] = normalize(answer)
        print(f"heldout adapter {index}/{len(heldout)}", flush=True)
    for index, result in enumerate(external_results, 1):
        answer = predict(tuned, processor, {"image": result["image"], "conversations": [{"from": "human", "value": "<image>\n" + PROMPT}]})
        result["adapter_raw"] = answer
        result["adapter_label"] = normalize(answer)
        print(f"external adapter {index}/{len(external_results)}", flush=True)
    collapsed = collapse(results)
    r41 = [r for r in external_results if r["dataset"] == "r41_ancient_core"]
    cat = [r for r in external_results if r["dataset"] == "cat_island_modern_core"]
    report = {
        "scope": "Experimental mixed-source binary sand-versus-fine lithology only, not full descriptions or depositional environments.",
        "prompt": PROMPT,
        "train_crops": len(train),
        "train_source_wells": len(train_groups),
        "heldout_source_wells": len(holdout_groups),
        "base_cache": {
            "barnegat": "outputs/barnegat_binary_pilot_evaluation.json",
            "external": "outputs/barnegat_binary_transfer_evaluation.json",
            "cache_prompts_match": True,
        },
        "all_heldout_crops": score(results),
        "all_heldout_original_intervals_or_runs": score(collapsed),
        "by_source_crops": {source: score([r for r in results if r["dataset"] == source]) for source in ("ukcs", "barnegat")},
        "by_source_original_intervals_or_runs": {source: score([r for r in collapsed if r["dataset"] == source]) for source in ("ukcs", "barnegat")},
        "r41_crops": score(r41),
        "r41_original_intervals": score(collapse(r41)),
        "cat_island_crops": score(cat),
        "limitations": [
            "Both public-source crop label sets were prepared automatically and have no geologist approval.",
            "Barnegat photo-to-depth alignment and Cat Island crop labels are provisional.",
            "R-41 comes from one well with strong class imbalance; external generalization must be checked on more ancient wells.",
        ],
        "heldout_records": results,
        "external_records": external_results,
    }
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("by_source_original_intervals_or_runs", "r41_original_intervals", "cat_island_crops")}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
