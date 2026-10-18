"""Rewrite the Fire Island pilot prompt with explicit numeric meter depths.

V1 asked for top_m/base_m after stating depths only in centimeters, which
confounded schema adherence with unit conversion. V1 files stay unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "data/usgs_fire_island_structure_pilot"
NEW = ROOT / "data/usgs_fire_island_structure_pilot_v2"


def rewrite(record: dict) -> dict:
    top, base = record["metadata"]["window_cm"]
    core_id = record["metadata"]["core_id"]
    target = json.loads(record["conversations"][1]["value"])["description_segments"][0]
    if target["top_m"] != round(top / 100, 5) or target["base_m"] != round(base / 100, 5):
        raise ValueError(f"Depth mismatch in {record['metadata']['record_id']}")
    prompt = (
        f"<image>\nЭто увеличенный фрагмент керна {core_id}, участок {top:g}–{base:g} см "
        f"от верха. Для ответа используй точно top_m={target['top_m']:.2f}, "
        f"base_m={target['base_m']:.2f}; не заменяй глубины словами. "
        "Кратко опиши по-русски видимый цвет и осадочную структуру: "
        "однородность, пятнистость, слои или их деформацию. Не угадывай "
        "минералы, точную зернистость, сортировку и обстановку осадконакопления. "
        "Верни только корректный JSON с ключом description_segments и ровно одним "
        "объектом с числовыми top_m, base_m и текстовым description."
    )
    record["conversations"][0]["value"] = prompt
    record["metadata"]["prompt_version"] = "explicit_meter_depths_v2"
    return record


def main() -> None:
    NEW.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ("train", "heldout"):
        rows = [rewrite(json.loads(line)) for line in (OLD / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
        (NEW / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        counts[split] = len(rows)
    if counts != {"train": 13, "heldout": 8}:
        raise ValueError(f"Unexpected v2 split: {counts}")
    summary = json.loads((OLD / "summary.json").read_text(encoding="utf-8"))
    summary.update({"prompt_version": "explicit_meter_depths_v2",
                    "change_from_v1": "Numerical meter depths are supplied verbatim in the user prompt; image windows and split unchanged."})
    (NEW / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"prompt_version": summary["prompt_version"], **counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
