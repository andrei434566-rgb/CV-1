"""Summarize the held-out shell-material pilot without promoting its adapter."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
BASELINE = OUT / "midnight_pass_shell_v8_baseline.json"
POSTTRAIN = OUT / "midnight_pass_shell_pilot_posttrain.json"
METRICS = OUT / "core_vlm_midnight_shell_pilot_lora/training_metrics.json"
CARD = OUT / "core_vlm_midnight_shell_pilot_lora/MODEL_CARD.md"


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    syntactic = 0
    raw_shell = 0
    for row in records:
        try:
            json.loads(row["output"])
            syntactic += 1
        except (ValueError, TypeError):
            pass
        raw_shell += bool(re.search(r"раковин|ракуш|створк", row["output"].lower()))
    return {"records": len(records),
            "heldout_cores": sorted({r["core_id"] for r in records}),
            "nonempty_outputs": sum(bool(r["output"].strip()) for r in records),
            "syntactically_valid_json": syntactic,
            "required_single_segment_schema": sum(r["valid_one_segment_json"] for r in records),
            "exact_given_depths": sum(r["exact_input_depths"] for r in records),
            "shell_word_in_raw_output": raw_shell,
            "output_chars": [len(r["output"]) for r in records]}


def main() -> None:
    before = summarize(BASELINE)
    after = summarize(POSTTRAIN)
    if before["heldout_cores"] != after["heldout_cores"] or before["records"] != after["records"]:
        raise ValueError("Pre/post held-out sets differ")
    metric_data = json.loads(METRICS.read_text(encoding="utf-8"))
    # The original pilot manifest used `well_id` for a sediment core ID. This
    # was corrected in the manifest generator; preserve the raw metric and add
    # a clear correction rather than pretending these four cores are wells.
    summary = {"data_source": "USGS Midnight Pass, DOI 10.5066/P1CQ6QRR, CC0",
               "train_core_ids": ["MP2-E", "MP2-F", "MP2-G", "MP3-D"],
               "train_windows": metric_data["candidate_rows_used"],
               "heldout": before["heldout_cores"],
               "epochs": metric_data["epoch"],
               "train_loss": metric_data["train_loss"],
               "before": before, "after": after,
               "decision": "do_not_promote",
               "reason": "Required JSON schema and user-supplied depths failed on every held-out window both before and after training; tiny single-site sample gives no reliable generalization evidence.",
               "provenance_note": "The training_metrics.json independent_wells=4 counts sediment cores because of an earlier metadata name; these are four cores, not four wells.",
               "metric_warning": "Shell-word presence and JSON shape do not measure geological accuracy; no 90-95% performance is established."}
    (OUT / "midnight_pass_shell_pilot_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    card = f"""# Эксперимент: Midnight Pass, раковинный материал

Источник: [USGS Midnight Pass](https://www.usgs.gov/data/sediment-data-cores-collected-2025-midnight-pass-florida), DOI 10.5066/P1CQ6QRR, CC0.

15 окон из четырёх кернов (MP2-E, MP2-F, MP2-G, MP3-D) использованы для продолжения адаптера v8: 3 эпохи, learning rate 5e-6, train loss {metric_data['train_loss']:.3f}. Метки «раковинный слой» и «песок с раковинным материалом» перенесены из официальных журналов на внутренние участки интервалов и визуально просмотрены. Геолог их не утверждал; `training_approved=false` во всём манифесте. Керны MP3-B и MP3-E целиком отложены (8 окон).

| Проверка на 8 окнах | v8 до обучения | пилот после |
|---|---:|---:|
| Синтаксически корректный JSON | {before['syntactically_valid_json']}/8 | {after['syntactically_valid_json']}/8 |
| Заданная схема `description_segments` с одним сегментом | {before['required_single_segment_schema']}/8 | {after['required_single_segment_schema']}/8 |
| Точно повторены предоставленные границы | {before['exact_given_depths']}/8 | {after['exact_given_depths']}/8 |
| Слово о раковинах в сыром ответе | {before['shell_word_in_raw_output']}/8 | {after['shell_word_in_raw_output']}/8 |

**Решение:** не включать этот адаптер в приложение. На отложенных кернах модель продолжает нарушать схему и подставлять текст вместо числовых глубин. Наличие слова «раковины» не подтверждает правильность геологического описания. Изменение на столь малой выборке одного района не даёт оценки общей точности.

Ранее в `training_metrics.json` поле `independent_wells=4` фактически означало **четыре керна**, а не скважины. Генератор манифеста исправлен и теперь использует `core_id`.

Исходные ответы и фото: `../midnight_pass_shell_v8_baseline.json`, `../midnight_pass_shell_pilot_posttrain.json`, `../../data/usgs_midnight_pass_shell_pilot/contact_sheet.jpg`.
"""
    CARD.write_text(card, encoding="utf-8")
    print(json.dumps({"before": before, "after": after,
                      "decision": summary["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
