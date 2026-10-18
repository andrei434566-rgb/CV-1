"""Create a tiny in-sample diagnostic for the training pipeline, never for quality claims."""

import json
from collections import defaultdict
from pathlib import Path


root = Path(__file__).resolve().parent
source = root / "data/r281_lithology_experiment/train.jsonl"
output = root / "data/r281_lithology_experiment/canary_train.jsonl"
wanted = {"песчаник": 2, "аргиллит": 2, "переслаивание": 2, "алевролит": 1, "уголь": 1}
by_class = defaultdict(list)
for line in source.read_text(encoding="utf-8").splitlines():
    if line:
        row = json.loads(line)
        by_class[row["metadata"]["rock_type"]].append(row)
selected = []
for label, count in wanted.items():
    if len(by_class[label]) < count:
        raise ValueError(label)
    selected.extend(by_class[label][:count])
output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
print(json.dumps({"output": str(output), "rows": len(selected), "classes": wanted}, ensure_ascii=False))
