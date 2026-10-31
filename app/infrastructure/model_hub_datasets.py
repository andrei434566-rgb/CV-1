"""Create candidate datasets exclusively from centrally accepted expert packets."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from app.infrastructure.model_hub import digest, inside, read_json, verify_files, atomic_json


def prepare_datasets(hub: Path, output: Path) -> dict:
    hub, output = Path(hub), Path(output)
    rows, identities, photo_wells = [], {}, defaultdict(set)
    for folder in sorted((hub / "training/approved").iterdir()):
        if not folder.is_dir() or not (folder / "approval.json").is_file():
            continue
        if read_json(folder / "approval.json").get("central_approved") is not True:
            continue
        receipt = read_json(folder / "receipt.json")
        if not verify_files(folder, receipt["files"]):
            raise ValueError("Принятый пакет повреждён: " + folder.name)
        for row in read_json(folder / "packet.json")["annotations"]:
            if row.get("geometry_approved") is not True:
                continue
            identity = json.dumps([row["source_photo_sha256"], row["polygon"], row.get("depth_from"), row.get("depth_to")], sort_keys=True)
            old = identities.get(identity)
            if old is not None:
                if old["label"] != row["label"]:
                    raise ValueError("Один контур имеет разные подтверждённые фации. Устраните конфликт перед обучением")
                semantic = lambda attrs: {k: v for k, v in attrs.items() if not k.startswith(("__", "Источник")) or k in {
                    "__vlm_description", "__vlm_lithology", "__vlm_texture", "__vlm_facies_association", "__vlm_depositional_environment", "__vlm_hydrodynamic_regime"}}
                if old.get("text_approved") and row.get("text_approved") and semantic(old.get("attributes", {})) != semantic(row.get("attributes", {})):
                    raise ValueError("Один контур имеет разные подтверждённые описания. Нужна центральная сверка")
                if old.get("text_approved") or not row.get("text_approved"):
                    continue
                rows.remove(old)
            key = hashlib.sha256(json.dumps([row["source_photo_sha256"], row["polygon"], row["label"],
                row.get("depth_from"), row.get("depth_to"), row.get("attributes")], sort_keys=True).encode()).hexdigest()
            photo_wells[row["source_photo_sha256"]].add(row.get("well") or "")
            entry = {**row, "source": inside(folder, row["image"]), "key": key}
            rows.append(entry)
            identities[identity] = entry
    if not rows:
        raise ValueError("Нет центрально принятых интервалов")
    if any(len(wells) > 1 for wells in photo_wells.values()):
        raise ValueError("Одинаковое фото подписано разными скважинами; нужна проверка")
    groups = sorted({row.get("well") or row["source_photo_sha256"] for row in rows})
    validation = set(groups[-max(1, round(len(groups) * .2)):]) if len(groups) >= 2 else set()
    classes = sorted({row["label"] for row in rows})
    output.mkdir(parents=True, exist_ok=False)
    vlm = {"train": [], "val": []}
    exported = []
    for row in rows:
        group = row.get("well") or row["source_photo_sha256"]
        split = "val" if group in validation else "train"
        with Image.open(row["source"]) as image:
            image = image.convert("RGB")
            sx, sy = image.width / row["canvas_size"][0], image.height / row["canvas_size"][1]
            points = [(float(x) * sx, float(y) * sy) for x, y in row["polygon"]]
            if len(points) < 3 or any(x < 0 or y < 0 or x > image.width or y > image.height for x, y in points):
                raise ValueError("Некорректная подтверждённая геометрия")
            left, top = max(0, int(min(x for x, _ in points))), max(0, int(min(y for _, y in points)))
            right, bottom = min(image.width, int(max(x for x, _ in points)) + 1), min(image.height, int(max(y for _, y in points)) + 1)
            if right - left < 8 or bottom - top < 8:
                continue
            crop = image.crop((left, top, right, bottom))
            name = row["key"] + ".jpg"
            images = output / "yolo/images" / split
            labels = output / "yolo/labels" / split
            images.mkdir(parents=True, exist_ok=True); labels.mkdir(parents=True, exist_ok=True)
            crop.save(images / name, quality=96)
            normalized = [(max(0, min(1, (x - left) / crop.width)), max(0, min(1, (y - top) / crop.height))) for x, y in points]
            coordinates = " ".join(f"{value:.8f}" for point in normalized for value in point)
            (labels / (row["key"] + ".txt")).write_text(f"{classes.index(row['label'])} {coordinates}\n", encoding="utf-8")
            attrs = row.get("attributes", {})
            description = str(attrs.get("__vlm_description") or attrs.get("Краткое описание") or attrs.get("Литологическое описание (16 параметров)") or "").strip()
            if row.get("text_approved") is True and description:
                mask = Image.new("L", crop.size)
                ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in points], fill=255)
                clean = Image.new("RGB", crop.size, "white"); clean.paste(crop, mask=mask)
                target = output / "vlm/images" / name
                target.parent.mkdir(parents=True, exist_ok=True); clean.save(target, quality=96)
                answer = description + "\nФация: " + row["label"]
                if attrs.get("__vlm_hydrodynamic_regime"):
                    answer += "\nГидродинамический режим: " + attrs["__vlm_hydrodynamic_regime"]
                vlm[split].append({"image": "images/" + name, "conversations": [
                    {"from": "human", "value": "<image>Опиши видимые признаки керна и предложи фацию по-русски. Отделяй наблюдения от интерпретации."},
                    {"from": "gpt", "value": answer}],
                    "metadata": {"training_approved": True, "geology_approved": True, "source_well": row.get("well"),
                                 "source_photo_sha256": row["source_photo_sha256"], "group": group, "split": split}})
            exported.append({"key": row["key"], "group": group, "split": split, "label": row["label"]})
    for split, samples in vlm.items():
        destination = output / "vlm" / (split + ".jsonl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples), encoding="utf-8")
    import yaml
    (output / "yolo").mkdir(exist_ok=True)
    (output / "yolo/data.yaml").write_text(yaml.safe_dump({"path": str((output / "yolo").resolve()),
        "train": "images/train", "val": "images/val", "names": dict(enumerate(classes))}, allow_unicode=True), encoding="utf-8")
    report = {"schema": "cv1-central-dataset-v1", "samples": exported, "validation_groups": sorted(validation),
              "independent_validation_available": bool(validation), "vlm_train": len(vlm["train"]), "vlm_val": len(vlm["val"]),
              "note": "Не публикует модель. Обучение и независимая оценка выполняются отдельно."}
    atomic_json(output / "dataset_report.json", report)
    return report
