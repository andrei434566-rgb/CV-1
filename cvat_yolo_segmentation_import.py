"""Local CVAT YOLO Segmentation 1.1 audit; no training or source mutation.

Supports exported ZIP or directory, images and YOLO polygon TXT labels.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import tempfile
import zipfile

IMAGES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def audit(source: str | Path, destination: str | Path) -> dict:
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cv1_cvat_") as temp:
        root = source
        if source.is_file():
            if not zipfile.is_zipfile(source):
                raise ValueError("Нужен ZIP из CVAT или распакованная папка")
            root = Path(temp)
            with zipfile.ZipFile(source) as archive:
                for item in archive.infolist():
                    target = (root / item.filename).resolve()
                    if not target.is_relative_to(root.resolve()):
                        raise ValueError("Небезопасный путь в ZIP")
                    if item.is_dir():
                        continue
                    if item.file_size > 1024 * 1024 * 1024:
                        raise ValueError("Слишком большой файл в архиве")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as inp, target.open("wb") as out:
                        import shutil
                        shutil.copyfileobj(inp, out)
        images = list(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGES)
        labels = list(p for p in root.rglob("*.txt") if p.is_file() and p.parent.name.lower() in ("labels", "obj_train_data", "annotations"))
        if not labels:
            labels = [p for p in root.rglob("*.txt") if p.is_file() and p.stem in {im.stem for im in images}]
        counts = Counter()
        invalid = []
        labeled_images = set()
        by_stem = {}
        for im in images:
            by_stem.setdefault(im.stem, []).append(im)
        for label in labels:
            matches = by_stem.get(label.stem, [])
            if not matches:
                invalid.append({"file": str(label.relative_to(root)), "error": "image_missing"})
                continue
            labeled_images.add(matches[0])
            for line_number, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    cls = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    if cls < 0 or len(coords) < 6 or len(coords) % 2 or any(not 0 <= v <= 1 for v in coords):
                        raise ValueError("invalid polygon")
                    counts[str(cls)] += 1
                except (ValueError, IndexError):
                    invalid.append({"file": str(label.relative_to(root)), "line": line_number, "error": "invalid_polygon"})
        report = {
            "format": "CVAT YOLO Segmentation 1.1",
            "source": str(source),
            "images": len(images),
            "label_files": len(labels),
            "images_with_labels": len(labeled_images),
            "images_without_labels": len(images) - len(labeled_images),
            "polygons_per_class": dict(sorted(counts.items(), key=lambda x: int(x[0]))),
            "invalid_count": len(invalid),
            "invalid_examples": invalid[:100],
            "training_ready": False,
            "note": "Audit only. Check class mapping, image-label pairing and train/val split before training."
        }
        output = destination / "cvat_yolo_segmentation_audit.json"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["report_file"] = str(output)
        return report
