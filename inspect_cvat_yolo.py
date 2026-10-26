"""Validate CVAT YOLO Segmentation 1.1 exports without changing source data."""
from __future__ import annotations
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
import yaml

IMAGE_EXT={".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"}

def inspect(source):
    source=Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_file():
        if not zipfile.is_zipfile(source):
            raise ValueError("Выберите ZIP-архив CVAT или распакованную папку")
        with tempfile.TemporaryDirectory() as temp:
            with zipfile.ZipFile(source) as z:
                for member in z.infolist():
                    target=(Path(temp)/member.filename).resolve()
                    if not target.is_relative_to(Path(temp).resolve()):
                        raise ValueError("Небезопасный путь внутри ZIP")
                    if not member.is_dir() and member.file_size > 500_000_000:
                        raise ValueError("Слишком большой файл в ZIP")
                z.extractall(temp)
            return _inspect_folder(Path(temp))
    return _inspect_folder(source)

def _inspect_folder(root):
    files=[p for p in root.rglob("*") if p.is_file()]
    yaml_files=[p for p in files if p.name in ("data.yaml","dataset.yaml")]
    names={}
    if yaml_files:
        metadata=yaml.safe_load(yaml_files[0].read_text(encoding="utf-8")) or {}
        raw=metadata.get("names",{})
        if isinstance(raw,list):
            names={str(i):str(name) for i,name in enumerate(raw)}
        elif isinstance(raw,dict):
            names={str(k):str(v) for k,v in raw.items()}
    images=[p for p in files if p.suffix.lower() in IMAGE_EXT]
    labels=[p for p in files if p.suffix.lower()==".txt" and ("labels" in p.parts or "labels" in str(p.parent).lower())]
    by_class=Counter()
    errors=[]
    polygons=0
    for label in labels:
        for line_number,line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(),1):
            if not line.strip():
                continue
            parts=line.split()
            try:
                class_id=int(parts[0])
                coords=[float(x) for x in parts[1:]]
                if class_id < 0 or len(coords)<6 or len(coords)%2 or any(not 0<=x<=1 for x in coords):
                    raise ValueError("invalid polygon")
                if names and str(class_id) not in names:
                    raise ValueError("unknown class")
                by_class[str(class_id)]+=1
                polygons+=1
            except (ValueError,IndexError):
                errors.append({"file":str(label.relative_to(root)),"line":line_number})
    return {"format":"YOLO Segmentation","images":len(images),"label_files":len(labels),
            "polygons":polygons,"classes":names,"polygons_by_class":dict(by_class),
            "invalid_annotations":len(errors),"error_examples":errors[:25],
            "training_started":False,"source_unchanged":True}

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("source")
    args=parser.parse_args()
    print(json.dumps(inspect(args.source),ensure_ascii=False,indent=2))
