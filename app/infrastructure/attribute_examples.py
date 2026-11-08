"""Local reviewed image/attribute examples, separate from automatic guesses."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
from uuid import uuid4

EVIDENCE_FIELDS = ("Слоистость", "Биотурбация", "Текстура")
EVIDENCE_KEY = "__feature_evidence"
SCHEMA = "cv1-reviewed-attributes-v1"


def default_root():
    from app.runtime_paths import user_data_root
    return user_data_root() / "training" / "attribute_examples"


def pixel_digest(pixmap):
    """Digest pixels, avoiding PNG encoder/platform-dependent metadata."""
    from PySide6.QtGui import QImage
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    return hashlib.sha256(f"{image.width()}:{image.height()}:".encode() + bytes(image.constBits())).hexdigest()


def png_bytes(pixmap):
    from PySide6.QtCore import QBuffer, QIODevice
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pixmap.save(buffer, "PNG"):
        raise ValueError("Не удалось сохранить фрагмент керна")
    return bytes(buffer.data())


def read_evidence(attributes):
    try:
        value = json.loads(attributes.get(EVIDENCE_KEY, "{}"))
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def valid_evidence(entry, value, image_digest):
    if not isinstance(entry, dict) or entry.get("value") != value or entry.get("image_sha256") != image_digest:
        return False
    try:
        l, t, r, b = map(float, entry["roi"])
        return all(math.isfinite(v) for v in (l,t,r,b)) and 0 <= l < r <= 1 and 0 <= t < b <= 1
    except (KeyError, TypeError, ValueError):
        return False


def _atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _descriptor(image):
    import numpy as np
    from PIL import Image
    image = image.convert("RGB"); image.thumbnail((160,160))
    a = np.asarray(image)
    pixels = a.reshape(-1,3)
    pixels = pixels[pixels.min(axis=1) < 240]
    if len(pixels) < 20:
        return None
    features = []
    for c in range(3):
        hist, _ = np.histogram(pixels[:,c], bins=16, range=(0,256))
        features.extend((hist / max(1,hist.sum())).tolist())
    vector = np.asarray(features, dtype=float)
    return (vector / max(float(np.linalg.norm(vector)), 1e-9)).tolist()


def write_example(payload, image_png, *, root=None):
    """Replace this interval's latest review; unchecked edits revoke approval."""
    from PIL import Image
    from app.infrastructure.description_template import observation, read_assumptions, UNKNOWN_FACIES
    root = Path(root or default_root())
    identifier = str(payload["id"])
    if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
        raise ValueError("Некорректный идентификатор учебного интервала")
    attrs = payload["attributes"]
    assumed = read_assumptions(attrs)
    image_digest = payload["pixel_sha256"]
    with Image.open(io.BytesIO(image_png)) as source:
        source.load()
        actual_digest = hashlib.sha256(f"{source.width}:{source.height}:".encode() + source.convert("RGBA").tobytes()).hexdigest()
        if actual_digest != image_digest:
            raise ValueError("Фрагмент фото изменился после выбора признака")
        descriptor = _descriptor(source)
        examples = []
        for field, entry in read_evidence(attrs).items():
            value, inline_assumption = observation(attrs.get(field))
            if (field not in EVIDENCE_FIELDS or not value or inline_assumption or field in assumed
                    or not valid_evidence(entry, value, image_digest) or entry.get("reviewed") is not True):
                continue
            l,t,r,b = entry["roi"]
            box = (round(l*source.width), round(t*source.height), round(r*source.width), round(b*source.height))
            crop = source.crop(box).convert("RGB")
            if min(crop.size) < 16:
                continue
            if _descriptor(crop) is None:
                continue
            buf = io.BytesIO(); crop.save(buf,format="PNG")
            data = buf.getvalue(); sha = hashlib.sha256(data).hexdigest()
            file = f"images/{sha}.png"
            _atomic(root/file, data)
            examples.append({"field":field, "value":value, "image":file, "sha256":sha,
                             "source":"expert_selected_reviewed_roi", "roi":entry["roi"]})
    reviewed = payload.get("description_reviewed") is True
    observations = {key: observation(value)[0] for key,value in attrs.items()
                    if not key.startswith("__") and key not in assumed and observation(value)[0]
                    and not observation(value)[1]}
    if attrs.get("__description_mode") != "template":
        # Reviewing free text does not approve unrelated, unmentioned form fields.
        text=str(attrs.get("Краткое описание", "")).casefold()
        evidence_fields={item['field'] for item in examples}
        observations={key:value for key,value in observations.items()
                      if key in evidence_fields or value.casefold() in text}
    image_sha = hashlib.sha256(image_png).hexdigest()
    file = f"images/{image_sha}.png"
    if reviewed or examples:
        _atomic(root/file, image_png)
    row = {"schema":SCHEMA, "id":identifier, "well":payload["well"],
           "project":payload.get("project", ""), "source_photo":payload.get("source_photo", ""),
           "record_id":payload.get("record_id", ""), "geometry":payload.get("geometry", []),
           "depth_from":payload.get("depth_from"), "depth_to":payload.get("depth_to"),
           "image":file, "sha256":image_sha, "pixel_sha256":image_digest,
           "description_reviewed":reviewed, "description":attrs.get("Краткое описание", "") if reviewed else "",
           "observations":observations if reviewed else {}, "targets":examples,
           "facies":payload.get("facies", ""), "facies_name":attrs.get("Название фации", ""),
           "facies_index":attrs.get("Индекс фации", ""),
           "facies_reviewed":bool(payload.get("training_ready") is True and "Название фации" not in assumed
                                  and str(payload.get("facies", "")).casefold() not in UNKNOWN_FACIES),
           "visual_descriptor":descriptor if reviewed else None,
           "image_kind":"displayed_mask_crop", "automatic_training":False}
    _atomic(root/"items"/(identifier+".json"), json.dumps(row,ensure_ascii=False,allow_nan=False).encode("utf-8"))
    return row


def read_examples(*, root=None):
    root = Path(root or default_root()); rows=[]
    for path in sorted((root/"items").glob("*.json"))[:10000]:
        try:
            row=json.loads(path.read_text(encoding="utf-8"))
            if isinstance(row,dict) and row.get("schema")==SCHEMA and (row.get("description_reviewed") is True or row.get("targets")):
                image=(root/row["image"]).resolve()
                if image.is_relative_to(root.resolve()) and image.is_file():
                    rows.append(row)
        except (OSError,KeyError,ValueError,TypeError):
            continue
    return rows


def revoke_example(identifier, *, root=None):
    root=Path(root or default_root())
    if len(identifier)!=32 or any(c not in '0123456789abcdef' for c in identifier):
        raise ValueError("Некорректный идентификатор учебного интервала")
    _atomic(root/'items'/(identifier+'.json'),json.dumps({"schema":SCHEMA,"id":identifier,
            "description_reviewed":False,"targets":[],"revoked":True}).encode('utf-8'))
    return {"revoked":True}


def find_similar(image_png, *, root=None, exclude_id="", limit=4):
    import numpy as np
    from PIL import Image
    root=Path(root or default_root()); query=_descriptor(Image.open(io.BytesIO(image_png)))
    if query is None: return []
    query=np.asarray(query); ranked=[]; seen=set()
    for row in read_examples(root=root):
        # The same crop/interval is not evidence of generalisation.
        if row["id"]==exclude_id or not row.get("description_reviewed") or row.get("sha256")==hashlib.sha256(image_png).hexdigest(): continue
        descriptor=np.asarray(row.get("visual_descriptor") or [])
        if descriptor.shape != query.shape or not np.isfinite(descriptor).all(): continue
        score=float(np.dot(query,descriptor))
        ranked.append((score,row))
    results=[]
    for score,row in sorted(ranked,key=lambda item:-item[0]):
        if row["sha256"] in seen: continue
        image=root/row["image"]
        if hashlib.sha256(image.read_bytes()).hexdigest()!=row["sha256"]: continue
        seen.add(row["sha256"]); results.append(dict(row,similarity=score,image_path=str(image)))
        if len(results)>=limit: break
    return results


def export_attribute_dataset(destination, *, root=None):
    """Export explicit reviewed ROIs. No labels are mined from prose or theory."""
    import shutil
    root=Path(root or default_root()); destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    rows=[]; seen=set(); counts={}; candidates=[]; labels={}
    for item in read_examples(root=root):
        for target in item.get("targets", []):
            image=(root/target["image"]).resolve()
            if not image.is_relative_to(root.resolve()) or not image.is_file(): continue
            if hashlib.sha256(image.read_bytes()).hexdigest()!=target["sha256"]: continue
            key=(target["sha256"],target["field"],target["value"])
            if key in seen: continue
            seen.add(key)
            labels.setdefault(key[:2],set()).add(target['value'])
            candidates.append((item,target,image))
    conflicts=[{'sha256':key[0],'field':key[1],'values':sorted(values)} for key,values in labels.items() if len(values)>1]
    for item,target,image in candidates:
        if len(labels[(target['sha256'],target['field'])])>1:continue
        folder=destination/"images";folder.mkdir(exist_ok=True)
        name=image.name;shutil.copyfile(image,folder/name)
        rows.append({**target,"image":"images/"+name,"well":item["well"],
                     "source_photo":item.get("source_photo", ""),"interval_id":item["id"],
                     "source_crop_sha256":item["sha256"],"label_provenance":"expert_selected_reviewed_roi"})
        counts[target["field"]]=counts.get(target["field"],0)+1
    report={"schema":"cv1-attribute-dataset-v1","examples":len(rows),"by_field":counts,
            "class_counts":{field:{value:sum(row['field']==field and row['value']==value for row in rows)
                                   for value in sorted({row['value'] for row in rows if row['field']==field})} for field in counts},
            "wells":sorted({row['well'] for row in rows}),"conflicting_labels_excluded":conflicts,
            "labels_from_theory":False,"trained":False,
            "split_policy":"Before training: group by well and source photo; keep duplicate images in one group.",
            "needs":"Several independent wells, at least two reviewed classes per trained property, and a separate test."}
    _atomic(destination/"examples.jsonl", "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows).encode("utf-8"))
    _atomic(destination/"audit.json",json.dumps(report,ensure_ascii=False,indent=2).encode("utf-8"))
    return report
