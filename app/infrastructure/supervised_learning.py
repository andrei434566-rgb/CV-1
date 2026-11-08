"""User-reviewed CNN feature heads. No text generation, automatic prose labels or downloads."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
from uuid import uuid4

from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS

REVIEW_BASIS = {'Степень цементации': 'sample_description_or_test', 'Контакт': 'expert_boundary_both_sides'}
FIELDS = ("Название породы", "Название фации", "Слоистость", "Биотурбация", "Текстура", "Зернистость", *REVIEW_BASIS)


def canonical_label(value, field):
    if field not in REVIEW_BASIS:
        return value
    key = ' '.join(value.casefold().replace('ё', 'е').split())
    return next((option for option in LITHOLOGY_ATTRIBUTE_OPTIONS[field]
                 if option.casefold().replace('ё', 'е') == key), '')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def suggested_label(text, field):
    """Only an explicit labelled line is a suggestion, never an approved target."""
    if field not in FIELDS:
        raise ValueError('Неизвестный признак')
    titles = {'Название породы': '(?:Название породы|Порода)', 'Название фации': '(?:Название фации|Фация)'}
    values = re.findall(r'^\s*' + titles.get(field, re.escape(field)) + r'\s*:\s*([^\n;]+)', text, flags=re.I | re.M)
    values = {canonical_label(v.strip().rstrip('.'), field) for v in values if v.strip()}
    if len(values) != 1:
        return ''
    value = values.pop()
    if len(value) > 120 or re.search(r'предполож|вероят|возможно|не установлен|не определ|\?', value, re.I):
        return ''
    return value


def rows_from_word_manifest(manifest):
    manifest = Path(manifest)
    rows = []
    for line in manifest.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        item = json.loads(line); meta = item.get('metadata', {})
        if meta.get('label_provenance') != 'expert_word_photo_number_reviewed' or meta.get('training_approved') is not True:
            raise ValueError('Сначала проверьте и подготовьте пары фото + Word.')
        image = (manifest.parent / str(item['image'])).resolve()
        if not image.is_relative_to(manifest.parent.resolve()) or not image.is_file():
            raise ValueError('Фото отсутствует в подготовленной базе: ' + image.name)
        text = '\n'.join(str(turn['value']) for turn in item.get('conversations', []) if turn.get('from') == 'gpt')
        rows.append({'id': str(meta['photo_id']), 'image': str(image), 'image_sha256': meta['image_sha256'],
                     'description': text, 'well': '', 'label': '', 'roi': None, 'reviewed': False})
    if not rows:
        raise ValueError('Нет проверенных пар фото + Word.')
    return rows


def validate_rows(rows, field, validation_well):
    if field not in FIELDS or not str(validation_well).strip():
        raise ValueError('Выберите признак и контрольную скважину.')
    approved = [dict(row) for row in rows if row.get('reviewed') is True]
    if not approved:
        raise ValueError('Нет проверенных участков. Укажите метку, скважину и выделите участок породы.')
    signatures = {}; unique = []; source_wells = {}; unique_by_region={}
    for row in approved:
        label, well = str(row.get('label', '')).strip(), str(row.get('well', '')).strip()
        roi = row.get('roi')
        if not label or not well or len(label) > 120 or len(well) > 120:
            raise ValueError('Каждому проверенному участку нужны метка класса и название скважины.')
        if re.search(r'^(?:предполож|вероятно|возможно|не определ|не установлен|требует проверки)|\?', label, re.I):
            raise ValueError('Предположительное или неизвестное значение нельзя использовать как проверенный класс: ' + label)
        if field in REVIEW_BASIS:
            label = canonical_label(label, field)
            if not label:
                raise ValueError('Допустимые классы: ' + ', '.join(LITHOLOGY_ATTRIBUTE_OPTIONS[field]))
            if row.get('review_basis') != REVIEW_BASIS[field]:
                raise ValueError('Откройте выделение участка и подтвердите основание для метки «' + field + '».')
        if field=='Название породы':
            from app.domain.rock_types import rock_training_label
            original_label=label;label,properties=rock_training_label(label)
            if properties:
                umbrella=properties['bituminous'] is None and properties['silicified'] is None
                source_labels=[] if umbrella else [original_label]
                # Revalidating a portable dataset must preserve the original
                # explicit subtype labels, not infer flags from its umbrella.
                previous_labels=row.get('rock_source_labels',[]) if umbrella and isinstance(row.get('rock_source_labels'),list) else []
                for source_label in previous_labels:
                    source_class,source_properties=rock_training_label(source_label)
                    if source_class==label and source_properties:
                        source_labels.append(source_label)
                        for prop in ('bituminous','silicified'):
                            if source_properties[prop] is True:properties[prop]=True
                row['rock_properties']=properties;row['rock_source_labels']=list(dict.fromkeys(source_labels or [original_label]))
            else:
                row.pop('rock_properties',None);row.pop('rock_source_labels',None)
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4 or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in roi)
                or roi[2] <= roi[0] or roi[3] <= roi[1]):
            raise ValueError('Выделите участок породы для каждого проверенного примера.')
        source = str(row.get('image_sha256') or '')
        if not source:
            raise ValueError('Нет контрольной суммы исходного фото.')
        if source in source_wells and source_wells[source] != well.casefold():
            raise ValueError('Одинаковое фото отнесено к разным скважинам. Уберите дубликаты.')
        source_wells[source] = well.casefold()
        key = (source, tuple(roi))
        if key in signatures:
            if signatures[key] != label:
                raise ValueError('Один участок имеет противоречивые метки: ' + row['id'])
            previous=unique_by_region[key]
            if row.get('rock_properties') and previous.get('rock_properties'):
                for prop in ('bituminous','silicified'):
                    if row['rock_properties'][prop] is True:previous['rock_properties'][prop]=True
                previous['rock_source_labels']=list(dict.fromkeys(previous['rock_source_labels']+row['rock_source_labels']))
            continue
        signatures[key] = label
        row.update(label=label, well=well, split='val' if well.casefold() == validation_well.strip().casefold() else 'train')
        unique.append(row);unique_by_region[key]=row
    classes = sorted({r['label'] for r in unique})
    if field in REVIEW_BASIS:
        missing = [label for label in LITHOLOGY_ATTRIBUTE_OPTIONS[field] if label not in classes]
        if missing:
            raise ValueError('Добавьте подтверждённые участки для классов: ' + ', '.join(missing))
    if not 2 <= len(classes) <= 128:
        raise ValueError('Нужны хотя бы два класса. Один класс не учит различать породы или фации.')
    counts = {split: Counter(r['label'] for r in unique if r['split'] == split) for split in ('train', 'val')}
    for label in classes:
        if counts['train'][label] < 5 or counts['val'][label] < 3:
            raise ValueError(f'Класс «{label}»: нужно минимум 5 учебных и 3 контрольных участка с другой скважины. '
                             f'Сейчас {counts["train"][label]} и {counts["val"][label]}. Это технический минимум, для качества нужно больше примеров.')
    return unique, classes, {split: dict(count) for split, count in counts.items()}


def export_dataset(rows, field, validation_well, destination):
    from PIL import Image, ImageOps
    approved, classes, counts = validate_rows(rows, field, validation_well)
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    (destination / 'images').mkdir()
    samples = []
    for i, row in enumerate(approved):
        image = Path(row['image'])
        if digest(image) != row['image_sha256']:
            raise ValueError('Фото изменилось после проверки: ' + row['id'])
        with Image.open(image) as original:
            if original.width * original.height > 100_000_000:
                raise ValueError('Фото больше 100 Мп: ' + row['id'])
            picture = ImageOps.exif_transpose(original).convert('RGB')
            l, t, r, b = row['roi']; w, h = picture.size
            box = (round(l*w), round(t*h), round(r*w), round(b*h))
            if min(box[2]-box[0], box[3]-box[1]) < 32:
                raise ValueError('Участок меньше 32 пикселей: ' + row['id'])
            crop = picture.crop(box)
            name = f'images/{i:06}.png'; crop.save(destination / name)
        samples.append({**row, 'image': name, 'crop_sha256': digest(destination / name)})
    # Pixel-identical crops cannot cross the control boundary, even if their
    # source files have different metadata or compression containers.
    seen = {}
    for row in samples:
        with Image.open(destination / row['image']) as im:
            pixels = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        if pixels in seen:
            raise ValueError('Повторяются одинаковые пиксели участков. Уберите копию, чтобы не завышать число примеров.')
        seen[pixels] = row
    dataset = {'schema': 'cv1-supervised-regions-v1', 'field': field, 'classes': classes, 'counts': counts,
               'validation_well': validation_well, 'samples': samples, 'generative': False,
               'limitation': 'Контроль на отдельной скважине используется для выбора модели; отдельного финального теста нет.'}
    write_json(destination / 'dataset.json', dataset)
    return destination / 'dataset.json'


def classifier_input_layer(model, architecture):
    return model.classifier[2 if architecture == 'convnext_tiny' else 1]


def extract_embedding(model, architecture, image, device, *, whole_region=False):
    import torch
    if whole_region:
        from app.infrastructure.bedding_classifier import prepare_image
        batch = prepare_image(image).unsqueeze(0)
    else:
        from torchvision import transforms
        from app.infrastructure.local_vision import image_tiles
        from app.infrastructure.vision_models import MEAN, STD
        tiles, _ = image_tiles(image, [[0, 0, image.width, image.height]], max_tiles=6)
        if not tiles:
            raise ValueError('Не найден пригодный участок породы.')
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        batch = torch.stack([transform(tile) for tile in tiles])
    captured = []
    hook = classifier_input_layer(model, architecture).register_forward_pre_hook(lambda _m, x: captured.append(x[0].detach()))
    try:
        with torch.inference_mode():
            model(batch.to(device))
        return torch.nn.functional.normalize(torch.cat(captured).float().mean(0), dim=0).cpu()
    finally:
        hook.remove()


def classification_metrics(target, prediction, classes):
    matrix = [[0 for _ in classes] for _ in classes]
    for a, b in zip(target, prediction):
        matrix[int(a)][int(b)] += 1
    per_class = []
    for c, name in enumerate(classes):
        tp = matrix[c][c]; support = sum(matrix[c]); predicted = sum(row[c] for row in matrix)
        precision = tp / predicted if predicted else 0.; recall = tp / support if support else 0.
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.
        per_class.append({'name': name, 'support': support, 'precision': precision, 'recall': recall, 'f1': f1})
    return {'accuracy': sum(matrix[c][c] for c in range(len(classes))) / max(1, len(target)),
            'macro_f1': sum(r['f1'] for r in per_class) / len(classes), 'per_class': per_class, 'confusion_matrix': matrix}


def train_feature_head(dataset_path, base_folder, destination, progress=lambda message: None, cancelled=lambda: False, *, steps=180):
    """Freeze the existing CNN and train a small linear classification head."""
    import torch
    from PIL import Image
    from app.infrastructure.local_vision import VisionEngine
    dataset_path = Path(dataset_path); data = json.loads(dataset_path.read_text(encoding='utf-8'))
    if data.get('schema') != 'cv1-supervised-regions-v1' or data.get('generative') is not False:
        raise ValueError('Нужна база проверенных участков CV-1.')
    rows, classes, counts = validate_rows(data['samples'], data['field'], data['validation_well'])
    review_only = data['field'] in REVIEW_BASIS
    engine = VisionEngine(); progress('Загружаю локальный классификатор…'); engine._load(base_folder)
    features = []; targets = []; splits = []
    for i, row in enumerate(rows):
        if cancelled():
            raise InterruptedError('Обучение остановлено; рабочая модель не изменена.')
        path = (dataset_path.parent / row['image']).resolve()
        if not path.is_relative_to(dataset_path.parent.resolve()) or digest(path) != row['crop_sha256']:
            raise ValueError('Учебный участок изменился или находится вне базы.')
        with Image.open(path) as image:
            features.append(extract_embedding(engine.model, engine.card['architecture'], image.convert('RGB'),
                                              engine.device, whole_region=review_only))
        targets.append(classes.index(row['label'])); splits.append(row['split'])
        progress(f'Признаки участка {i+1}/{len(rows)}')
    x = torch.stack(features); y = torch.tensor(targets)
    training = torch.tensor([s == 'train' for s in splits]); validation = ~training
    torch.manual_seed(20261104)
    head = torch.nn.Linear(x.shape[1], len(classes))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.03, weight_decay=.01)
    class_counts = torch.bincount(y[training], minlength=len(classes)).float()
    criterion = torch.nn.CrossEntropyLoss(weight=class_counts.sum() / class_counts)
    best = None; best_metrics = None; best_loss = float('inf')
    for step in range(steps):
        if cancelled():
            raise InterruptedError('Обучение остановлено; рабочая модель не изменена.')
        optimizer.zero_grad(); loss = criterion(head(x[training]), y[training]); loss.backward(); optimizer.step()
        if not torch.isfinite(loss):
            raise ValueError('Обучение вернуло некорректную ошибку.')
        with torch.no_grad():
            logits = head(x[validation]); val_loss = float(torch.nn.functional.cross_entropy(logits, y[validation]))
            metrics = classification_metrics(y[validation].tolist(), logits.argmax(1).tolist(), classes)
        if best_metrics is None or (metrics['macro_f1'], -val_loss) > (best_metrics['macro_f1'], -best_loss):
            best = {key: value.detach().clone() for key, value in head.state_dict().items()}
            best_metrics, best_loss = metrics, val_loss
        if step % 20 == 0:
            progress(f'Обучение {step+1}/{steps} · контроль F1 {metrics["macro_f1"]:.3f}')
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=False)
    torch.save(best, destination / 'head.pt')
    card = {'schema': 'cv1-feature-head-v1', 'generative': False, 'field': data['field'], 'classes': classes,
            'created_at': datetime.now().isoformat(timespec='seconds'), 'architecture': engine.card['architecture'],
            'base_sha256': engine.card['model_sha256'], 'head_sha256': digest(destination / 'head.pt'),
            'input_features': int(x.shape[1]), 'validation': best_metrics, 'counts': counts,
            'validation_well': data['validation_well'], 'dataset_sha256': digest(dataset_path),
            'limitation': data['limitation'], 'review_threshold': .8, 'margin_threshold': .15,
            'deployment': 'review_only' if review_only else 'suggestions_only', 'backbone_modified': False}
    if review_only:
        card['input_region'] = 'whole_region'
    write_json(destination / 'config.json', card)
    progress('Готово. Сохранён отдельный классификатор; исходная модель не изменена.')
    return {'folder': str(destination), 'card': card}


def activate_head(folder, root):
    folder = Path(folder).resolve(); root = Path(root).resolve()
    if not folder.resolve().is_relative_to(root.resolve()):
        raise ValueError('Классификатор должен находиться в локальной папке моделей.')
    card = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if card.get('field') in REVIEW_BASIS:
        raise ValueError('Для этого признака используется отдельно проверяемая модель целого фрагмента.')
    if (card.get('schema') != 'cv1-feature-head-v1' or card.get('generative') is not False or
            card.get('deployment') != 'suggestions_only' or card.get('field') not in FIELDS or
            digest(folder / 'head.pt') != card.get('head_sha256')):
        raise ValueError('Не прошла проверка классификатора признака.')
    pointer = root / 'active.json'
    active = json.loads(pointer.read_text(encoding='utf-8')) if pointer.is_file() else {}
    active[card['field']] = folder.relative_to(root).as_posix()
    write_json(pointer, active)


def head_suggestions(embedding, base_sha, root):
    """Small checked heads return proposals only, never accepted geological facts."""
    import torch
    root = Path(root); pointer = root / 'active.json'
    if not pointer.is_file():
        return {}, []
    suggestions = {}; details = []
    try:
        active = json.loads(pointer.read_text(encoding='utf-8'))
        if not isinstance(active, dict):
            return {}, []
    except (OSError, ValueError):
        return {}, []
    for field, name in list(active.items())[:len(FIELDS)]:
        try:
            folder = (root / name).resolve()
            if field not in FIELDS or field in REVIEW_BASIS or not folder.is_relative_to(root.resolve()):
                continue
            card = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
            if (card.get('schema') != 'cv1-feature-head-v1' or card.get('generative') is not False or
                    card.get('base_sha256') != base_sha or card.get('field') != field or
                    card.get('deployment') != 'suggestions_only' or
                    digest(folder / 'head.pt') != card.get('head_sha256') or
                    not isinstance(card.get('classes'), list) or not 2 <= len(card['classes']) <= 128 or
                    not all(isinstance(label, str) and 0 < len(label) <= 120 for label in card['classes']) or
                    card.get('input_features') != embedding.numel()):
                continue
            head = torch.nn.Linear(card['input_features'], len(card['classes']))
            head.load_state_dict(torch.load(folder / 'head.pt', map_location='cpu', weights_only=True))
            with torch.inference_mode():
                probabilities = head(embedding.cpu()).softmax(-1)
            if not torch.isfinite(probabilities).all():
                continue
            scores, indices = probabilities.topk(2)
            score = float(scores[0]); label = card['classes'][int(indices[0])]
            validation = next((r for r in card['validation']['per_class'] if r['name'] == label), {})
            class_checked = validation.get('support', 0) >= 3 and validation.get('precision', 0) >= .8 and validation.get('recall', 0) >= .5
            usable = class_checked and score >= .8 and float(scores[0]-scores[1]) >= .15
            if usable:
                suggestions[field] = label
            details.append({'field': field, 'label': label, 'score': score, 'suggested': usable,
                            'validation': card['validation'], 'model': name, 'confirmed': False,
                            'class_passed_validation_gate': class_checked})
        except (OSError, ValueError, TypeError, KeyError, RuntimeError):
            continue
    return suggestions, details


def deactivate_head(field, root):
    if field not in FIELDS:
        raise ValueError('Неизвестный признак')
    pointer = Path(root) / 'active.json'
    if pointer.is_file():
        active = json.loads(pointer.read_text(encoding='utf-8'))
        active.pop(field, None)
        write_json(pointer, active)
