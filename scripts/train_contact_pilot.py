"""Fit isolated research classifiers on author-annotated core photos."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.local_vision import VisionEngine
from app.infrastructure.supervised_learning import REVIEW_BASIS, classification_metrics, digest, extract_embedding, write_json

def fit_head(features, labels, num_classes, *, publications=None):
    # Class-balanced ridge regression with a fixed penalty and an unpenalized intercept.
    x = torch.cat([features.double(), torch.ones(len(features), 1)], dim=1)
    y = torch.nn.functional.one_hot(labels, num_classes=num_classes).double()
    counts = y.sum(0)
    if not counts.all():
        raise ValueError('В каждой учебной части нужны все классы из манифеста.')
    if publications is None:
        weight = (1 / counts[labels]).unsqueeze(1)
    else:
        # For each class, give each publication the same total influence.
        keys = list(zip(labels.tolist(), publications, strict=True))
        publication_counts = Counter(keys)
        class_publications = Counter(label for label, _ in publication_counts)
        weight = x.new_tensor([1 / (publication_counts[key] * class_publications[key[0]])
                               for key in keys]).unsqueeze(1)
    regularizer = 0.1 * torch.eye(x.shape[1], dtype=x.dtype)
    regularizer[-1, -1] = 0
    coefficients = torch.linalg.solve(x.T @ (x * weight) + regularizer, x.T @ (y * weight))
    return {'weight': coefficients[:-1].T.float().contiguous(), 'bias': coefficients[-1].float()}


def extract_features(engine, rows, data_root):
    features = []; seen = set()
    for row in rows:
        path = (data_root / row['asset']).resolve()
        if not path.is_relative_to(data_root.resolve()):
            raise ValueError('Фото находится за пределами базы.')
        with Image.open(path) as image:
            l, t, r, b = row['box']
            if not (0 <= l < r <= image.width and 0 <= t < b <= image.height and min(r-l, b-t) >= 32):
                raise ValueError('Неверные координаты фрагмента: ' + row['id'])
            crop = image.convert('RGB').crop((l, t, r, b))
            pixels = (crop.size, crop.tobytes())
            if pixels in seen:
                raise ValueError('Повторяются пиксели участков.')
            seen.add(pixels)
            features.append(extract_embedding(engine.model, engine.card['architecture'], crop,
                                              engine.device, whole_region=True))
        print(row['id'], flush=True)
    return torch.stack(features)


def run(manifest, data_root, base_folder, destination, *, baseline_folder=None, balance_publications=False):
    data = json.loads(manifest.read_text(encoding='utf-8'))
    rows = data['samples']
    field = data.get('field', 'Контакт')
    if field not in REVIEW_BASIS:
        raise ValueError('Выберите признак «Контакт» или «Степень цементации».')
    present = {row['label'] for row in rows}
    classes = [label for label in LITHOLOGY_ATTRIBUTE_OPTIONS[field] if label in present]
    if len(classes) < 2 or set(classes) != present:
        raise ValueError('Нужны как минимум два класса из справочника признака.')
    sample_groups = [row.get('validation_group', row['publication']) for row in rows]
    groups = sorted(set(sample_groups))
    if len(groups) < 3:
        raise ValueError('Нужно хотя бы три независимых группы снимков.')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Повторяются идентификаторы участков.')
    asset_digests = {}; asset_groups = {}; sample_digests = []
    for row, group in zip(rows, sample_groups):
        asset = (data_root / row['asset']).resolve()
        if not asset.is_relative_to(data_root.resolve()):
            raise ValueError('Фото находится за пределами базы.')
        if asset not in asset_digests:
            asset_digests[asset] = digest(asset)
        sha = asset_digests[asset]
        if asset_groups.setdefault(sha, group) != group:
            raise ValueError('Участки одного исходного снимка должны быть в одной группе.')
        sample_digests.append(sha)
    held_out = set(data.get('holdout_groups', []))
    if held_out and not held_out < set(groups):
        raise ValueError('Укажите непустые учебную и контрольную группы.')
    if baseline_folder and not held_out:
        raise ValueError('Предыдущий классификатор сравнивается только на фиксированном новом контроле.')
    engine = VisionEngine(); engine._load(base_folder)
    x = extract_features(engine, rows, data_root)
    y = torch.tensor([classes.index(row['label']) for row in rows])

    def fit_subset(subset):
        publications = [row['publication'] for row, keep in zip(rows, subset) if keep] if balance_publications else None
        return fit_head(x[subset], y[subset], len(classes), publications=publications)

    predictions = torch.full_like(y, -1); baseline = torch.full_like(y, -1)
    folds = [held_out] if held_out else [{group} for group in groups]
    for fold in folds:
        test = torch.tensor([sample_group in fold for sample_group in sample_groups])
        state = fit_subset(~test)
        predictions[test] = (x[test] @ state['weight'].T + state['bias']).argmax(1)
        baseline[test] = torch.bincount(y[~test], minlength=len(classes)).argmax()
    evaluated = predictions >= 0
    final_train = torch.tensor([group not in held_out for group in sample_groups])
    result = {
        'schema': 'cv1-core-attribute-pilot-v2', 'field': field, 'classes': classes, 'generative': False,
        'deployment': 'research_only', 'input_region': 'whole_region',
        'method': 'publication_balanced_ridge_lambda_0.1' if balance_publications else 'balanced_ridge_lambda_0.1',
        'architecture': engine.card['architecture'], 'base_sha256': engine.card['model_sha256'],
        'manifest_sha256': digest(manifest),
        'counts': dict(Counter(row['label'] for row in rows)), 'validation_groups': groups,
        'evaluation': 'fixed_holdout' if held_out else 'leave_one_group_out',
        'holdout_groups': sorted(held_out), 'final_train_count': int(final_train.sum()),
        'training_asset_sha256': sorted({sha for sha, keep in zip(sample_digests, final_train) if keep}),
        'metrics': classification_metrics(y[evaluated].tolist(), predictions[evaluated].tolist(), classes),
        'majority_baseline': classification_metrics(y[evaluated].tolist(), baseline[evaluated].tolist(), classes),
    }
    prior_predictions = None
    if baseline_folder:
        card = json.loads((baseline_folder / 'evaluation.json').read_text(encoding='utf-8'))
        old_weights = baseline_folder / 'head.pt'
        if (card['field'] != field or card['classes'] != classes or card['base_sha256'] != result['base_sha256']
                or card['head_sha256'] != digest(old_weights)):
            raise ValueError('Предыдущие веса несовместимы с новым контролем.')
        old_assets = card.get('training_asset_sha256')
        if old_assets is None and card['evaluation'] == 'leave_one_group_out':
            old_assets = [row['asset_sha256'] for row in card['predictions']]
        if not old_assets:
            raise ValueError('Нет контрольных сумм учебных снимков предыдущего классификатора.')
        if set(old_assets) & {sha for sha, keep in zip(sample_digests, evaluated) if keep}:
            raise ValueError('Снимок контрольной группы встречался в предыдущем эксперименте.')
        old_state = torch.load(old_weights, map_location='cpu', weights_only=True)
        prior_predictions = (x @ old_state['weight'].T + old_state['bias']).argmax(1)
        result['previous_head_sha256'] = card['head_sha256']
        result['previous_head_metrics'] = classification_metrics(y[evaluated].tolist(), prior_predictions[evaluated].tolist(), classes)
    result['predictions'] = []
    for i, row in enumerate(rows):
        if not evaluated[i]:
            continue
        prediction = {**row, 'prediction': classes[int(predictions[i])], 'asset_sha256': sample_digests[i]}
        if prior_predictions is not None:
            prediction['previous_prediction'] = classes[int(prior_predictions[i])]
        result['predictions'].append(prediction)
    destination.mkdir(parents=True, exist_ok=False)
    weights = destination / 'head.pt'; torch.save(fit_subset(final_train), weights)
    result['head_sha256'] = digest(weights)
    write_json(destination / 'evaluation.json', result)
    print(json.dumps(result['metrics'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('base_folder', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--baseline-folder', type=Path)
    parser.add_argument('--balance-publications', action='store_true')
    args = parser.parse_args()
    run(args.manifest, args.data_root, args.base_folder, args.destination,
        baseline_folder=args.baseline_folder, balance_publications=args.balance_publications)
