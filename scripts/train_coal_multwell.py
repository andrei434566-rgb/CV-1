"""Train an isolated coal detector and validate it on complete held-out wells."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.infrastructure.local_vision import VisionEngine, image_tiles
from app.infrastructure.supervised_learning import classification_metrics, classifier_input_layer, digest, write_json
from app.infrastructure.vision_models import MEAN, STD
from train_contact_pilot import fit_head

BASE = ROOT / "models/vision/vision_20261104_143516"
PREVIOUS = ROOT / "models/features/coal_core_research_20261106/head.pt"
DATA = ROOT.parent / "vlm_core_v1/data"
LABELS = ("not_coal", "coal")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def source_rows(new_manifest, *, include_prior=True):
    prior_manifest = ROOT / "open_datasets/rock_web_20261106/nt1h_core_manifest.json"
    prior = json.loads(prior_manifest.read_text(encoding="utf-8"))["rows"] if include_prior else []
    new = json.loads(new_manifest.read_text(encoding="utf-8"))
    seen = {}
    for row in prior + new:
        path = (ROOT / row["image"]).resolve()
        fingerprint = digest(path)
        expected = row["crop_sha256"] if "crop_sha256" in row else row["image_sha256"]
        if fingerprint != expected or fingerprint in seen or not path.is_relative_to(ROOT / "open_datasets"):
            raise ValueError(f"Invalid or duplicate training crop: {row['id']}")
        seen[fingerprint] = row["id"]
        row["image"] = str(path)
    return prior + new


def external_r41():
    benchmark = ROOT / "research_models/ukcs_lithology_20261108/transfer_diagnostic.json"
    chosen = {r["source_row"]: r["depth_m"]
              for r in json.loads(benchmark.read_text(encoding="utf-8"))["r41"]["rows"]}
    source = DATA / "e_drive_interval_crops_v2/qwen3_vl_interval_crops.jsonl"
    by_row = {}
    for line in source.read_text(encoding="utf-8-sig").splitlines():
        row = json.loads(line)
        m = row["metadata"]
        if m.get("well_id") != "R-41" or m.get("pairing_approved_by_user") is not True:
            continue
        key = m["source_row"]
        if key not in chosen or m["photo_tile_depth_interval_m"] != chosen[key]:
            continue
        first = m["source_description_full"].strip().split(".", 1)[0].lower()
        match = re.match(r"^(песчаники|алевролиты)\b", first)
        path = Path(row["image"]).resolve()
        if match and path.is_relative_to(DATA) and path.is_file():
            by_row.setdefault(key, (path, match[1]))
    if set(by_row) != set(chosen):
        raise ValueError("Some fixed R-41 control rows are missing.")
    return [(key, path, label) for key, (path, label) in sorted(by_row.items())]


@torch.inference_mode()
def encode(engine, photos):
    raw, texture = [], []
    capture = []
    layer = classifier_input_layer(engine.model, "convnext_tiny")
    hook = layer.register_forward_pre_hook(lambda _module, inputs: capture.append(inputs[0].float().cpu()))
    try:
        for i in range(0, len(photos), 48):
            block = photos[i:i + 48]
            rgb = [TRANSFORM(photo) for photo in block]
            gray = [TRANSFORM(ImageOps.autocontrast(ImageOps.grayscale(photo), cutoff=3).convert("RGB"))
                    for photo in block]
            with torch.autocast("cuda", dtype=torch.float16):
                engine.model(torch.stack(rgb + gray).to("cuda"))
            rgb_features, gray_features = capture.pop().chunk(2)
            raw.append(rgb_features)
            texture.append(nn.functional.normalize(torch.cat([
                nn.functional.normalize(rgb_features), nn.functional.normalize(gray_features),
            ], dim=1)))
    finally:
        hook.remove()
    return torch.cat(raw), torch.cat(texture)


def fit(x, target, wells, subset):
    return fit_head(x[subset], target[subset], 2,
                    publications=[w for w, use in zip(wells, subset.tolist(), strict=True) if use])


def coal_score(features, state):
    return (features @ state["weight"].T + state["bias"])[:, 1]


def calibrate(x, target, wells, training):
    # Calibration scores come from wells excluded from the corresponding fit.
    negatives = []
    for well in sorted({w for w, used in zip(wells, training.tolist(), strict=True) if used}):
        validation = training & torch.tensor([w == well for w in wells])
        scores = coal_score(x[validation], fit(x, target, wells, training & ~validation))
        negatives.extend(scores[target[validation] == 0].tolist())
    if not negatives:
        raise ValueError("Independent negative calibration samples are required.")
    return max(0.5, max(negatives) + 0.02)


def metrics(targets, predicted):
    return classification_metrics(targets.tolist(), predicted.long().tolist(), LABELS)


def fit_adjustment(x, target, wells, logits, training):
    # Fit a small correction to the original coal log-odds with equal class/well weight.
    features = x[training].double()
    labels = target[training].double()
    offset = logits[training].double()
    keys = list(zip(labels.tolist(), [w for w, used in zip(wells, training.tolist(), strict=True) if used],
                    strict=True))
    counts = Counter(keys)
    groups = Counter(label for label, _ in counts)
    weights = features.new_tensor([0.5 / (counts[key] * groups[key[0]]) for key in keys])
    head = nn.Linear(features.shape[1], 1, dtype=features.dtype)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=50, line_search_fn="strong_wolfe")

    def step():
        optimizer.zero_grad()
        correction = head(features).squeeze(1)
        loss = (nn.functional.binary_cross_entropy_with_logits(offset + correction, labels, reduction="none")
                * weights).sum() + 0.02 * (head.weight.square().sum() + head.bias.square().sum())
        loss.backward()
        return loss

    optimizer.step(step)
    return {k: v.float() for k, v in head.state_dict().items()}


def adjusted_score(features, prior_logit, adjustment):
    delta = (features @ adjustment["weight"].T + adjustment["bias"]).squeeze(1)
    return (prior_logit + delta).sigmoid()


def read_photos(rows):
    photos = []
    for row in rows:
        with Image.open(row["image"]) as photo:
            photos.append(photo.convert("RGB"))
    return photos


def prior_logit(raw, old_state):
    logits = raw @ old_state["weight"].T + old_state["bias"]
    return logits[:, 1] - logits[:, 0]


def run(manifest, output, external_manifest=None):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "research_models"):
        raise ValueError("Research weights must be stored under research_models.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this research run.")
    rows = source_rows(manifest)
    control = external_r41()
    output.mkdir(parents=True, exist_ok=False)
    engine = VisionEngine(device="cuda")
    engine._load(BASE)
    raw, x = encode(engine, read_photos(rows))
    target = torch.tensor([r["binary_label"] for r in rows])
    wells = [r.get("validation_group", r["well"]) for r in rows]
    if "R-41" in wells or len(set(wells)) < 4:
        raise ValueError("At least four source wells and separate R-41 control are required.")
    for well in sorted(set(wells)):
        if {r["binary_label"] for r, group in zip(rows, wells, strict=True) if group == well} != {0, 1}:
            raise ValueError(f"Both classes must be present in {well}.")

    old_state = torch.load(PREVIOUS, map_location="cpu", weights_only=True)
    offsets = prior_logit(raw, old_state)
    old_scores = offsets.sigmoid()
    scores = torch.zeros(len(rows))
    adapted_scores = torch.zeros(len(rows))
    guard = torch.zeros(len(rows), dtype=torch.bool)
    folds = []
    for well in sorted(set(wells)):
        test = torch.tensor([w == well for w in wells])
        state = fit(x, target, wells, ~test)
        threshold = calibrate(x, target, wells, ~test)
        scores[test] = coal_score(x[test], state)
        guard[test] = scores[test] >= threshold
        adjustment = fit_adjustment(x, target, wells, offsets, ~test)
        adapted_scores[test] = adjusted_score(x[test], offsets[test], adjustment)
        fold = {
            "well": well, "threshold_from_training_wells": threshold,
            "old": metrics(target[test], old_scores[test] >= 0.5) if well != "NT1H" else None,
            "new": metrics(target[test], scores[test] >= 0.5),
            "new_guarded": metrics(target[test], guard[test]),
            "adapted": metrics(target[test], adapted_scores[test] >= 0.5) if well != "NT1H" else None,
        }
        folds.append(fold)
        print(well, fold["new"]["confusion_matrix"], fold["new_guarded"]["confusion_matrix"], flush=True)

    train_all = torch.ones(len(rows), dtype=torch.bool)
    final = fit(x, target, wells, train_all)
    adjustment = fit_adjustment(x, target, wells, offsets, train_all)
    threshold = calibrate(x, target, wells, train_all)
    photo_tiles, spans = [], []
    for _, path, _ in control:
        with Image.open(path) as image:
            image = image.convert("RGB")
            tiles, _ = image_tiles(image, [(0, 0, image.width, image.height)], max_tiles=3)
        if not tiles:
            raise ValueError(f"No usable tile for {path}.")
        start = len(photo_tiles)
        photo_tiles.extend(tiles)
        spans.append(slice(start, len(photo_tiles)))
    raw, x_control = encode(engine, photo_tiles)
    control_offset = prior_logit(raw, old_state)
    previous_r41 = control_offset.sigmoid()
    adapted_r41 = adjusted_score(x_control, control_offset, adjustment)
    current_r41 = coal_score(x_control, final)
    external = []
    for (key, _, lith), span in zip(control, spans, strict=True):
        external.append({"source_row": key, "lithology": lith,
                         "old": float(previous_r41[span].mean()),
                         "new": float(current_r41[span].mean()),
                         "adapted": float(adapted_r41[span].mean())})

    transfer = torch.tensor([w != "NT1H" for w in wells])
    torch.save(final, output / "head.pt")
    torch.save(adjustment, output / "prior_adjustment.pt")
    card = {
        "schema": "cv1-coal-multwell-research-v1", "generative": False, "deployment": "research_only",
        "base_sha256": digest(BASE / "model.pt"), "previous_head_sha256": digest(PREVIOUS),
        "head_sha256": digest(output / "head.pt"),
        "prior_adjustment_sha256": digest(output / "prior_adjustment.pt"),
        "labels": LABELS, "score_threshold": threshold,
        "samples_per_well": dict(Counter(wells)), "folds": folds,
        "all_wells_loo": {"new": metrics(target, scores >= 0.5), "new_guarded": metrics(target, guard)},
        "comparison_on_wells_unseen_by_previous_head": {
            "old": metrics(target[transfer], old_scores[transfer] >= 0.5),
            "new": metrics(target[transfer], scores[transfer] >= 0.5),
            "new_guarded": metrics(target[transfer], guard[transfer]),
            "adapted": metrics(target[transfer], adapted_scores[transfer] >= 0.5),
        },
        "r41": {"samples": len(external), "old_false_coal": sum(r["old"] >= 0.5 for r in external),
                "new_false_coal": sum(r["new"] >= 0.5 for r in external),
                "guarded_false_coal": sum(r["new"] >= threshold for r in external),
                "adapted_false_coal": sum(r["adapted"] >= 0.5 for r in external),
                "rows": external},
        "predictions": [dict(row, score=float(score), accepted=bool(allowed),
                             old_score=float(old), adapted_score=float(adapted))
                        for row, score, allowed, old, adapted in
                        zip(rows, scores, guard, old_scores, adapted_scores, strict=True)],
    }
    if external_manifest is not None:
        held_out = source_rows(external_manifest, include_prior=False)
        if ({r["well"] for r in held_out} & {r["well"] for r in rows}
                or {r["image"] for r in held_out} & {r["image"] for r in rows}
                or {r["crop_sha256"] for r in held_out} &
                {r.get("crop_sha256", r.get("image_sha256")) for r in rows}):
            raise ValueError("The external control must be separate from all training and calibration.")
        target_out = torch.tensor([r["binary_label"] for r in held_out])
        raw, features = encode(engine, read_photos(held_out))
        offset = prior_logit(raw, old_state)
        adapted = adjusted_score(features, offset, adjustment)
        card["external"] = {"old": metrics(target_out, offset >= 0),
                            "adapted": metrics(target_out, adapted >= 0.5),
                            "samples": [dict(r, old=float(s), adapted=float(t)) for r, s, t in
                                        zip(held_out, offset.sigmoid(), adapted, strict=True)]}
    write_json(output / "evaluation.json", card)
    print("external R-41", {k: v for k, v in card["r41"].items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--external", type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output, args.external)
