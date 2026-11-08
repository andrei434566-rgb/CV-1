"""Measure coal recall with held-out publications and a frozen R-41 control."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from itertools import product
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch

from train_coal_multwell import (
    BASE, PREVIOUS, ROOT, adjusted_score, coal_score, digest, encode, external_r41,
    fit, fit_adjustment, image_tiles, prior_logit, read_photos, source_rows,
    VisionEngine, write_json,
)

PREVIOUS_RUN = ROOT / "research_models/coal_multwell_20261108_adapted"
REFERENCE = ROOT / "open_datasets/coal_multwell_20261108"
LEGACY_GROUP = "NT1H"


def classify(old, support, policy, *, veto_only=False):
    floor, promote, ceiling, veto = policy
    accepted = (old >= 0.5) & ~((old < ceiling) & (support < veto))
    if veto_only:
        return accepted
    return accepted | ((old >= floor) & (support >= promote))


def counts(target, predicted):
    return {
        "coal_detected": int((target & predicted).sum()),
        "coal_missed": int((target & ~predicted).sum()),
        "false_coal": int((~target & predicted).sum()),
        "other_correct": int((~target & ~predicted).sum()),
    }


def select_policy(target, old, support, *, rescue_only, veto_only=False, veto_margin=0.0):
    if veto_only:
        # On calibration wells, no existing coal detection may be lost.
        detected = old >= 0.5
        selected = (0, 0.5, 0.0)
        for ceiling in (0.55, 0.65, 0.75, 0.85, 0.95, 1.01):
            eligible = detected & (old < ceiling)
            coal = eligible & target
            veto = float(support[coal].min() if coal.any() else support.max() + 1e-6) - veto_margin
            removed = int((eligible & ~target & (support < veto)).sum())
            if removed > selected[0]:
                selected = (removed, ceiling, veto)
        return 1.0, 1.0, selected[1], selected[2]

    # A no-op is in the grid. Use out-of-publication data to limit extra false alarms.
    baseline = counts(target, old >= 0.5)
    candidates = []
    ceilings = (0.50,) if rescue_only else (0.50, 0.58, 0.66, 0.75, 1.00)
    vetoes = (0.25,) if rescue_only else (0.25, 0.30, 0.35, 0.40, 0.45)
    floors = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20) if rescue_only else (0.50, 0.45, 0.40, 0.35, 0.30)
    promote_scores = (0.48, 0.50, 0.52, 0.54, 0.56, 0.60, 0.65) if rescue_only else (0.50, 0.55, 0.60, 0.65)
    for floor, promote, ceiling, veto in product(floors, promote_scores, ceilings, vetoes):
        policy = (floor, promote, ceiling, veto)
        candidate = classify(old, support, policy)
        result = counts(target, candidate)
        if result["false_coal"] <= baseline["false_coal"]:
            changes = int(((old >= 0.5) != candidate).sum())
            candidates.append((result["coal_missed"], result["false_coal"], changes, policy))
    return min(candidates)[-1]


def group(row):
    return row.get("validation_group", row["well"])


def photo_statistics(photos):
    rows = []
    for photo in photos:
        gray = ImageOps.grayscale(photo.resize((96, 96), Image.Resampling.BILINEAR))
        pixels = torch.frombuffer(bytearray(gray.tobytes()), dtype=torch.uint8).reshape(96, 96).float() / 255
        tones = torch.quantile(pixels, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
        row_edges = (pixels[1:] - pixels[:-1]).abs().mean()
        col_edges = (pixels[:, 1:] - pixels[:, :-1]).abs().mean()
        rows.append(torch.cat((tones, torch.stack((row_edges, col_edges)))))
    return torch.stack(rows)


def score_rows(rows, target, old, support, residual, predictions, previous=None):
    result = []
    for index, row in enumerate(rows):
        record = {key: row[key] for key in ("id", "well", "source_lithology", "focus_label") if key in row}
        record.update(
            label=int(target[index]), old=float(old[index]), support=float(support[index]),
            residual=float(residual[index]), candidate_coal=bool(predictions[index]),
        )
        if previous is not None:
            record["previous_adjustment"] = float(previous[index])
        result.append(record)
    return result


def run(data, output, *, rescue_only=False, veto_only=False, texture_stats=False, focused_negatives=False,
        rotation_average=False, veto_margin=0.0):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "research_models"):
        raise ValueError("Research weights must be stored under research_models.")
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    legacy = source_rows(REFERENCE / "expanded_training_crops.json")
    added = source_rows(data / "new_training_crops.json", include_prior=False)
    fresh = source_rows(data / "new_fixed_holdout.json", include_prior=False)
    texas = source_rows(REFERENCE / "texas_fixed_holdout.json", include_prior=False)
    rows = legacy + added
    all_rows = rows + fresh + texas
    if len({digest(Path(row["image"])) for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate crops across the experiment.")
    if {group(r) for r in rows} & {group(r) for r in fresh + texas}:
        raise ValueError("An external validation group is in the training data.")
    if {r["well"] for r in rows} & {r["well"] for r in fresh + texas}:
        raise ValueError("An external well is in the training data.")
    if "R-41" in {r["well"] for r in all_rows}:
        raise ValueError("R-41 must stay independent.")
    groups = [group(r) for r in rows]
    publications = frozenset(groups)
    masks = {name: torch.tensor([g == name for g in groups]) for name in publications}
    reference_mask = ~masks[LEGACY_GROUP]
    target = torch.tensor([r["binary_label"] for r in rows], dtype=torch.bool)
    focused = target | torch.tensor([
        any(name in r["source_lithology"].lower()
            for name in ("shale", "argillite", "carbonaceous mudstone", "grey-black mudstone"))
        for r in rows
    ])
    engine = VisionEngine(device="cuda")
    engine._load(BASE)

    def extract(photos):
        raw, features = encode(engine, photos)
        unrotated = features
        if rotation_average:
            rotated = [photo.transpose(Image.Transpose.ROTATE_90) for photo in photos]
            _, rotated_features = encode(engine, rotated)
            features = torch.nn.functional.normalize(features + rotated_features, dim=1)
        if texture_stats:
            features = torch.cat((features, photo_statistics(photos)), dim=1)
        return raw, features, unrotated

    raw, features, _ = extract(read_photos(rows))
    base = torch.load(PREVIOUS, map_location="cpu", weights_only=True)
    prior_adjustment = torch.load(PREVIOUS_RUN / "prior_adjustment.pt", map_location="cpu", weights_only=True)
    offset = prior_logit(raw, base)
    old = offset.sigmoid()

    @lru_cache(maxsize=None)
    def ridge(excluded):
        train = torch.tensor([g not in excluded for g in groups])
        if focused_negatives:
            train &= focused
        return fit(features, target.long(), groups, train)

    def policy_without(excluded):
        inner = torch.zeros(len(rows))
        selected = torch.zeros(len(rows), dtype=torch.bool)
        for name in sorted(publications - excluded - {LEGACY_GROUP}):
            test = masks[name]
            inner[test] = coal_score(features[test], ridge(excluded | {name}))
            selected |= test
        return select_policy(
            target[selected], old[selected], inner[selected], rescue_only=rescue_only, veto_only=veto_only,
            veto_margin=veto_margin,
        )

    supports = torch.zeros(len(rows))
    residuals = torch.zeros(len(rows))
    predictions = torch.zeros(len(rows), dtype=torch.bool)
    folds = {}
    for name in sorted(publications - {LEGACY_GROUP}):
        test = masks[name]
        policy = policy_without(frozenset({name}))
        supports[test] = coal_score(features[test], ridge(frozenset({name})))
        adjustment = fit_adjustment(features, target.long(), groups, offset, ~test)
        residuals[test] = adjusted_score(features[test], offset[test], adjustment)
        predictions[test] = classify(old[test], supports[test], policy, veto_only=veto_only)
        folds[name] = {
            "policy": policy, "old": counts(target[test], old[test] >= 0.5),
            "candidate": counts(target[test], predictions[test]),
        }
        print(name, folds[name], flush=True)

    policy = policy_without(frozenset())
    state = ridge(frozenset())
    adjustment = fit_adjustment(features, target.long(), groups, offset, torch.ones(len(rows), dtype=torch.bool))
    result = {
        "schema": "cv1-coal-recall-research-v1", "deployment": "research_only", "generative": False,
        "base_sha256": digest(BASE / "model.pt"), "previous_head_sha256": digest(PREVIOUS),
        "rescue_only": rescue_only, "veto_only": veto_only, "texture_stats": texture_stats,
        "focused_negatives": focused_negatives, "rotation_average": rotation_average, "veto_margin": veto_margin,
        "training_manifest_sha256": digest(data / "new_training_crops.json"),
        "control_manifest_sha256": digest(data / "new_fixed_holdout.json"),
        "policy": dict(zip(("rescue_floor", "support_floor", "veto_ceiling", "veto_support"), policy, strict=True)),
        "samples_per_group": dict(Counter(groups)), "folds": folds, "added_training_samples": len(added),
    }
    for key, mask in (
        ("held_out_publications", reference_mask),
        ("new_held_out_publications", torch.arange(len(rows)) >= len(legacy)),
    ):
        result[key] = {
            "old": counts(target[mask], old[mask] >= 0.5),
            "candidate": counts(target[mask], predictions[mask]),
            "residual": counts(target[mask], residuals[mask] >= 0.5),
            "ridge_at_half": counts(target[mask], supports[mask] >= 0.5),
            "rows": score_rows([r for r, keep in zip(rows, mask.tolist(), strict=True) if keep],
                               target[mask], old[mask], supports[mask], residuals[mask], predictions[mask]),
        }
    for key, sample_rows in (("fresh_fixed_control", fresh), ("texas_fixed_control", texas)):
        raw, encoded, unrotated = extract(read_photos(sample_rows))
        logits = prior_logit(raw, base)
        base_score = logits.sigmoid()
        support = coal_score(encoded, state)
        residual = adjusted_score(encoded, logits, adjustment)
        previous = adjusted_score(unrotated, logits, prior_adjustment)
        predicted = classify(base_score, support, policy, veto_only=veto_only)
        labels = torch.tensor([r["binary_label"] for r in sample_rows], dtype=torch.bool)
        result[key] = {
            "old": counts(labels, base_score >= 0.5),
            "previous_adjustment": counts(labels, previous >= 0.5),
            "candidate": counts(labels, predicted),
            "residual": counts(labels, residual >= 0.5),
            "rows": score_rows(sample_rows, labels, base_score, support, residual, predicted, previous),
        }

    control = external_r41()
    tiles, spans = [], []
    for _, path, _ in control:
        with Image.open(path) as photo:
            prepared, _ = image_tiles(photo.convert("RGB"), [(0, 0, photo.width, photo.height)], max_tiles=3)
        if not prepared:
            raise ValueError(f"R-41 tile extraction failed for {path}")
        spans.append(slice(len(tiles), len(tiles) + len(prepared)))
        tiles.extend(prepared)
    raw, encoded, unrotated = extract(tiles)
    logits = prior_logit(raw, base)
    old, support, residual, previous = [
        torch.stack([scores[span].mean() for span in spans])
        for scores in (logits.sigmoid(), coal_score(encoded, state),
                       adjusted_score(encoded, logits, adjustment),
                       adjusted_score(unrotated, logits, prior_adjustment))
    ]
    predicted = classify(old, support, policy, veto_only=veto_only)
    labels = torch.zeros(len(control), dtype=torch.bool)
    result["r41"] = {
        "samples": len(control), "old": counts(labels, old >= 0.5),
        "previous_adjustment": counts(labels, previous >= 0.5),
        "candidate": counts(labels, predicted), "residual": counts(labels, residual >= 0.5),
        "rows": score_rows([{"id": k, "source_lithology": lith} for k, _, lith in control],
                           labels, old, support, residual, predicted, previous),
    }
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "state": state, "policy": policy, "veto_only": veto_only, "texture_stats": texture_stats,
        "rotation_average": rotation_average,
    }, output / "coal_recall_head.pt")
    torch.save(adjustment, output / "coal_residual_head.pt")
    result["candidate_sha256"] = digest(output / "coal_recall_head.pt")
    result["residual_sha256"] = digest(output / "coal_residual_head.pt")
    write_json(output / "evaluation.json", result)
    print(json.dumps({k: {a: b for a, b in v.items() if a != "rows"}
                      for k, v in result.items() if isinstance(v, dict) and "rows" in v}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rescue-only", action="store_true")
    mode.add_argument("--veto-only", action="store_true")
    parser.add_argument("--texture-stats", action="store_true")
    parser.add_argument("--focused-negatives", action="store_true")
    parser.add_argument("--rotation-average", action="store_true")
    parser.add_argument("--veto-margin", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.veto_margin <= 1.0 or (args.veto_margin and not args.veto_only):
        parser.error("--veto-margin requires --veto-only and a value between 0 and 1")
    run(args.data, args.output, rescue_only=args.rescue_only, veto_only=args.veto_only,
        texture_stats=args.texture_stats, focused_negatives=args.focused_negatives,
        rotation_average=args.rotation_average, veto_margin=args.veto_margin)
