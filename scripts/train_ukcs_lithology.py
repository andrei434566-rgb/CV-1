"""Train a local research lithology classifier on depth-aligned UKCS core photos."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.infrastructure.open_dataset_intervals import _npy_header, _skip
from app.infrastructure.supervised_learning import classification_metrics, digest, write_json
from app.infrastructure.vision_models import MEAN, STD, create_network
from train_contact_pilot import fit_head

SOURCE = "https://figshare.com/articles/dataset/14265689"
ARCHIVE_SHA256 = "2957abe45d046c7206b4c8c55d769fceb6aa26eedf20f8f29ceb8f28211ecf41"
BASE = ROOT / "models/vision/vision_20261104_143516/model.pt"
CLASSES = ("s", "is", "ih", "os", "sh")
BLOCKS = {
    "204-20": "train", "204-20a": "train",
    "204-19": "validation",
    "204-24a": "test", "205-21b": "test",
}
SIDE = 160
WINDOW = 192
BATCH = 48
SEED = 20261108


def prepare(archive_path, destination):
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError("The archive differs from the verified UKCS release.")
    rng = np.random.default_rng(SEED)
    records, crops, seen = [], [], {}
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.namelist()):
            if not member.endswith("_labels.npy"):
                continue
            prefix = member.removesuffix("_labels.npy")
            well = prefix.rsplit("/", 1)[-1]
            split = BLOCKS[well.rsplit("-", 1)[0]]
            labels = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
            depth = np.load(io.BytesIO(archive.read(prefix + "_depth.npy")), allow_pickle=False)
            if labels.ndim != 1 or labels.shape != depth.shape or not np.isfinite(depth).all():
                raise ValueError(f"{well}: invalid label/depth alignment")
            delta = np.diff(depth)
            boundaries = np.r_[0, np.flatnonzero(
                (labels[1:] != labels[:-1]) | (delta < 0) | (delta > 0.01)) + 1, len(labels)]
            windows = {code: [] for code in CLASSES}
            for top, bottom in zip(boundaries[:-1], boundaries[1:], strict=True):
                code = labels[top].decode("ascii")
                if code in windows:
                    windows[code].extend(range(int(top) + 32, int(bottom) - WINDOW - 31, WINDOW))
            chosen = []
            for code, starts in windows.items():
                chosen.extend((int(top), code) for top in rng.choice(
                    starts, size=min(len(starts), 190), replace=False))

            with archive.open(prefix + "_image.npy") as stream:
                shape, fortran, dtype = _npy_header(stream)
                if (fortran or dtype != np.uint8 or len(shape) != 3 or shape[0] != len(depth)
                        or shape[2] != 3 or shape[1] < WINDOW
                        or archive.getinfo(prefix + "_image.npy").file_size != stream.tell() + np.prod(shape)):
                    raise ValueError(f"{well}: invalid image array")
                cursor = 0
                width = shape[1]
                left = (width - WINDOW) // 2
                for top, code in sorted(chosen):
                    _skip(stream, (top - cursor) * width * 3, None)
                    raw = stream.read(WINDOW * width * 3)
                    if len(raw) != WINDOW * width * 3:
                        raise ValueError(f"{well}: incomplete image")
                    cursor = top + WINDOW
                    core = np.frombuffer(raw, dtype=np.uint8).reshape(WINDOW, width, 3)
                    crop = core[:, left:left + WINDOW]
                    if np.mean(np.max(crop, axis=2) < 12) > 0.2 or crop.std() < 4:
                        continue
                    resized = np.asarray(Image.fromarray(crop).resize((SIDE, SIDE), Image.Resampling.BILINEAR))
                    sha = hashlib.sha256(resized.tobytes()).hexdigest()
                    if sha in seen:
                        if seen[sha] != split:
                            raise ValueError("Duplicate pixels between splits.")
                        continue
                    seen[sha] = split
                    crops.append(resized)
                    records.append({
                        "well": well, "split": split, "label": code, "crop_sha256": sha,
                        "image_member": prefix + "_image.npy", "rows": [top, cursor],
                        "depth_m": [float(depth[top]), float(depth[cursor - 1])],
                    })
            print("prepared", well, split, Counter(r["label"] for r in records if r["well"] == well),
                  flush=True)
    write_json(destination / "samples.json", records)
    return torch.from_numpy(np.stack(crops)), records


def run(archive, destination, epochs):
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT / "research_models"):
        raise ValueError("Write research weights under research_models only.")
    destination.mkdir(parents=True, exist_ok=False)
    images, records = prepare(archive, destination)
    ids = {name: torch.tensor([i for i, r in enumerate(records) if r["split"] == name])
           for name in ("train", "validation", "test")}
    y = torch.tensor([CLASSES.index(r["label"]) for r in records])
    for indices in ids.values():
        if set(y[indices].tolist()) != set(range(len(CLASSES))):
            raise ValueError("Every split must cover all five classes.")

    mean = torch.tensor(MEAN, device="cuda").view(1, -1, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, -1, 1, 1)

    def batch(index):
        pixels = images[index].to("cuda").permute(0, 3, 1, 2).float() / 255
        return (pixels - mean) / std

    base = create_network("convnext_tiny")
    base.load_state_dict(torch.load(BASE, map_location="cpu", weights_only=True))
    base = base.to("cuda", memory_format=torch.channels_last).eval()
    model = base
    source_head = model.classifier[2]
    model.classifier[2] = nn.Identity()

    @torch.inference_mode()
    def encode(indices):
        with torch.autocast("cuda", dtype=torch.float16):
            return torch.cat([model(batch(i)).float().cpu() for i in indices.split(BATCH)])

    x_train = nn.functional.normalize(encode(ids["train"]))
    x_val = nn.functional.normalize(encode(ids["validation"]))
    print("encoded training and validation", len(x_train), len(x_val), flush=True)
    head = fit_head(x_train, y[ids["train"]], len(CLASSES),
                    publications=[records[i]["well"] for i in ids["train"].tolist()])

    def predict_head(features):
        return (features @ head["weight"].T + head["bias"]).argmax(1)

    def measure(split, prediction):
        return classification_metrics(y[ids[split]].tolist(), prediction.tolist(), CLASSES)

    frozen_val = measure("validation", predict_head(x_val))
    torch.save(head, destination / "frozen_head.pt")
    model.classifier[2] = nn.Linear(source_head.in_features, len(CLASSES)).cuda()
    model.features.requires_grad_(False)
    model.features[-1].requires_grad_(True)
    optimizer = torch.optim.AdamW([
        {"params": model.features[-1].parameters(), "lr": 3e-5},
        {"params": model.classifier.parameters(), "lr": 4e-4},
    ], weight_decay=1e-4)
    train = ids["train"]
    well_class = Counter((records[i]["well"], records[i]["label"]) for i in train.tolist())
    wells_by_class = Counter(code for _, code in well_class)
    sampling_weights = torch.tensor([
        1 / (well_class[records[i]["well"], records[i]["label"]] * wells_by_class[records[i]["label"]])
        for i in train.tolist()])
    history, best = [], -1.0
    scaler = torch.amp.GradScaler("cuda")
    for epoch in range(1, epochs + 1):
        model.eval()
        model.features[-1].train()
        chosen = train[torch.multinomial(sampling_weights, len(train), replacement=True)]
        for indices in chosen.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = nn.functional.cross_entropy(model(batch(indices)), y[indices].cuda(), label_smoothing=0.02)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        model.eval()
        val = measure("validation", encode(ids["validation"]).argmax(1))
        history.append({"epoch": epoch, "validation": val})
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save(model.state_dict(), destination / "fine_tuned.pt")
        print("epoch", epoch, "validation_macro_f1", round(val["macro_f1"], 4), flush=True)

    selected = "fine_tuned" if best > frozen_val["macro_f1"] else "frozen"
    model.load_state_dict(torch.load(destination / "fine_tuned.pt", map_location="cpu", weights_only=True))
    fine_test = encode(ids["test"]).argmax(1)
    model.classifier[2] = nn.Identity()
    model.load_state_dict({k: v for k, v in torch.load(BASE, map_location="cpu", weights_only=True).items()
                           if not k.startswith("classifier.2.")})
    frozen_test = predict_head(nn.functional.normalize(encode(ids["test"])))
    test_predictions = fine_test if selected == "fine_tuned" else frozen_test
    test = ids["test"]
    report = {
        "source": SOURCE, "attribution": "Contains British Geological Survey materials © UKRI 2019 & 2020",
        "classes": CLASSES, "generative": False, "deployment": "research_only", "base_sha256": digest(BASE),
        "archive_sha256": ARCHIVE_SHA256, "image_side": SIDE, "blocks": BLOCKS,
        "samples": {name: dict(Counter(records[i]["label"] for i in indices.tolist())) for name, indices in ids.items()},
        "wells": {name: sorted({records[i]["well"] for i in indices.tolist()}) for name, indices in ids.items()},
        "frozen_validation": frozen_val, "fine_tuning": history, "selected_on_validation": selected,
        "frozen_test": measure("test", frozen_test), "fine_tuned_test": measure("test", fine_test),
        "majority_test": measure("test", torch.full_like(y[test], int(y[train].bincount().argmax()))),
        "test_predictions": [dict(records[i], prediction=CLASSES[p]) for i, p in
                             zip(test.tolist(), test_predictions.tolist(), strict=True)],
    }
    write_json(destination / "evaluation.json", report)
    print("test", selected, measure("test", test_predictions), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--epochs", type=int, default=4, choices=range(1, 21))
    args = parser.parse_args()
    run(args.archive, args.destination, args.epochs)
