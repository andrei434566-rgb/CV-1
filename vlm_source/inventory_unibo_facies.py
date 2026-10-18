"""Inventory the noncommercial Bologna facies benchmark without extracting it."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "external_raw" / "unibo_facies_7308"
OUTPUT = ROOT / "data" / "unibo_facies_research_inventory.json"
BASE_URL = "https://amsacta.unibo.it/7308/"
SPLITS = ("Training", "Validation", "Test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    split_info = {}
    wells_by_split = {}
    all_colors = Counter()
    for i, split in enumerate(SPLITS, 1):
        archive = SOURCE / f"{split}_subset.zip"
        with zipfile.ZipFile(archive) as z:
            images = {PurePosixPath(name).name: name for name in z.namelist()
                      if name.startswith(f"{split}/Images/") and name.lower().endswith(".png")}
            masks = {PurePosixPath(name).name: name for name in z.namelist()
                     if name.startswith(f"{split}/Annotations/") and name.lower().endswith(".png")}
            if set(images) != set(masks):
                raise ValueError(f"Image/mask mismatch in {split}")
            wells = {name.split("_")[0] for name in images}
            wells_by_split[split] = wells
            colors = Counter()
            sizes = Counter()
            for filename in sorted(masks):
                with Image.open(io.BytesIO(z.read(masks[filename]))) as mask:
                    sizes[str(mask.size)] += 1
                    counts = mask.convert("RGB").getcolors(maxcolors=256)
                    if counts is None:
                        raise ValueError(f"More than 256 colors in {filename}")
                    for count, rgb in counts:
                        colors["#%02X%02X%02X" % rgb] += count
            all_colors.update(colors)
        split_info[split] = {
            "source_url": f"{BASE_URL}{i}/{split}_subset.zip",
            "archive": str(archive),
            "archive_bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "image_mask_pairs": len(images),
            "unique_borehole_ids": len(wells),
            "borehole_ids": sorted(wells),
            "mask_colors_pixels": dict(colors.most_common()),
            "image_sizes": dict(sizes),
        }
    overlap = {}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            overlap[f"{first}_{second}"] = sorted(wells_by_split[first] & wells_by_split[second])
    output = {
        "doi": "10.6092/unibo/amsacta/7308",
        "source_url": BASE_URL,
        "license": "CC BY-NC-SA 4.0 shown beside each archive in the repository",
        "use": "Research inventory and benchmark only; exclude from deployable model training pending permission.",
        "readme": str(SOURCE / "Sedimentary_Facies_README.rtf"),
        "split_info": split_info,
        "total_images": sum(row["image_mask_pairs"] for row in split_info.values()),
        "unique_boreholes": len(set.union(*wells_by_split.values())),
        "borehole_overlap_between_original_splits": overlap,
        "all_mask_colors_pixels": dict(all_colors.most_common()),
        "warning": "Original train/validation/test splits are image-disjoint but not borehole-disjoint; re-split by borehole for generalization claims.",
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_images": output["total_images"], "unique_boreholes": output["unique_boreholes"],
                      "overlap": overlap, "colors": output["all_mask_colors_pixels"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
