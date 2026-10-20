"""Unit tests for the CoreBreakout segmentation pipeline; no GPU needed."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_corebreakout_yolo import prepare
from scripts.train_core_columns_yolo import validate_dataset


class CorebreakoutYoloTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.rows = []

    def add(self, well, n, same_pixels=False):
        path = self.source / f"{well}_{n}.png"
        Image.new("RGB", (100, 150), (22 if same_pixels else 22+n*20, 30, 40)).save(path)
        data = {"imageWidth": 100, "imageHeight": 150, "imagePath": path.name, "shapes": [
            {"label": "col1", "points": [[10, 10], [90, 10], [90, 140], [10, 140]]},
            {"label": "tray1", "points": [[0, 0], [9, 0], [9, 140], [0, 140]]},
        ]}
        path.with_suffix(".json").write_text(json.dumps(data), encoding="utf-8")
        self.rows.append((path.name, well))
        return path

    def groups_csv(self):
        path = self.root / "groups.csv"
        with path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(["image", "well_id"])
            writer.writerows(self.rows)
        return path

    def test_groupkfold_holds_out_whole_wells(self):
        for n in range(5):
            self.add(f"W{n}", n)
        output = self.root / "dataset"
        report = prepare(self.source, output, groups_csv=self.groups_csv())
        self.assertEqual("well", report["split_strategy"])
        self.assertEqual(5, report["total_columns"])
        self.assertEqual(report, validate_dataset(output))
        train = {r["well_group"] for r in report["samples"] if r["split"] == "train"}
        val = {r["well_group"] for r in report["samples"] if r["split"] == "val"}
        self.assertFalse(train & val)
        for row in report["samples"]:
            self.assertEqual(1, len((output / row["labels"]).read_text().splitlines()))

    def test_photo_holdout_is_explicit(self):
        self.add("W1", 1)
        self.add("W2", 2)
        with self.assertRaisesRegex(ValueError, "groups-csv"):
            prepare(self.source, self.root / "blocked")
        report = prepare(self.source, self.root / "research", allow_photo_holdout=True)
        self.assertEqual("photo_only", report["split_strategy"])

    def test_conflicting_wells_for_identical_photo_are_rejected(self):
        self.add("W1", 1, same_pixels=True)
        self.add("W2", 2, same_pixels=True)
        with self.assertRaisesRegex(ValueError, "Duplicate photo"):
            prepare(self.source, self.root / "conflict", groups_csv=self.groups_csv())


if __name__ == "__main__":
    unittest.main()
