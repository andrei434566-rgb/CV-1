"""Small offline archives test provenance, independent splits and failure cleanup."""

from __future__ import annotations

import io
import json
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from app.infrastructure.open_dataset_import import (
    DatasetPreparationCancelled, IGNORE_INDEX, UNIBO_CLASSES, prepare_dataset,
)


def png(color, size=(32, 40)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


class OpenDatasetImportTests(unittest.TestCase):
    def setUp(self):
        # Windows sandbox's redirected system TEMP has a very long prefix.
        self.temp = tempfile.TemporaryDirectory(prefix=".odt-", dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, result):
        return [json.loads(line) for line in Path(result["manifest"]).read_text(encoding="utf-8").splitlines()]

    def assert_no_destination(self, path):
        self.assertFalse(path.exists())
        self.assertFalse(list(path.parent.glob(f".{path.name}.preparing-*")))

    def test_unibo_groups_whole_wells_before_tiles_and_preserves_ignore(self):
        archive = self.root / "unibo.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for well in range(1, 5):
                for part, split in ((1, "Training"), (2, "Test")):
                    name = f"S{well}_{part}.png"
                    z.writestr(f"{split}/Images/{name}", png((well * 20, part * 40, 120)))
                    arr = np.full((40, 32, 3), UNIBO_CLASSES[0]["rgb"], np.uint8)
                    arr[3:6, 3:6] = (255, 0, 255)  # not a source palette color
                    stream = io.BytesIO()
                    Image.fromarray(arr).save(stream, format="PNG")
                    z.writestr(f"{split}/Annotations/{name}", stream.getvalue())
        events = []
        result = prepare_dataset("unibo", [archive], self.root / "prepared", progress=events.append)
        self.assertEqual("ready", result["readiness"])
        self.assertEqual("facies", result["model_task"])
        self.assertEqual("CC BY-NC-SA 4.0", result["license"])
        rows = self.manifest(result)
        assignments = {}
        for row in rows:
            assignments.setdefault(row["well_group"], set()).add(row["split"])
            mask = np.asarray(Image.open(Path(result["output_dir"]) / row["mask"]))
            self.assertIn(IGNORE_INDEX, np.unique(mask))
            self.assertTrue((Path(result["output_dir"]) / row["original_annotation"]).exists())
            self.assertTrue((Path(result["output_dir"]) / row["labels"]).read_text().strip())
        self.assertTrue(all(len(splits) == 1 for splits in assignments.values()))
        self.assertEqual(4, len(result["split_notes"]["original_split_overlapping_wells"]))
        self.assertEqual("finalizing", events[-1]["stage"])

    def test_incomplete_unibo_pair_is_not_published(self):
        archive = self.root / "missing.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("Training/Images/S1_1.png", png((20, 30, 40)))
        target = self.root / "missing_out"
        with self.assertRaisesRegex(ValueError, "Images/Annotations"):
            prepare_dataset("unibo", [archive], target)
        self.assert_no_destination(target)

    def test_unibo_wrong_size_pair_is_quarantined_without_resizing_to_fit(self):
        archive = self.root / "sizes.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for well in range(1, 5):
                name = f"S{well}_1.png"
                z.writestr(f"Training/Images/{name}", png((well * 20, 80, 120)))
                size = (30, 40) if well == 4 else (32, 40)
                z.writestr(f"Training/Annotations/{name}", png(tuple(UNIBO_CLASSES[0]["rgb"]), size))
        result = prepare_dataset("unibo", [archive], self.root / "size_out")
        rows = self.manifest(result)
        self.assertEqual(3, len(rows))
        self.assertNotIn("S4", {r["well_group"] for r in rows})
        issues = [json.loads(line) for line in (Path(result["output_dir"]) / "issues.jsonl").read_text().splitlines()]
        self.assertEqual("image_annotation_size_mismatch_quarantined", issues[0]["reason"])
        self.assertEqual([32, 40], issues[0]["image_size"])
        self.assertEqual([30, 40], issues[0]["annotation_size"])

    def test_corebreakout_exports_only_columns_and_discards_weights_uv(self):
        archive = self.root / "assets.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for n in range(3):
                z.writestr(f"assets/data/train/photo{n}.png", png((20 + n * 30, 40, 60)))
                obj = {"imageWidth": 32, "imageHeight": 40, "imagePath": f"../old/photo{n}.png", "shapes": [
                    {"label": "col1", "points": [[4, 2], [12, 2], [12, 38], [4, 38]]},
                    {"label": "tray", "points": [[0, 0], [31, 0], [31, 39], [0, 39]]},
                ]}
                z.writestr(f"assets/data/train/photo{n}.json", json.dumps(obj))
            z.writestr("assets/data/train/photo_uv.json", json.dumps(obj))
            z.writestr("assets/models/mask_rcnn.h5", b"must never load")
        result = prepare_dataset("corebreakout", [archive], self.root / "core_out")
        self.assertEqual("core_columns", result["model_task"])
        self.assertEqual("photo_only", result["split_notes"]["strategy"])
        self.assertEqual(3, result["sample_count"])
        self.assertGreater(result["issue_count"], 0)
        for row in self.manifest(result):
            self.assertEqual(1, row["objects"])
            self.assertEqual(["col1"], row["original_shape_labels"])
            self.assertEqual("0", (Path(result["output_dir"]) / row["labels"]).read_text().split()[0])
        self.assertFalse(list(Path(result["output_dir"]).rglob("*.h5")))

    def test_classification_deduplicates_across_splits_and_reserves_val(self):
        archive = self.root / "DCID.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for n in range(5):
                z.writestr(f"DCID-512-7/train/sandstone/{n}.png", png((n * 30 + 20, 50, 90)))
            z.writestr("DCID-512-7/test/sandstone/duplicate.png", png((20, 50, 90)))
            z.writestr("DCID-512-7/test/sandstone/test.png", png((210, 50, 90)))
            z.writestr("DCID-512-7/evil.pkl", b"pickle not opened or executed")
            z.writestr("DCID-7.jpg", png((250, 250, 250)))
        result = prepare_dataset("dcid", [archive], self.root / "classification")
        self.assertEqual("classification", result["task"])
        self.assertIsNone(result["data_yaml"])
        rows = self.manifest(result)
        self.assertEqual(6, len(rows))
        self.assertEqual(6, len({r["source_sha256"] for r in rows}))
        self.assertEqual(1, result["decode_cache_hits"])
        self.assertEqual(6, result["decoded_unique_encodings"])
        self.assertEqual(2, sum(r["source_split"] == "test" for r in rows))
        self.assertTrue(any(r["source_split"] == "train" and r["split"] == "val" for r in rows))
        self.assertTrue(all(r["class_name"] == "sandstone" for r in rows))
        self.assertTrue(all("description" not in r for r in rows))
        with zipfile.ZipFile(archive) as z:
            for row in rows:
                self.assertEqual(z.read(row["source"]["member"]), (Path(result["output_dir"]) / row["image"]).read_bytes())
        self.assertFalse(list(Path(result["output_dir"]).rglob("*.pkl")))

    def test_different_encodings_still_use_pixel_duplicate_detection(self):
        archive = self.root / "different_png_encoding.zip"
        first, second = io.BytesIO(), io.BytesIO()
        image = Image.new("RGB", (32, 40), (20, 50, 90))
        image.save(first, format="PNG", compress_level=0)
        image.save(second, format="PNG", compress_level=9)
        self.assertNotEqual(first.getvalue(), second.getvalue())
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("DCID-512-7/train/sandstone/a.png", first.getvalue())
            z.writestr("DCID-512-7/test/sandstone/b.png", second.getvalue())
            z.writestr("DCID-512-7/train/sandstone/c.png", png((80, 50, 90)))
        result = prepare_dataset("dcid", [archive], self.root / "different_encoding_out")
        self.assertEqual(2, result["sample_count"])
        self.assertEqual(3, result["decoded_unique_encodings"])
        self.assertEqual(0, result["decode_cache_hits"])
        self.assertEqual(1, sum(r["source_split"] == "test" for r in self.manifest(result)))

    def test_conflicting_class_labels_are_quarantined(self):
        archive = self.root / "classes.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("train/sandstone/a.png", png((20, 50, 90)))
            z.writestr("test/shale/a.png", png((20, 50, 90)))
            z.writestr("train/limestone/b.png", png((80, 50, 90)))
        result = prepare_dataset("mendeley", [archive], self.root / "conflict")
        self.assertEqual(1, result["sample_count"])
        self.assertEqual("limestone", self.manifest(result)[0]["class_name"])
        self.assertEqual("needs_validation", result["readiness"])

    def test_crc_damaged_classification_member_is_quarantined(self):
        archive = self.root / "damaged_member.zip"
        damaged = png((15, 45, 75))
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("DCID-512-7/train/sandstone/bad.png", damaged)
            z.writestr("DCID-512-7/train/sandstone/good1.png", png((30, 60, 90)))
            z.writestr("DCID-512-7/train/sandstone/good2.png", png((50, 80, 100)))
            z.writestr("DCID-512-7/test/sandstone/good3.png", png((70, 100, 120)))
        payload = bytearray(archive.read_bytes())
        payload[payload.index(damaged) + len(damaged) - 1] ^= 1
        archive.write_bytes(payload)
        result = prepare_dataset("dcid", [archive], self.root / "damaged_out")
        self.assertEqual(3, result["sample_count"])
        self.assertEqual("ready", result["readiness"])
        issues = [json.loads(line) for line in (Path(result["output_dir"]) / "issues.jsonl").read_text().splitlines()]
        self.assertEqual("source_member_crc_error_quarantined", issues[0]["reason"])
        self.assertTrue(issues[0]["member"].endswith("bad.png"))

    def test_mendeley_reuses_whole_well_holdouts_across_original_splits(self):
        archive = self.root / "wells.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for index, well in enumerate(("HODD2", "MMDD1", "07PD01", "COD011")):
                for part, split in enumerate(("train", "test")):
                    z.writestr(f"Lithology dataset/{split}/sandstone/{well}_{part}.png", png((index * 40 + 20, part * 80 + 20, 90)))
        result = prepare_dataset("mendeley", [archive], self.root / "well_out")
        self.assertEqual("ready", result["readiness"])
        rows = self.manifest(result)
        assignments = {}
        for row in rows:
            assignments.setdefault(row["well_group"], set()).add(row["split"])
        self.assertTrue(all(len(values) == 1 for values in assignments.values()))
        self.assertEqual({"test"}, assignments["HODD2"])
        self.assertEqual({"val"}, assignments["MMDD1"])
        self.assertEqual("verified_filename_prefix", result["split_notes"]["well_disjointness"])
        self.assertEqual(4, len(result["split_notes"]["original_split_overlapping_wells"]))
        self.assertEqual("manifest.jsonl", result["manifest_relative"])

    def test_mendeley_cross_well_pixel_copies_are_quarantined(self):
        archive = self.root / "cross_well.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("train/sandstone/HODD2_1.png", png((20, 50, 90)))
            z.writestr("test/sandstone/MMDD1_1.png", png((20, 50, 90)))
            z.writestr("val/sandstone/07PD01_1.png", png((80, 50, 90)))
        result = prepare_dataset("mendeley", [archive], self.root / "cross_well_out")
        rows = self.manifest(result)
        self.assertEqual(1, len(rows))
        self.assertEqual("07PD01", rows[0]["well_group"])
        self.assertEqual("needs_validation", result["readiness"])

    def test_unknown_mendeley_well_names_block_training_even_with_source_val(self):
        archive = self.root / "unknown_wells.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("train/sandstone/arbitrary_1.png", png((20, 50, 90)))
            z.writestr("val/sandstone/arbitrary_2.png", png((80, 50, 90)))
        result = prepare_dataset("mendeley", [archive], self.root / "unknown_out")
        self.assertEqual("needs_validation", result["readiness"])
        self.assertTrue(result["requires_independent_validation"])

    def test_archive_traversal_and_symlink_are_rejected(self):
        for name, symlink in (("../escape.png", False), ("link.png", True)):
            archive = self.root / ("link.zip" if symlink else "traversal.zip")
            with zipfile.ZipFile(archive, "w") as z:
                if symlink:
                    info = zipfile.ZipInfo(name)
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    z.writestr(info, b"outside")
                else:
                    z.writestr(name, png((20, 30, 40)))
            target = self.root / archive.stem
            with self.assertRaises(ValueError):
                prepare_dataset("dcid", [archive], target)
            self.assert_no_destination(target)

    def test_cancellation_and_existing_destination_never_publish_partial(self):
        archive = self.root / "cancel.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("train/sandstone/a.png", png((20, 50, 90)))
        cancelled = {"value": False}
        def progress(event):
            if event["stage"] == "preparing":
                cancelled["value"] = True
        target = self.root / "cancelled"
        with self.assertRaises(DatasetPreparationCancelled):
            prepare_dataset("dcid", [archive], target, progress=progress, cancel=lambda: cancelled["value"])
        self.assert_no_destination(target)
        target.mkdir()
        (target / "user.txt").write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare_dataset("dcid", [archive], target)
        self.assertEqual("keep", (target / "user.txt").read_text())


if __name__ == "__main__":
    unittest.main()
