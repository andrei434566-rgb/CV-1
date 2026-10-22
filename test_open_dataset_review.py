"""Review approval cannot arise from source labels or leak into holdout wells."""

from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.infrastructure.open_dataset_review import export_reviewed_pairs, read_candidates, save_review

_TEMP_ROOT = Path(__file__).resolve().parent / "open_datasets" / "test_tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def _write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _read(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _fixture(root: Path, wells=("W-1", "W-2")):
    rows = []
    for number, well in enumerate(wells):
        image = f"photo_{number}.png"
        Image.new("RGB", (8, 12), ((20 + number * 37) % 256, (80 + number * 11) % 256, 100)).save(root / image)
        rows.append({"image": image, "source_label": "silty clay", "original_description": "Original source phrase.",
                     "metadata": {"record_id": f"pair_{number}", "source_id": "iodp362", "well_id": well,
                                  "depth_top_m": 100.0, "depth_bottom_m": 101.0,
                                  "photo_depth_top_m": 100.0, "photo_depth_bottom_m": 101.0,
                                  "source_path": "original_supplement.xlsx", "source_row": f"Sheet!{number+2}",
                                  "pairing_status": "source_identifier_and_depth_matched_unreviewed",
                                  "training_approved": False, "geology_approved": False,
                                  "needs_interval_crop": False}})
    _write(root / "candidates.jsonl", rows)
    return {"output_dir": str(root), "manifest": str(root / "candidates.jsonl")}, rows


class OpenDatasetReviewTests(unittest.TestCase):
    def test_source_text_never_becomes_approved_russian_answer(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, _ = _fixture(root)
            before = (root / "candidates.jsonl").read_bytes()
            candidates = read_candidates(preparation)
            self.assertFalse(candidates[0]["review"]["approved"])
            self.assertNotIn("answer_ru", candidates[0]["review"])
            save_review(preparation, "pair_0", "Черновик специалиста", False)
            with self.assertRaises(ValueError):
                export_reviewed_pairs(preparation)
            self.assertEqual(before, (root / "candidates.jsonl").read_bytes())
            self.assertEqual("Черновик специалиста", read_candidates(preparation)[0]["review"]["answer_ru"])

    def test_explicit_russian_reviews_export_with_whole_well_holdout(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, _ = _fixture(root)
            for number in range(2):
                save_review(preparation, f"pair_{number}", f"Проверенное описание {number}: порода серая.", True)
            report = export_reviewed_pairs(preparation)
            train, holdout = _read(report["manifest"]), _read(report["holdout_manifest"])
            self.assertEqual((1, 1), (len(train), len(holdout)))
            self.assertTrue(train[0]["metadata"]["training_approved"])
            self.assertTrue(train[0]["metadata"]["geology_approved"])
            self.assertFalse(holdout[0]["metadata"]["training_approved"])
            self.assertFalse(Path(train[0]["image"]).is_absolute())
            self.assertTrue((Path(report["manifest"]).parent / train[0]["image"]).is_file())
            self.assertNotEqual(train[0]["metadata"]["well_id"], holdout[0]["metadata"]["well_id"])
            self.assertNotEqual(train[0]["metadata"]["image_pixel_sha256"], holdout[0]["metadata"]["image_pixel_sha256"])
            self.assertEqual("Original source phrase.", train[0]["metadata"]["original_source_description"])
            self.assertIn("Проверенное описание", train[0]["conversations"][1]["value"])
            self.assertEqual(report["manifest"], export_reviewed_pairs(preparation)["manifest"])

    def test_approval_rejects_uncropped_interval_english_and_missing_image(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, rows = _fixture(root)
            with self.assertRaises(ValueError):
                save_review(preparation, "pair_0", "silty clay", True)
            rows[0]["metadata"]["needs_interval_crop"] = True
            _write(root / "candidates.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "вырезку"):
                save_review(preparation, "pair_0", "Серая порода", True)
            rows[0]["metadata"]["needs_interval_crop"] = False
            _write(root / "candidates.jsonl", rows)
            (root / "photo_0.png").unlink()
            with self.assertRaisesRegex(ValueError, "не найдено"):
                save_review(preparation, "pair_0", "Серая порода", True)

    def test_changed_source_revokes_approval_and_changed_pixels_block_export(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, rows = _fixture(root)
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            Image.new("RGB", (8, 12), "pink").save(root / "photo_0.png")
            with self.assertRaisesRegex(ValueError, "изменилось фото"):
                export_reviewed_pairs(preparation)
            rows[0]["original_description"] = "A corrected original source phrase."
            _write(root / "candidates.jsonl", rows)
            review = read_candidates(preparation)[0]["review"]
            self.assertFalse(review["approved"])
            self.assertTrue(review["source_changed"])
            self.assertEqual("Серая порода", review["answer_ru"])

    def test_same_pixels_in_different_wells_block_export(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, _ = _fixture(root)
            Image.open(root / "photo_0.png").save(root / "photo_1.png")
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            with self.assertRaisesRegex(ValueError, "разных скважинах"):
                export_reviewed_pairs(preparation)

    def test_one_well_is_review_only_and_source_crop_bounds_must_match(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, rows = _fixture(root, ("W-1", "W-1"))
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            with self.assertRaisesRegex(ValueError, "двух разных скважин"):
                export_reviewed_pairs(preparation)
            rows[0]["metadata"].update(pairing_status="source_arrays_row_aligned", pixel_alignment_verified=True,
                                       row_top=0, row_bottom=100, crop_row_top=10, crop_row_bottom=90,
                                       crop_depth_top_m=100.1, crop_depth_bottom_m=100.9)
            _write(root / "candidates.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "не соответствуют вырезке"):
                save_review(preparation, "pair_0", "Серая порода", True)

    def test_adding_wells_never_moves_previous_train_into_holdout(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            wells = tuple(f"W-{number}" for number in range(8))
            preparation, _ = _fixture(root, wells[:7])
            for number in range(7):
                save_review(preparation, f"pair_{number}", "Проверенная серая порода", True)
            first = export_reviewed_pairs(preparation)
            initial = {well: "train" for well in first["train_wells"]}
            initial.update({well: "holdout" for well in first["holdout_wells"]})
            persisted = json.loads((root / "well_split_assignments.json").read_text(encoding="utf-8"))
            preparation, _ = _fixture(root, wells)
            save_review(preparation, "pair_7", "Проверенная серая порода", True)
            second = export_reviewed_pairs(preparation)
            for well, split in initial.items():
                self.assertEqual(split, second["split_assignment_provenance"][well]["split"])
                self.assertEqual(persisted["assignments"][well], second["split_assignment_provenance"][well])
            self.assertTrue(set(first["train_wells"]).isdisjoint(second["holdout_wells"]))
            self.assertEqual(second["manifest"], export_reviewed_pairs(preparation)["manifest"])

    def test_new_preparation_run_reuses_source_level_assignments(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            source = Path(folder) / "iodp362"
            first_root, second_root = source / "prepared" / "run1", source / "prepared" / "run2"
            first_root.mkdir(parents=True)
            second_root.mkdir()
            wells = tuple(f"W-{number}" for number in range(8))
            preparation, _ = _fixture(first_root, wells[:7])
            preparation["source_id"] = "iodp362"
            for number in range(7):
                save_review(preparation, f"pair_{number}", "Проверенная серая порода", True)
            first = export_reviewed_pairs(preparation)
            preparation, _ = _fixture(second_root, wells)
            preparation["source_id"] = "iodp362"
            for number in range(8):
                save_review(preparation, f"pair_{number}", "Проверенная серая порода", True)
            second = export_reviewed_pairs(preparation)
            self.assertEqual(str(source / "well_split_assignments.json"), second["split_assignments_file"])
            self.assertFalse((first_root / "well_split_assignments.json").exists())
            for well, assignment in first["split_assignment_provenance"].items():
                self.assertEqual(assignment, second["split_assignment_provenance"][well])
            self.assertTrue(set(first["train_wells"]).isdisjoint(second["holdout_wells"]))

    def test_legacy_exports_in_other_source_preparations_are_migrated(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            source = Path(folder) / "iodp362"
            old = source / "prepared" / "run1" / "reviewed_exports" / "legacy"
            old.mkdir(parents=True)
            (old / "export_report.json").write_text(json.dumps({"train_wells": ["W-1"], "holdout_wells": ["W-2"]}), encoding="utf-8")
            current = source / "prepared" / "run2"
            current.mkdir()
            preparation, _ = _fixture(current)
            preparation["source_id"] = "iodp362"
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            exported = export_reviewed_pairs(preparation)
            self.assertEqual(["w-1"], exported["train_wells"])
            self.assertEqual(["w-2"], exported["holdout_wells"])
            self.assertIn("prepared/run1/reviewed_exports/legacy/export_report.json", [
                item["path"] for item in exported["split_assignment_provenance"]["w-1"]["evidence"]])

    def test_sibling_review_versions_share_only_their_qualified_source_history(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            parent = Path(folder) / "open_dataset_review"
            for name, source_id, train, holdout in (("iodp362_v1", "iodp362", "W-1", "W-2"),
                                                   ("ukcs_q204_q205_v1", "ukcs_q204_q205_coremdlr", "W-2", "W-1")):
                old = parent / name
                legacy = old / "reviewed_exports" / "legacy"
                legacy.mkdir(parents=True)
                (old / "report.json").write_text(json.dumps({"source_id": source_id}), encoding="utf-8")
                (legacy / "export_report.json").write_text(json.dumps({"train_wells": [train], "holdout_wells": [holdout]}), encoding="utf-8")
            for name, source_id, train in (("iodp362_v2", "iodp362", "w-1"), ("ukcs_v2", "ukcs", "w-2")):
                current = parent / name
                current.mkdir()
                preparation, rows = _fixture(current)
                preparation["source_id"] = source_id
                (current / "report.json").write_text(json.dumps({"source_id": source_id}), encoding="utf-8")
                for row in rows:
                    row["metadata"]["source_id"] = source_id
                _write(current / "candidates.jsonl", rows)
                for number in range(2):
                    save_review(preparation, f"pair_{number}", "Серая порода", True)
                exported = export_reviewed_pairs(preparation)
                self.assertEqual([train], exported["train_wells"])
                self.assertEqual(str(parent / f"well_split_assignments.{source_id}.json"), exported["split_assignments_file"])

    def test_missing_current_holdout_does_not_reassign_previously_trained_well(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, rows = _fixture(root, ("W-1", "W-2", "W-3"))
            for number in range(3):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            first = export_reviewed_pairs(preparation)
            held = set(first["holdout_wells"])
            registry = (root / "well_split_assignments.json").read_bytes()
            for row in rows:
                if row["metadata"]["well_id"].casefold() in held:
                    save_review(preparation, row["metadata"]["record_id"], "Нужна повторная проверка", False)
            with self.assertRaisesRegex(ValueError, "ранее закреплённых"):
                export_reviewed_pairs(preparation)
            self.assertEqual(registry, (root / "well_split_assignments.json").read_bytes())

    def test_legacy_split_is_migrated_even_when_old_absolute_paths_are_stale(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, _ = _fixture(root)
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            ordered = sorted(("w-1", "w-2"), key=lambda value: hashlib.sha256(value.encode()).hexdigest())
            legacy = root / "reviewed_exports" / "old_revision"
            legacy.mkdir(parents=True)
            # Deliberately reverse the old hash algorithm: history must win.
            report = {"schema": "kern-open-reviewed-photo-pairs-v1", "train_wells": [ordered[0]],
                      "holdout_wells": [ordered[1]], "manifest": "Z:/old/train.jsonl"}
            (legacy / "export_report.json").write_text(json.dumps(report), encoding="utf-8")
            _write(legacy / "train.jsonl", [{"metadata": {"well_id": ordered[0]}}])
            _write(legacy / "holdout.jsonl", [{"metadata": {"well_id": ordered[1]}}])
            exported = export_reviewed_pairs(preparation)
            self.assertEqual([ordered[0]], exported["train_wells"])
            self.assertEqual([ordered[1]], exported["holdout_wells"])
            provenance = exported["split_assignment_provenance"][ordered[0]]
            self.assertEqual("legacy_export_migration", provenance["origin"])
            self.assertTrue(any(item["path"].endswith("train.jsonl") for item in provenance["evidence"]))
            self.assertTrue(all(len(item["sha256"]) == 64 for item in provenance["evidence"]))
            self.assertEqual(exported["manifest"], export_reviewed_pairs(preparation)["manifest"])

    def test_conflicting_legacy_exports_block_migration_without_overwriting_history(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, _ = _fixture(root)
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            for revision, train, holdout in (("old_a", "W-1", "W-2"), ("old_b", "W-2", "W-1")):
                legacy = root / "reviewed_exports" / revision
                legacy.mkdir(parents=True)
                (legacy / "export_report.json").write_text(json.dumps({"train_wells": [train], "holdout_wells": [holdout]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "и в обучении, и в контроле"):
                export_reviewed_pairs(preparation)
            self.assertFalse((root / "well_split_assignments.json").exists())
            self.assertTrue((root / "reviewed_exports" / "old_a" / "export_report.json").is_file())

    def test_unreadable_persistent_split_never_silently_recreates_assignments(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            root = Path(folder)
            preparation, _ = _fixture(root)
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            path = root / "well_split_assignments.json"
            path.write_text("broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "история разделения|историю разделения"):
                export_reviewed_pairs(preparation)
            self.assertEqual("broken", path.read_text(encoding="utf-8"))

    def test_relocated_absolute_candidate_images_preserve_source_and_approval_signatures(self):
        with tempfile.TemporaryDirectory(dir=_TEMP_ROOT) as folder:
            base = Path(folder)
            original = base / "original_library"
            original.mkdir()
            preparation, rows = _fixture(original)
            for row in rows:
                row["image"] = str(original / row["image"])
                row["metadata"]["original_crop_path"] = row["image"]
            _write(original / "candidates.jsonl", rows)
            (original / "report.json").write_text(json.dumps({"output_dir": str(original)}), encoding="utf-8")
            for number in range(2):
                save_review(preparation, f"pair_{number}", "Серая порода", True)
            old_export = export_reviewed_pairs(preparation)
            before = (original / "candidates.jsonl").read_bytes()
            copied = base / "copied_library"
            shutil.copytree(original, copied)
            current = {"output_dir": str(copied), "manifest": str(copied / "candidates.jsonl")}
            candidates = read_candidates(current)
            for number, row in enumerate(candidates):
                self.assertEqual(str(copied / f"photo_{number}.png"), row["image_path"])
                self.assertEqual(rows[number]["image"], row["image"])
                self.assertEqual(rows[number]["metadata"], row["metadata"])
                self.assertTrue(row["review"]["approved"])
            self.assertEqual(before, (copied / "candidates.jsonl").read_bytes())
            new_export = export_reviewed_pairs(current)
            self.assertEqual(Path(old_export["manifest"]).parent.name, Path(new_export["manifest"]).parent.name)
            self.assertEqual(Path(old_export["manifest"]).read_bytes(), Path(new_export["manifest"]).read_bytes())
            for row in _read(new_export["manifest"]) + _read(new_export["holdout_manifest"]):
                self.assertFalse(Path(row["image"]).is_absolute())
                self.assertTrue((Path(new_export["manifest"]).parent / row["image"]).is_file())


if __name__ == "__main__":
    unittest.main()
