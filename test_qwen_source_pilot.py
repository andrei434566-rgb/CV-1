"""Small CPU-only regression checks for the source-labelled Qwen pilot."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from PIL import Image

from scripts import prepare_qwen_source_pilot as pilot


TEMP_ROOT = Path(__file__).resolve().parent / "open_datasets/test_tmp"


class SourcePilotTests(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="qwen_source_", dir=TEMP_ROOT)
        self.root = Path(self.temp.name)
        self.storage = self.root / "library"
        self.source_rows = {}
        image_number = 0
        for source_id, classes in pilot.LABELS.items():
            prepared = self.storage / source_id / "prepared" / f"{source_id}_v1"
            prepared.mkdir(parents=True)
            rows = []
            for label in classes:
                for split in ("train", "val", "test"):
                    image_number += 1
                    image = prepared / f"sample_{image_number}.png"
                    Image.new("RGB", (3, 4), (image_number, image_number * 2, image_number * 3)).save(image)
                    rows.append({
                        "source_id": source_id,
                        "dataset_subset": "DCID-512-7" if source_id == "dcid" else "mendeley",
                        "class_name": label,
                        "split": split,
                        "source_split": split,
                        "image": image.name,
                        "source_sha256": pilot.pixel_hash(image),
                        "well_group": f"well_{split}" if source_id == "mendeley" else None,
                        "source": {"member": image.name},
                    })
            self.source_rows[source_id] = rows
            self.write_rows(prepared / "manifest.jsonl", rows)
        self.theory_train = self.root / "theory_train.jsonl"
        self.theory_holdout = self.root / "theory_holdout.jsonl"
        self.write_rows(self.theory_train, [self.theory("Учебный вопрос?", approved=True)])
        self.write_rows(self.theory_holdout, [self.theory("Проверочный вопрос?", approved=False)])

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_rows(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def theory(question, *, approved):
        return {
            "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": "Краткий ответ."}],
            "metadata": {
                "record_type": "text_only_knowledge", "sources": [{"source_id": "synthetic_fixture"}],
                "training_approved": approved, "geology_approved": False,
            },
        }

    def build(self):
        return pilot.build_plan(self.storage, self.theory_train, self.theory_holdout,
                                train_per_class=1, val_per_class=1)

    def rewrite_source(self, source):
        self.write_rows(self.storage / source / "prepared" / f"{source}_v1/manifest.jsonl", self.source_rows[source])

    def test_small_plan_keeps_test_and_expert_status_separate(self):
        exports, summary = self.build()
        self.assertEqual(summary["counts"], {
            "train_visual": 11, "val_visual": 11, "train_theory": 1,
            "holdout_theory": 1, "train_conversation": 12, "holdout_conversation": 4,
        })
        self.assertEqual(summary["selected_unique_actual_pixel_hashes"], 22)
        self.assertEqual(summary["selected_original_source_split_counts"], {
            "dcid/train/train": 7, "dcid/val/val": 7,
            "mendeley/train/train": 4, "mendeley/val/val": 4,
        })
        self.assertIn("all prepared project/well test records", summary["excluded"])
        self.assertNotIn("all test images", summary["excluded"])
        self.assertIn("prepared project partitions", summary["split_definition"]["cross_split_guards_apply_to"])
        for split in ("train", "val"):
            for row in exports[f"{split}_visual"]:
                self.assertEqual(row["metadata"]["training_approved"], split == "train")
                self.assertFalse(row["metadata"]["geology_approved"])
                self.assertIn("Допустимые классы", row["conversations"][0]["value"])
                self.assertNotIn("/test/", row["image"])
        self.assertFalse((self.root / "output").exists())

    def test_global_source_hash_leakage_is_rejected(self):
        rows = self.source_rows["mendeley"]
        rows[1]["source_sha256"] = rows[0]["source_sha256"]
        self.rewrite_source("mendeley")
        with self.assertRaisesRegex(ValueError, "cross-split"):
            self.build()

    def test_whole_well_leakage_is_rejected(self):
        self.source_rows["mendeley"][1]["well_group"] = "well_train"
        self.rewrite_source("mendeley")
        with self.assertRaisesRegex(ValueError, "well appears"):
            self.build()

    def test_original_source_test_records_can_belong_to_disjoint_project_train(self):
        before, _summary = self.build()
        for row in self.source_rows["mendeley"]:
            if row["split"] == "train":
                row["source_split"] = "test"
                row["source"] = {"member": f"original/test/{row['image']}"}
            elif row["split"] == "val":
                row["source_split"] = "train"
        self.rewrite_source("mendeley")
        exports, summary = self.build()
        self.assertEqual(summary["selected_original_source_split_counts"]["mendeley/train/test"], 4)
        self.assertEqual(summary["selected_original_source_split_counts"]["mendeley/val/train"], 4)
        self.assertIn("Original-source test records may occur", summary["split_definition"]["mendeley"])
        project_train = [row for row in exports["train_visual"] if row["metadata"]["dataset"] == "mendeley"]
        self.assertTrue(all(row["metadata"]["partition"] == "train" and row["metadata"]["source_split"] == "test"
                            and row["metadata"]["training_approved"] for row in project_train))
        train_wells = {row["metadata"]["well_id"] for row in project_train}
        val_wells = {row["metadata"]["well_id"] for row in exports["val_visual"]
                     if row["metadata"]["dataset"] == "mendeley"}
        self.assertFalse(train_wells & val_wells)
        key = lambda row: (row["metadata"]["record_id"], row["image"], row["conversations"])
        self.assertEqual([key(row) for row in before["train_visual"]], [key(row) for row in exports["train_visual"]])
        self.assertEqual([key(row) for row in before["val_visual"]], [key(row) for row in exports["val_visual"]])

    def test_original_source_test_label_never_relaxes_project_whole_well_guard(self):
        for row in self.source_rows["mendeley"]:
            if row["split"] == "train":
                row["source_split"] = "test"
        self.source_rows["mendeley"][1]["well_group"] = "well_train"
        self.rewrite_source("mendeley")
        with self.assertRaisesRegex(ValueError, "well appears"):
            self.build()

    def test_actual_prepared_pixel_leakage_is_rejected(self):
        prepared = self.storage / "mendeley/prepared/mendeley_v1"
        train, val = self.source_rows["mendeley"][:2]
        (prepared / val["image"]).write_bytes((prepared / train["image"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "Selected prepared images"):
            self.build()

    def test_same_theory_question_in_holdout_is_rejected(self):
        self.write_rows(self.theory_holdout, [self.theory("Учебный вопрос?", approved=False)])
        with self.assertRaisesRegex(ValueError, "identical theory question"):
            self.build()

    def test_prompt_extraction_does_not_execute_python(self):
        path = self.root / "source.py"
        path.write_text('raise RuntimeError("must not execute")\nsystem_text = ("Привет, " "мир.")\n', encoding="utf-8")
        self.assertEqual(pilot.application_system_prompt(path), "Привет, мир.")

    def test_write_never_overwrites_and_preserves_system_text(self):
        exports, summary = self.build()
        output = self.root / "output"
        pilot.write_plan(output, exports, summary)
        self.assertEqual((output / "system_prompt.txt").read_text(encoding="utf-8"), summary["system_prompt"]["text"])
        self.assertEqual(len(pilot.read_jsonl(output / "train_visual.jsonl")), 11)
        combined = pilot.read_jsonl(output / "mixed.jsonl")
        self.assertEqual(len(combined), 24)
        self.assertTrue(all(row["metadata"]["split"] == "train" for row in combined))
        self.assertTrue(all(row["metadata"]["training_approved"] for row in combined))
        self.assertNotIn("split", exports["train_theory"][0]["metadata"])
        with self.assertRaises(FileExistsError):
            pilot.write_plan(output, exports, summary)

    def test_friendly_holdout_has_no_identical_questions(self):
        train, holdout = pilot.friendly_rows()
        train_questions = {row["conversations"][0]["value"] for row in train}
        self.assertTrue(all(row["conversations"][0]["value"] not in train_questions for row in holdout))
        self.assertTrue(all(row["metadata"]["training_approved"] for row in train))
        self.assertTrue(all(not row["metadata"]["training_approved"] for row in holdout))

    def test_library_selects_timestamped_run_over_legacy(self):
        original = self.storage / "mendeley/prepared/mendeley_v1"
        current = original.parent / "20261023_112233_a1b2c3"
        shutil.copytree(original, current)
        preparation = {
            "source_id": "mendeley", "task": "classification", "readiness": "ready",
            "output_dir": str(current), "manifest": str(current / "manifest.jsonl"),
        }
        (current / "summary.json").write_text(json.dumps(preparation), encoding="utf-8")
        (self.storage / "mendeley/library.json").write_text(json.dumps({
            "source_id": "mendeley", "local_files": [], "preparation": preparation,
        }), encoding="utf-8")
        self.assertEqual(pilot.resolve_prepared(self.storage, "mendeley"), current.resolve())
        exports, summary = self.build()
        self.assertEqual(summary["source_provenance"]["mendeley"]["manifest"], str(current / "manifest.jsonl"))
        self.assertTrue(all(Path(row["image"]).parent == current for row in exports["train_visual"]
                            if row["metadata"]["dataset"] == "mendeley"))

    def test_unready_library_never_falls_back_to_ready_legacy(self):
        original = self.storage / "mendeley/prepared/mendeley_v1"
        current = original.parent / "20261023_112233_c0ffee"
        shutil.copytree(original, current)
        for task, readiness in (("classification", "needs_validation"), ("segmentation", "ready")):
            with self.subTest(task=task, readiness=readiness):
                preparation = {
                    "source_id": "mendeley", "task": task, "readiness": readiness,
                    "output_dir": str(current), "manifest": str(current / "manifest.jsonl"),
                }
                (current / "summary.json").write_text(json.dumps(preparation), encoding="utf-8")
                (self.storage / "mendeley/library.json").write_text(json.dumps({
                    "source_id": "mendeley", "local_files": [], "preparation": preparation,
                }), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "not ready classification"):
                    self.build()

    def test_invalidated_library_never_uses_legacy_preparation(self):
        (self.storage / "mendeley/library.json").write_text(json.dumps({
            "source_id": "mendeley", "local_files": [],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No current valid library preparation"):
            self.build()


if __name__ == "__main__":
    unittest.main()
