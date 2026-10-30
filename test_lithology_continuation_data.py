"""Reject training leakage and changed publisher images before GPU allocation."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import json
import zipfile
from types import SimpleNamespace
from scripts.train_lithology_facies_continuation import LABELS, balanced, complete_checkpoint, restore_recipe, validate_split


class ContinuationDataTests(unittest.TestCase):
    def test_resume_skips_an_interrupted_newer_checkpoint(self):
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for step in (50, 100):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text("{}")
                (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
                save_file({"weight": torch.ones(1)}, str(checkpoint / "adapter_model.safetensors"))
                for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
                    with zipfile.ZipFile(checkpoint / name, "w") as archive:
                        archive.writestr("state", "complete")
            (root / "checkpoint-100/adapter_model.safetensors").write_bytes(b"partial save")
            self.assertEqual(complete_checkpoint(root), root / "checkpoint-50")

    def test_resume_restores_recipe_and_refuses_another_learning_rate_or_count(self):
        saved = {"image_rows": 800, "learning_rate": 5e-6, "seed": 20261030}
        args = SimpleNamespace(count=None, learning_rate=None, seed=None)
        restore_recipe(args, saved)
        self.assertEqual((args.count, args.learning_rate, args.seed), (800, 5e-6, 20261030))
        for overrides in ({"count": 400}, {"learning_rate": 1e-4}, {"seed": 42}):
            args = SimpleNamespace(count=None, learning_rate=None, seed=None)
            vars(args).update(overrides)
            with self.assertRaisesRegex(ValueError, "saved training recipe"):
                restore_recipe(args, saved)

    def test_balanced_sample_is_reproducible_and_has_four_classes(self):
        rows = [{"metadata": {"record_id": f"{label}-{i}"},
                 "conversations": [{}, {"value": label}]}
                for label in LABELS for i in range(5)]
        selected = balanced(rows, 12, 42)
        self.assertEqual(selected, balanced(rows, 12, 42))
        self.assertEqual({label: sum(row["conversations"][1]["value"] == label for row in selected)
                          for label in LABELS}, dict.fromkeys(LABELS, 3))

    def test_checksum_approval_and_well_split_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            def row(name, well, split):
                path = Path(folder) / name
                path.write_bytes(name.encode())
                return {"image": str(path), "metadata": {"image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "well_id": well, "well_split": split, "training_approved": split == "train",
                    "label_provenance": "source_direct_folder"}}
            train, validation, previous = row("new", "train-well", "train"), row("val", "val-well", "val"), row("old", "train-well", "train")
            validate_split([train], [validation], [previous])
            contaminated = copy.deepcopy(validation)
            contaminated["metadata"]["well_id"] = "train-well"
            with self.assertRaisesRegex(ValueError, "wells overlap"):
                validate_split([train], [contaminated], [previous])
            unapproved = copy.deepcopy(train)
            unapproved["metadata"]["training_approved"] = False
            with self.assertRaisesRegex(ValueError, "approved"):
                validate_split([unapproved], [validation], [previous])
            with self.assertRaisesRegex(ValueError, "repeated"):
                validate_split([train], [validation], [train])
            Path(train["image"]).write_bytes(b"unexpected replacement")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_split([train], [validation], [previous])


if __name__ == "__main__":
    unittest.main()
