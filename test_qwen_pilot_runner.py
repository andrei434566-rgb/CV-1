"""CPU-only safety and metric regressions for the offline Qwen pilot runner."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
_TEMP_ROOT = ROOT / "open_datasets" / "test_tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def load_runner():
    """Import the helpers without enabling the CLI's global network audit hook."""
    spec = importlib.util.spec_from_file_location(
        "qwen_source_pilot_cpu_checks", ROOT / "scripts" / "train_qwen_source_pilot.py"
    )
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    try:
        with patch.dict(os.environ), patch.object(sys, "addaudithook") as audit_hook:
            original_environment = dict(os.environ)
            spec.loader.exec_module(module)
            audit_hook.assert_not_called()
            if dict(os.environ) != original_environment:
                raise AssertionError("Importing pilot helpers must not alter the process environment")
    finally:
        sys.path[:] = old_path
    return module


runner = load_runner()


class OfflineConfigurationTests(unittest.TestCase):
    def test_import_does_not_change_environment_or_install_an_audit_hook(self):
        initial = {"QWEN_TEST_SENTINEL": "unchanged", "HF_HUB_OFFLINE": "original-value"}
        with patch.dict(os.environ, initial, clear=True), patch.object(sys, "addaudithook") as audit_hook:
            imported = load_runner()
            self.assertEqual(dict(os.environ), initial)
            audit_hook.assert_not_called()
            self.assertTrue(callable(imported.configure_offline_environment))

    def test_explicit_offline_configuration_sets_all_required_flags(self):
        expected = {
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "WANDB_DISABLED": "true", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
            "TOKENIZERS_PARALLELISM": "false", "QWEN_TEST_SENTINEL": "unchanged",
        }
        with patch.dict(os.environ, {"QWEN_TEST_SENTINEL": "unchanged"}, clear=True):
            with patch.object(sys, "addaudithook") as audit_hook:
                runner.configure_offline_environment()
                self.assertEqual(dict(os.environ), expected)
                audit_hook.assert_not_called()


def example(identifier="row1", *, split="train", approved=True, source="source_a", label="песчаник"):
    return {
        "conversations": [
            {"from": "human", "value": f"Вопрос {identifier}?"},
            {"from": "gpt", "value": label},
        ],
        "metadata": {
            "record_id": identifier,
            "source_id": source,
            "split": split,
            "training_approved": approved,
            "label_provenance": "source_direct_folder",
        },
    }


class ReadRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=_TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.manifest = self.folder / "examples.jsonl"

    def write(self, rows, *, bom=False):
        self.manifest.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8-sig" if bom else "utf-8",
        )
        return self.manifest

    def test_accepts_only_explicitly_approved_training_rows(self):
        rows = runner.read_records(self.write([example()], bom=True), training=True)
        self.assertEqual(runner.record_id(rows[0]), "row1")
        self.assertEqual(rows[0]["conversations"][1]["value"], "песчаник")

    def test_validation_holdout_and_test_are_not_training_data(self):
        for split in ("val", "validation", "holdout", "test", "", "VAL"):
            with self.subTest(split=split):
                with self.assertRaises(ValueError):
                    runner.read_records(self.write([example(split=split)]), training=True)

    def test_false_missing_and_string_approval_are_refused(self):
        for approval in (False, None, "true", 1):
            with self.subTest(approval=approval):
                row = example(approved=approval)
                if approval is None:
                    del row["metadata"]["training_approved"]
                with self.assertRaisesRegex(ValueError, "explicitly approved"):
                    runner.read_records(self.write([row]), training=True)

    def test_validation_can_be_unapproved_but_never_test_or_train(self):
        for split in ("val", "validation", "holdout", ""):
            with self.subTest(split=split):
                result = runner.read_records(
                    self.write([example(split=split, approved=False)]), training=False
                )
                self.assertEqual(len(result), 1)
        for split in ("test", "train"):
            with self.subTest(refused_split=split):
                with self.assertRaises(ValueError):
                    runner.read_records(self.write([example(split=split)]), training=False)

    def test_partition_field_and_uppercase_train_are_supported(self):
        row = example()
        del row["metadata"]["split"]
        row["metadata"]["partition"] = "TRAIN"
        self.assertEqual(len(runner.read_records(self.write([row]), training=True)), 1)

    def test_duplicate_metadata_or_top_level_ids_are_refused(self):
        for top_level in (False, True):
            with self.subTest(top_level=top_level):
                rows = [example("a"), example("b")]
                if top_level:
                    rows[0]["id"] = rows[1]["id"] = "same"
                else:
                    rows[0]["metadata"]["record_id"] = rows[1]["metadata"]["record_id"] = "same"
                with self.assertRaisesRegex(ValueError, "Duplicate record IDs"):
                    runner.read_records(self.write(rows), training=True)

    def test_fallback_conversation_ids_also_deduplicate(self):
        row = example()
        del row["metadata"]["record_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate record IDs"):
            runner.read_records(self.write([row, copy.deepcopy(row)]), training=True)

    def test_relative_image_resolves_against_manifest_folder(self):
        image = self.folder / "images" / "core.png"
        image.parent.mkdir()
        image.write_bytes(b"existence check only; no image library is imported")
        row = example()
        row["image"] = "images/core.png"
        result = runner.read_records(self.write([row]), training=True)
        self.assertEqual(Path(result[0]["image"]), image.resolve())

    def test_missing_image_is_refused(self):
        row = example()
        row["image"] = "missing.png"
        with self.assertRaises(FileNotFoundError):
            runner.read_records(self.write([row]), training=True)

    def test_training_image_requires_direct_source_label(self):
        image = self.folder / "core.png"
        image.write_bytes(b"not opened by the manifest reader")
        row = example()
        row["image"] = str(image)
        for provenance in ("", "model_prediction", "assistant_generated"):
            with self.subTest(provenance=provenance):
                row["metadata"]["label_provenance"] = provenance
                with self.assertRaisesRegex(ValueError, "direct source-label provenance"):
                    runner.read_records(self.write([row]), training=True)

    def test_empty_or_incomplete_pairs_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Empty manifest"):
            runner.read_records(self.write([]), training=True)
        for turns in ([], [{"value": "question"}], [{"value": "question"}, {"value": " "}]):
            with self.subTest(turns=turns):
                row = example()
                row["conversations"] = turns
                with self.assertRaisesRegex(ValueError, "complete question/answer pair"):
                    runner.read_records(self.write([row]), training=True)


class ClassificationMetricTests(unittest.TestCase):
    @staticmethod
    def response(identifier, answer, *, truncated=False, kind="image_classification"):
        return {"record_id": identifier, "response": answer, "truncated": truncated, "kind": kind}

    def test_normalization_preserves_exact_answer_semantics(self):
        self.assertEqual(runner.normalized_label("  **ИЗВЕСТНЯК**.\n"), "известняк")
        self.assertEqual(runner.normalized_label("`ТЁМНЫЙ   СЛАНЕЦ`!"), "темный сланец")
        self.assertNotEqual(runner.normalized_label("Это песчаник."), "песчаник")

    def test_accuracy_confusion_unknown_and_macro_f1(self):
        rows = [example("a1"), example("a2"),
                example("b1", label="известняк"), example("b2", label="известняк")]
        responses = [self.response("a1", " **ПЕСЧАНИК**. "), self.response("a2", "песчаник"),
                     self.response("b1", "песчаник"), self.response("b2", "не знаю", truncated=True)]
        metrics = runner.classification_metrics(rows, responses)["source_a"]
        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["correct"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.4)
        self.assertEqual(metrics["unknown_output"], 1)
        self.assertEqual(metrics["truncated"], 1)
        self.assertEqual(metrics["confusion_matrix"]["песчаник"]["песчаник"], 2)
        self.assertEqual(metrics["confusion_matrix"]["известняк"]["песчаник"], 1)
        self.assertEqual(metrics["confusion_matrix"]["известняк"]["__unknown__"], 1)

    def test_sources_keep_their_own_class_spaces(self):
        rows = [example("a", source="mendeley"), example("b", source="dcid", label="известняк")]
        metrics = runner.classification_metrics(
            rows, [self.response("a", "песчаник"), self.response("b", "известняк")]
        )
        self.assertEqual(set(metrics), {"mendeley", "dcid"})
        for source in metrics.values():
            self.assertEqual(source["accuracy"], 1.0)
            self.assertEqual(source["macro_f1"], 1.0)
            self.assertEqual(len(source["confusion_matrix"]), 1)

    def test_explicit_aliases_are_accepted_without_substring_matching(self):
        row = example()
        row["metadata"]["label_aliases"] = ["Sandstone", "песчаная порода"]
        self.assertEqual(runner.classification_metrics(
            [row], [self.response("row1", "SANDSTONE.")]
        )["source_a"]["correct"], 1)
        metrics = runner.classification_metrics([row], [self.response("row1", "Вероятно, песчаник.")])
        self.assertEqual(metrics["source_a"]["unknown_output"], 1)

    def test_ambiguous_alias_is_counted_unknown(self):
        rows = [example("a"), example("b", label="известняк")]
        for row in rows:
            row["metadata"]["label_aliases"] = "порода"
        result = runner.classification_metrics(rows, [self.response("a", "порода"), self.response("b", "порода")])
        self.assertEqual(result["source_a"]["correct"], 0)
        self.assertEqual(result["source_a"]["unknown_output"], 2)
        self.assertEqual(result["source_a"]["macro_f1"], 0.0)

    def test_non_classification_responses_do_not_replace_class_answers(self):
        row = example()
        responses = [self.response("row1", "песчаник"), self.response("row1", "incorrect", kind="theory")]
        self.assertEqual(runner.classification_metrics([row], responses)["source_a"]["correct"], 1)


class TheorySelectionTests(unittest.TestCase):
    @staticmethod
    def rows():
        result = []
        for topic in ("gamma", "alpha", "beta"):
            for number in range(3):
                row = example(f"{topic}_{number}", split="holdout", approved=False)
                row["metadata"]["topic"] = topic
                result.append(row)
        return result

    def test_baseline_and_candidate_choose_identical_ids_without_mutation(self):
        rows = self.rows()
        original = copy.deepcopy(rows)
        random_state = random.getstate()
        first = [runner.record_id(row) for row in runner.choose_theory(rows, 5)]
        second = [runner.record_id(row) for row in runner.choose_theory(rows, 5)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)
        self.assertEqual(rows, original)
        self.assertEqual(random.getstate(), random_state)

    def test_selection_round_robins_topics_in_sorted_order(self):
        selected = runner.choose_theory(self.rows(), 5)
        self.assertEqual([row["metadata"]["topic"] for row in selected],
                         ["alpha", "beta", "gamma", "alpha", "beta"])

    def test_zero_limit_and_excess_limit_are_bounded(self):
        self.assertEqual(runner.choose_theory(self.rows(), 0), [])
        selected = runner.choose_theory(self.rows(), 100)
        self.assertEqual(len(selected), 9)
        self.assertEqual(len({runner.record_id(row) for row in selected}), 9)

    def test_theory_validation_rejects_images(self):
        row = example()
        row["image"] = "core.png"
        with self.assertRaisesRegex(ValueError, "contains images"):
            runner.choose_theory([row], 1)


class TrainingArgumentsTests(unittest.TestCase):
    def test_warmup_uses_supported_integer_steps_for_855_rows(self):
        kwargs = runner.training_arguments_kwargs(
            Path("candidate"), epochs=1.0, learning_rate=5e-5, rows=855, seed=20261022
        )
        self.assertEqual(kwargs["warmup_steps"], 7)
        self.assertNotIn("warmup_ratio", kwargs)
        self.assertEqual(kwargs["gradient_accumulation_steps"], 4)
        self.assertEqual(kwargs["per_device_train_batch_size"], 1)
        self.assertEqual(kwargs["seed"], kwargs["data_seed"])
        self.assertEqual(kwargs["output_dir"], str(Path("candidate") / "trainer"))
        self.assertEqual(kwargs["gradient_checkpointing_kwargs"], {"use_reentrant": False})

    def test_warmup_accounts_for_fractional_epochs(self):
        kwargs = runner.training_arguments_kwargs(
            Path("candidate"), epochs=0.5, learning_rate=1e-5, rows=855, seed=42
        )
        self.assertEqual(kwargs["warmup_steps"], 4)
        self.assertEqual(kwargs["num_train_epochs"], 0.5)
        self.assertEqual(kwargs["learning_rate"], 1e-5)


class BaselineReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=_TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        image = example("image", split="val", source="mendeley")
        image["image"] = "not-opened-by-reuse-check.png"
        theory = example("theory", split="holdout", source="source_theory")
        casual = example("casual", split="holdout", source="conversation")
        casual["metadata"]["evaluation_kind"] = "casual_guard"
        self.rows = [image, theory, casual]
        self.settings = {
            "schema": "qwen_source_pilot_v1",
            "model": "local/model", "model_config_sha256": "model-config-hash",
            "max_image_side": 280, "seed": 20261022,
            "generation": {"do_sample": False, "repetition_penalty": 1.05,
                           "classification_max_new_tokens": 24, "theory_max_new_tokens": 260},
            "theory_evaluated_ids": ["theory"], "image_validation_rows": 1,
            "torch": "2.9-fixture", "transformers": "5.18-fixture",
            "peft": "0.21-fixture", "cuda": "12.8-fixture",
            "manifests": {name: {"path": f"original/{name}", "sha256": f"hash-{name}"}
                          for name in ("training", "image_validation", "theory_validation",
                                       "system_prompt", "conversation_validation")},
        }
        self.responses = [{
            "stage": "baseline", "record_id": runner.record_id(row),
            "kind": runner.evaluation_kind(row), "source_id": runner.source_id(row),
            "question": row["conversations"][0]["value"],
            "reference": row["conversations"][1]["value"],
            "response": row["conversations"][1]["value"],
            "truncated": False, "generated_tokens": 7, "seconds": 0.25,
        } for row in self.rows]
        self.save()

    def save(self, *, settings=None, responses=None):
        (self.folder / "run_config.json").write_text(
            json.dumps(self.settings if settings is None else settings, ensure_ascii=False), encoding="utf-8"
        )
        (self.folder / "baseline_responses.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False)
                      for row in (self.responses if responses is None else responses)) + "\n",
            encoding="utf-8",
        )

    def test_identical_baseline_reuses_raw_responses_with_hash_provenance(self):
        responses, provenance = runner.load_reusable_baseline(self.folder, self.settings, self.rows)
        self.assertEqual(responses, self.responses)
        self.assertEqual(provenance["rows"], 3)
        self.assertEqual(provenance["run_config_sha256"], runner.sha256(self.folder / "run_config.json"))
        self.assertEqual(provenance["baseline_responses_sha256"],
                         runner.sha256(self.folder / "baseline_responses.jsonl"))
        self.assertTrue(provenance["metrics_recomputed_from_raw_responses"])

    def test_model_seed_generation_size_and_version_changes_are_refused(self):
        changed_values = {
            "schema": "v2", "model": "different/model", "model_config_sha256": "changed",
            "max_image_side": 448, "seed": 11,
            "generation": {**self.settings["generation"], "do_sample": True},
            "theory_evaluated_ids": ["other"], "image_validation_rows": 2,
            "torch": "other", "transformers": "other", "peft": "other", "cuda": "other",
        }
        for field, value in changed_values.items():
            with self.subTest(field=field):
                settings = copy.deepcopy(self.settings)
                settings[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    runner.load_reusable_baseline(self.folder, settings, self.rows)

    def test_each_manifest_including_system_requires_matching_checksum(self):
        for name in self.settings["manifests"]:
            with self.subTest(manifest=name):
                settings = copy.deepcopy(self.settings)
                settings["manifests"][name]["sha256"] = "different-bytes"
                with self.assertRaisesRegex(ValueError, name):
                    runner.load_reusable_baseline(self.folder, settings, self.rows)

    def test_missing_or_extra_manifest_set_is_refused(self):
        for add in (False, True):
            with self.subTest(add=add):
                settings = copy.deepcopy(self.settings)
                if add:
                    settings["manifests"]["extra"] = {"sha256": "new"}
                else:
                    del settings["manifests"]["system_prompt"]
                with self.assertRaisesRegex(ValueError, "manifest set"):
                    runner.load_reusable_baseline(self.folder, settings, self.rows)

    def test_same_bytes_can_move_without_invalidating_comparison(self):
        settings = copy.deepcopy(self.settings)
        for manifest in settings["manifests"].values():
            manifest["path"] = "relocated/path"
        responses, _provenance = runner.load_reusable_baseline(self.folder, settings, self.rows)
        self.assertEqual(responses, self.responses)

    def test_missing_response_and_duplicate_expected_id_are_refused(self):
        self.save(responses=self.responses[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete or duplicate"):
            runner.load_reusable_baseline(self.folder, self.settings, self.rows)
        self.save()
        with self.assertRaisesRegex(ValueError, "incomplete or duplicate"):
            runner.load_reusable_baseline(self.folder, self.settings, [self.rows[0], self.rows[0], self.rows[2]])

    def test_duplicate_or_unknown_response_id_is_refused(self):
        for identifier in ("image", "unknown"):
            with self.subTest(identifier=identifier):
                responses = copy.deepcopy(self.responses)
                responses[1]["record_id"] = identifier
                self.save(responses=responses)
                with self.assertRaisesRegex(ValueError, "unexpected or duplicate response ID"):
                    runner.load_reusable_baseline(self.folder, self.settings, self.rows)

    def test_changed_question_reference_kind_source_or_stage_is_refused(self):
        changes = {"question": "Another question?", "reference": "another target",
                   "kind": "theory", "source_id": "another source", "stage": "candidate"}
        for field, value in changes.items():
            with self.subTest(field=field):
                responses = copy.deepcopy(self.responses)
                responses[0][field] = value
                self.save(responses=responses)
                with self.assertRaisesRegex(ValueError, "response identity differs"):
                    runner.load_reusable_baseline(self.folder, self.settings, self.rows)

    def test_malformed_response_types_and_nonfinite_times_are_refused(self):
        changes = [("response", None), ("response", 123), ("truncated", 0),
                   ("generated_tokens", True), ("generated_tokens", 2.5), ("generated_tokens", -1),
                   ("seconds", -0.1), ("seconds", "0.2"), ("seconds", float("nan")),
                   ("seconds", float("inf"))]
        for field, value in changes:
            with self.subTest(field=field, value=value):
                responses = copy.deepcopy(self.responses)
                responses[0][field] = value
                self.save(responses=responses)
                with self.assertRaisesRegex(ValueError, "malformed response"):
                    runner.load_reusable_baseline(self.folder, self.settings, self.rows)


if __name__ == "__main__":
    unittest.main()
