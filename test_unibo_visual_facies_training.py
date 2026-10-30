"""CPU checks for restricted UniBo task encoding, scoring and split guards."""
from copy import deepcopy
import json
import unittest

from scripts.train_unibo_visual_facies import (
    CODES, LABELS, SOURCE_CODES, PROMPT, code_to_russian, raw_code,
    transform_facies, validate_facies_split, facies_weights,
    classification_metrics, facies_stage_metrics, select_replay,
)


def source_row(index, split="train", suffix=""):
    identity = f"{split}_{index}{suffix}"
    return {
        "image": identity + ".png",
        "conversations": [{"from": "human", "value": "original source question"},
                          {"from": "gpt", "value": json.dumps({"facies_name": LABELS[index]}, ensure_ascii=False)}],
        "metadata": {"record_id": identity, "source_id": "unibo7308",
                     "source_license": "CC BY-NC-SA 4.0",
                     "usage_scope": "noncommercial_research_only_no_automatic_deployment",
                     "label_provenance": "source_direct_mask_exact_palette",
                     "training_approved": split == "train", "geology_approved": False,
                     "depths_available": False, "split": split,
                     "source_group": "S_train" if split == "train" else "S_val",
                     "source_sha256": "origin_" + identity,
                     "pixel_sha256": "pixel_" + identity,
                     "image_sha256": "file_" + identity,
                     "source_class_code": SOURCE_CODES[index],
                     "source_class_display_ru": LABELS[index]},
    }


class VisualFaciesTaskTests(unittest.TestCase):
    def test_all_six_source_codes_have_stable_russian_letter_mapping(self):
        self.assertEqual(SOURCE_CODES, ("WDF", "PDF", "P", "PL", "FS", "Sw"))
        for index, code in enumerate(CODES):
            row = source_row(index)
            transformed = transform_facies(row)
            self.assertEqual(transformed["conversations"][1]["value"], code)
            self.assertEqual(code_to_russian(code), LABELS[index])
            self.assertEqual(transformed["metadata"], row["metadata"])
            self.assertEqual(transformed["original_conversations"], row["conversations"])
            self.assertEqual(transformed["conversations"][0]["value"], "<image>\n" + PROMPT)
            self.assertIn("голоценового", PROMPT)
        self.assertIsNone(code_to_russian(None))
        self.assertIsNone(code_to_russian("WDF"))

    def test_task_transform_does_not_mutate_source_metadata_or_prompt(self):
        row = source_row(0)
        before = deepcopy(row)
        result = transform_facies(row)
        result["metadata"]["source_class_code"] = "changed"
        self.assertEqual(row, before)

    def test_source_target_mismatch_is_refused(self):
        row = source_row(0)
        row["metadata"]["source_class_code"] = "PDF"
        with self.assertRaisesRegex(ValueError, "source code"):
            transform_facies(row)

    def test_raw_code_scoring_rejects_formatted_and_multiple_answers(self):
        for answer in ("A", " A\n", "F"):
            self.assertIn(raw_code(answer), CODES)
        for answer in ("", "a", "А", "A.", "A B", '"A"', '{"code":"A"}', "Ответ: A", "WDF"):
            self.assertIsNone(raw_code(answer), answer)

    def test_inverse_frequency_weights_balance_facies_without_oversampling(self):
        counts = (64, 64, 32, 64, 64, 64)
        rows = [{"conversations": [{"value": PROMPT}, {"value": code}]} for code, count in zip(CODES, counts) for _ in range(count)]
        weights = facies_weights(rows)
        self.assertEqual(len(rows), 352)
        self.assertAlmostEqual(weights["C"], 11 / 6)
        self.assertAlmostEqual(weights["A"], 11 / 12)
        for code, count in zip(CODES, counts):
            self.assertAlmostEqual(count * weights[code], 352 / 6)

    def test_replay_is_balanced_and_preserves_raw_question_and_answer(self):
        from scripts.train_unibo_visual_facies import LITHOLOGY
        original = [{"conversations": [{"value": f"old raw question {i}"}, {"value": label}],
                     "metadata": {"record_id": f"{label}_{i}"}} for label in LITHOLOGY for i in range(40)]
        selected = select_replay(original, 20261030)
        self.assertEqual(len(selected), 128)
        self.assertEqual(selected, select_replay(original, 20261030))
        for label in LITHOLOGY:
            self.assertEqual(sum(row["conversations"][1]["value"] == label for row in selected), 32)
        for row in selected:
            self.assertIn(row, original)


class VisualFaciesMetricTests(unittest.TestCase):
    def test_invalid_generation_is_a_false_negative_and_remains_in_denominator(self):
        metrics = classification_metrics(["A", "B", "C", "D", "E", "F"], ["A", "B", None, "D", "A", "F"])
        self.assertEqual(metrics["total"], 6)
        self.assertEqual(metrics["correct"], 4)
        self.assertEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(metrics["invalid_outputs"], 1)
        self.assertEqual(metrics["confusion_expected_rows_predicted_columns"]["C"]["INVALID"], 1)
        self.assertEqual(metrics["per_class_recall"]["C"], 0)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 1 + 0 + 1 + 0 + 1) / 6)

    def test_gray_prediction_is_fixed_even_when_validation_labels_change(self):
        answers = [{"expected_code": code, "forced_code": code, "raw_code": None, "source_group": "S10"} for code in CODES]
        answers.append({"expected_code": "F", "forced_code": "F", "raw_code": "F", "source_group": "S11"})
        metrics = facies_stage_metrics(answers, {"forced_code": "B", "raw_code": None}, ["A"] * 2 + list(CODES))
        self.assertEqual(metrics["forced_choice"]["accuracy"], 1)
        self.assertEqual(metrics["gray_forced_choice"]["correct"], 1)
        self.assertEqual(metrics["gray_forced_choice"]["prediction_distribution"], {"B": 7})
        self.assertEqual(metrics["gray_raw_generation"]["invalid_outputs"], 7)
        self.assertEqual(metrics["training_majority"]["code"], "A")
        self.assertEqual(metrics["validation_majority_reference"]["code"], "F")
        self.assertAlmostEqual(metrics["forced_accuracy_minus_gray"], 6 / 7)
        self.assertAlmostEqual(metrics["forced_accuracy_minus_validation_majority"], 5 / 7)
        self.assertEqual(set(metrics["per_source_group"]), {"S10", "S11"})

    def test_format_only_collapse_cannot_beat_gray_and_majority(self):
        answers = [{"expected_code": code, "forced_code": "A", "raw_code": "A", "source_group": "S10"} for code in CODES]
        metrics = facies_stage_metrics(answers, {"forced_code": "A", "raw_code": "A"}, list(CODES))
        self.assertEqual(metrics["forced_accuracy_minus_gray"], 0)
        self.assertEqual(metrics["forced_accuracy_minus_training_majority"], 0)

    def test_misaligned_or_unknown_references_are_refused(self):
        for expected, guesses in (([], []), (["A"], []), (["unknown"], ["A"])):
            with self.assertRaises(ValueError):
                classification_metrics(expected, guesses)


class VisualFaciesSplitTests(unittest.TestCase):
    def setUp(self):
        self.train = [source_row(i) for i in range(6)]
        self.val = [source_row(i, "val") for i in range(6)]

    def test_actual_groups_and_counts_are_computed_from_selected_rows(self):
        result = validate_facies_split(self.train, self.val, verify_images=False)
        self.assertEqual(result["source_groups"], {"train": ["S_train"], "val": ["S_val"]})
        self.assertEqual(result["class_counts"]["train"], dict.fromkeys(SOURCE_CODES, 1))

    def test_each_source_identity_and_hash_leak_is_refused(self):
        for field in ("record_id", "source_group", "source_sha256", "pixel_sha256", "image_sha256"):
            val = deepcopy(self.val)
            val[0]["metadata"][field] = self.train[0]["metadata"][field]
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_facies_split(self.train, val, verify_images=False)

    def test_duplicate_training_pixels_are_refused(self):
        self.train[1]["metadata"]["pixel_sha256"] = self.train[0]["metadata"]["pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_facies_split(self.train, self.val, verify_images=False)

    def test_test_split_and_unapproved_or_relicensed_rows_are_refused(self):
        for field, value in (("split", "test"), ("training_approved", False), ("geology_approved", True),
                             ("source_license", "CC BY 4.0"), ("depths_available", True)):
            train = deepcopy(self.train)
            train[0]["metadata"][field] = value
            with self.assertRaises(ValueError):
                validate_facies_split(train, self.val, verify_images=False)


if __name__ == "__main__":
    unittest.main()
