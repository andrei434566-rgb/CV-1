"""Numerical CPU checks for causal facies discrimination and honest gates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn.functional as F

from scripts.train_unibo_facies_discrimination import (
    RECIPE, CRITERIA, discrimination_loss, pairwise_probes, research_gates, restore,
)
from test_sft_supervised_logits import TinySelectedModel

IDS = tuple(range(32, 38))


def example(target, count=3):
    labels = torch.tensor([[-100] * 4 + [IDS[target], 2, 3][:count]])
    return {"input_ids": torch.where(labels == -100, torch.tensor(1), labels), "labels": labels}


class FaciesDiscriminationTests(unittest.TestCase):
    def test_full_logits_loss_and_gradient_match_independent_objective_for_all_classes(self):
        for target in range(6):
            torch.manual_seed(7)
            initial = torch.randn(1, 7, 44)
            model = TinySelectedModel(initial)
            inputs = example(target)
            loss, _, classification = discrimination_loss(model, inputs, IDS, target)
            loss.backward()
            direct = initial.clone().requires_grad_()
            class_ce = F.cross_entropy(direct[:, 3, list(IDS)], torch.tensor([target]))
            shifted = F.pad(inputs["labels"], (0, 1), value=-100)[:, 1:]
            sft = F.cross_entropy(direct.reshape(-1, 44), shifted.reshape(-1), ignore_index=-100)
            reference = class_ce + .1 * sft
            reference.backward()
            self.assertTrue(torch.allclose(loss, reference, atol=1e-6))
            self.assertTrue(torch.allclose(model.raw.grad, direct.grad, atol=1e-7))
            self.assertTrue(torch.allclose(classification, class_ce))
            self.assertEqual(model.positions, [3, 4, 5])

    def test_classification_gradient_favours_true_class_over_every_wrong_alternative(self):
        model = TinySelectedModel(torch.zeros(1, 7, 44))
        _, _, class_ce = discrimination_loss(model, example(2), IDS, 2)
        class_ce.backward()
        grad = model.raw.grad[0, 3]
        self.assertLess(grad[IDS[2]].item(), 0)
        for index in (0, 1, 3, 4, 5):
            self.assertGreater(grad[IDS[index]].item(), 0)
        self.assertEqual(grad[0].item(), 0)
        self.assertTrue(torch.all(model.raw.grad[:, 4:] == 0))

    def test_eos_targets_do_not_dilute_classification_term(self):
        initial = torch.randn(1, 7, 44)
        values = []
        for count in (1, 2, 3):
            inputs = example(3, count)
            model = TinySelectedModel(initial[:, :inputs["labels"].shape[1]])
            _, _, class_ce = discrimination_loss(model, inputs, IDS, 3)
            values.append(class_ce.item())
        self.assertAlmostEqual(values[0], values[1])
        self.assertAlmostEqual(values[0], values[2])

    def test_raw_lithology_replay_remains_full_sft_with_same_gradients(self):
        initial = torch.randn(1, 7, 44)
        model = TinySelectedModel(initial)
        inputs = example(0)
        inputs["labels"][0, 4] = 11
        inputs["input_ids"][0, 4] = 11
        loss, _, class_ce = discrimination_loss(model, inputs, IDS, -1)
        loss.backward()
        direct = initial.clone().requires_grad_()
        shifted = F.pad(inputs["labels"], (0, 1), value=-100)[:, 1:]
        expected = F.cross_entropy(direct.reshape(-1, 44), shifted.reshape(-1), ignore_index=-100)
        expected.backward()
        self.assertIsNone(class_ce)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.allclose(model.raw.grad, direct.grad))

    def test_sample_mean_gradient_accumulation_does_not_count_tokens_twice(self):
        initial = torch.randn(1, 7, 44)
        accumulated, direct = TinySelectedModel(initial), TinySelectedModel(initial)
        losses = []
        for target in (0, 2, 3, 5):
            loss, _, _ = discrimination_loss(accumulated, example(target), IDS, target)
            (loss / 4).backward()
            other, _, _ = discrimination_loss(direct, example(target), IDS, target)
            losses.append(other)
        torch.stack(losses).mean().backward()
        self.assertTrue(torch.allclose(accumulated.raw.grad, direct.raw.grad, atol=1e-7))

    def test_wrong_class_target_and_duplicate_ids_are_refused(self):
        for ids, target in ((IDS, 1), ((32,) * 6, 0)):
            with self.assertRaises(ValueError):
                discrimination_loss(TinySelectedModel(torch.randn(1, 7, 44)), example(0), ids, target)

    def test_multi_image_batch_and_unknown_task_are_refused(self):
        inputs = {key: value.repeat(2, 1) for key, value in example(0).items()}
        with self.assertRaises(ValueError):
            discrimination_loss(TinySelectedModel(torch.randn(2, 7, 44)), inputs, IDS, 0)
        with self.assertRaises(ValueError):
            discrimination_loss(TinySelectedModel(torch.randn(1, 7, 44)), example(0), IDS, 6)

    def test_pair_probes_use_all15_pairs_but_only6_unique_images(self):
        rows = [{"expected_code": c, "probabilities_A_to_F": {x: .9 if x == c else .02 for x in "ABCDEF"}} for c in "ABCDEF"]
        report = pairwise_probes(rows)
        self.assertEqual(report["unique_validation_images"], 6)
        self.assertEqual(len(report["pairs"]), 15)
        self.assertEqual(report["pairs"]["BC"]["correct"], 2)
        self.assertTrue(all(v["accuracy"] == 1 for v in report["pairs"].values()))
        self.assertIn("not_six_way_accuracy", report["role"])

    def test_high_aggregate_score_does_not_hide_unlearned_prodelta_or_regression(self):
        before = {"facies": {"forced_choice": {"accuracy": .32, "macro_f1": .26}}, "raw_lithology_guard": {"correct": 60}}
        after = {"facies": {"forced_choice": {"accuracy": .6, "macro_f1": .5, "per_class_recall": dict.fromkeys("ABCDEF", .5)}, "gray_forced_choice": {"accuracy": .24}}, "raw_lithology_guard": {"correct": 58}}
        self.assertTrue(all(research_gates(before, after).values()))
        after["facies"]["forced_choice"]["per_class_recall"]["C"] = 0
        self.assertFalse(research_gates(before, after)["all_six_nonzero_recall"])
        after["raw_lithology_guard"]["correct"] = 57
        self.assertFalse(research_gates(before, after)["lithology_preserved"])

    def test_resume_cannot_change_epoch_count_or_objective(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            config = {"recipe": deepcopy(RECIPE), "criteria": CRITERIA}
            config["recipe"]["epochs"] += 1
            (output / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recipe"):
                restore(output, output / "parent", output / "runtime")


if __name__ == "__main__":
    unittest.main()
