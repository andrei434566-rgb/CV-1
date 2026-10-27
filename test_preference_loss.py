"""CPU tensors only: causal DPO, detached same-model reference and validation."""
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from app.infrastructure.preference_loss import (
    PreferencePairCollator, dpo_loss, forward_assistant_logps, model_preference_loss,
    preference_labels, sequence_assistant_logps, supports_target_logits, validate_preference_records,
)
from scripts.train_qwen_source_pilot import load_reusable_baseline, validate_loss_records


def row():
    return {"conversations": [{"value": "Чем различаются эти породы?"}, {"value": "Окатанностью обломков."}],
            "preference_rejected": "Только цветом.",
            "metadata": {"split": "train", "training_approved": True}}


def encoded(target=2):
    return {"input_ids": torch.tensor([[0, 1, target, 3]]), "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.tensor([[-100, -100, target, 3]])}


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([0., 0., .2, 0., -.2, 0.]))
        self.enabled = True
        self.calls = []
        self.fail_reference = False

    @contextmanager
    def disable_adapter(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
        self.calls.append({"enabled": self.enabled, "training": self.training, "grad": torch.is_grad_enabled(),
                           "positions": logits_to_keep.tolist() if logits_to_keep is not None else None, "use_cache": use_cache})
        if self.fail_reference and not self.enabled:
            raise RuntimeError("reference failed")
        bias = self.adapter if self.enabled else torch.zeros_like(self.adapter)
        logits = bias.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1)
        if logits_to_keep is not None:
            logits = logits.index_select(1, logits_to_keep)
        return SimpleNamespace(logits=logits)


class PreferenceLossTests(unittest.TestCase):
    def test_labels_mask_prompt_and_template_newline_but_keep_assistant_eos(self):
        labels = preference_labels([10, 11, 20, 21, 99, 198], [10, 11], eos_token_id=99,
                                   max_sequence_tokens=6, identity="pair")
        self.assertEqual(labels, [-100, -100, 20, 21, 99, -100])

    def test_prefix_eos_empty_answer_and_length_errors_are_rejected(self):
        cases = [([10, 11, 20, 99], [10, 12], 8),
                 ([10, 11, 20, 21], [10, 11], 8),
                 ([10, 11, 99], [10, 11], 8),
                 ([10, 11, 20, 99, 21, 99], [10, 11], 8),
                 ([10, 11, 20, 99], [10, 11], 3)]
        for full, prompt, limit in cases:
            with self.subTest(full=full, prompt=prompt, limit=limit), self.assertRaises(ValueError):
                preference_labels(full, prompt, eos_token_id=99, max_sequence_tokens=limit, identity="bad")

    def test_logps_use_causal_shift_only_unmasked_targets_and_sum_not_mean(self):
        logits = torch.tensor([[[9., 0., 0., 0.], [0., 0., 2., 0.], [0., 0., 0., 3.], [9., 0., 0., 0.]]],
                              requires_grad=True)
        labels = torch.tensor([[-100, -100, 2, 3]])
        result = sequence_assistant_logps(logits, labels)
        expected = torch.log_softmax(logits[0, 1], -1)[2] + torch.log_softmax(logits[0, 2], -1)[3]
        torch.testing.assert_close(result, expected.reshape(1))
        result.sum().backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0)
        self.assertEqual(float(logits.grad[0, 3].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, 1:3].abs().sum()), 0)

    def test_selected_logits_do_not_receive_a_second_causal_shift(self):
        logits = torch.randn(1, 5, 8)
        labels = torch.tensor([[-100, -100, 2, 3, -100]])
        full = sequence_assistant_logps(logits, labels)
        positions = torch.tensor([1, 2])
        selected = sequence_assistant_logps(logits[:, positions, :], labels, prediction_positions=positions)
        torch.testing.assert_close(selected, full)

    def test_missing_supervision_and_unpredictable_first_token_are_refused(self):
        for labels in (torch.tensor([[-100, -100, -100]]), torch.tensor([[0, 1, 2]])):
            with self.assertRaises(ValueError):
                sequence_assistant_logps(torch.zeros(1, 3, 6), labels)

    def test_forward_uses_target_positions_before_lm_head_and_disables_cache(self):
        model = TinyAdapterModel()
        self.assertTrue(supports_target_logits(model))
        result = forward_assistant_logps(model, encoded())
        self.assertEqual(model.calls[0]["positions"], [1, 2])
        self.assertIs(model.calls[0]["use_cache"], False)
        self.assertTrue(result.requires_grad)

    def test_fallback_without_logits_to_keep_matches_selected_forward(self):
        class Fallback(TinyAdapterModel):
            def forward(self, input_ids, attention_mask=None, use_cache=False):
                return super().forward(input_ids, attention_mask, use_cache)
        selected, fallback = TinyAdapterModel(), Fallback()
        self.assertFalse(supports_target_logits(fallback))
        torch.testing.assert_close(forward_assistant_logps(selected, encoded()), forward_assistant_logps(fallback, encoded()))

    def test_dpo_formula_and_reference_detachment_preserve_policy_gradients(self):
        chosen = torch.tensor([-2.], requires_grad=True)
        rejected = torch.tensor([-4.], requires_grad=True)
        ref_chosen = torch.tensor([-3.], requires_grad=True)
        ref_rejected = torch.tensor([-4.], requires_grad=True)
        loss = dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=.1)
        self.assertAlmostEqual(float(loss), math.log1p(math.exp(-.1)), places=6)
        loss.backward()
        self.assertLess(float(chosen.grad), 0)
        self.assertGreater(float(rejected.grad), 0)
        self.assertIsNone(ref_chosen.grad)
        self.assertIsNone(ref_rejected.grad)
        for beta in (0, -1, float("nan"), float("inf")):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=beta)

    def test_same_model_reference_runs_first_without_adapter_or_grad_and_restores_train(self):
        model = TinyAdapterModel()
        loss, outputs = model_preference_loss(model, {"chosen": encoded(2), "rejected": encoded(4)}, beta=.1)
        self.assertEqual(len(model.calls), 4)
        for call in model.calls[:2]:
            self.assertFalse(call["enabled"])
            self.assertFalse(call["training"])
            self.assertFalse(call["grad"])
        for call in model.calls[2:]:
            self.assertTrue(call["enabled"])
            self.assertTrue(call["training"])
            self.assertTrue(call["grad"])
        self.assertTrue(model.training)
        self.assertTrue(model.enabled)
        self.assertTrue(all(not value.requires_grad for value in outputs.values()))
        loss.backward()
        self.assertGreater(float(model.adapter.grad.abs().sum()), 0)

    def test_reference_failure_restores_adapter_and_training_state(self):
        model = TinyAdapterModel()
        model.fail_reference = True
        with self.assertRaisesRegex(RuntimeError, "reference failed"):
            model_preference_loss(model, {"chosen": encoded(2), "rejected": encoded(4)}, beta=.1)
        self.assertTrue(model.enabled)
        self.assertTrue(model.training)

    def test_pair_validation_accepts_only_approved_distinct_text_answers(self):
        validate_preference_records([row()])
        mutations = [lambda value: value.update(image="photo.jpg"), lambda value: value.update(images=["photo.jpg"]),
                     lambda value: value.update(preference_rejected=""), lambda value: value.update(preference_rejected=None),
                     lambda value: value.update(preference_rejected=value["conversations"][1]["value"]),
                     lambda value: value["metadata"].update(split="test"),
                     lambda value: value["metadata"].update(training_approved=False)]
        for mutate in mutations:
            item = row()
            mutate(item)
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_preference_records([item])

    def test_sft_default_rejects_preference_manifest_instead_of_ignoring_rejected(self):
        with self.assertRaisesRegex(ValueError, "--loss-mode dpo"):
            validate_loss_records([row()], "sft")
        validate_loss_records([row()], "dpo")
        ordinary = row()
        ordinary.pop("preference_rejected")
        validate_loss_records([ordinary], "sft")

    def test_collator_preserves_nested_chosen_and_rejected_tensors_and_batch_one(self):
        def text_collator(rows):
            return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
        collator = PreferencePairCollator(text_collator)
        pair = {branch: {key: value.squeeze(0) for key, value in encoded(target).items()}
                for branch, target in (("chosen", 2), ("rejected", 4))}
        batch = collator([pair])
        self.assertEqual(set(batch), {"chosen", "rejected"})
        self.assertEqual(batch["chosen"]["input_ids"].shape, (1, 4))
        self.assertEqual(int(batch["rejected"]["labels"][0, 2]), 4)
        with self.assertRaises(ValueError):
            collator([pair, pair])

    def test_dpo_baseline_reuse_allows_only_training_manifest_to_change(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            validation_row = {"id": "test-reference", "conversations": [{"value": "Q"}, {"value": "A"}], "metadata": {}}
            previous = {"schema": "qwen_source_pilot_v1", "model": "same-base", "model_source_fingerprint": {"digest": "same"},
                        "manifests": {"training": {"sha256": "old"}, "theory_validation": {"sha256": "identical"}}}
            answer = {"stage": "baseline", "record_id": "test-reference", "kind": "theory", "source_id": "unknown_source",
                      "question": "Q", "reference": "A", "response": "A", "truncated": False, "generated_tokens": 2, "seconds": .1}
            (path / "run_config.json").write_text(json.dumps(previous), encoding="utf-8")
            (path / "baseline_responses.jsonl").write_text(json.dumps(answer), encoding="utf-8")
            current = copy.deepcopy(previous)
            current["loss_mode"] = "dpo"
            current["dpo_beta"] = .1
            current["manifests"]["training"]["sha256"] = "new-preference-data"
            with self.assertRaisesRegex(ValueError, "training"):
                load_reusable_baseline(path, current, [validation_row])
            _, provenance = load_reusable_baseline(path, current, [validation_row], allow_different_training_manifest=True)
            self.assertEqual(provenance["training_manifest_is_not_a_baseline_input"]["current_sha256"], "new-preference-data")
            current["manifests"]["theory_validation"]["sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "theory_validation"):
                load_reusable_baseline(path, current, [validation_row], allow_different_training_manifest=True)


if __name__ == "__main__":
    unittest.main()
