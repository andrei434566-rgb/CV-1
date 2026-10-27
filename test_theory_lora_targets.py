"""CPU-only checks for text LoRA recipes and untruncated assistant supervision."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.train_qwen_source_pilot import (
    load_reusable_baseline, select_lora_targets, source_fingerprint, validate_training_tokens,
)


class TheoryLoraTargetsTests(unittest.TestCase):
    def modules(self):
        language = "model.language_model.layers.0"
        names = [f"{language}.self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "o_proj")]
        names += [f"{language}.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")]
        names += ["model.visual.blocks.0.attn.q_proj", "model.visual.blocks.0.attn.v_proj",
                  "model.visual.language_model.q_proj", "model.vision_tower.language_model.up_proj",
                  "model.not_language_model.layers.0.q_proj", f"{language}.mlp.other_proj",
                  "lm_head", "model.language_model.norm"]
        return [(name, object()) for name in names]

    def test_default_preserves_language_qv_recipe(self):
        self.assertEqual(select_lora_targets(self.modules()),
                         ["model.language_model.layers.0.self_attn.q_proj",
                          "model.language_model.layers.0.self_attn.v_proj"])

    def test_all_text_linear_contains_exactly_seven_language_projections(self):
        names = select_lora_targets(self.modules(), "all-text-linear")
        self.assertEqual(len(names), 7)
        self.assertEqual({name.split(".")[-1] for name in names},
                         {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"})
        self.assertTrue(all(name.startswith("model.language_model.") for name in names))

    def test_no_language_module_cannot_fall_back_to_visual(self):
        with self.assertRaisesRegex(ValueError, "refusing implicit visual"):
            select_lora_targets([("model.visual.q_proj", object())], "all-text-linear")

    def test_unknown_recipe_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown LoRA"):
            select_lora_targets(self.modules(), "all-linear")

    def validate(self, full=None, prompt=None, labels=None, limit=1024):
        return validate_training_tokens(
            [1, 2, 3, 4] if full is None else full,
            [1, 2] if prompt is None else prompt,
            [-100, -100, 3, 4] if labels is None else labels,
            max_sequence_tokens=limit, identity="test-row",
        )

    def test_valid_supervision_at_exact_limit(self):
        self.assertEqual(self.validate(limit=4), {"sequence_tokens": 4, "prompt_tokens": 2,
                                                 "supervised_tokens": 2, "final_token_supervised": True})

    def test_oversize_is_rejected_without_truncating_caller_data(self):
        full = [1, 2, 3, 4]
        labels = [-100, -100, 3, 4]
        with self.assertRaisesRegex(ValueError, "exceeds 3 tokens.*test-row"):
            self.validate(full=full, labels=labels, limit=3)
        self.assertEqual(full, [1, 2, 3, 4])
        self.assertEqual(labels, [-100, -100, 3, 4])

    def test_prefix_mismatch_and_empty_prompt_are_rejected(self):
        for prompt in ([1, 9], [], [1, 2, 3, 4, 5]):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "prefix mismatch"):
                self.validate(prompt=prompt)

    def test_label_length_must_match_tokens(self):
        with self.assertRaisesRegex(ValueError, "Label/token length"):
            self.validate(labels=[-100, -100, 3])

    def test_prompt_labels_must_be_ignored(self):
        with self.assertRaisesRegex(ValueError, "Prompt tokens must be masked"):
            self.validate(labels=[1, -100, 3, 4])

    def test_empty_supervision_and_wrong_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No supervised assistant tokens"):
            self.validate(labels=[-100] * 4)
        with self.assertRaisesRegex(ValueError, "labels differ"):
            self.validate(labels=[-100, -100, 99, 4])

    def test_terminal_token_must_remain_supervised(self):
        with self.assertRaisesRegex(ValueError, "Final assistant token"):
            self.validate(labels=[-100, -100, 3, -100])

    def test_nonpositive_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.validate(limit=0)

    def test_fingerprint_records_actual_weights_and_processor_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            names = ("config.json", "tokenizer.json", "preprocessor_config.json", "model-00001.safetensors")
            for name in names:
                (base / name).write_bytes(name.encode())
            previous = source_fingerprint(base)
            self.assertEqual(set(previous["files"]), set(names))
            for name in names:
                with self.subTest(name=name):
                    (base / name).write_bytes(b"new content " + name.encode())
                    current = source_fingerprint(base)
                    self.assertNotEqual(previous["digest"], current["digest"])
                    self.assertNotEqual(previous["files"][name], current["files"][name])
                    previous = current

    def test_reused_baseline_must_match_actual_model_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = {"model_source_fingerprint": {"digest": "old weights"}}
            (root / "run_config.json").write_text(json.dumps(previous), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_source_fingerprint differs"):
                load_reusable_baseline(root, {"model_source_fingerprint": {"digest": "changed weights"}}, [])


if __name__ == "__main__":
    unittest.main()
