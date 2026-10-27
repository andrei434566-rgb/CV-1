"""Small CPU fixtures; no converter execution, model allocation or server."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from scripts import prepare_qwen_lora_gguf as prep


class QwenLoRAGGUFTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.base, self.adapter, self.runtime = root / "base", root / "run/adapter", root / "runtime"
        for folder in (self.base, self.adapter, self.runtime):
            folder.mkdir(parents=True)
        self.source, self.output = root / "source", root / "new-conversion"
        self.config = {"peft_type": "LORA", "r": 1, "lora_alpha": 2, "bias": "none"}
        self.write(self.base / "config.json", {"text_config": {"num_hidden_layers": 28}})
        self.write(self.adapter / "adapter_config.json", self.config)
        self.fp = {"digest": "fixture", "files": {"config.json": prep.sha256(self.base / "config.json")}}
        self.training = {"status": "candidate_requires_geological_validation", "model_source_fingerprint": self.fp}
        self.write(self.adapter.parent / "run_config.json", self.training)
        self.write(self.adapter / "cv1_adapter_manifest.json", {
            "base_revision": prep.BASE_REVISION, "base_config_sha256": self.fp["files"]["config.json"]})
        (self.runtime / "base.gguf").write_bytes(b"GGUF-fixture-base")
        (self.runtime / "mmproj.gguf").write_bytes(b"GGUF-fixture-mmproj")
        self.runtime_config = {"base_revision": prep.BASE_REVISION, "base_model": "Qwen/Qwen3-VL-2B-Instruct",
                               "release": prep.TAG, "quantization": "Q8_0", "source_fingerprint": self.fp,
                               "model": "base.gguf", "mmproj": "mmproj.gguf", "sha256": {
                                   name: prep.sha256(self.runtime / name) for name in ("base.gguf", "mmproj.gguf")}}
        self.write(self.runtime / "runtime.json", self.runtime_config)
        self.tensors = {}
        for name in prep.GGUF_NAMES:
            branch = "self_attn" if name in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            prefix = f"base_model.model.model.language_model.layers.0.{branch}.{name}"
            self.tensors[prefix + ".lora_A.weight"] = np.ones((1, 3), dtype=np.float32)
            self.tensors[prefix + ".lora_B.weight"] = np.ones((2, 1), dtype=np.float32)
        self.save_tensors()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def save_tensors(self):
        save_file(self.tensors, self.adapter / "adapter_model.safetensors")

    def plan(self):
        with patch.object(prep, "verify_source", return_value={"tag": prep.TAG}), \
                patch.object(prep, "validate_local_models", return_value={"base_revision": prep.BASE_REVISION}), \
                patch.object(prep, "source_fingerprint", return_value=copy.deepcopy(self.fp)):
            return prep.build_plan(self.base, self.adapter, self.runtime, self.source, self.output)

    def test_all_attention_and_mlp_pairs_are_prepared_without_writes_or_activation(self):
        result = self.plan()
        self.assertEqual(len(result["expected_tensor_names"]), 14)
        self.assertIn("blk.0.ffn_gate.weight.lora_a", result["expected_tensor_names"])
        self.assertFalse(result["active_in_application"])
        self.assertFalse(self.output.exists())

    def test_running_training_and_changed_hf_base_are_refused(self):
        for changes in ({"status": "running"}, {"model_source_fingerprint": {"digest": "changed"}}):
            with self.subTest(changes=changes):
                self.write(self.adapter.parent / "run_config.json", {**self.training, **changes})
                with self.assertRaises(ValueError):
                    self.plan()

    def test_runtime_model_tampering_is_refused(self):
        (self.runtime / "base.gguf").write_bytes(b"different-model")
        with self.assertRaisesRegex(ValueError, "Runtime model bytes"):
            self.plan()

    def test_existing_output_is_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()

    def test_visual_non_lora_and_incomplete_tensor_sets_are_refused(self):
        original = copy.deepcopy(self.tensors)
        for bad in ("base_model.model.visual.linear.lora_A.weight", "not.a.lora.tensor"):
            self.tensors = {**original, bad: np.ones((1, 3), dtype=np.float32)}
            self.save_tensors()
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "Unsupported, visual"):
                prep.language_tensor_names(self.adapter, prep.read_json(self.base / "config.json"))
        self.tensors = dict(original)
        self.tensors.pop(next(key for key in original if key.endswith("lora_B.weight")))
        self.save_tensors()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            prep.language_tensor_names(self.adapter, prep.read_json(self.base / "config.json"))

    def test_nonuniform_or_alternative_lora_scaling_is_refused(self):
        for changes in ({"use_rslora": True}, {"use_dora": True}, {"rank_pattern": {"q_proj": 4}},
                        {"modules_to_save": ["lm_head"]}, {"alora_invocation_tokens": [10]}):
            self.write(self.adapter / "adapter_config.json", {**self.config, **changes})
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "ordinary language-only"):
                prep.language_tensor_names(self.adapter, prep.read_json(self.base / "config.json"))

    def test_pinned_source_hash_and_unknown_package_files_are_checked(self):
        self.source.mkdir()
        for name in ("convert_lora_to_gguf.py", "conversion/__init__.py", "gguf-py/gguf/__init__.py"):
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"# fixture\n")
        files = {path.relative_to(self.source).as_posix(): {"sha256": prep.sha256(path)}
                 for path in self.source.rglob("*.py")}
        lock = self.source / "lock.json"
        self.write(lock, {"tag": prep.TAG, "commit": prep.COMMIT, "files": files})
        with patch.object(prep, "LOCK", lock), patch.object(prep, "LOCK_SHA256", prep.sha256(lock)):
            self.assertEqual(prep.verify_source(self.source)["verified_source_files"], 3)
            (self.source / "conversion/unreviewed.py").write_bytes(b"# unknown\n")
            with self.assertRaisesRegex(ValueError, "unexpected or missing"):
                prep.verify_source(self.source)

    def test_gguf_architecture_and_complete_converted_tensor_names_are_checked(self):
        expected = {"blk.0.attn_q.weight.lora_a", "blk.0.attn_q.weight.lora_b"}
        metadata = {"general.architecture": "qwen3vl", "general.type": "adapter",
                    "adapter.type": "lora", "adapter.lora.alpha": 2.0}
        class Field:
            def __init__(self, value): self.value = value
            def contents(self): return self.value
        reader = SimpleNamespace(get_field=lambda name: Field(metadata[name]),
                                 tensors=[SimpleNamespace(name=name) for name in expected])
        package = SimpleNamespace(__file__=str(self.source / "gguf-py/gguf/__init__.py"),
                                  GGUFReader=lambda _: reader)
        with patch.object(prep.importlib, "import_module", return_value=package):
            self.assertEqual(prep.inspect_gguf(self.root_path(), self.source, expected, 2)["tensor_count"], 2)
            metadata["general.architecture"] = "wrong-model"
            with self.assertRaisesRegex(ValueError, "metadata or complete"):
                prep.inspect_gguf(self.root_path(), self.source, expected, 2)

    def root_path(self):
        return self.output / "adapter-f16.gguf"


if __name__ == "__main__":
    unittest.main()
