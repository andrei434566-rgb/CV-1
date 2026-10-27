"""CPU fixtures for the adapter-update measurement, never a geological test."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

from scripts.audit_language_lora_updates import TensorReader, audit


class LoRAUpdateAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base, self.adapter = self.root / "base", self.root / "experiment/adapter"
        self.base.mkdir()
        self.adapter.mkdir(parents=True)
        self.module = "model.language_model.layers.0.self_attn.q_proj"
        self.a = np.array([[.1, .2, -.1]], dtype=np.float32)
        self.b = np.array([[.3], [-.4]], dtype=np.float32)
        self.weight = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        save_file({self.module + ".weight": self.weight}, self.base / "model.safetensors")
        save_file({"base_model.model." + self.module + ".lora_A.weight": self.a,
                   "base_model.model." + self.module + ".lora_B.weight": self.b},
                  self.adapter / "adapter_model.safetensors")
        (self.adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 2}), encoding="utf-8")

    def test_rank_products_equal_the_explicit_scaled_update_norm(self):
        result = audit(self.base, self.adapter)
        expected = np.linalg.norm(2 * self.b.astype(np.float64) @ self.a.astype(np.float64))
        self.assertEqual(result["modules"], 1)
        self.assertAlmostEqual(result["module_measurements"][0]["delta_norm"], expected, places=12)
        self.assertAlmostEqual(result["relative_delta_min_median_max"][1],
                               expected / np.linalg.norm(self.weight.astype(np.float64)), places=12)

    def test_bfloat16_is_decoded_without_torch_or_gpu(self):
        values = np.array([1., -.5, 3.25], dtype="<f4")
        packed = (values.view("<u4") >> 16).astype("<u2").tobytes()
        header = json.dumps({"weight": {"dtype": "BF16", "shape": [3], "data_offsets": [0, len(packed)]}}).encode()
        header += b" " * (-len(header) % 8)
        path = self.root / "bf16.safetensors"
        path.write_bytes(len(header).to_bytes(8, "little") + header + packed)
        np.testing.assert_array_equal(TensorReader(path).tensor("weight"), values)

    def test_unsupported_scaling_is_refused(self):
        for extra in ({"use_rslora": True}, {"use_dora": True}, {"rank_pattern": {"q_proj": 2}},
                      {"alpha_pattern": {"q_proj": 4}}):
            with self.subTest(extra=extra):
                (self.adapter / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 2, **extra}))
                with self.assertRaisesRegex(ValueError, "uniform ordinary LoRA"):
                    audit(self.base, self.adapter)

    def test_nonfinite_weights_are_refused(self):
        bad = self.weight.copy()
        bad[0, 0] = np.nan
        save_file({self.module + ".weight": bad}, self.base / "model.safetensors")
        with self.assertRaisesRegex(ValueError, "Nonfinite tensor"):
            audit(self.base, self.adapter)


if __name__ == "__main__":
    unittest.main()
