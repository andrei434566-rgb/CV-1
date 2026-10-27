"""Local-runtime selection must preserve adapters and per-object results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.infrastructure.local_llama_runtime import available_runtime, DESCRIPTION_SCHEMA
from app.infrastructure.local_vlm_runner import LocalModel


class LocalLlamaRuntimeTests(unittest.TestCase):
    def test_specialist_adapter_cannot_be_replaced_by_base_gguf(self):
        request = {"base": "cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/rev", "model_mode": "lora", "adapter": "expert"}
        self.assertIsNone(available_runtime(request))

    def test_runtime_is_selected_only_for_matching_source_revision_and_contained_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = {"base_model": "Qwen/Qwen3-VL-2B-Instruct", "base_revision": "rev",
                        "server": "server.exe", "model": "model.gguf", "mmproj": "mmproj.gguf"}
            for key in ("server", "model", "mmproj"):
                (root / manifest[key]).touch()
            path = root / "runtime.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            request = {"base": "cache/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/rev", "model_mode": "base", "turbo_root": str(root)}
            self.assertIsNotNone(available_runtime(request))
            request["base"] += "-changed"
            self.assertIsNone(available_runtime(request))
            request["base"] = request["base"].removesuffix("-changed")
            manifest["model"] = "../outside.gguf"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertIsNone(available_runtime(request))

    def test_detailed_schema_keeps_all_six_original_fields(self):
        self.assertEqual(set(DESCRIPTION_SCHEMA["required"]),
                         {"facies_name", "lithology", "texture", "facies_association", "depositional_environment", "description"})

    def test_batch_keeps_item_identity_and_isolates_failures(self):
        class Model(LocalModel):
            def answer(self, request, hits):
                if request["image"] == "bad":
                    raise ValueError("bad image")
                return {"fields": {"description": request["image"]}}
        model = Model()
        with patch("app.infrastructure.local_vlm_runner.available_runtime", return_value=None):
            response = model._answer_batch({"items": [{"id": "2", "image": "good"}, {"id": "9", "image": "bad"}]})
        self.assertEqual(response["items"]["2"]["fields"]["description"], "good")
        self.assertFalse(response["items"]["9"]["ok"])
        self.assertEqual(set(response["items"]), {"2", "9"})

    def test_duplicate_batch_ids_are_rejected_before_inference(self):
        with self.assertRaises(ValueError):
            LocalModel()._answer_batch({"items": [{"id": "1"}, {"id": "1"}]})


if __name__ == "__main__":
    unittest.main()
