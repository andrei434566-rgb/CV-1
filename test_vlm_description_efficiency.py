"""Duplicate crops reuse only complete outputs from the same model/context."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from app.infrastructure.local_vlm_runner import LocalModel, DESCRIPTION_FIELDS


class DescriptionEfficiencyTests(unittest.TestCase):
    def test_identical_attachments_reuse_result_but_context_and_model_changes_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first, copy = root / "one.png", root / "copy.png"
            first.write_bytes(b"same crop")
            copy.write_bytes(first.read_bytes())
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"version-one")
            model = LocalModel()
            result = {"fields": {key: "Серый слоистый керн." for key in DESCRIPTION_FIELDS}, "incomplete": False}
            request = {"operation": "describe", "base": "local-base", "adapter": str(adapter),
                       "image": str(first), "context": "Скважина 1; интервал 1–2"}
            with patch.object(model, "_answer_uncached", return_value=result) as infer:
                answer = model.answer(request, [])
                answer["fields"]["description"] = "Правка специалиста"
                cached = model.answer({**request, "image": str(copy)}, [])
                self.assertEqual(infer.call_count, 1)
                self.assertTrue(cached["timing"]["cache_hit"])
                self.assertNotEqual(cached["fields"]["description"], "Правка специалиста")
                model.answer({**request, "context": "Другой интервал"}, [])
                self.assertEqual(infer.call_count, 2)
                weights.write_bytes(b"new-version-longer")
                model.answer(request, [])
                self.assertEqual(infer.call_count, 3)

    def test_incomplete_or_failed_outputs_are_never_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "crop.png"
            image.write_bytes(b"crop")
            model = LocalModel()
            for response in ({"ok": False}, {"incomplete": True, "fields": {key: "X" for key in DESCRIPTION_FIELDS}}):
                with patch.object(model, "_answer_uncached", return_value=response) as infer:
                    for _ in range(2):
                        model.answer({"operation": "describe", "image": str(image)}, [])
                    self.assertEqual(infer.call_count, 2)

    def test_description_uses_full_final_allowance_once_even_when_limit_is_hit(self):
        import torch
        class Inputs(dict):
            def to(self, _device):
                return self
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "prompt"
            def __call__(self, **kwargs):
                return Inputs(input_ids=torch.tensor([[1, 2]]))
            def batch_decode(self, *args, **kwargs):
                return [json.dumps({name: "Серый керн." for name in DESCRIPTION_FIELDS})]
        model = LocalModel()
        model.processor = Processor()
        fake = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=2))
        from unittest.mock import Mock
        fake.generate = Mock(return_value=torch.tensor([[1, 2] + [3] * 320]))
        model.model = fake
        with (patch.object(model, "load"), patch("app.infrastructure.local_vlm_runner.available_runtime", return_value=None),
              patch("torch.cuda.synchronize")):
            response = model.answer({"operation": "describe"}, [])
        self.assertEqual(fake.generate.call_count, 1)
        self.assertEqual(fake.generate.call_args.kwargs["max_new_tokens"], 320)
        self.assertTrue(response["incomplete"])
        self.assertFalse(response["timing"]["retried"])


if __name__ == "__main__":
    unittest.main()
