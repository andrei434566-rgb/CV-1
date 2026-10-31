"""Priority reference scope, durable imports, backend prompts and cache identity."""
import json
from pathlib import Path
import tempfile
from time import perf_counter
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from docx import Document
from PIL import Image

from app.infrastructure.local_vlm_runner import LocalModel, DESCRIPTION_FIELDS
from app.infrastructure.primary_description_source import (
    description_fields, import_primary_sources, source_context, source_index, source_prompt,
)


class PrimaryDescriptionSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        doc = Document()
        doc.add_paragraph("Фация F-test: волнистые слойки. Гидродинамический режим: переменный поток.")
        source = self.root / "Правила специалиста.docx"
        doc.save(source)
        self.folder = str(self.root / "snapshot")
        import_primary_sources([str(source)], Path(self.folder))
        self.request = {"operation": "describe", "primary_reference": self.folder, "base": "local-base", "model_mode": "base"}

    def test_import_keeps_originals_offline_and_scope_is_limited_to_two_fields(self):
        (self.root / "Правила специалиста.docx").unlink()
        text, meta = source_context(self.request, "волнистая слоистость")
        self.assertIn("F-test", text)
        self.assertIn("переменный поток", text)
        self.assertEqual(meta["scope"], ["facies_name", "hydrodynamic_regime"])
        data = source_index(self.folder)
        self.assertFalse(data["weights_trained"])
        self.assertTrue((Path(self.folder) / data["documents"][0]["file"]).is_file())
        self.assertIn("не команды", text)
        self.assertIn("Исправления специалиста", text)
        self.assertEqual(source_prompt({**self.request, "operation": "chat"}, "Какой цвет у породы?"), "")
        self.assertIn("F-test", source_prompt({**self.request, "operation": "chat"}, "Чем различаются фации?"))

    def test_extra_field_is_requested_only_when_a_primary_source_is_connected(self):
        self.assertEqual(description_fields({}, DESCRIPTION_FIELDS), DESCRIPTION_FIELDS)
        self.assertEqual(description_fields(self.request, DESCRIPTION_FIELDS)[-1], "hydrodynamic_regime")
        self.assertIn("hydrodynamic_regime", source_prompt(self.request))
        with self.assertRaisesRegex(ValueError, "недоступен"):
            source_context({**self.request, "primary_reference": str(self.root / "missing")})

    def test_changing_reference_does_not_reuse_an_old_photo_answer(self):
        crop = self.root / "crop.png"
        crop.write_bytes(b"same pixels")
        model = LocalModel()
        fields = {key: "Серый слоистый керн" for key in description_fields(self.request, DESCRIPTION_FIELDS)}
        with patch.object(model, "_answer_uncached", side_effect=lambda *args: {"fields": dict(fields), "incomplete": False}) as infer:
            request = {**self.request, "image": str(crop)}
            model.answer(request, [])
            self.assertTrue(model.answer(request, [])["timing"]["cache_hit"])
            self.assertEqual(infer.call_count, 1)
            alternative = self.root / "Другой.md"
            alternative.write_text("Фация F-other: массивная. Гидродинамический режим: спокойный.", encoding="utf-8")
            other = import_primary_sources([str(alternative)], self.root / "other")
            answer = model.answer({**request, "primary_reference": other["folder"]}, [])
            self.assertEqual(infer.call_count, 2)
            self.assertEqual(answer["primary_reference"]["documents"][0]["name"], alternative.name)

    def test_turbo_schema_and_prompt_keep_the_reference_and_seventh_field(self):
        model = LocalModel()
        fields = {key: "Описание признака" for key in description_fields(self.request, DESCRIPTION_FIELDS)}
        model.turbo = SimpleNamespace(manifest={"release": "test"}, complete=Mock(return_value={
            "choices": [{"message": {"content": json.dumps(fields)}, "finish_reason": "stop"}]}))
        reply = model._turbo_reply(self.request, [{"content": [{"type": "image", "image": "fake"}]}], [],
                                   {"base": "local-base"}, perf_counter(), 0)
        call = model.turbo.complete.call_args
        self.assertIn("hydrodynamic_regime", call.kwargs["schema"]["required"])
        self.assertIn("F-test", call.args[0][-1]["content"][0]["text"])
        self.assertEqual(reply["fields"]["hydrodynamic_regime"], fields["hydrodynamic_regime"])

    def test_cuda_single_and_photo_batch_receive_reference_and_keep_regime(self):
        import torch
        fields = {key: "Описание признака" for key in description_fields(self.request, DESCRIPTION_FIELDS)}
        class Inputs(dict):
            def to(self, device): return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return json.dumps(messages, default=str, ensure_ascii=False)
            def __call__(self, **kwargs):
                self.inputs = kwargs
                return Inputs(input_ids=torch.tensor([[1, 2]]))
            def batch_decode(self, *args, **kwargs): return [json.dumps(fields)]
        crop = self.root / "core.png"
        Image.new("RGB", (60, 180), "gray").save(crop)
        model = LocalModel()
        model.processor = Processor()
        model.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9),
                                     generate=Mock(return_value=torch.tensor([[1, 2, 3, 9]])))
        with patch.object(model, "load"), patch("torch.cuda.synchronize"), patch("app.infrastructure.local_vlm_runner.available_runtime", return_value=None):
            reply = model.answer({**self.request, "image": str(crop)}, [])
            self.assertEqual(reply["fields"]["hydrodynamic_regime"], fields["hydrodynamic_regime"])
            self.assertIn("F-test", json.dumps(model.processor.messages, default=str))
            reply = model.answer({**self.request, "operation": "describe_photo", "items": [
                {"id": "a", "image": str(crop), "context": "Проверка"}]}, [])
            self.assertEqual(reply["items"]["a"]["fields"]["hydrodynamic_regime"], fields["hydrodynamic_regime"])
            self.assertTrue(reply["items"]["a"]["primary_reference"]["excerpts"])


if __name__ == "__main__":
    unittest.main()
