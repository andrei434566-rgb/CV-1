"""Check the photo-colour hint is used without falsifying model responses."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app.infrastructure import local_vlm_runner as worker


class VlmColorHintTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.crop = Path(self.folder.name) / "crop.png"
        Image.new("RGB", (80, 200), (180, 178, 173)).save(self.crop)
        fields = {key: "описание" for key in worker.DESCRIPTION_FIELDS}
        fields["description"] = "Породы имеют буровато-коричневый цвет. Видна слоистость."
        self.original = fields["description"]
        self.model = worker.LocalModel()
        self.model.turbo = SimpleNamespace(
            start=Mock(return_value=0), manifest={"release": "test"},
            complete=Mock(return_value={"choices": [{"message": {"content": json.dumps(fields, ensure_ascii=False)},
                                                       "finish_reason": "stop"}]}),
        )
        self.request = {"operation": "describe", "image": str(self.crop), "base": "base", "model_mode": "base"}

    def test_grey_crop_guides_model_and_warns_on_contradiction_without_rewriting(self):
        with patch.object(worker, "available_runtime", return_value=(Path("fake"), {})):
            result = self.model.answer(self.request, [])
        prompt = self.model.turbo.complete.call_args.args[0][-1]["content"][0]["text"]
        self.assertIn("светло-серый", prompt)
        self.assertIn("освещения", prompt)
        self.assertTrue(result["color_review_required"])
        self.assertTrue(result["fields"]["description"].startswith(self.original))
        self.assertIn("не совпадает", result["fields"]["description"])
        self.assertEqual(json.loads(result["answer"])["description"], self.original)

    def test_brown_core_does_not_generate_grey_hint_or_wrong_conflict(self):
        Image.new("RGB", (80, 200), (140, 90, 55)).save(self.crop)
        with patch.object(worker, "available_runtime", return_value=(Path("fake"), {})):
            result = self.model.answer(self.request, [])
        prompt = self.model.turbo.complete.call_args.args[0][-1]["content"][0]["text"]
        self.assertIn("оценка цвета по внутренней части вырезки: коричневый", prompt)
        self.assertNotIn("color_review_required", result)
        self.assertEqual(result["fields"]["description"], self.original)

    def test_whole_chat_photo_does_not_get_interval_color_hint(self):
        request = {**self.request, "operation": "photo_chat", "question": "Опиши фото"}
        with patch.object(worker, "available_runtime", return_value=(Path("fake"), {})), \
             patch.object(worker, "estimate_core_color") as estimate:
            self.model.answer(request, [])
        estimate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
