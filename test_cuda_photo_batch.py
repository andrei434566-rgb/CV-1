"""Independent image batches preserve interval identity and all six fields."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from PIL import Image
from app.infrastructure.core_description_prompt import description_prompt, check_interpretation_roles
from app.infrastructure.local_vlm_runner import LocalModel, DESCRIPTION_FIELDS
from app.infrastructure.structured_generation import lithology_logits_processor


class CudaPhotoBatchTests(unittest.TestCase):
    def test_rock_filter_rejects_unknown_name_and_preserves_free_observations(self):
        import torch
        class CharacterTokenizer:
            alphabet = list(dict.fromkeys('0{}[],:" \\abcdefghijklmnopqrstuvwxyz' +
                                         'Предположительно: ПесчаникНе установлена по фотосуглинок'))
            all_special_ids = [0]
            eos_token_id = 0
            def __len__(self):
                return len(self.alphabet) + 1
            def encode(self, text, **kwargs):
                return [self.alphabet.index(char) + 1 for char in text]
            def decode(self, ids, **kwargs):
                return ''.join(self.alphabet[i - 1] for i in ids if i)
        tokenizer = CharacterTokenizer()
        processor = lithology_logits_processor(tokenizer, [
            "Предположительно: Песчаник", "Не установлена по фото"], 1)
        def accepts(text):
            ids = [1]
            for token_id in tokenizer.encode(text):
                scores = processor(torch.tensor([ids]), torch.zeros((1, len(tokenizer))))
                if not torch.isfinite(scores[0, token_id]):
                    return False
                ids.append(token_id)
            return True
        self.assertTrue(accepts('{"description":"Песчаник","lithology":"Предположительно: Песчаник"}'))
        self.assertFalse(accepts('{"description":"Песчаник","lithology":"суглинок"}'))
        self.assertTrue(accepts('{"description":"Песчаник","lithology":"Не установлена по фото"}'))
        inputs = torch.tensor([[1] + tokenizer.encode('{"description":"Песчаник')])
        original = torch.randn(1, len(tokenizer))
        self.assertTrue(torch.equal(processor(inputs, original.clone()), original))

    def test_prompt_does_not_offer_rare_textures_as_a_menu(self):
        prompt = description_prompt("Скважина 1", True)
        self.assertNotIn("ролл", prompt.casefold())
        self.assertNotIn("пуддин", prompt.casefold())
        self.assertIn("facies_association", prompt)
        self.assertIn("Не установлена по фото", prompt)

    def test_observations_survive_but_clear_role_errors_are_rejected(self):
        fields = {name: "песчаник" for name in DESCRIPTION_FIELDS}
        fields["description"] = "Серый керн с горизонтальными слойками."
        fields["depositional_environment"] = "вулканическая или речная среда"
        checked, notes = check_interpretation_roles(fields)
        self.assertEqual(checked["description"], fields["description"])
        self.assertEqual(checked["lithology"], "песчаник")
        self.assertEqual(checked["facies_association"], "Не установлена по фото")
        self.assertEqual(checked["depositional_environment"], "Не установлена по фото")
        self.assertEqual(len(notes), 3)
        self.assertEqual(fields["facies_association"], "песчаник")

    def test_bedding_description_is_not_a_depositional_setting(self):
        fields = {name: "Не установлен" for name in DESCRIPTION_FIELDS}
        fields.update(description="Серый керн со слойками.", lithology="Предположительно: Аргиллит",
                      facies_association="Слойчатая аргиллитовая фация", depositional_environment="Разделение по горизонтали")
        checked, notes = check_interpretation_roles(fields)
        self.assertEqual(checked["description"], fields["description"])
        self.assertEqual(checked["facies_name"], "Не установлена по фото")
        self.assertEqual(checked["facies_association"], "Не установлена по фото")
        self.assertEqual(checked["depositional_environment"], "Не установлена по фото")
        self.assertEqual(len(notes), 2)

    def test_five_images_decode_once_and_keep_independent_results(self):
        import torch
        class Inputs(dict):
            def to(self, device):
                return self
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                return messages[-1]["content"][-1]["text"]
            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return Inputs(input_ids=torch.tensor([[1, 2]] * 5))
            def batch_decode(self, *args, **kwargs):
                return [json.dumps({name: f"Независимое описание {i}" for name in DESCRIPTION_FIELDS}) for i in range(5)]
        model = LocalModel()
        model.processor = Processor()
        model.model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=9),
                                     generate=Mock(return_value=torch.tensor([[1, 2, 3, 9]] * 5)))
        with tempfile.TemporaryDirectory() as folder:
            items = []
            for i in range(5):
                path = Path(folder) / f"core{i}.png"
                Image.new("RGB", (80 + i, 700), (90 + i * 20,) * 3).save(path)
                items.append({"id": str(i + 7), "image": str(path), "context": f"Участок {i}", "whole_column": True})
            with patch.object(model, "load"), patch("torch.cuda.synchronize"):
                result = model.answer({"operation": "describe_photo", "items": items, "model_mode": "base"}, [])
        self.assertEqual(model.model.generate.call_count, 1)
        self.assertEqual(len(model.processor.kwargs["images"]), 5)
        self.assertTrue(model.processor.kwargs["padding"])
        self.assertEqual(len({image.getpixel((74, 224)) for image in model.processor.kwargs["images"]}), 5)
        for i, item in enumerate(items):
            self.assertTrue(result["items"][item["id"]]["ok"])
            self.assertEqual(result["items"][item["id"]]["fields"]["description"], f"Независимое описание {i}")
            self.assertEqual(set(result["items"][item["id"]]["fields"]), set(DESCRIPTION_FIELDS))

    def test_duplicate_ids_are_rejected_before_model_load(self):
        model = LocalModel()
        with patch.object(model, "load") as load:
            with self.assertRaises(ValueError):
                model.answer({"operation": "describe_photo", "items": [{"id": "1"}, {"id": "1"}]}, [])
        load.assert_not_called()

    def test_cuda_oom_falls_back_without_losing_the_fifth_interval(self):
        import torch
        class Inputs(dict):
            def to(self, device):
                return self
        model = LocalModel()
        class Processor:
            def apply_chat_template(self, *args, **kwargs):
                return "prompt"
            def __call__(self, **kwargs):
                return Inputs(input_ids=torch.tensor([[1]] * 5))
        model.processor = Processor()
        model.model = SimpleNamespace(generate=Mock(side_effect=torch.cuda.OutOfMemoryError("test OOM")))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "core.png"
            Image.new("RGB", (100, 200), "gray").save(path)
            items = [{"id": str(i), "image": str(path)} for i in range(5)]
            response = {"items": {str(i): {"ok": True} for i in range(4)}, "timing": {}}
            with (patch.object(model, "load"), patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"),
                  patch.object(model, "_answer_batch", return_value=response) as first,
                  patch.object(model, "answer", return_value={"ok": True, "fields": {"description": "Пятый"}}) as last):
                result = model._answer_photo_batch({"items": items, "model_mode": "base"})
        self.assertEqual(len(first.call_args.args[0]["items"]), 4)
        self.assertEqual(last.call_args.args[0]["id"], "4")
        self.assertEqual(result["items"]["4"]["fields"]["description"], "Пятый")
        self.assertEqual(set(result["items"]), {"0", "1", "2", "3", "4"})
        self.assertTrue(result["timing"]["smaller_batch_fallback"])


if __name__ == "__main__":
    unittest.main()
