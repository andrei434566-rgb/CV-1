"""CPU-only worker integration; references do not change selected model weights."""
from contextlib import nullcontext
import copy
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from app.infrastructure import local_vlm_runner as worker


QUESTION = "Как отличить конгломерат от осадочной брекчии?"
OLD_HITS = [{"path": "C:/unrelated/project.txt", "excerpt": "Посторонняя выдержка проекта.", "kind": "document"}]


class GeologyReferenceWorkerTests(unittest.TestCase):
    def test_general_question_uses_at_most_two_notes_and_replaces_folder_hits(self):
        request = {"operation": "chat", "question": QUESTION, "context": "Посторонний проектный контекст"}
        original = copy.deepcopy(request)
        with patch.object(worker, "retrieve_notes", wraps=worker.retrieve_notes) as retrieve:
            prompt, hits, ids = worker.prepare_geology_reference_chat(request, OLD_HITS)
        retrieve.assert_called_once_with(QUESTION, max_notes=2)
        self.assertEqual(ids, ["coarse_clasts"])
        self.assertIn("Локальная геологическая справка:", prompt)
        self.assertTrue(prompt.endswith("Вопрос: " + QUESTION))
        self.assertTrue(hits)
        self.assertTrue(all(hit["path"].startswith("https://") and hit["reference_note_id"] in ids for hit in hits))
        self.assertTrue(all("Конгломерат" in hit["excerpt"] for hit in hits))
        self.assertNotIn(OLD_HITS[0], hits)
        self.assertNotIn("Посторонний", prompt)
        self.assertNotIn("https://", prompt)
        self.assertEqual(request, original)

    def test_photo_description_batch_and_closed_book_requests_are_not_retrieved(self):
        requests = [
            {"operation": operation, "question": QUESTION} for operation in ("photo_chat", "describe", "describe_batch", "search")
        ]
        requests += [{"operation": "chat", "question": QUESTION, "image": "photo.jpg"},
                     {"operation": "chat", "question": QUESTION, "use_geology_reference": False}]
        with patch.object(worker, "retrieve_notes") as retrieve:
            for request in requests:
                with self.subTest(request=request):
                    prompt, hits, ids = worker.prepare_geology_reference_chat(request, OLD_HITS)
                    self.assertEqual(prompt, QUESTION)
                    self.assertIs(hits, OLD_HITS)
                    self.assertEqual(ids, [])
        retrieve.assert_not_called()

    def test_specific_well_field_folder_file_project_or_interval_keeps_original_scope(self):
        questions = [
            "Чем конгломерат отличается от брекчии в скважине Р-31?",
            "Какой уголь на этом месторождении?",
            "Найди в папках описание конгломерата.",
            "Что говорит файл про конгломерат?",
            "Сохрани проект с описанием конгломерата.",
            "Есть ли уголь в интервале 3000–3001 м?",
            "Конгломерат в скв. Р-31 — что записано?",
            "Find conglomerate in this project file.",
        ]
        with patch.object(worker, "retrieve_notes") as retrieve:
            for question in questions:
                with self.subTest(question=question):
                    prompt, hits, ids = worker.prepare_geology_reference_chat({"operation": "chat", "question": question}, OLD_HITS)
                    self.assertEqual(prompt, question)
                    self.assertIs(hits, OLD_HITS)
                    self.assertEqual(ids, [])
        retrieve.assert_not_called()

    def test_greetings_and_unrelated_questions_remain_unwrapped(self):
        for question in ("Привет, работаешь?", "Спасибо!", "Почему он такого цвета?", "Сколько будет два плюс два?"):
            with self.subTest(question=question):
                prompt, hits, ids = worker.prepare_geology_reference_chat({"operation": "chat", "question": question}, OLD_HITS)
                self.assertEqual(prompt, question)
                self.assertIs(hits, OLD_HITS)
                self.assertEqual(ids, [])

    def test_notes_without_real_source_url_do_not_replace_existing_evidence(self):
        note = {"id": "bad", "text": "Неподтверждённая справка", "source_records": [{"url": "C:/local.txt"}]}
        with patch.object(worker, "retrieve_notes", return_value=[note]):
            prompt, hits, ids = worker.prepare_geology_reference_chat({"operation": "chat", "question": QUESTION}, OLD_HITS)
        self.assertEqual(prompt, QUESTION)
        self.assertIs(hits, OLD_HITS)
        self.assertEqual(ids, [])

    def test_real_worker_messages_and_model_info_are_grounded_in_turbo_mode(self):
        model = worker.LocalModel()
        model.turbo = SimpleNamespace(
            start=Mock(return_value=0), manifest={"release": "test", "gpu_backend": "fake"},
            complete=Mock(return_value={"choices": [{"message": {"content": "Различаются преобладающей окатанностью обломков."},
                                                      "finish_reason": "stop"}]}),
        )
        request = {"operation": "chat", "question": QUESTION, "base": "base", "model_mode": "base",
                   "context": "Посторонний проектный контекст", "show_sources": False}
        with patch.object(worker, "available_runtime", return_value=(Path("fake"), {})):
            result = model.answer(request, OLD_HITS)
        messages = model.turbo.complete.call_args.args[0]
        prompt = messages[-1]["content"][0]["text"]
        self.assertIn("[coarse_clasts]", prompt)
        self.assertNotIn("Контекст текущего проекта:", prompt)
        self.assertNotIn("Посторонний", prompt)
        self.assertNotIn("Найденные локальные выдержки", prompt)
        self.assertNotIn("C:/", prompt)
        self.assertEqual(result["model_info"]["scope"], "general_geology_reference")
        self.assertEqual(result["model_info"]["reference_note_ids"], ["coarse_clasts"])
        self.assertEqual(result["model_info"]["model_mode"], "base")
        self.assertEqual(result["model_info"]["adapter"], "")
        self.assertTrue(all(hit["path"].startswith("https://") for hit in result["hits"]))

    def test_closed_book_turbo_request_keeps_original_context_and_metadata(self):
        model = worker.LocalModel()
        model.turbo = SimpleNamespace(
            start=Mock(return_value=0), manifest={"release": "test"},
            complete=Mock(return_value={"choices": [{"message": {"content": "Ответ без справки."}, "finish_reason": "stop"}]}),
        )
        request = {"operation": "chat", "question": QUESTION, "base": "base", "model_mode": "base",
                   "context": "Первоначальный контекст", "use_geology_reference": False}
        with patch.object(worker, "available_runtime", return_value=(Path("fake"), {})), patch.object(worker, "retrieve_notes") as retrieve:
            result = model.answer(request, OLD_HITS)
        retrieve.assert_not_called()
        prompt = model.turbo.complete.call_args.args[0][-1]["content"][0]["text"]
        self.assertIn("Первоначальный контекст", prompt)
        self.assertIn(OLD_HITS[0]["excerpt"], prompt)
        self.assertNotIn("Локальная геологическая справка:", prompt)
        self.assertNotIn("reference_note_ids", result["model_info"])
        self.assertNotIn("scope", result["model_info"])
        self.assertEqual(result["hits"], OLD_HITS)

    def test_transformers_mode_preserves_the_selected_specialist_adapter(self):
        class Inputs(dict):
            def to(self, device):
                return self

        model = worker.LocalModel()
        model.load = Mock()
        model.processor = Mock()
        model.processor.return_value = Inputs(input_ids=np.array([[1, 2]]))
        model.processor.batch_decode.return_value = ["У конгломерата преобладают окатанные обломки."]
        model.model = Mock()
        model.model.generation_config = SimpleNamespace(eos_token_id=99)
        model.model.generate.return_value = np.array([[1, 2, 5, 99]])
        fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext(), cuda=SimpleNamespace(synchronize=lambda: None))
        request = {"operation": "chat", "question": QUESTION, "base": "base", "model_mode": "lora", "adapter": "previously-selected"}
        with patch.object(worker, "available_runtime", return_value=None), patch.dict("sys.modules", {"torch": fake_torch}):
            result = model.answer(request, OLD_HITS)
        model.model.set_adapter.assert_called_once_with("theory")
        self.assertEqual(result["model_info"]["adapter"], "previously-selected")
        self.assertEqual(result["model_info"]["model_mode"], "lora")
        self.assertEqual(result["model_info"]["reference_note_ids"], ["coarse_clasts"])
        prompt = model.processor.apply_chat_template.call_args.args[0][-1]["content"][0]["text"]
        self.assertNotIn(OLD_HITS[0]["path"], prompt)

    def test_standalone_worker_import_supports_the_same_reference_preparation(self):
        folder = Path(worker.__file__).resolve().parent
        with patch.object(sys, "path", [str(folder), *sys.path]):
            standalone = runpy.run_path(str(folder / "local_vlm_runner.py"), run_name="cv1_standalone_import_test")
        _, hits, ids = standalone["prepare_geology_reference_chat"]({"operation": "chat", "question": QUESTION}, [])
        self.assertEqual(ids, ["coarse_clasts"])
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
