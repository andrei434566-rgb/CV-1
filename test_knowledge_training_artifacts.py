import json
from pathlib import Path
import tempfile
import unittest

from app.infrastructure.local_knowledge_search import _document_text


class KnowledgeTrainingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def put(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_authored_false_preference_cards_are_not_knowledge(self):
        path = self.put("examples.json", {"rejected_training_role": "rejected_false_statement_not_ground_truth",
                                          "cards": [{"chosen": "Окатанность", "rejected": "Выдуманный признак"}]})
        self.assertEqual(_document_text(path), "")

    def test_paired_jsonl_is_not_knowledge_even_with_unknown_filename(self):
        path = self.put("geology_train.jsonl", {"conversations": [], "preference_rejected": "Ложное объяснение"})
        self.assertEqual(_document_text(path), "")

    def test_evaluation_only_cards_are_never_used_as_reference(self):
        for role in ("evaluation_only_never_train", "evaluation_source_audit"):
            path = self.put("unknown_evaluation.json", {"record_role": role,
                                                         "items": [{"question": "Q", "answer": "A"}]})
            self.assertEqual(_document_text(path), "")

    def test_known_facies_drafts_with_errors_are_quarantined(self):
        folder = self.root / "facies_reasoning_v1"
        folder.mkdir()
        for name in ("training_cards.json", "training_reasoned_cards.json", "training_reasoned_cards_v2.json",
                     "training_positive_extension.json", "training_positive_extension_v2.json"):
            path = folder / name
            path.write_text('{"answer":"unsupported observation"}', encoding="utf-8")
            self.assertEqual(_document_text(path), "")
        fixed = folder / "training_reasoned_cards_v3.json"
        fixed.write_text('{"answer":"corrected"}', encoding="utf-8")
        self.assertIn("corrected", _document_text(fixed))

    def test_model_responses_are_not_geological_sources(self):
        for name in ("baseline_responses.jsonl", "candidate_responses.jsonl", "blind_review.json", "QUALITATIVE_REVIEW.json",
                     "baseline_metrics.json", "candidate_metrics.json", "validation_questions.json", "evaluation_challenge.json",
                     "holdout_theory.jsonl", "holdout_conversation.jsonl", "val_visual.jsonl"):
            with self.subTest(name=name):
                path = self.put(name, {"response": "Модель может ошибаться"})
                self.assertEqual(_document_text(path), "")

    def test_source_notes_and_legitimate_geological_json_remain_searchable(self):
        path = self.put("geology.json", {"well": "Р-31", "description": "Песчаник с горизонтальной слоистостью"})
        self.assertIn("Песчаник", _document_text(path))
        path = self.put("source_notes.json", {"schema_version": 1, "notes": [{"text": "Конгломерат: окатанные обломки"}]})
        self.assertIn("окатанные", _document_text(path))


if __name__ == "__main__":
    unittest.main()
