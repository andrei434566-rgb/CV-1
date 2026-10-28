"""Expert edits are durable and cannot leak across wells, photos or intervals."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

from PySide6.QtCore import QPointF

from app.infrastructure import expert_corrections as memory
from app.infrastructure.vlm_training_data import expert_description_digest, expert_description_is_confirmed, _pair_provenance


class ExpertCorrectionMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.patch = patch.object(memory, "_root", return_value=self.root / "memory")
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.photo = self.root / "photo.jpg"
        self.photo.write_bytes(b"original photo")
        self.record = SimpleNamespace(path=str(self.photo), well_name="Р-31",
                                      pixmap=SimpleNamespace(width=lambda: 100, height=lambda: 200))
        self.layer = SimpleNamespace(label="Русловая", polygon=[QPointF(10, 10), QPointF(50, 10), QPointF(50, 80), QPointF(10, 80)],
                                     depth_from=100.0, depth_to=101.0, training_ready=True,
                                     attributes={"__geometry_only": "1", "__vlm_description": "Серый песчаник."})

    def confirmed(self):
        self.layer.attributes["__expert_description_confirmed"] = "1"
        self.layer.attributes["__expert_description_sha256"] = expert_description_digest(self.layer)

    def save(self):
        return memory.save_interval_edit(self.record, self.layer, project="project-a", well="Р-31")

    def find(self, **kwargs):
        return memory.exact_interval_edit(self.record, self.layer, project="project-a", well="Р-31", **kwargs)

    def test_saved_manual_draft_is_durable_but_not_approved_context(self):
        row = self.save()
        self.assertFalse(row["description_expert_confirmed"])
        self.assertIsNone(self.find())
        self.assertEqual(self.find(require_confirmed=False)["attributes"]["__vlm_description"], "Серый песчаник.")

    def test_memory_confirmation_for_regular_interval_does_not_approve_training(self):
        self.layer.attributes.pop("__geometry_only")
        self.layer.training_ready = False
        self.layer.attributes["__expert_memory_confirmed"] = "1"
        self.layer.attributes["__expert_memory_sha256"] = memory.expert_memory_digest(self.layer)
        self.assertTrue(memory.expert_memory_is_confirmed(self.layer))
        self.assertFalse(expert_description_is_confirmed(self.layer))
        self.assertEqual(_pair_provenance(self.layer)[0], "")
        self.assertTrue(self.save()["description_expert_confirmed"])
        self.assertEqual(self.find()["facies"], "Русловая")

    def test_memory_confirmation_for_whole_column_without_training(self):
        self.layer.training_ready = False
        self.layer.attributes["__core_column_description"] = "1"
        self.layer.attributes["__expert_memory_confirmed"] = "1"
        self.layer.attributes["__expert_memory_sha256"] = memory.expert_memory_digest(self.layer)
        self.assertTrue(memory.expert_memory_is_confirmed(self.layer))
        self.assertEqual(_pair_provenance(self.layer)[0], "")
        self.save()
        self.assertIsNotNone(self.find())

    def test_stale_memory_confirmation_after_text_or_geometry_changes_is_rejected(self):
        self.layer.training_ready = False
        self.layer.attributes["__expert_memory_confirmed"] = "1"
        self.layer.attributes["__expert_memory_sha256"] = memory.expert_memory_digest(self.layer)
        self.save()
        self.layer.attributes["__vlm_description"] = "Изменённое описание."
        self.assertFalse(memory.expert_memory_is_confirmed(self.layer))
        self.save()
        self.assertIsNone(self.find())
        self.layer.attributes["__expert_memory_sha256"] = memory.expert_memory_digest(self.layer)
        self.layer.polygon[0] = QPointF(12, 10)
        self.assertFalse(memory.expert_memory_is_confirmed(self.layer))

    def test_memory_confirmation_requires_nonempty_description(self):
        self.layer.attributes["__vlm_description"] = " "
        self.layer.attributes["__expert_memory_confirmed"] = "1"
        self.layer.attributes["__expert_memory_sha256"] = memory.expert_memory_digest(self.layer)
        self.assertFalse(memory.expert_memory_is_confirmed(self.layer))

    def test_memory_uncheck_does_not_reuse_older_training_confirmation(self):
        self.confirmed()
        self.assertTrue(expert_description_is_confirmed(self.layer))
        self.layer.attributes["__expert_memory_confirmed"] = "0"
        self.assertFalse(self.save()["description_expert_confirmed"])
        self.assertIsNone(self.find())

    def test_confirmed_edit_only_matches_same_well_project_image_and_geometry(self):
        self.confirmed()
        self.save()
        self.assertEqual(self.find()["facies"], "Русловая")
        self.assertIsNone(memory.exact_interval_edit(self.record, self.layer, project="project-b", well="Р-31"))
        self.assertIsNone(memory.exact_interval_edit(self.record, self.layer, project="project-a", well="Р-32"))
        self.photo.write_bytes(b"another photo")
        self.assertIsNone(self.find())
        self.photo.write_bytes(b"original photo")
        self.layer.polygon[0] = QPointF(11, 10)
        self.assertIsNone(self.find())
        self.layer.polygon[0] = QPointF(10, 10)
        self.layer.depth_to = 102.0
        self.assertIsNone(self.find())

    def test_new_unconfirmed_edit_does_not_restore_stale_confirmed_text(self):
        self.confirmed()
        self.save()
        self.layer.attributes["__vlm_description"] = "Новая ручная поправка."
        self.save()
        self.assertIsNone(self.find())
        self.assertEqual(self.find(require_confirmed=False)["attributes"]["__vlm_description"], "Новая ручная поправка.")

    def test_project_photo_copy_with_identical_bytes_keeps_interval_memory(self):
        self.confirmed()
        self.save()
        copied = self.root / "project-copy.jpg"
        copied.write_bytes(self.photo.read_bytes())
        self.record.path = str(copied)
        self.assertEqual(self.find()["facies"], "Русловая")

    def test_photo_chat_exact_correction_is_scoped_to_well_and_project(self):
        memory.save_correction("Опиши", "Старое", "Новое", operation="photo_chat", image_path=str(self.photo), project="p", well="Р-31")
        call = dict(operation="photo_chat", image_path=str(self.photo), project="p", well="Р-31")
        self.assertEqual(memory.exact_correction("Опиши", **call), "Новое")
        self.assertIsNone(memory.exact_correction("Опиши", **{**call, "well": "Р-32"}))
        self.assertIsNone(memory.exact_correction("Опиши", **{**call, "project": "q"}))
        self.assertIsNone(memory.exact_correction("Опиши", **{**call, "project": "", "well": ""}))
        with self.assertRaises(ValueError):
            memory.save_correction("Опиши", "Старое", "Новое", operation="photo_chat", image_path=str(self.photo), general=True)

    def test_explicit_global_text_correction_remains_available_across_projects(self):
        memory.save_correction("Вопрос", "Старое", "Правильное общее правило", general=True)
        self.assertEqual(memory.exact_correction("Вопрос", project="another-project", well="another-well"),
                         "Правильное общее правило")

    def test_project_reassignment_keeps_unapproved_and_malformed_lines(self):
        self.confirmed()
        memory.save_interval_edit(self.record, self.layer, project="unsaved:1", well="Р-31")
        root = memory._root()
        path = root / "corrections.jsonl"
        path.write_text('not-json\n' + json.dumps({"scope": "project", "project": "unsaved:1", "approved": False}) + '\n', encoding="utf-8")
        memory.reassign_project("unsaved:1", "saved-project")
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "not-json")
        self.assertFalse(json.loads(lines[1])["approved"])
        self.assertEqual(json.loads(lines[1])["project"], "saved-project")
        self.assertIsNotNone(memory.exact_interval_edit(self.record, self.layer, project="saved-project", well="Р-31"))


if __name__ == "__main__":
    unittest.main()
