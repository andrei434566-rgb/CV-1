"""Local theory import: persistence, attribution, search and honest limitations."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

from app.infrastructure.theory_chat_import import import_theory_document
from app.infrastructure.local_knowledge_search import search_folders
from app.infrastructure.local_vlm_runner import prepare_geology_reference_chat


class TheoryChatImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_complete_text_is_indexed_persistently_and_searchable_after_source_removal(self):
        source = self.root / "учебник.txt"
        source.write_text("Описание осадков. " * 1000 + "\nПсаммитовая уникальная текстура в этом фрагменте.", encoding="utf-8")
        result = import_theory_document(source, self.root / "library")
        self.assertGreater(result["chunks"], 1)
        self.assertFalse(result["weights_trained"])
        self.assertEqual(result["warnings"], [])
        source.unlink()
        hits = search_folders("псаммитовая текстура", [result["index_folder"]])
        self.assertTrue(any("Псаммитовая уникальная текстура" in hit["excerpt"] for hit in hits))
        self.assertTrue(Path(result["original"]).is_file())

    def test_import_is_deduplicated_by_content_and_preserves_sources(self):
        a = self.root / "a.txt"
        b = self.root / "b.txt"
        a.write_text("Рябь течения в песке.", encoding="utf-8")
        b.write_bytes(a.read_bytes())
        one = import_theory_document(a, self.root / "library")
        two = import_theory_document(b, self.root / "library")
        self.assertEqual(one["sha256"], two["sha256"])
        self.assertEqual(one["original"], two["original"])
        self.assertEqual(len(list(Path(one["index_folder"]).rglob("*.txt"))), 1)

    def test_changed_source_during_extraction_does_not_publish_partial_index(self):
        from app.infrastructure import theory_chat_import as importer
        source = self.root / "moving.txt"
        source.write_text("Первоначальный текст", encoding="utf-8")
        original_extract = importer._extract
        def mutate_after_extract(path):
            value = original_extract(path)
            source.write_text("Другое содержимое", encoding="utf-8")
            return value
        with patch.object(importer, "_extract", side_effect=mutate_after_extract), \
             self.assertRaisesRegex(ValueError, "изменился"):
            import_theory_document(source, self.root / "library")
        self.assertFalse((self.root / "library" / "theory_index").exists())

    def test_incomplete_write_never_publishes_document_chunks(self):
        source = self.root / "two_chunks.txt"
        source.write_text("Текст " * 3000, encoding="utf-8")
        original_write = Path.write_text
        def fail_second_chunk(path, *args, **kwargs):
            if path.name.endswith("_0002.txt"):
                raise OSError("test failure")
            return original_write(path, *args, **kwargs)
        with patch.object(Path, "write_text", fail_second_chunk), self.assertRaises(OSError):
            import_theory_document(source, self.root / "library")
        index = self.root / "library" / "theory_index"
        self.assertFalse(index.exists())

    def test_russian_legacy_encoding(self):
        source = self.root / "old.txt"
        source.write_bytes("Текстура слоистая".encode("cp1251"))
        result = import_theory_document(source, self.root / "library")
        self.assertIn("Текстура слоистая", next(Path(result["index_folder"]).rglob("*.txt")).read_text(encoding="utf-8"))

    def test_docx_paragraphs_are_extracted(self):
        source = self.root / "theory.docx"
        xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
               '<w:body><w:p><w:r><w:t>Слоистость и песчаник</w:t></w:r></w:p></w:body></w:document>')
        with zipfile.ZipFile(source, "w") as doc:
            doc.writestr("word/document.xml", xml)
        result = import_theory_document(source, self.root / "library")
        self.assertIn("Слоистость и песчаник", next(Path(result["index_folder"]).rglob("*.txt")).read_text(encoding="utf-8"))

    def test_blank_and_untrusted_model_outputs_are_not_added_to_knowledge(self):
        for name, content in [("empty.txt", "  \n"), ("baseline_responses.jsonl", '{"response":"wrong"}')]:
            source = self.root / name
            source.write_text(content, encoding="utf-8")
            with self.subTest(name=name), self.assertRaises(ValueError):
                import_theory_document(source, self.root / "library")

    def test_scanned_or_empty_pdf_does_not_claim_success(self):
        import sys
        vendor = str(Path(__file__).resolve().parent / "third_party")
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        from pypdf import PdfWriter
        source = self.root / "scan.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=500)
        with source.open("wb") as output:
            writer.write(output)
        with self.assertRaisesRegex(ValueError, "OCR"):
            import_theory_document(source, self.root / "library")

    def test_found_user_theory_is_not_replaced_with_generic_notes(self):
        source = self.root / "theory.txt"
        source.write_text("Конгломерат содержит окатанные обломки.", encoding="utf-8")
        result = import_theory_document(source, self.root / "library")
        hits = search_folders("Чем конгломерат отличается от брекчии?", [result["index_folder"]])
        self.assertTrue(hits)
        question = "Чем конгломерат отличается от брекчии?"
        effective, preserved, notes = prepare_geology_reference_chat({"operation": "chat", "question": question}, hits)
        self.assertEqual(effective, question)
        self.assertEqual(preserved, hits)
        self.assertEqual(notes, [])

    def test_background_import_reports_success_and_failure_individually(self):
        from PySide6.QtCore import QEventLoop, QTimer
        from PySide6.QtWidgets import QApplication
        from app.infrastructure.theory_chat_import import TheoryImportThread
        app = QApplication.instance() or QApplication([])
        source = self.root / "source.txt"
        source.write_text("Песчаник с косой слоистостью.", encoding="utf-8")
        loop = QEventLoop()
        worker = TheoryImportThread([str(source), str(self.root / "missing.txt")], self.root / "library")
        results = []
        worker.document_result.connect(results.append)
        worker.finished.connect(loop.quit)
        QTimer.singleShot(10000, loop.quit)
        worker.start()
        loop.exec()
        self.assertTrue(worker.wait(10000))
        app.processEvents()
        self.assertEqual([entry["ok"] for entry in results], [True, False])
        self.assertTrue(Path(results[0]["original"]).is_file())

    def test_drop_over_the_chat_input_or_transcript_emits_file(self):
        from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
        from PySide6.QtGui import QDragEnterEvent, QDropEvent
        from PySide6.QtWidgets import QApplication
        from app.ui.widgets.assistant_panel import AssistantPanel
        app = QApplication.instance() or QApplication([])
        panel = AssistantPanel()
        self.addCleanup(panel.close)
        received = []
        panel.files_dropped.connect(received.append)
        file_path = str(self.root / "theory.txt")
        for target in (panel.composer.viewport(), panel.transcript.viewport()):
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(file_path)])
            enter = QDragEnterEvent(QPoint(2, 2), Qt.DropAction.CopyAction, mime,
                                    Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            app.sendEvent(target, enter)
            self.assertTrue(enter.isAccepted())
            drop = QDropEvent(QPointF(2, 2), Qt.DropAction.CopyAction, mime,
                              Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            app.sendEvent(target, drop)
            self.assertTrue(drop.isAccepted())
        self.assertEqual([[Path(value) for value in group] for group in received],
                         [[Path(file_path)], [Path(file_path)]])


if __name__ == "__main__":
    unittest.main()
