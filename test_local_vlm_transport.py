"""Regression check for UTF-8 questions and Windows paths over QProcess."""

import sys
import tempfile
import unittest
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from app.infrastructure.local_vlm_client import LocalVlmClient


class LocalVlmTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QCoreApplication.instance() or QCoreApplication([])

    def test_russian_search_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "Теория керна"
            folder.mkdir()
            (folder / "песчаник.txt").write_text("Песчаник содержит обломочные зёрна.", encoding="utf-8")
            client = LocalVlmClient()
            loop = QEventLoop()
            results = []
            client.reply.connect(lambda result: (results.append(result), loop.quit()))
            QTimer.singleShot(15000, loop.quit)
            client.send(sys.executable, {
                "operation": "search", "question": "песчаник", "folders": [str(folder)],
                "image": "", "base": "", "adapter": "", "theory_adapter": "",
            })
            loop.exec()
            client.stop()
            self.assertEqual(len(results), 1, "Локальный процесс не ответил за 15 секунд")
            self.assertTrue(results[0]["ok"], results[0].get("error"))
            self.assertEqual(len(results[0]["hits"]), 1)
            self.assertIn("песчаник", results[0]["hits"][0]["path"])


if __name__ == "__main__":
    unittest.main()
