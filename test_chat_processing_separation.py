"""Chat admission, source binding and shared-GPU scheduling regressions."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest

from app.ui.windows.main_window import MainWindow
from app.ui.widgets.assistant_panel import AssistantPanel


class ChatProcessingSeparationTests(unittest.TestCase):
    def test_photo_question_does_not_start_project_processing(self):
        sent = []
        window = SimpleNamespace(_chat_photo_path=Path("original.jpg"),
                                 _send_vlm_request=lambda *args, **kwargs: sent.append((args, kwargs)))
        MainWindow._analyze_chat_photo(window, "Опиши цвет")
        self.assertEqual(sent[0][0], ("photo_chat", "Опиши цвет"))
        self.assertEqual(sent[0][1]["image_override"], "original.jpg")

    def make_scheduler(self, *, pending=None):
        sent, batch = [], []
        request = {"python": "local-python", "payload": {"operation": "photo_chat", "question": "Цвет?"},
                   "target": None, "selected_job": None, "photo_path": "old-photo.jpg", "temporary_image": ""}
        window = SimpleNamespace(
            _vlm_dispatch_scheduled=True, _vlm_closing=False, _vlm_pending=None, _analysis_pending=pending,
            _segmentation_thread=None, _training_thread=None, _column_training_thread=None,
            _vlm_chat_queue=deque([request]), _analysis_active=True, _queued_records=[],
            _vlm_history=[], _chat_photo_histories={},
            _vlm_client=SimpleNamespace(send=lambda *args: sent.append(args) or "request-id"),
            _advance_cv1_description=lambda: batch.append(True), _refresh_chat_status=lambda: None,
        )
        return window, sent, batch

    def test_chat_has_priority_after_current_batch(self):
        window, sent, batch = self.make_scheduler()
        MainWindow._drain_vlm_requests(window)
        self.assertEqual(len(sent), 1)
        self.assertEqual(batch, [])
        self.assertEqual(window._vlm_pending_meta["photo_path"], "old-photo.jpg")

    def test_only_one_request_is_submitted_while_gpu_batch_is_active(self):
        window, sent, batch = self.make_scheduler(pending=("batch", [], []))
        MainWindow._drain_vlm_requests(window)
        self.assertFalse(sent)
        self.assertFalse(batch)
        self.assertEqual(len(window._vlm_chat_queue), 1)

    def test_busy_indicator_does_not_disable_composer_actions(self):
        values = []
        button = SimpleNamespace(setEnabled=values.append)
        panel = SimpleNamespace(send_button=button, attach_file_button=button, clear_photo_button=None,
                                status=SimpleNamespace(setText=lambda value: None))
        AssistantPanel.set_busy(panel, True)
        self.assertEqual(values, [True, True])


if __name__ == "__main__":
    unittest.main()
