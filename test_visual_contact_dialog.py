"""Headless Qt regression: visual review must not alter approved masks."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from app.domain.models import FaciesDetection, PhotoRecord
from app.ui.dialogs.visual_contact_review_dialog import VisualContactReviewDialog


_APP = QApplication.instance() or QApplication([])


def _draft(index, top, bottom):
    return FaciesDetection(
        label="Новый контур",
        confidence=0.0,
        polygon=[
            QPointF(5, top), QPointF(20, top), QPointF(35, top),
            QPointF(35, bottom), QPointF(20, bottom), QPointF(5, bottom),
        ],
        attributes={"__structural_interval": f"1:{index}"},
        depth_from=100.0 + top / 100,
        depth_to=100.0 + bottom / 100,
        training_ready=False,
    )


class ContactDialogTests(unittest.TestCase):
    def test_preview_does_not_mutate_existing_facies(self):
        image = QPixmap(100, 200)
        image.fill()
        existing = FaciesDetection(
            "DWCh", 1.0,
            [QPointF(5, 5), QPointF(40, 5), QPointF(40, 90), QPointF(5, 90)],
            training_ready=True,
        )
        record = PhotoRecord("sample", "test.jpg", image, detections=[existing], well_name="W-1")
        draft = [_draft(0, 10, 60), _draft(1, 60, 120)]
        with patch(
            "app.ui.dialogs.visual_contact_review_dialog.create_automatic_interval_detections",
            return_value=(draft, []),
        ):
            dialog = VisualContactReviewDialog([record], lambda rec: (100.0, 102.0))
            dialog._refresh()
            self.assertEqual(1, dialog.table.rowCount())
            self.assertEqual(1, dialog._audit["contact_count"])
            self.assertIs(record.detections[0], existing)
            self.assertTrue(record.detections[0].training_ready)
            self.assertFalse(any(d.training_ready for d in dialog._drafts))
            self.assertIsNotNone(dialog.preview.pixmap())
            dialog.close()


if __name__ == "__main__":
    unittest.main()
