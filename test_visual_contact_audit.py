"""No-GPU regression tests for non-destructive candidate contact audits."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from app.infrastructure.visual_contact_audit import audit_visual_contacts, is_uv_photo


@dataclass
class DummyDetection:
    polygon: list
    attributes: dict = field(default_factory=dict)
    depth_from: float | None = None
    depth_to: float | None = None
    training_ready: bool = False


def draft(index, y0, y1, top_depth, bottom_depth, curve=None, col=1):
    if curve is None:
        curve = [(10, y0), (20, y0), (30, y0)]
    polygon = [
        (5, curve[0][1]), *curve, (35, curve[-1][1]),
        (35, y1), (30, y1), (20, y1), (10, y1), (5, y1),
    ]
    return DummyDetection(
        polygon,
        {"__structural_interval": f"{col}:{index}"},
        top_depth, bottom_depth,
    )


class VisualContactAuditTests(unittest.TestCase):
    def test_preserves_curved_contact_and_depth(self):
        first = draft(0, 0, 45, 100.0, 100.45)
        second = draft(1, 45, 100, 100.45, 101.0, curve=[(10, 42), (20, 46), (30, 50)])
        report = audit_visual_contacts([first, second], source_photo="well_100-101.jpg")
        self.assertEqual(2, report["interval_count"])
        self.assertEqual(1, report["contact_count"])
        contact = report["contacts"][0]
        self.assertEqual(100.45, contact["depth_m"])
        self.assertEqual("curved", contact["geometry"])
        self.assertEqual(8, contact["amplitude_px"])
        self.assertFalse(contact["training_approved"])
        self.assertFalse(contact["geologist_reviewed"])
        self.assertFalse(first.training_ready)
        self.assertFalse(second.training_ready)

    def test_no_contact_between_independent_columns(self):
        a = draft(0, 0, 100, None, None, col=1)
        b = draft(0, 0, 100, None, None, col=2)
        report = audit_visual_contacts([a, b])
        self.assertEqual(2, report["column_count"])
        self.assertEqual([], report["contacts"])

    def test_missing_depth_never_fabricates_one(self):
        first = draft(0, 0, 45, None, None)
        second = draft(1, 45, 100, None, None)
        report = audit_visual_contacts([first, second])
        self.assertIsNone(report["contacts"][0]["depth_m"])

    def test_gap_in_interval_sequence_is_not_fabricated_contact(self):
        first = draft(0, 0, 45, 1, 2)
        second = draft(2, 45, 100, 2, 3)
        report = audit_visual_contacts([first, second])
        self.assertEqual([], report["contacts"])

    def test_uv_filters_keep_normal_wells(self):
        for filename in ("R-31_UV_3000.jpg", "УФ_керн.jpg", "ультрафиолетовая_серия.png"):
            self.assertTrue(is_uv_photo(filename), filename)
        self.assertFalse(is_uv_photo("Р-31_3002,00-3004,96.jpg"))


if __name__ == "__main__":
    unittest.main()
