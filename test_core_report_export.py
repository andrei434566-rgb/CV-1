"""Regression test for the 31-column client Excel export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from app.infrastructure.core_report_export import export_core_description_report


class CoreReportExportTests(unittest.TestCase):
    def test_final_excel_preserves_template_and_only_explicit_vlm_description(self):
        wells = [{
            "name": "Р-31",
            "core_interval": (3002.0, 3004.0),
            "excel_report_inputs": {
                "field": "Уренгойское", "well_number": "Р-31", "drilling_number": "1",
                "core_top": 3002.0, "core_base": 3004.0, "stratigraphy": "К2m",
                "gis_shift": -1.0, "penetration": 2.0, "core_recovered": 1.5,
            },
            "layers": [{
                "label": "L",
                "depth_from": 3002.0,
                "depth_to": 3002.5,
                "attributes": {
                    "Название фации": "Лагуна",
                    "Название породы": "Песчаник",
                    "Цвет": "Серый",
                    "Цемент": "Карбонатный",
                    "Индекс фации": "07", "Код фации": "L",
                    "Смещение внутри интервала, м": "−0,25",
                    "Индекс макрофации": "M1", "Название макрофации": "Лагунная",
                    "Код макрофации": "Lg", "Индекс ассоциации фаций": "A1",
                    "Комплекс осадконакопления": "Прибрежный", "Примечания": "=Текст специалиста",
                },
            }],
        }]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.xlsx"
            export_core_description_report(path, "Тест", wells, "xlsx")
            book = load_workbook(path)
            self.addCleanup(book.close)
            sheet = book.active

            self.assertEqual("Краткое описание", sheet["AD1"].value)
            self.assertEqual(31, sheet.max_column)
            self.assertEqual("Уренгойское", sheet["A4"].value)
            self.assertEqual("Р-31", sheet["B4"].value)
            self.assertEqual(-1.0, sheet["G4"].value)
            self.assertEqual((sheet["N4"].value, sheet["O4"].value), (3002.0, 3002.5))
            self.assertEqual(1, sheet["R4"].value)
            self.assertEqual([sheet.cell(4, column).value for column in range(20, 27)],
                             ["07", "Лагуна", "L", "M1", "Лагунная", "Lg", "A1"])
            self.assertEqual(sheet["AC4"].value, "Прибрежный")
            self.assertIsNone(sheet["AD4"].value)
            self.assertEqual((sheet["AE4"].value, sheet["AE4"].data_type), ("=Текст специалиста", "s"))
            self.assertEqual(sheet["H4"].value, -.25)
            self.assertIn("N4+G4+SUM(H4)", sheet["P4"].value)
            self.assertIn("O4+G4+SUM(H4)", sheet["Q4"].value)


if __name__ == "__main__":
    unittest.main()
