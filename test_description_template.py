"""Descriptions stay tied to explicit observations and preserve expert edits."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QPointF, QSettings
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication, QDialog

from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.domain.models import FaciesDetection, PhotoRecord
from app.infrastructure.description_template import ASSUMPTIONS_KEY, compose_description
from app.infrastructure.project_storage import load_project
from app.ui.dialogs.facies_dialog import FaciesDialog, make_interval_preview


class DescriptionTemplateTests(unittest.TestCase):
    def test_empty_is_unknown_not_absence_and_facies_typicals_are_not_observations(self):
        draft = compose_description({"Предполагаемые литотипы": "Песчаник", "Энергия среды": "High"})
        self.assertEqual(draft.text, "Признаки участка не заполнены.")
        self.assertIn("Название породы", draft.missing)
        self.assertIn("Название фации", draft.missing)
        self.assertNotIn("биотурбация отсутствует", draft.text.lower())

    def test_all_observations_keep_custom_words_and_explicit_uncertainty(self):
        attrs = {field: options[0] for field, options in LITHOLOGY_ATTRIBUTE_OPTIONS.items()}
        attrs.update({"Биотурбация": "Planolites и Chondrites", "Название породы": "Предположительно: Песчаник"})
        draft = compose_description(attrs, 3188.98, 3189.24123, facies_name="Пойменная",
                                    assumptions=["Биотурбация", "Название фации"])
        self.assertIn("3188,98–3189,241 м", draft.text)
        self.assertIn("Порода: предположительно Песчаник.", draft.text)
        self.assertIn("предположительно Planolites и Chondrites", draft.text)
        self.assertIn("Фация: предположительно Пойменная.", draft.text)
        self.assertIn("Крупнозернистая", draft.text)
        self.assertFalse(draft.missing)
        for field in LITHOLOGY_ATTRIBUTE_OPTIONS:
            self.assertNotIn(field, draft.missing)

    def test_missing_markers_and_bad_depths_do_not_create_fake_values(self):
        draft = compose_description({"Название породы": "Литология требует проверки", "Цвет": "Серый",
                                     "Гидродинамический режим": "Не установлен по фото"}, float("nan"), 100)
        self.assertEqual("Цвет: Серый.", draft.text)
        self.assertIn("Название породы", draft.missing)
        self.assertNotIn("Интервал", draft.text)


class DescriptionCardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = QSettings(str(self.root / "settings.ini"), QSettings.Format.IniFormat)
        patcher = patch("app.ui.dialogs.facies_dialog.QSettings", return_value=self.settings)
        patcher.start(); self.addCleanup(patcher.stop)

    def card(self, *args, **kwargs):
        dialog = FaciesDialog(*args, **kwargs)
        self.addCleanup(dialog.close)
        return dialog

    def choose(self, dialog, field, value):
        combo = dialog.attribute_combos[field]
        combo.setCurrentIndex(combo.findData(value))

    def test_live_description_and_clearing_a_property_updates_both_storage_aliases(self):
        dialog = self.card("Интервал YOLO", .84, {"__geometry_only": "1"}, 100, 101)
        self.choose(dialog, "Название породы", "Песчаник")
        self.choose(dialog, "Цвет", "Серый")
        self.choose(dialog, "Зернистость", "Мелкозернистый")
        dialog.assumption_checks["Зернистость"].setChecked(True)
        self.assertIn("Зернистость: предположительно Мелкозернистая.", dialog.vlm_description.toPlainText())
        self.assertIn("Название фации", dialog.missing_fields.text())
        attrs = dialog.selected_attributes()
        self.assertEqual(attrs["Краткое описание"], attrs["__vlm_description"])
        reopened = self.card("Не определена", .84, attrs, 100, 101)
        reopened.attribute_combos["Цвет"].setCurrentIndex(0)
        cleared = reopened.selected_attributes()
        self.assertNotIn("Цвет", cleared)
        self.assertNotIn("Серый", cleared["__vlm_description"])
        self.assertEqual(cleared["Название фации"], "Не определена")

    def test_manual_text_survives_template_toggle_reopen_and_property_edits(self):
        text = "Описание специалиста: Planolites, нижняя граница неровная."
        dialog = self.card("", .5, {"Краткое описание": text, "__geometry_only": "1"}, description_confirmed=True)
        self.assertFalse(dialog.template_mode.isChecked())
        self.choose(dialog, "Цвет", "Серый")
        self.assertEqual(text, dialog.vlm_description.toPlainText())
        self.assertFalse(dialog.description_confirmed.isChecked())
        dialog.template_mode.setChecked(True)
        self.assertNotEqual(text, dialog.vlm_description.toPlainText())
        attrs = dialog.selected_attributes()
        reopened = self.card("", .5, attrs)
        reopened.template_mode.setChecked(False)
        self.assertEqual(text, reopened.vlm_description.toPlainText())

    def test_facies_selection_shows_reference_without_inventing_rock_bedding_or_regime(self):
        dialog = self.card("", .5)
        dialog.facies.setCurrentIndex(1)
        self.assertIn("Справочная характеристика", dialog.facies_reference.text())
        attrs = dialog.selected_attributes()
        self.assertFalse(attrs.get("Название породы"))
        self.assertFalse(attrs.get("Слоистость"))
        self.assertFalse(attrs.get("Гидродинамический режим"))
        self.assertIn("Фация:", attrs["Краткое описание"])
        self.assertNotIn("High", attrs["Краткое описание"])
        dialog.training_confirmed.setChecked(True)
        dialog.facies_assumed.setChecked(True)
        self.assertFalse(dialog.training_confirmed.isChecked())

    def test_measured_color_can_be_confirmed_and_stays_confirmed_on_reopen(self):
        dialog = self.card("", .5, {"Цвет": "Серый", "__vision_color": "Серый"})
        self.assertTrue(dialog.assumption_checks["Цвет"].isChecked())
        dialog.assumption_checks["Цвет"].setChecked(False)
        attrs = dialog.selected_attributes()
        self.assertEqual(json.loads(attrs[ASSUMPTIONS_KEY]), [])
        reopened = self.card("", .5, attrs)
        self.assertFalse(reopened.assumption_checks["Цвет"].isChecked())

    def test_invalid_depth_keeps_card_open_and_valid_empty_facies_can_be_saved(self):
        dialog = self.card("", .5)
        dialog.depth_from.setText("nan"); dialog.depth_to.setText("101")
        with patch("app.ui.dialogs.facies_dialog.QMessageBox.warning") as warning:
            dialog.accept()
        warning.assert_called_once()
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
        dialog.depth_from.clear(); dialog.depth_to.clear()
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)

    def test_preview_hides_pixels_outside_the_selected_polygon(self):
        photo = QPixmap(100, 100); photo.fill(QColor("red"))
        preview = make_interval_preview(photo, [QPointF(10, 10), QPointF(90, 10), QPointF(10, 90)])
        self.assertFalse(preview.isNull())
        image = preview.toImage()
        self.assertEqual(image.pixelColor(5, 5), QColor("red"))
        self.assertEqual(image.pixelColor(70, 70), QColor("white"))

    def test_saved_template_without_facies_round_trips_and_reaches_excel(self):
        from app.infrastructure.core_report_export import export_core_description_report
        from app.ui.windows.main_window import MainWindow
        from openpyxl import load_workbook
        with patch("app.ui.windows.main_window.QSettings", return_value=self.settings), \
             patch("app.infrastructure.expert_corrections._root", return_value=self.root / "memory"):
            window = MainWindow()
            self.addCleanup(window.close)
            photo = QPixmap(100, 200); photo.fill(QColor("gray"))
            path = self.root / "core.png"; photo.save(str(path))
            detection = FaciesDetection("Интервал YOLO", .8,
                [QPointF(10, 10), QPointF(80, 10), QPointF(80, 180)], {"__geometry_only": "1"})
            record = PhotoRecord("new", str(path), photo, [detection], well_name="Проверка", analysis_mode="yolo_vlm")
            window._records = [record]; window._active_well_name = record.well_name
            window._project_folder = self.root / "project"
            window._refresh_facies_views = Mock(side_effect=lambda _record: window._schedule_edit_autosave())
            dialog = self.card("Интервал YOLO", .8, detection.attributes, 100, 101)
            self.choose(dialog, "Название породы", "Песчаник")
            self.choose(dialog, "Цвет", "Серый")
            dialog.assumption_checks["Название породы"].setChecked(True)
            polygon_before = list(detection.polygon)
            window._save_lithology(record, detection, dialog)
            self.assertEqual(detection.label, "Не определена")
            self.assertEqual(detection.polygon, polygon_before)
            self.assertFalse(detection.training_ready)
            self.assertNotIn("__expert_facies_mapping", detection.attributes)
            self.assertTrue(window._flush_edit_autosave())
            _, records, _, missing, _ = load_project(window._project_folder)
            self.assertFalse(missing)
            attrs = records[0].detections[0].attributes
            self.assertIn("предположительно Песчаник", attrs["Краткое описание"])
            self.assertEqual(attrs["__description_mode"], "template")
            target = self.root / "report.xlsx"
            with patch.object(window, "_show_stack"):
                wells = window._core_description_report_wells()
            export_core_description_report(target, "Проверка", wells, "xlsx")
            workbook = load_workbook(target)
            try:
                self.assertEqual(workbook.active["V4"].value, attrs["Краткое описание"])
                rules = [formula for group in workbook.active.conditional_formatting._cf_rules.values()
                         for rule in group for formula in rule.formula]
                self.assertTrue(any('SEARCH("предполож"' in formula for formula in rules))
            finally:
                workbook.close()

    def test_assumption_change_invalidates_a_confirmation_even_in_manual_text_mode(self):
        from app.infrastructure.vlm_training_data import expert_description_digest, expert_description_is_confirmed
        detection = FaciesDetection("Русловая", .8, [QPointF(0, 0), QPointF(10, 0), QPointF(10, 10)],
            {"__geometry_only": "1", "__vlm_description": "Серый песчаник.",
             "__description_assumptions": "[]", "__expert_description_confirmed": "1"},
            training_ready=True)
        detection.attributes["__expert_description_sha256"] = expert_description_digest(detection)
        self.assertTrue(expert_description_is_confirmed(detection))
        detection.attributes[ASSUMPTIONS_KEY] = json.dumps(["Название фации"], ensure_ascii=False)
        self.assertFalse(expert_description_is_confirmed(detection))


if __name__ == "__main__":
    unittest.main()
