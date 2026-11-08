"""Render the interval card with private settings and illustrative field values."""
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6 import QtCore


def main():
    native = QtCore.QSettings
    original = native(native.NativeFormat, native.UserScope, "Kern Analyzer", "KernAnalyzer")
    before = {key: original.value(key) for key in original.allKeys()}
    output = ROOT / "research_reports/vision_20261104"
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="description_card_") as folder:
        class Settings(native):
            def __init__(self, *args, **kwargs):
                if args == ("Kern Analyzer", "KernAnalyzer"):
                    super().__init__(str(Path(folder) / "settings.ini"), native.IniFormat)
                else:
                    super().__init__(*args, **kwargs)
        QtCore.QSettings = Settings
        from PySide6.QtGui import QFont, QFontDatabase, QPixmap
        from PySide6.QtWidgets import QApplication, QScrollArea
        from app.ui.windows.main_window import MainWindow
        from app.ui.dialogs.facies_dialog import FaciesDialog
        app = QApplication.instance() or QApplication([])
        # The offscreen Qt plugin does not enumerate Windows system fonts.
        font = Path("C:/Windows/Fonts/segoeui.ttf")
        if font.is_file():
            font_id = QFontDatabase.addApplicationFont(str(font))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                app.setFont(QFont(families[0], 10))
        parent = MainWindow()
        photo = QPixmap(str(ROOT / "models/vision/pipeline_worker_benchmark_20261104/tape_01.jpg"))
        if not photo.isNull():
            photo = photo.copy(0, 0, photo.width(), min(photo.height(), photo.width() * 3))
        dialog = FaciesDialog("Интервал YOLO", .84, {"__geometry_only": "1"},
                              3188.98, 3189.24, parent, preview_pixmap=photo)
        dialog.setWindowTitle("Карточка участка · демонстрационные признаки")
        for field, value in {"Название породы": "Песчаник", "Цвет": "Серый",
                             "Зернистость": "Мелкозернистый", "Слоистость": "Волнистая"}.items():
            combo = dialog.attribute_combos[field]
            combo.setCurrentIndex(combo.findData(value))
        dialog.assumption_checks["Зернистость"].setChecked(True)
        dialog.show(); app.processEvents()
        scroll = dialog.findChild(QScrollArea)
        assert scroll.horizontalScrollBar().maximum() == 0, "Observation controls must fit the card"
        assert dialog.grab().save(str(output / "description_card.png"))
        combo=dialog.attribute_combos['Текстура'];combo.setCurrentIndex(combo.findData('Массивная'))
        dialog.preview_tabs.setCurrentIndex(1)
        scroll.ensureWidgetVisible(dialog._evidence_buttons['Слоистость'])
        app.processEvents()
        assert 'разным прослоям' in dialog.review_notes.toPlainText()
        assert dialog.grab().save(str(output/'description_checks.png'))
        from app.ui.dialogs.feature_evidence_dialog import FeatureEvidenceDialog
        from PySide6.QtCore import QRectF
        evidence=FeatureEvidenceDialog(photo,'Слоистость','Волнистая',parent=dialog)
        evidence.show();evidence.view.rectangle.setRect(QRectF(0,0,photo.width(),photo.height()/2));app.processEvents()
        assert evidence.grab().save(str(output/'feature_evidence.png'))
        evidence.close()
        result = {"gui": "passed", "example_fields_are_illustrative_not_model_predictions": True,
                  "live_preview": dialog.vlm_description.toPlainText(),
                  "review_tab":True,"feature_evidence_zoom":True,
                  "settings_unchanged": {key: original.value(key) for key in original.allKeys()} == before}
        assert result["settings_unchanged"]
        (output / "description_card_smoke.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        dialog.close(); parent.close(); app.processEvents(); app.quit()
        QtCore.QSettings = native


if __name__ == "__main__":
    main()
