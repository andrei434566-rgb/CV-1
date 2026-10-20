"""CV-1 review station for photo/depth/Excel candidate links.

Run after index_geology_archive.py:
    python review_geology_pairs.py "D:\\Archive\\_index_cv1\\archive_index.json"
Requires PySide6. Never changes source photos or spreadsheets.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSplitter, QTextEdit, QVBoxLayout, QWidget)


class ReviewWindow(QMainWindow):
    def __init__(self, index_file: Path):
        super().__init__()
        self.index_file = index_file.resolve()
        self.data = json.loads(self.index_file.read_text(encoding="utf-8"))
        self.pairs = self.data.get("pairs", [])
        self.review_file = self.index_file.with_name("pair_decisions.json")
        self.decisions = self._load_decisions()
        self.setWindowTitle("CV-1 — Проверка соответствия фото и Excel")
        self.resize(1300, 780)

        self.items = QListWidget()
        self.items.currentRowChanged.connect(self.show_pair)
        self.photo = QLabel("Выберите фотографию")
        self.original_pixmap = QPixmap()
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo.setMinimumSize(320, 260)
        self.photo.setStyleSheet("background: #171b25; color: #dde5f0;")
        self.photo_scroll = QScrollArea()
        self.photo_scroll.setWidgetResizable(True)
        self.photo_scroll.setWidget(self.photo)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.candidates = QComboBox()
        self.candidates.currentIndexChanged.connect(self.show_candidate)
        self.candidate_details = QTextEdit()
        self.candidate_details.setReadOnly(True)
        self.candidate_details.setMinimumHeight(180)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Фотографии по скважинам"))
        left_layout.addWidget(self.items)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Фото керна"))
        right_layout.addWidget(self.photo_scroll, 5)
        right_layout.addWidget(QLabel("Интервал фотографии / статус"))
        right_layout.addWidget(self.details, 1)
        right_layout.addWidget(QLabel("Кандидат из Excel (выбери соответствующий интервал)"))
        right_layout.addWidget(self.candidates)
        right_layout.addWidget(self.candidate_details, 2)
        buttons = QHBoxLayout()
        for title, callback in (
            ("Подтвердить", self.confirm),
            ("Исправить / выбрать другой", self.correct),
            ("Отклонить", self.reject),
            ("Пропустить", self.skip),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        right_layout.addLayout(buttons)
        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 4)
        self.setCentralWidget(splitter)
        for i, pair in enumerate(self.pairs):
            self.items.addItem(self._label(i, pair))
        if self.pairs:
            self.items.setCurrentRow(0)

    def _load_decisions(self):
        if not self.review_file.exists():
            return {}
        try:
            data = json.loads(self.review_file.read_text(encoding="utf-8"))
            return data.get("decisions", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _label(self, i, pair):
        name = Path(pair.get("photo", "")).name
        status = self.decisions.get(pair.get("photo", ""), {}).get("decision", "не проверено")
        return f"{i+1}. {pair.get('well', '?')} / {name} [{status}]"

    def show_pair(self, index):
        self.candidates.blockSignals(True)
        self.candidates.clear()
        if index < 0 or index >= len(self.pairs):
            self.candidates.blockSignals(False)
            return
        pair = self.pairs[index]
        path = Path(pair["photo"])
        pixmap = QPixmap(str(path))
        self.original_pixmap = pixmap
        if pixmap.isNull():
            self.photo.setText("Не удалось открыть фотографию:\n" + str(path))
        else:
            self.render_photo()
        self.details.setPlainText(
            f"Скважина: {pair.get('well')}\nФото: {path}\n"
            f"Глубина: {pair.get('photo_depth', 'не определена')}\n"
            f"Автоматический статус: {pair.get('status')}\n"
            "Совпадение глубин не означает подтверждённую геологическую разметку."
        )
        for j, match in enumerate(pair.get("matches", [])):
            label = (f"{j+1}. {match.get('top')}–{match.get('bottom')} м "
                     f"| {Path(match.get('excel','')).name} | строка {match.get('row')}")
            self.candidates.addItem(label, j)
        previous = self.decisions.get(pair["photo"], {})
        selected = previous.get("candidate_index")
        if isinstance(selected, int) and 0 <= selected < self.candidates.count():
            self.candidates.setCurrentIndex(selected)
        self.candidates.blockSignals(False)
        self.show_candidate()

    def show_candidate(self, *_):
        i = self.items.currentRow()
        if i < 0 or i >= len(self.pairs):
            return
        matches = self.pairs[i].get("matches", [])
        j = self.candidates.currentIndex()
        if not (0 <= j < len(matches)):
            self.candidate_details.setPlainText(
                "Нет автоматических кандидатов. Проверьте имя фото и глубины в Excel.")
            return
        m = matches[j]
        self.render_photo()
        self.candidate_details.setPlainText(
            f"Excel: {m.get('excel')}\nЛист: {m.get('sheet')}\n"
            f"Строка: {m.get('row')}\nПерекрытие: {m.get('overlap_fraction')}\n\n"
            f"Описание:\n{m.get('description','')}\n\n"
            f"Индекс: {m.get('facies_index','')} | Код: {m.get('facies_code','')}\n"
            f"Толщина: {m.get('thickness','')} м\n"
            f"Кровля: {m.get('top')} | Подошва: {m.get('bottom')}\n"
            f"Фация: {m.get('facies','')}\nЛитология: {m.get('lithology','')}"
        )

    def render_photo(self):
        if self.original_pixmap.isNull():
            return
        canvas = self.original_pixmap.scaled(
            850, 620, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        i = self.items.currentRow()
        if not (0 <= i < len(self.pairs)):
            self.photo.setPixmap(canvas)
            return
        pair = self.pairs[i]
        depth = pair.get("photo_depth")
        if not depth or depth[1] <= depth[0]:
            self.photo.setPixmap(canvas)
            return
        painter = QPainter(canvas)
        for k, match in enumerate(pair.get("matches", [])):
            top = max(depth[0], float(match["top"]))
            bottom = min(depth[1], float(match["bottom"]))
            if bottom <= top:
                continue
            y1 = round((top-depth[0])/(depth[1]-depth[0])*canvas.height())
            y2 = round((bottom-depth[0])/(depth[1]-depth[0])*canvas.height())
            selected = k == self.candidates.currentIndex()
            color = QColor(255, 194, 57) if selected else QColor(60, 190, 245)
            painter.fillRect(0, y1, canvas.width(), max(2,y2-y1),
                             QColor(color.red(),color.green(),color.blue(),75 if selected else 28))
            painter.setPen(QPen(color, 3 if selected else 1))
            painter.drawRect(1, y1, canvas.width()-3, max(2,y2-y1))
            painter.drawText(8, max(18,y1+16), str(match.get("facies_index") or match.get("facies") or "фация")[:30])
        painter.fillRect(0,0,canvas.width(),25,QColor(25,25,25,220))
        painter.setPen(QColor("white"))
        painter.drawText(8,18,"ПРЕДВАРИТЕЛЬНАЯ СХЕМА ПО ГЛУБИНАМ — НЕ МАСКА КЕРНА")
        painter.end()
        self.photo.setPixmap(canvas)

    def _save(self, decision, candidate_index=None):
        i = self.items.currentRow()
        if not (0 <= i < len(self.pairs)):
            return
        pair = self.pairs[i]
        self.decisions[pair["photo"]] = {
            "decision": decision,
            "candidate_index": candidate_index,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "well": pair.get("well"),
        }
        payload = {"source_index": str(self.index_file), "decisions": self.decisions}
        temporary = self.review_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.review_file)
        self.items.item(i).setText(self._label(i, pair))
        if i + 1 < len(self.pairs):
            self.items.setCurrentRow(i + 1)

    def confirm(self):
        j = self.candidates.currentIndex()
        if j < 0:
            QMessageBox.information(self, "Нет кандидата", "Нет выбранного интервала Excel.")
            return
        self._save("confirmed", j)

    def correct(self):
        if self.candidates.count() == 0:
            QMessageBox.information(self, "Нет кандидатов", "Сначала уточните глубины в исходном индексе.")
            return
        self._save("corrected_candidate", self.candidates.currentIndex())

    def reject(self):
        self._save("rejected")

    def skip(self):
        self._save("skipped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", nargs="?", type=Path)
    args = parser.parse_args()
    app = QApplication([])
    index = args.index
    if index is None:
        filename, _ = QFileDialog.getOpenFileName(None, "Открыть archive_index.json", "",
                                                   "JSON (*.json)")
        if not filename:
            return
        index = Path(filename)
    window = ReviewWindow(index)
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
