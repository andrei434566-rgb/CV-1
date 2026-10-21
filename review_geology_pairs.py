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
from index_geology_archive import uv_filename, likely_uv_visual

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
        excluded = set(self.data.get("excluded_uv", [])) | set(self.data.get("suspected_uv", []))
        self.pairs = [
            p for p in self.data.get("pairs", [])
            if p.get("photo") and p["photo"] not in excluded
            and not uv_filename(Path(p["photo"]))
            and not likely_uv_visual(Path(p["photo"]))
        ]
        self.hidden_uv_count = len(self.data.get("pairs", [])) - len(self.pairs)
        self.review_file = self.index_file.with_name("pair_decisions.json")
        self.decisions = self._load_decisions()
        self.setWindowTitle("CV-1 — Проверка соответствия фото и Excel")
        self.resize(1300, 780)

        self.items = QListWidget()
        self.items.currentRowChanged.connect(self.show_pair)
        self.photo = QLabel("Выберите фотографию")
        self.original_pixmap = QPixmap()
        self.core_boxes = []
        self.recognizer = None
        self.photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.photo.setMinimumSize(320, 260)
        self.photo.setStyleSheet("background: #171b25; color: #dde5f0;")
        self.photo_scroll = QScrollArea()
        self.photo_scroll.setWidgetResizable(True)
        self.photo_scroll.setWidget(self.photo)
        self.detect_button = QPushButton("Найти столбики — Kern Analyzer")
        self.detect_button.clicked.connect(lambda: self.detect_columns(silent=False))
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
        left_layout.addWidget(QLabel(f"UV-фотографий скрыто: {self.hidden_uv_count}"))
        left_layout.addWidget(self.items)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Фото керна"))
        right_layout.addWidget(self.photo_scroll, 5)
        right_layout.addWidget(self.detect_button)
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
        self.core_boxes = []
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
        span = pair.get("photo_depth")
        all_rows = self.data.get("descriptions", [])
        overlapping = []
        if span and len(span) == 2:
            for row in all_rows:
                if row.get("well") != pair.get("well"): continue
                try:
                    top, bottom = float(row["top"]), float(row["bottom"])
                    if min(float(span[1]), bottom) > max(float(span[0]), top):
                        overlapping.append(row)
                except (KeyError, ValueError, TypeError):
                    pass
        sublayers = [r for r in overlapping
                     if float(r["bottom"]) - float(r["top"]) < float(span[1]) - float(span[0]) - 0.01] if span else []
        self.details.append(
            f"\nДИАГНОСТИКА ФАЦИЙ: в Excel пересекаются с фото {len(overlapping)} строк; "
            f"отдельных интервалов {len(sublayers)}; "
            f"прошли фильтр сопоставления {len(pair.get('matches', []))}."
        )
        for row in overlapping[:12]:
            self.details.append(
                f"  Excel строка {row.get('row')}: {row.get('top')}–{row.get('bottom')} м "
                f"| индекс={row.get('facies_index') or 'ПУСТО'} "
                f"| мощность={row.get('thickness')} "
                f"| {str(row.get('facies') or '')[:45]}"
            )
        if len(overlapping) > 12:
            self.details.append(f"  ...ещё {len(overlapping)-12} строк")
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
        if not pixmap.isNull():
            self.detect_columns(silent=True)

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

    def detect_columns(self, silent=False):
        i = self.items.currentRow()
        if not (0 <= i < len(self.pairs)):
            return
        try:
            from app.infrastructure.ml.core_column_service import CoreColumnRecognizer, read_bgr_image
            image = read_bgr_image(self.pairs[i]["photo"])
            if self.recognizer is None:
                self.recognizer = CoreColumnRecognizer()
            self.core_boxes = self.recognizer.recognize(image, orientation="auto")
            self.render_photo()
            self.details.append(f"\nKern Analyzer: найдено {len(self.core_boxes)} столбиков/рядов. Проверьте контуры.")
        except Exception as exc:
            self.details.append(f"\nДетектор: {exc}")
            if not silent:
                QMessageBox.warning(self, "Детектор керна", str(exc))

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
        painter = QPainter(canvas)
        sx = canvas.width() / max(1,self.original_pixmap.width())
        sy = canvas.height() / max(1,self.original_pixmap.height())
        painter.setPen(QPen(QColor(90,245,140),2))
        for n,box in enumerate(self.core_boxes,1):
            x=round(box["left"]*sx)
            y=round(box["top"]*sy)
            w=round((box["right"]-box["left"])*sx)
            h=round((box["bottom"]-box["top"])*sy)
            painter.drawRect(x,y,w,h)
            painter.drawText(x+4,y+18,f"Керн {n}")
        # Provisional mapping: lane lengths define relative photo depth.
        # Never treat these bands as validated segmentation labels.
        span = pair.get("photo_depth")
        matches = pair.get("matches", [])
        lanes = sorted(self.core_boxes, key=lambda b:(b["left"],b["top"]))
        if span and len(span)==2 and span[1]>span[0] and lanes:
            widths=[max(1,b["bottom"]-b["top"]) for b in lanes]
            full=sum(widths)
            cursor=float(span[0])
            displayed=0
            for box,length in zip(lanes,widths):
                lane_end=cursor+(float(span[1])-float(span[0]))*length/full
                for m in matches:
                    try:
                        top,bottom=float(m["top"]),float(m["bottom"])
                    except (TypeError,ValueError,KeyError):
                        continue
                    if bottom-top >= float(span[1])-float(span[0])-0.01:
                        continue
                    lo,hi=max(cursor,top),min(lane_end,bottom)
                    if hi<=lo: continue
                    y1=round((box["top"]+(lo-cursor)/(lane_end-cursor)*(box["bottom"]-box["top"]))*sy)
                    y2=round((box["top"]+(hi-cursor)/(lane_end-cursor)*(box["bottom"]-box["top"]))*sy)
                    x1=round(box["left"]*sx)
                    x2=round(box["right"]*sx)
                    painter.fillRect(x1,y1,max(2,x2-x1),max(2,y2-y1),QColor(255,185,40,85))
                    painter.setPen(QPen(QColor(255,190,40),2))
                    painter.drawRect(x1,y1,max(2,x2-x1),max(2,y2-y1))
                    painter.drawText(x1+2,y1+14,str(m.get("facies_index") or m.get("facies") or "фация")[:14])
                    displayed+=1
                cursor=lane_end
            if displayed==0:
                self.details.append("Нет отдельных фаций для подсветки: проверьте столбцы Excel и повторите сканирование.")
        painter.fillRect(0,0,canvas.width(),25,QColor(25,25,25,220))
        painter.setPen(QColor("white"))
        painter.drawText(8,18,"ЗЕЛЁНЫЙ: КЕРН | ЖЁЛТЫЙ: ПРЕДВАРИТЕЛЬНЫЕ ФАЦИИ")
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
