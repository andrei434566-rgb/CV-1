"""CV-1 review station for photo/depth/Excel candidate links.

Run after index_geology_archive.py:
    python review_geology_pairs.py "D:\\Archive\\_index_cv1\\archive_index.json"
Requires PySide6. Never changes source photos or spreadsheets.
"""
from __future__ import annotations
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from index_geology_archive import uv_filename, likely_uv_visual

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QPainter, QColor, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea,
    QSplitter, QTextEdit, QVBoxLayout, QWidget)


class ReviewWindow(QMainWindow):
    def __init__(self, index_file: Path):
        super().__init__()
        self.index_file = index_file.resolve()
        self.data = json.loads(self.index_file.read_text(encoding="utf-8"))
        excluded = set(self.data.get("excluded_uv", []))
        self.pairs = [
            p for p in self.data.get("pairs", [])
            if p.get("photo") and p["photo"] not in excluded
            and not uv_filename(Path(p["photo"]))
        ]
        self.hidden_uv_count = len(self.data.get("pairs", [])) - len(self.pairs)
        self.suspected_uv = set(self.data.get("suspected_uv", []))
        self.review_file = self.index_file.with_name("pair_decisions.json")
        self.decisions = self._load_decisions()
        self.box_depth_file = self.index_file.with_name('verified_box_depths.json')
        try:
            self.box_depths = json.loads(self.box_depth_file.read_text(encoding='utf-8')).get('photos', {})
        except (OSError, ValueError):
            self.box_depths = {}
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
        depths = QHBoxLayout()
        depths.addWidget(QLabel('Паспорт ящика: от'))
        self.box_top = QLineEdit()
        self.box_top.setMaximumWidth(110)
        depths.addWidget(self.box_top)
        depths.addWidget(QLabel('до'))
        self.box_bottom = QLineEdit()
        self.box_bottom.setMaximumWidth(110)
        depths.addWidget(self.box_bottom)
        save_depth = QPushButton('Подтвердить глубины ящика')
        save_depth.clicked.connect(self.save_box_depth)
        depths.addWidget(save_depth)
        right_layout.addLayout(depths)
        ocr_button = QPushButton("Распознать паспорт (OCR) — проверить результат")
        ocr_button.clicked.connect(self.read_box_label)
        right_layout.addWidget(ocr_button)
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

    def _box_span(self, pair):
        entry = self.box_depths.get(pair['photo'])
        if entry and entry.get('confirmed'):
            return [float(entry['top']), float(entry['bottom'])]
        # Filename intervals are drilling metadata or unverified candidates.
        # Only a manually confirmed printed box label authorizes depth mapping.
        return None

    def _box_matches(self, pair):
        span = self._box_span(pair)
        if not span:
            return []
        rows = []
        for d in self.data.get('descriptions', []):
            if d.get('well') != pair.get('well'):
                continue
            try:
                a, b = float(d['top']), float(d['bottom'])
                if a >= b or min(span[1], b) <= max(span[0], a):
                    continue
                if b-a >= span[1]-span[0]-0.01:
                    continue
                if not (d.get('facies_index') or d.get('facies')):
                    continue
                thickness = d.get('thickness')
                if thickness is not None and abs(float(thickness)-(b-a)) > max(0.05,(b-a)*0.05):
                    continue
                rows.append(d)
            except (TypeError, ValueError, KeyError):
                continue
        result = []
        for d in sorted(rows, key=lambda x:float(x['bottom'])-float(x['top'])):
            a,b=float(d['top']),float(d['bottom'])
            if not any(min(b,float(v['bottom']))-max(a,float(v['top']))>0.02 for v in result):
                result.append(d)
        return sorted(result, key=lambda x:float(x['top']))

    def read_box_label(self):
        """Offer OCR suggestions only; never confirm or export them automatically."""
        i = self.items.currentRow()
        if not (0 <= i < len(self.pairs)):
            return
        try:
            import cv2
            import pytesseract
            from app.infrastructure.ml.core_column_service import read_bgr_image
            image = read_bgr_image(self.pairs[i]["photo"])
            if image is None or image.size == 0:
                raise ValueError("Не удалось прочитать изображение.")
            height, width = image.shape[:2]
            # Printed placards are usually above the tray or beside its first
            # lane. Search both the original orientation and a rotated copy.
            regions = [
                image[:max(1, int(height * 0.28)), :],
                image[:, :max(1, int(width * 0.27))],
            ]
            found = []
            for region in regions:
                for candidate in (region, cv2.rotate(region, cv2.ROTATE_90_CLOCKWISE)):
                    gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.GaussianBlur(gray, (3, 3), 0)
                    for psm in (6, 11):
                        raw = pytesseract.image_to_string(
                            gray, config=f"--psm {psm}")
                        for match in re.finditer(
                            r"(?<!\d)(\d{2,5}[.,]\d{1,3})\s*[-–—]\s*(\d{2,5}[.,]\d{1,3})(?!\d)",
                            raw,
                        ):
                            a, b = (float(v.replace(",", ".")) for v in match.groups())
                            if a < b and b - a <= 5.0:
                                found.append((a, b))
            suggestions = sorted(set(found))
            if suggestions:
                self.box_top.setText(str(suggestions[0][0]))
                self.box_bottom.setText(str(suggestions[0][1]))
                self.details.append(
                    "\nOCR: предварительные интервалы " + str(suggestions[:8])
                    + ". Сверьте цифры с паспортом и нажмите «Подтвердить» вручную."
                )
            else:
                self.details.append(
                    "\nOCR не нашёл надёжно читаемую пару глубин. "
                    "Введите глубины с паспорта вручную."
                )
        except ImportError:
            QMessageBox.warning(self, "OCR", "Установите pytesseract и Tesseract OCR. "
                                "Без них можно ввести глубины вручную.")
        except Exception as exc:
            QMessageBox.warning(self, "OCR", "Не удалось прочитать паспорт: " + str(exc)
                                + "\\nГлубины можно ввести вручную.")

    def save_box_depth(self):
        i=self.items.currentRow()
        if i < 0 or i >= len(self.pairs):
            return
        pair=self.pairs[i]
        try:
            a=float(self.box_top.text().replace(',','.').strip())
            b=float(self.box_bottom.text().replace(',','.').strip())
        except ValueError:
            QMessageBox.warning(self,'Глубины','Введите две числовые глубины с таблички.')
            return
        drilling=pair.get('photo_depth') if pair.get('depth_source') != 'filename_short_interval' else None
        if not a < b or b-a > 5.0:
            QMessageBox.warning(self,'Глубины','Неверный интервал ящика.')
            return
        if drilling and (a < float(drilling[0])-0.1 or b > float(drilling[1])+0.1):
            QMessageBox.warning(self,'Глубины','Интервал ящика выходит за пределы долбления.')
            return
        self.box_depths[pair['photo']]={'top':a,'bottom':b,'confirmed':True,
            'source':'manual_from_printed_box_label',
            'confirmed_at':datetime.now(timezone.utc).isoformat()}
        tmp=self.box_depth_file.with_suffix('.tmp')
        tmp.write_text(json.dumps({'photos':self.box_depths},ensure_ascii=False,indent=2),encoding='utf-8')
        tmp.replace(self.box_depth_file)
        self.show_pair(i)

    def _label(self, i, pair):
        name = Path(pair.get("photo", "")).name
        status = self.decisions.get(pair.get("photo", ""), {}).get("decision", "не проверено")
        uv_note = " [УФ?]" if pair.get("photo") in self.suspected_uv else ""
        return f"{i+1}. {pair.get('well', '?')} / {name} [{status}]{uv_note}"

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
        verified=self._box_span(pair)
        self.box_top.setText(str(verified[0]) if verified else '')
        self.box_bottom.setText(str(verified[1]) if verified else '')
        if pixmap.isNull():
            self.photo.setText("Не удалось открыть фотографию:\n" + str(path))
        else:
            self.render_photo()
        self.details.setPlainText(
            f"Скважина: {pair.get('well')}\nФото: {path}\n"
            f"Долбление из имени: {pair.get('photo_depth', 'не определено')}\n"
            f"Глубины ящика: {verified if verified else 'НЕ ПОДТВЕРЖДЕНЫ'}\n"
            f"Автоматический статус: {pair.get('status')}\n"
            "Совпадение глубин не означает подтверждённую геологическую разметку."
        )
        span = self._box_span(pair)
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
            f"прошли фильтр сопоставления {len(self._box_matches(pair))}."
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
        for j, match in enumerate(self._box_matches(pair)):
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
        matches = self._box_matches(self.pairs[i])
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
            self.core_boxes = self._select_lanes(self.recognizer.recognize(image, orientation="auto"))
            self.render_photo()
            self.details.append(f"\nKern Analyzer: найдено {len(self.core_boxes)} столбиков/рядов. Проверьте контуры.")
        except Exception as exc:
            self.details.append(f"\nДетектор: {exc}")
            if not silent:
                QMessageBox.warning(self, "Детектор керна", str(exc))

    @staticmethod
    def _select_lanes(boxes):
        """Choose complete parallel core rows, not transverse strips or foreground objects.

        The detector can return either vertical columns or horizontal rows.
        Require a dominant orientation, similar long-axis lengths, and
        non-overlapping parallel lanes. An outlier (e.g. shoes below the tray)
        is not automatically accepted as another core row.
        """
        candidates = []
        for b in boxes:
            try:
                w = float(b["right"]) - float(b["left"])
                h = float(b["bottom"]) - float(b["top"])
            except (KeyError, TypeError, ValueError):
                continue
            if min(w, h) <= 0:
                continue
            ratio = max(w, h) / min(w, h)
            if ratio < 2.4:
                continue
            orientation = "horizontal" if w > h else "vertical"
            candidates.append((b, orientation, max(w, h), min(w, h)))
        if not candidates:
            return []
        # Majority by coherent total long-axis support, not the widest
        # isolated rectangle, which may be the ruler or the tray itself.
        scores = {}
        for _, orient, length, _ in candidates:
            scores[orient] = scores.get(orient, 0) + length
        dominant = max(scores, key=scores.get)
        group = [c for c in candidates if c[1] == dominant]
        lengths = sorted(c[2] for c in group)
        median = lengths[len(lengths) // 2]
        group = [c for c in group if 0.65 * median <= c[2] <= 1.45 * median]
        if not group:
            return []
        # Prefer boxes with consistent transverse width, rejecting spurious
        # skinny dividers and oversized regions.
        thicknesses = sorted(c[3] for c in group)
        thickness = thicknesses[len(thicknesses) // 2]
        group = [c for c in group if 0.45 * thickness <= c[3] <= 1.8 * thickness]
        group.sort(key=lambda c: (c[0]["top"], c[0]["left"]) if dominant == "horizontal"
                   else (c[0]["left"], c[0]["top"]))
        selected = []
        for b, _, _, _ in group:
            x1, x2 = float(b["left"]), float(b["right"])
            y1, y2 = float(b["top"]), float(b["bottom"])
            area = (x2 - x1) * (y2 - y1)
            duplicate = False
            for a in selected:
                ax1, ax2 = float(a["left"]), float(a["right"])
                ay1, ay2 = float(a["top"]), float(a["bottom"])
                overlap = max(0, min(x2, ax2) - max(x1, ax1)) * max(
                    0, min(y2, ay2) - max(y1, ay1))
                other_area = (ax2 - ax1) * (ay2 - ay1)
                if overlap > 0.45 * min(area, other_area):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(b)
        return selected

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
        # Provisional mapping after explicit confirmation of printed box depths.
        # Never treat these bands as validated segmentation labels.
        span = self._box_span(pair)
        matches = self._box_matches(pair)
        preliminary = False
        if not span:
            # Display unverified filename overlap as a visual hint only.
            # It is NEVER a confirmed depth assignment or training label.
            proposed = pair.get("photo_depth")
            if (isinstance(proposed, list) and len(proposed) == 2
                    and pair.get("depth_source") == "filename_short_interval"):
                try:
                    a, b = float(proposed[0]), float(proposed[1])
                    if a < b:
                        span = [a, b]
                        preliminary = True
                        matches = [d for d in self.data.get("descriptions", [])
                                   if d.get("well") == pair.get("well")
                                   and d.get("top") is not None and d.get("bottom") is not None
                                   and float(d["top"]) < b and float(d["bottom"]) > a
                                   and (d.get("facies_index") or d.get("facies"))]
                except (TypeError, ValueError):
                    span = None
                    matches = []
        # Multiple Excel rows can represent the same layer or overlap because
        # of aggregate records. Never paint overlapping facies as independent
        # ground truth; choose the narrowest interval and flag uncertainty.
        filtered=[]
        for m in sorted(matches, key=lambda v: (float(v["bottom"])-float(v["top"]), -float(v.get("overlap_fraction") or 0))):
            try:
                a,b=float(m["top"]),float(m["bottom"])
            except (TypeError,ValueError,KeyError):
                continue
            if b<=a: continue
            if any(min(b,float(n["bottom"]))-max(a,float(n["top"])) > 0.02 for n in filtered):
                continue
            filtered.append(m)
        matches=sorted(filtered,key=lambda v:float(v["top"]))
        # Depth runs along each row's long axis; row order follows tray layout.
        horizontal = sum((b["right"]-b["left"]) > (b["bottom"]-b["top"]) for b in self.core_boxes) > len(self.core_boxes)/2
        lanes = sorted(self.core_boxes, key=(lambda b:(b["top"],b["left"])) if horizontal else (lambda b:(b["left"],b["top"])))
        if span and len(span)==2 and span[1]>span[0] and lanes:
            widths=[max(1,(b["right"]-b["left"]) if horizontal else (b["bottom"]-b["top"])) for b in lanes]
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
                    if horizontal:
                        x1=round((box["left"]+(lo-cursor)/(lane_end-cursor)*(box["right"]-box["left"]))*sx)
                        x2=round((box["left"]+(hi-cursor)/(lane_end-cursor)*(box["right"]-box["left"]))*sx)
                        y1=round(box["top"]*sy)
                        y2=round(box["bottom"]*sy)
                    else:
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
            # Rendering must not mutate the diagnostic text panel.
        painter.fillRect(0,0,canvas.width(),25,QColor(25,25,25,220))
        painter.setPen(QColor("white"))
        painter.drawText(8,18,("ЖЁЛТЫЙ: ПРЕДВАРИТЕЛЬНО ПО ИМЕНИ, НЕ ДЛЯ ОБУЧЕНИЯ" if preliminary else "ЗЕЛЁНЫЙ: КЕРН | ЖЁЛТЫЙ: ПРОЕКЦИЯ ПО ПАСПОРТУ") if span else "ГЛУБИНЫ ЯЩИКА НЕ ПОДТВЕРЖДЕНЫ")
        painter.end()
        self.photo.setPixmap(canvas)

    def _save(self, decision, candidate_index=None):
        i = self.items.currentRow()
        if not (0 <= i < len(self.pairs)):
            return
        pair = self.pairs[i]
        if decision in ("confirmed","corrected_candidate") and not self._box_span(pair):
            QMessageBox.warning(self,"Глубины","Сначала подтвердите глубины с паспорта ящика.")
            return
        self.decisions[pair["photo"]] = {
            "decision": decision,
            "candidate_index": candidate_index,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "well": pair.get("well"),
            "box_depth": self._box_span(pair),
            "geometry_status": "provisional_projection",
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
