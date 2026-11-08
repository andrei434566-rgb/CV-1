"""Numbered Word/photos to explicit labels and locally trained CNN heads."""
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QPushButton, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget)

from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.supervised_learning import (FIELDS, REVIEW_BASIS, activate_head, canonical_label, export_dataset,
    rows_from_word_manifest, suggested_label, train_feature_head, validate_rows, write_json)
from app.runtime_paths import user_data_root
from app.ui.dialog_theme import apply_light_card
from app.ui.dialogs.feature_evidence_dialog import EVIDENCE_GUIDANCE, FeatureEvidenceDialog


class TrainingJob(QThread):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, rows, field, control, base, destination, model_root, parent=None):
        super().__init__(parent)
        self.args = rows, field, control, base, destination, model_root

    def run(self):
        rows, field, control, base, destination, model_root = self.args
        try:
            self.progress.emit('Копирую только проверенные участки…')
            dataset = export_dataset(rows, field, control, destination)
            if self.isInterruptionRequested():
                raise InterruptedError('Подготовка остановлена. Рабочая модель не изменена.')
            result = train_feature_head(dataset, base, model_root / destination.name,
                                        self.progress.emit, self.isInterruptionRequested)
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class SupervisedLearningDialog(QDialog):
    def __init__(self, manifest, base_folder, parent=None):
        super().__init__(parent)
        apply_light_card(self)
        self.setWindowTitle('Обучение на ваших фото · породы, признаки и фации')
        self.resize(1100, 820)
        self.manifest = Path(manifest); self.base = Path(base_folder)
        self.originals = rows_from_word_manifest(self.manifest)
        self.draft_path = self.manifest.parent / 'labels_cv1.json'
        try:
            self.draft = json.loads(self.draft_path.read_text(encoding='utf-8'))
            if not isinstance(self.draft, dict): self.draft = {}
        except (OSError, ValueError):
            self.draft = {}
        self.rows = []; self.loading = False; self.job = None; self.result = None; self.field_name = FIELDS[0]
        layout = QVBoxLayout(self)
        intro = QLabel('1. Выберите, чему учить. 2. Укажите метку и скважину, выделите на фото участок '
                       'и подтвердите его. 3. Выберите контрольную скважину и нажмите «Обучить». '
                       'Она не попадёт в обучение. Полное описание Word остаётся рядом для проверки.')
        intro.setWordWrap(True); layout.addWidget(intro)
        row = QHBoxLayout(); row.addWidget(QLabel('Чему учить:'))
        self.field = QComboBox(); self.field.addItems(FIELDS); row.addWidget(self.field)
        row.addWidget(QLabel('Контрольная скважина:'))
        self.control = QComboBox(); self.control.setMinimumWidth(160); row.addWidget(self.control)
        layout.addLayout(row)
        self.field_guide = QLabel(); self.field_guide.setWordWrap(True); layout.addWidget(self.field_guide)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Фото', 'Метка класса · исправьте', 'Скважина · заполните', 'Участок', 'Проверен'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(180); layout.addWidget(self.table, 1)
        bulk = QHBoxLayout(); self.bulk_well = QLineEdit(); self.bulk_well.setPlaceholderText('Скважина для выделенных строк')
        bulk.addWidget(self.bulk_well)
        self.apply_well = QPushButton('Назначить скважину'); self.apply_well.clicked.connect(self._assign_well); bulk.addWidget(self.apply_well)
        self.region = QPushButton('Выделить и проверить участок фото…'); self.region.clicked.connect(self._choose_region); bulk.addWidget(self.region)
        layout.addLayout(bulk)
        self.add_region = QPushButton('Ещё один участок на выбранном фото')
        self.add_region.clicked.connect(self._add_region); layout.addWidget(self.add_region)
        preview = QSplitter(Qt.Orientation.Horizontal)
        self.image = QLabel('Выберите строку'); self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumWidth(150); self.image.setMaximumWidth(260); preview.addWidget(self.image)
        self.description = QPlainTextEdit(); self.description.setReadOnly(True); preview.addWidget(self.description)
        preview.setMaximumHeight(200); preview.setMinimumHeight(140); layout.addWidget(preview)
        self.status = QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.report = QPlainTextEdit(); self.report.setReadOnly(True); self.report.setMaximumHeight(155); layout.addWidget(self.report)
        actions = QHBoxLayout()
        self.train = QPushButton('Обучить локальный классификатор'); self.train.clicked.connect(self._train); actions.addWidget(self.train)
        self.activate = QPushButton('Использовать как подсказки'); self.activate.setEnabled(False)
        self.activate.clicked.connect(self._activate); actions.addWidget(self.activate)
        self.disable = QPushButton('Отключить подсказки признака')
        self.disable.clicked.connect(self._deactivate); actions.addWidget(self.disable)
        self.stop = QPushButton('Остановить обучение'); self.stop.setEnabled(False); self.stop.clicked.connect(self._stop); actions.addWidget(self.stop)
        self.close_button = QPushButton('Закрыть'); self.close_button.clicked.connect(self.close); actions.addWidget(self.close_button)
        layout.addLayout(actions)
        note = QLabel('Обучается небольшой классификатор на признаках существующей CNN. Интернет не нужен. '
                      'Для обучения границ YOLO по-прежнему нужны контуры. '
                      'Минимум на класс: 5 учебных и 3 контрольных участка; для устойчивого качества этого обычно мало.')
        note.setWordWrap(True); layout.addWidget(note)
        self.field.currentTextChanged.connect(self._change_field)
        self.table.itemChanged.connect(self._item_changed)
        self.table.currentCellChanged.connect(self._preview)
        self.control.currentTextChanged.connect(self._control_changed)
        self._load_field(self.field_name)

    def _load_field(self, field):
        self.loading = True; self.field_name = field
        guidance = EVIDENCE_GUIDANCE.get(field)
        self.field_guide.setVisible(bool(guidance))
        if guidance:
            options = ', '.join(LITHOLOGY_ATTRIBUTE_OPTIONS[field])
            self.field_guide.setText(f'Классы: {options}. {guidance[0]} Подтвердите метку в окне выделения. '
                                    'Модель сохраняется для отдельной проверки.')
        self.activate.setEnabled(bool(self.result and self.result['card']['field'] == field
                                     and self.result['card']['deployment'] == 'suggestions_only'))
        saved = self.draft.get(field, {})
        if not isinstance(saved, dict): saved = {}
        by_id = {r.get('id'): r for r in saved.get('rows', []) if isinstance(r, dict)}
        self.rows = []
        for original in self.originals:
            row = dict(original); row['source_id'] = original['id']; stored = by_id.get(row['id'], {})
            if stored.get('image_sha256') == row['image_sha256']:
                for key in ('well', 'label', 'roi', 'reviewed', 'review_basis'): row[key] = stored.get(key, row.get(key))
            elif not stored:
                row['label'] = suggested_label(row['description'], field)
                for other_field, other in self.draft.items():
                    if not isinstance(other, dict): continue
                    previous = next((r for r in other.get('rows', []) if isinstance(r, dict)
                                     and r.get('id') == row['id'] and r.get('image_sha256') == row['image_sha256']), None)
                    if previous:
                        row['well'] = previous.get('well', '')
                        if field not in REVIEW_BASIS and other_field not in REVIEW_BASIS:
                            row['roi'] = previous.get('roi')
                        break
            self.rows.append(row)
        originals = {r['id']: r for r in self.originals}
        for stored in by_id.values():
            original = originals.get(stored.get('source_id'))
            if original and stored['id'] not in originals and original['image_sha256'] == stored.get('image_sha256'):
                self.rows.append({**original, **stored})
        if field in REVIEW_BASIS:
            for row in self.rows:
                row['reviewed'] = row['reviewed'] is True and row.get('review_basis') == REVIEW_BASIS[field]
        self._fill_table()
        self._refresh_wells(saved.get('validation_well', ''))
        self.loading = False
        if self.rows: self.table.selectRow(0); self._preview(0)
        self._save()

    def _fill_table(self):
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            values = (row['id'], row['label'], row['well'], 'Выделен' if row['roi'] else 'Не выделен', '')
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col not in (1, 2): item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if col == 4:
                    if self.field_name in REVIEW_BASIS:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                        item.setToolTip('Подтвердите основание метки в окне выделения участка.')
                    else:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if row['reviewed'] else Qt.CheckState.Unchecked)
                self.table.setItem(i, col, item)

    def _add_region(self):
        i = self.table.currentRow()
        if i < 0: return
        row = dict(self.rows[i]); row.update(id=row['source_id'] + ':' + uuid4().hex[:6], roi=None, reviewed=False)
        row.pop('review_basis', None)
        self.rows.append(row); self.loading = True; self._fill_table(); self.loading = False
        self.table.selectRow(len(self.rows)-1); self._save()

    def _change_field(self, field):
        if self.job: return
        self._save(); self._load_field(field)

    def _save(self):
        if self.loading: return
        keys = ('id', 'source_id', 'image_sha256', 'well', 'label', 'roi', 'reviewed', 'review_basis')
        self.draft[self.field_name] = {'rows': [{k: row[k] for k in keys if k in row}
                                              for row in self.rows], 'validation_well': self.control.currentText()}
        try:
            write_json(self.draft_path, self.draft)
            ready = sum(r['reviewed'] is True for r in self.rows)
            self.status.setText(f'Проверено участков: {ready}/{len(self.rows)}. Метки сохранены рядом с учебной базой; '
                                'при переносе копируйте всю её папку.')
        except OSError as exc:
            self.status.setText('Не удалось сохранить метки: ' + str(exc))

    def _refresh_wells(self, selected=None):
        selected = self.control.currentText() if selected is None else selected
        self.control.blockSignals(True); self.control.clear(); self.control.addItem('')
        self.control.addItems(sorted({r['well'].strip() for r in self.rows if r['well'].strip()}))
        self.control.setCurrentText(selected); self.control.blockSignals(False)

    def _control_changed(self, *_):
        self._save()

    def _item_changed(self, item):
        if self.loading: return
        row = self.rows[item.row()]; column = item.column()
        if column in (1, 2):
            row['label' if column == 1 else 'well'] = item.text().strip()
            row['reviewed'] = False
            row.pop('review_basis', None)
            self.loading = True; self.table.item(item.row(), 4).setCheckState(Qt.CheckState.Unchecked); self.loading = False
            self._refresh_wells()
        elif column == 4:
            if self.field_name in REVIEW_BASIS:
                return
            row['reviewed'] = item.checkState() == Qt.CheckState.Checked and bool(row['roi'] and row['label'] and row['well'])
            if not row['reviewed']:
                self.loading = True; item.setCheckState(Qt.CheckState.Unchecked); self.loading = False
        self._save()

    def _assign_well(self):
        value = self.bulk_well.text().strip()
        if value:
            for index in self.table.selectionModel().selectedRows():
                self.table.item(index.row(), 2).setText(value)

    def _preview(self, index, *_):
        if not 0 <= index < len(self.rows): return
        row = self.rows[index]; self.description.setPlainText(row['description'])
        reader = QImageReader(row['image']); reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid(): reader.setScaledSize(size.scaled(QSize(230, 190), Qt.AspectRatioMode.KeepAspectRatio))
        self.image.setPixmap(QPixmap.fromImage(reader.read()))

    def _choose_region(self):
        i = self.table.currentRow()
        if i < 0: return
        row = self.rows[i]
        if not row['label'] or not row['well']:
            QMessageBox.information(self, 'Укажите метку и скважину', 'Сначала заполните метку класса и скважину в выбранной строке.'); return
        label = canonical_label(row['label'], self.field_name)
        if not label:
            QMessageBox.information(self, 'Проверьте метку', self.field_guide.text()); return
        reader = QImageReader(row['image']); reader.setAutoTransform(True); size = reader.size()
        if not size.isValid() or size.width()*size.height() > 100_000_000:
            QMessageBox.warning(self, 'Фото', 'Не удалось прочитать размер или фото больше 100 Мп.'); return
        reader.setScaledSize(size.scaled(QSize(3600, 3600), Qt.AspectRatioMode.KeepAspectRatio))
        pixmap = QPixmap.fromImage(reader.read())
        if pixmap.isNull():
            QMessageBox.warning(self, 'Фото', 'Не удалось прочитать фотографию.'); return
        entry = {'roi': row['roi'], 'reviewed': row['reviewed']} if row['roi'] else None
        dialog = FeatureEvidenceDialog(pixmap, self.field_name, label, entry, parent=self)
        apply_light_card(dialog)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            result = dialog.result_entry
            row['roi'] = result['roi'] if result else None
            row['reviewed'] = bool(result and result['reviewed'])
            row.pop('review_basis', None)
            if result and result.get('review_basis'):
                row['review_basis'] = result['review_basis']
            self.loading = True
            self.table.item(i, 3).setText('Выделен' if row['roi'] else 'Не выделен')
            self.table.item(i, 4).setCheckState(Qt.CheckState.Checked if row['reviewed'] else Qt.CheckState.Unchecked)
            self.loading = False; self._save()

    def _train(self):
        try:
            validate_rows(self.rows, self.field_name, self.control.currentText())
        except ValueError as exc:
            QMessageBox.information(self, 'Что подготовить для обучения', str(exc)); return
        from app.infrastructure.local_vision import read_model_card
        try: read_model_card(self.base)
        except Exception as exc:
            QMessageBox.warning(self, 'Не найдена исходная CNN', str(exc)); return
        self._save(); self.result = None; self.activate.setEnabled(False)
        name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6]
        destination = user_data_root() / 'training' / 'supervised' / name
        self.job = TrainingJob([dict(r) for r in self.rows], self.field_name, self.control.currentText(), self.base,
                               destination, user_data_root() / 'models' / 'features', self)
        self.job.progress.connect(self.report.appendPlainText)
        self.job.failed.connect(self.report.appendPlainText)
        self.job.succeeded.connect(self._trained)
        self.job.finished.connect(self._finished)
        self._set_busy(True); self.report.clear(); self.job.start()

    def _set_busy(self, active):
        for control in (self.field, self.table, self.control, self.region, self.add_region, self.train, self.apply_well, self.bulk_well, self.disable):
            control.setEnabled(not active)
        self.stop.setEnabled(active)

    def _trained(self, result):
        self.result = result; metrics = result['card']['validation']
        self.report.appendPlainText(f'Контрольная скважина: {result["card"]["validation_well"]}. '
                                    f'Точность {metrics["accuracy"]:.1%}; macro F1 {metrics["macro_f1"]:.3f}.')
        for row in metrics['per_class']:
            self.report.appendPlainText(f'{row["name"]}: примеров {row["support"]}, precision {row["precision"]:.1%}, recall {row["recall"]:.1%}.')
        self.report.appendPlainText(result['card']['limitation'] + '\nСохранено: ' + result['folder'])
        self.activate.setEnabled(result['card']['deployment'] == 'suggestions_only')
        if not self.activate.isEnabled():
            self.report.appendPlainText('Результат для отдельной проверки по фрагментам. Автоподстановка в описание отключена.')

    def _finished(self):
        self.job = None; self._set_busy(False)

    def _activate(self):
        if not self.result: return
        try:
            activate_head(self.result['folder'], user_data_root() / 'models' / 'features')
            self.report.appendPlainText('Подсказки включены для следующих распознаваний. '
                                        'Результаты старых фото не перезаписываются. Оценка модели не является гарантией точности.')
        except Exception as exc:
            QMessageBox.warning(self, 'Классификатор не подключён', str(exc))

    def _stop(self):
        if self.job:
            self.job.requestInterruption(); self.stop.setEnabled(False)
            self.report.appendPlainText('Остановка после текущего участка…')

    def _deactivate(self):
        from app.infrastructure.supervised_learning import deactivate_head
        try:
            deactivate_head(self.field_name, user_data_root() / 'models/features')
            self.report.appendPlainText('Подсказки для «' + self.field_name + '» отключены. Сохранённые модели и разметка остаются.')
        except Exception as exc:
            QMessageBox.warning(self, 'Не удалось отключить подсказки', str(exc))

    def closeEvent(self, event):
        if self.job:
            self.status.setText('Сначала остановите обучение и дождитесь завершения текущего участка.'); event.ignore(); return
        self._save(); super().closeEvent(event)

    def reject(self):
        if self.job: self._stop(); return
        self._save(); super().reject()
