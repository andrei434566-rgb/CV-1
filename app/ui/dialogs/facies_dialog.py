from __future__ import annotations

import json
import math

from PySide6.QtCore import QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPixmap, QPolygonF
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSplitter, QTabWidget, QVBoxLayout, QWidget

from app.domain.facies_catalog import FACIES_MODEL_CLASSES, FACIES_REFERENCE_FIELDS, facies_metadata, facies_title, resolve_facies_class
from app.domain.lithology_attributes import LITHOLOGY_ATTRIBUTE_OPTIONS
from app.infrastructure.primary_facies_catalog import resolve_catalog_class
from app.infrastructure.description_template import ASSUMPTIONS_KEY, PHOTO_UNRESOLVED, SCHEMA, UNKNOWN_FACIES, compose_description, observation, read_assumptions
from app.infrastructure.description_review import review_description
from app.infrastructure.attribute_examples import EVIDENCE_FIELDS, EVIDENCE_KEY, pixel_digest, png_bytes, read_evidence, valid_evidence
from app.ui.dialogs.feature_evidence_dialog import EvidenceView, FeatureEvidenceDialog


CUSTOM_VALUE = "__kern_analyzer_custom_value__"
FACIES_METADATA_ROLE = Qt.ItemDataRole.UserRole + 1


def make_interval_preview(pixmap: QPixmap, polygon) -> QPixmap:
    """Show the selected geometry only, using the already loaded photo."""
    if pixmap.isNull() or len(polygon) < 3:
        return QPixmap()
    bounds = QPolygonF(polygon).boundingRect().toAlignedRect().intersected(pixmap.rect())
    if bounds.isEmpty():
        return QPixmap()
    result = QPixmap(bounds.size())
    result.fill(Qt.GlobalColor.white)
    painter = QPainter(result)
    painter.translate(-bounds.left(), -bounds.top())
    path = QPainterPath()
    path.addPolygon(QPolygonF(polygon))
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()
    return result


class FaciesDialog(QDialog):
    delete_requested = Signal()
    move_up_requested = Signal()
    move_down_requested = Signal()

    def __init__(
        self,
        current_facies: str,
        confidence: float,
        attributes: dict[str, str] | None = None,
        depth_from: float | None = None,
        depth_to: float | None = None,
        parent=None,
        *,
        training_ready: bool = False,
        primary_catalog: dict | None = None,
        description_confirmed: bool = False,
        preview_pixmap: QPixmap | None = None,
        photo_quality: dict | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("intervalDescriptionCard")
        from app.ui.dialog_theme import apply_light_card
        apply_light_card(self)
        self._uses_primary_catalog = primary_catalog is not None
        self._catalog_classes = list(primary_catalog["classes"]) if primary_catalog else list(FACIES_MODEL_CLASSES)
        self.setWindowTitle("Карточка участка керна")
        self.setMinimumSize(700, 520)
        self.resize(960, 800)
        attributes = dict(attributes or {})
        from app.infrastructure.core_color_hint import color_is_confirmed
        self._color_auto_allowed = not description_confirmed and not color_is_confirmed(attributes) and (
            attributes.get('Цвет') in (None, '', PHOTO_UNRESOLVED, attributes.get('__vision_color')))
        self._color_started = False
        self._color_revision = 0
        self._applying_color = False
        self._color_user_touched = False
        # Historical photo-contrast hints are not sedimentary texture labels.
        auto_hint = str(attributes.get("__vision_visual_hint") or "")
        if (not description_confirmed and attributes.get("__description_engine") == "vision"
                and str(attributes.get("Текстура", "")).startswith(("На фото заметны контрастные полосы", "Текстура и слоистость требуют"))):
            auto_hint = str(attributes.pop("Текстура"))
        self._preview_pixmap = preview_pixmap or QPixmap()
        self._preview_digest = pixel_digest(self._preview_pixmap) if not self._preview_pixmap.isNull() else ""
        self._evidence = read_evidence(attributes)
        self._evidence_buttons = {}
        self._example_id = attributes.get("__attribute_example_id", "")
        self._similar_rows = []
        self._similar_signature = None
        try:self._quality_report=json.loads(attributes.get('__image_quality','{}'))
        except (ValueError,TypeError):self._quality_report={}
        if not isinstance(self._quality_report,dict):self._quality_report={}
        if not self._quality_report:self._quality_report=dict(photo_quality or {})
        if str(current_facies).casefold() in UNKNOWN_FACIES:
            current_facies = ""
        is_whole_column = attributes.get("__core_column_description") == "1"
        if is_whole_column:
            self.setWindowTitle("Описание столбика керна")
        requested_index = str(attributes.get("Индекс фации") or "").strip()
        # Catalogue values are displayed as reference hints. They do not fill
        # observation controls or become image-derived attributes.
        self._preserved_attributes = {
            key: value
            for key, value in attributes.items()
            if key not in LITHOLOGY_ATTRIBUTE_OPTIONS
            and key not in {"Цвет насыщения", "Цвет литологии"}
            and key not in FACIES_REFERENCE_FIELDS
        }
        try:grain_entry=json.loads(self._preserved_attributes.get('__grain_measurement','{}'))
        except (ValueError,TypeError):grain_entry={}
        self._grain_measurement_stale=bool(isinstance(grain_entry,dict) and grain_entry
            and grain_entry.get('image_sha256')!=self._preview_digest)
        self.settings = QSettings("Kern Analyzer", "KernAnalyzer")
        self.attribute_combos: dict[str, QComboBox] = {}
        self.assumption_checks: dict[str, QCheckBox] = {}
        initial_assumptions = read_assumptions(attributes)

        layout = QVBoxLayout(self)
        confidence_title = (
            "Границы найденного столбика" if is_whole_column
            else "Оценка геометрии YOLO" if attributes.get("__geometry_only") == "1"
            else "Оценка модели"
        )
        confidence_text = (
            "Рамка целого столбика предварительная — проверьте по исходному фото."
            if is_whole_column else f"{confidence_title}: {confidence:.0%} (не вероятность правильности фации)"
        )
        confidence_label = QLabel(confidence_text)
        confidence_label.setStyleSheet("color: #687087;")
        layout.addWidget(confidence_label)
        source_class = str(attributes.get("__source_model_class") or "").strip()
        if source_class:
            source_id = str(attributes.get("__source_model_class_id") or "").strip()
            source_text = "Предложение YOLO: " + source_class
            if source_id:
                source_text += " (класс " + source_id + ")"
            if attributes.get("__geometry_only") == "1":
                source_text += ". Модель может выделять только геометрию; фацию укажите ниже."
            source_label = QLabel(source_text)
            source_label.setObjectName("yoloSourceClass")
            source_label.setWordWrap(True)
            layout.addWidget(source_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        form_container = QWidget()
        form = QFormLayout(form_container)
        form.setSpacing(10)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.facies = QComboBox()
        self.facies.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.facies.setMinimumContentsLength(12)
        self.facies.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.facies.addItem("— фация не выбрана —", "")
        for model_class in self._catalog_classes:
            item = model_class["metadata"]
            code = model_class["code"] or model_class["model_label"]
            title = facies_title(item) if model_class["code"] and model_class["index"] else f"{code} · {model_class['name']}"
            self.facies.addItem(title, code)
            self.facies.setItemData(self.facies.count() - 1, dict(item), FACIES_METADATA_ROLE)
        self.facies.addItem("Самостоятельный выбор…", CUSTOM_VALUE)
        resolved = self._resolve_class(current_facies, requested_index)
        if resolved:
            current_facies, requested_index = resolved["code"], resolved["index"]
        elif current_facies == "Новый контур":
            current_facies = ""
        current_index = next(
            (
                index
                for index in range(self.facies.count())
                if self.facies.itemData(index) == current_facies and (resolved or not current_facies)
                and (
                    not requested_index
                    or str((self.facies.itemData(index, FACIES_METADATA_ROLE) or {}).get("Индекс фации") or "")
                    == requested_index
                )
            ),
            -1,
        )
        if current_facies and current_index < 0:
            self.facies.insertItem(0, current_facies, current_facies)
            current_index = 0
        if current_index >= 0:
            self.facies.setCurrentIndex(current_index)
        self.facies.activated.connect(self._choose_custom_facies)
        self.facies.currentIndexChanged.connect(self._update_facies_reference)
        self.facies.setToolTip("Выберите код из справочника или «Самостоятельный выбор» для своего кода.")
        search = QLineEdit()
        search.setPlaceholderText("Код, индекс или часть названия…")
        search.textChanged.connect(self._filter_facies)
        form.addRow("Поиск фации:", search)
        form.addRow("Фация (основная метка):", self.facies)
        self.facies_assumed = QCheckBox("Фация предположительная")
        self.facies_assumed.setChecked("Название фации" in initial_assumptions)
        form.addRow("", self.facies_assumed)
        self.facies_reference = QLabel()
        self.facies_reference.setWordWrap(True)
        self.facies_reference.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.facies_reference.setStyleSheet("color: #596174; background: #f5f7fa; padding: 7px; border-radius: 6px;")
        form.addRow("Определение:", self.facies_reference)
        self._update_facies_reference()
        self.depth_from = QLineEdit("" if depth_from is None else self._format_depth(depth_from))
        self.depth_from.setPlaceholderText("например, 2450.5")
        self.depth_to = QLineEdit("" if depth_to is None else self._format_depth(depth_to))
        self.depth_to.setPlaceholderText("например, 2451.2")
        if depth_from is not None and depth_to is not None and depth_to > depth_from and self.depth_from.text() == self.depth_to.text():
            self.depth_from.setText(f"{float(depth_from):.10f}".rstrip('0').rstrip('.'))
            self.depth_to.setText(f"{float(depth_to):.10f}".rstrip('0').rstrip('.'))
        self._original_depths = (depth_from, depth_to)
        self._initial_depth_texts = (self.depth_from.text(), self.depth_to.text())
        for edit, value in zip((self.depth_from, self.depth_to), self._original_depths):
            edit.setToolTip("Полная отметка: " + (repr(value) if value is not None else "не задана")
                            + ". Показано с точностью до 0,001 м. Пока поле не изменено, исходная точность сохраняется.")
        form.addRow("Начало слоя, м:", self.depth_from)
        form.addRow("Конец слоя, м:", self.depth_to)
        self.hydrodynamic_regime = QLineEdit(str(attributes.get("__vlm_hydrodynamic_regime") or attributes.get("Гидродинамический режим") or ""))
        self.hydrodynamic_regime.setPlaceholderText("Гипотеза по структурам и основному источнику")
        form.addRow("Гидродинамический режим:", self.hydrodynamic_regime)
        for field_name, options in LITHOLOGY_ATTRIBUTE_OPTIONS.items():
            combo = QComboBox()
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            combo.addItem("— не выбрано —", "")
            for option in options:
                combo.addItem(option, option)
            combo.addItem("Самостоятельный выбор…", CUSTOM_VALUE)
            original_value = str(attributes.get(field_name) or "").strip()
            value, inline_assumption = observation(original_value)
            value_index = combo.findData(value)
            if value and value_index < 0:
                combo.insertItem(combo.count() - 1, value, value)
                value_index = combo.count() - 2
            combo.setCurrentIndex(value_index if value_index >= 0 else 0)
            combo.activated.connect(lambda index, field=field_name, widget=combo: self._choose_custom_attribute(field, widget, index))
            self.attribute_combos[field_name] = combo
            assumed = QCheckBox("Предположительно")
            assumed.setChecked(field_name in initial_assumptions or inline_assumption
                               or (ASSUMPTIONS_KEY not in attributes and field_name == "Цвет"
                                   and attributes.get("__vision_color") == value and bool(value)))
            assumed.setToolTip("Свойство является предположением. Эта отметка попадёт в описание.")
            self.assumption_checks[field_name] = assumed
            row = QVBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(1)
            if field_name == 'Цвет':
                color_row = QHBoxLayout(); color_row.addWidget(combo,1)
                self.color_chip = QLabel('—'); self.color_chip.setObjectName('photoColorChip')
                self.color_chip.setFixedSize(32,28);self.color_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
                color_row.addWidget(self.color_chip);row.addLayout(color_row)
            else:
                row.addWidget(combo)
            row.addWidget(assumed)
            if field_name == 'Название породы':
                button=QPushButton('Признаки пород и возможности модели…')
                button.setObjectName('rockTypesHelp');button.clicked.connect(self._open_rock_types)
                row.addWidget(button)
            if field_name == 'Зернистость':
                button=QPushButton('Измерить зернистость по масштабу…')
                button.setObjectName('grainMeasurementButton');button.setEnabled(not self._preview_pixmap.isNull())
                button.clicked.connect(self._open_grain_measurement);row.addWidget(button)
                if self._grain_measurement_stale:
                    note=QLabel('Сохранённые промеры относятся к другому фрагменту. Проверьте зернистость заново.')
                    note.setWordWrap(True);row.addWidget(note)
            if field_name == 'Цвет':
                self.color_note = QLabel('Цвет определяется по фотографии автоматически.');self.color_note.setWordWrap(True)
                self.color_note.setObjectName('photoColorNote');row.addWidget(self.color_note)
                self.color_button = QPushButton('Определить цвет по фото')
                self.color_button.setEnabled(not self._preview_pixmap.isNull())
                self.color_button.clicked.connect(lambda:self._request_photo_color(force=True));row.addWidget(self.color_button)
                combo.currentIndexChanged.connect(self._color_edited);assumed.toggled.connect(self._color_edited)
            if field_name in EVIDENCE_FIELDS:
                button = QPushButton("Фрагмент признака…")
                button.setEnabled(not self._preview_pixmap.isNull())
                button.clicked.connect(lambda _checked=False, field=field_name: self._edit_feature_evidence(field))
                row.addWidget(button)
                self._evidence_buttons[field_name] = button
            field_label = QLabel(f"{field_name}:")
            field_label.setWordWrap(True)
            field_label.setMaximumWidth(155)
            form.addRow(field_label, row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_container)
        splitter.addWidget(scroll)

        preview_panel = QWidget()
        panel_layout = QVBoxLayout(preview_panel)
        panel_layout.setContentsMargins(8, 0, 0, 0)
        self.preview_tabs = QTabWidget()
        panel_layout.addWidget(self.preview_tabs)
        description_page = QWidget()
        preview_layout = QVBoxLayout(description_page)
        self.preview_tabs.addTab(description_page, "Описание")
        splitter.addWidget(preview_panel)
        splitter.setSizes([510, 420])
        if preview_pixmap is not None and not preview_pixmap.isNull():
            photo_label = QLabel()
            photo_label.setObjectName("intervalPhotoPreview")
            photo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            photo_label.setPixmap(preview_pixmap.scaled(340, 170, Qt.AspectRatioMode.KeepAspectRatio,
                                                       Qt.TransformationMode.SmoothTransformation))
            preview_layout.addWidget(photo_label)
            zoom_photo = QPushButton("Рассмотреть породу крупно…")
            zoom_photo.clicked.connect(self._open_photo_zoom)
            preview_layout.addWidget(zoom_photo)

        vlm_note = QLabel("Типовое описание для Excel")
        vlm_note.setWordWrap(True)
        vlm_note.setStyleSheet("color: #2364a3; font-weight: 600;")
        preview_layout.addWidget(vlm_note)
        if auto_hint:
            visual_note = QLabel("Наблюдение по фотографии: " + auto_hint)
            visual_note.setWordWrap(True)
            visual_note.setToolTip("Контрастные полосы сами по себе не доказывают тип слоистости или фацию.")
            preview_layout.addWidget(visual_note)
        self.vlm_association = QLineEdit(str(attributes.get("__vlm_facies_association") or attributes.get("Фациальная ассоциация") or ""))
        self.vlm_association.setPlaceholderText("Ассоциация фаций")
        preview_layout.addWidget(self.vlm_association)
        self.vlm_environment = QLineEdit(str(attributes.get("__vlm_depositional_environment") or attributes.get("Обстановка седиментации") or ""))
        self.vlm_environment.setPlaceholderText("Обстановка осадконакопления")
        preview_layout.addWidget(self.vlm_environment)
        self._automatic_context = {
            field: (edit, edit.text()) for field, edit in (
                ("Фациальная ассоциация", self.vlm_association),
                ("Обстановка седиментации", self.vlm_environment),
                ("Гидродинамический режим", self.hydrodynamic_regime),
            ) if field in initial_assumptions
        }
        if attributes.get("__description_engine") == "vision" and not description_confirmed:
            confidence_note = QLabel("Уверенность интерпретации: низкая")
            confidence_note.setStyleSheet("color: #687087;")
            preview_layout.addWidget(confidence_note)
        self.template_mode = QCheckBox("Описание по выбранным признакам")
        self.template_mode.setObjectName("descriptionTemplateMode")
        existing_text = str(attributes.get("__vlm_description") or attributes.get("Краткое описание") or "")
        mode = attributes.get("__description_mode")
        use_template = mode == "template" or (mode != "manual" and (not existing_text or (
            attributes.get("__description_engine") == "vision" and not description_confirmed
            and attributes.get("__manual_interval_edited") != "1")))
        self.template_mode.setChecked(use_template)
        preview_layout.addWidget(self.template_mode)
        self._manual_description = str(attributes.get("__description_manual_text", existing_text))
        self.vlm_description = QPlainTextEdit(str(attributes.get("__vlm_description") or attributes.get("Краткое описание") or ""))
        self.vlm_description.setPlaceholderText("Краткое описание по фото")
        self.vlm_description.setObjectName("intervalDescriptionPreview")
        self.vlm_description.setMinimumHeight(150)
        preview_layout.addWidget(self.vlm_description, 1)
        self.description_mode_note = QLabel()
        self.description_mode_note.setWordWrap(True)
        preview_layout.addWidget(self.description_mode_note)
        self.missing_fields = QLabel()
        self.missing_fields.setObjectName("descriptionMissingFields")
        self.missing_fields.setWordWrap(True)
        self.missing_fields.setStyleSheet("color: #896015;")
        preview_layout.addWidget(self.missing_fields)
        self._build_review_tab()
        self._build_examples_tab()

        self.training_confirmed = QCheckBox("Фация и границы проверены — использовать в обучении")
        self.training_confirmed.setChecked(training_ready)
        self.training_confirmed.setToolTip("Отметьте после проверки класса и контура. Сохранить черновик можно без этой отметки.")
        self.training_confirmed.setVisible(not is_whole_column)
        layout.addWidget(self.training_confirmed)
        self.description_confirmed = QCheckBox("Описание проверено специалистом — сохранить подтверждение")
        self.description_confirmed.setChecked(description_confirmed)
        self.description_confirmed.setToolTip("Подтвердите текст отдельно от класса и границ. Для обучения модели фаций также нужна проверка контура и класса.")
        self.description_confirmed.setVisible(True)
        layout.addWidget(self.description_confirmed)
        try:
            suggestions = json.loads(str(attributes.get("__attribute_suggestions") or "{}"))
        except (ValueError, TypeError):
            suggestions = {}
        if isinstance(suggestions, dict) and suggestions:
            suggested = QLabel("Визуальные подсказки (не подтверждены): " + "; ".join(f"{k}: {v}" for k, v in suggestions.items()))
            suggested.setWordWrap(True)
            layout.addWidget(suggested)
            apply_suggestions = QPushButton("Перенести подсказки в пустые поля")
            apply_suggestions.clicked.connect(lambda: self._apply_suggestions(suggestions))
            layout.addWidget(apply_suggestions)

        order_row = QHBoxLayout()
        move_up = QPushButton("↑ Слой выше")
        move_up.setObjectName("moveFaciesLayerUp")
        move_up.setToolTip("Поднять этот слой на одну позицию в собранном столбике керна")
        move_down = QPushButton("↓ Слой ниже")
        move_down.setObjectName("moveFaciesLayerDown")
        move_down.setToolTip("Опустить этот слой на одну позицию в собранном столбике керна")
        move_up.clicked.connect(self.move_up_requested)
        move_down.clicked.connect(self.move_down_requested)
        order_row.addWidget(move_up)
        order_row.addWidget(move_down)
        layout.addLayout(order_row)

        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        actions.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        actions.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        delete_button = QPushButton("Удалить")
        delete_button.setStyleSheet("color: #b33b4d;")
        actions.addButton(delete_button, QDialogButtonBox.ButtonRole.DestructiveRole)
        actions.accepted.connect(self.accept)
        actions.rejected.connect(self.reject)
        delete_button.clicked.connect(self._request_delete)
        layout.addWidget(actions)
        for combo in self.attribute_combos.values():
            combo.currentIndexChanged.connect(self._description_inputs_changed)
        for check in (*self.assumption_checks.values(), self.facies_assumed):
            check.toggled.connect(self._description_inputs_changed)
        for edit in (self.depth_from, self.depth_to, self.hydrodynamic_regime,
                     self.vlm_association, self.vlm_environment):
            edit.textChanged.connect(self._description_inputs_changed)
        self.facies.currentIndexChanged.connect(self._description_inputs_changed)
        self.facies.currentIndexChanged.connect(lambda: self.training_confirmed.setChecked(False))
        self.facies_assumed.toggled.connect(lambda checked: self.training_confirmed.setChecked(False) if checked else None)
        self.depth_from.textChanged.connect(lambda: self.training_confirmed.setChecked(False))
        self.depth_to.textChanged.connect(lambda: self.training_confirmed.setChecked(False))
        self.template_mode.toggled.connect(self._toggle_template_mode)
        self.vlm_description.textChanged.connect(self._manual_description_changed)
        self._refresh_description()
        self._opening_signature=self._editor_signature()
        try:self._show_photo_color(json.loads(attributes.get('__color_measurement','{}')))
        except (ValueError,TypeError):pass

    def showEvent(self,event):
        super().showEvent(event)
        if self._color_auto_allowed and not self._color_started and not self._preview_pixmap.isNull():
            self._color_started = True
            QTimer.singleShot(0,lambda:self._request_photo_color(force=False))

    def _open_grain_measurement(self):
        from app.ui.dialogs.grain_measurement_dialog import GrainMeasurementDialog
        try:entry=json.loads(self._preserved_attributes.get('__grain_measurement','{}'))
        except (ValueError,TypeError):entry=None
        dialog=GrainMeasurementDialog(self._preview_pixmap,entry=entry,parent=self)
        if dialog.exec()!=QDialog.DialogCode.Accepted or not dialog.result_entry:return
        self._apply_grain_measurement(dialog.result_entry)

    def _apply_grain_measurement(self,entry):
        # Explicit user action, never an automatic replacement on opening a card.
        if entry.get('image_sha256')!=self._preview_digest:return
        self._preserved_attributes['__grain_measurement']=json.dumps(entry,ensure_ascii=False,allow_nan=False)
        self._grain_measurement_stale=False
        if entry.get('label'):
            combo=self.attribute_combos['Зернистость'];index=combo.findData(entry['label'])
            if index<0:return
            combo.setCurrentIndex(index);self.assumption_checks['Зернистость'].setChecked(True)
            self.description_confirmed.setChecked(False);self._refresh_description()

    def _open_rock_types(self):
        from app.infrastructure.local_vision import default_model_folder, read_model_card
        from app.ui.dialogs.rock_types_dialog import RockTypesDialog
        folder=str(self.settings.value('vision/model_folder','') or '') or default_model_folder()
        try:card=read_model_card(folder)
        except (OSError,ValueError):card={}
        RockTypesDialog(self,selected=self.attribute_combos['Название породы'].currentData(),card=card).exec()

    def _color_edited(self,*_):
        if not self._applying_color:
            self._color_revision += 1
            self._color_user_touched = True

    def _request_photo_color(self,*,force=False):
        if self._preview_pixmap.isNull():return
        if not force and (self._color_user_touched or not self._color_auto_allowed):return
        from app.infrastructure.attribute_tasks import submit
        self.color_button.setEnabled(False);self.color_note.setText('Определяю цвет по пикселям породы…')
        submit('color',{'image_png':png_bytes(self._preview_pixmap),'signature':self._color_revision},self._photo_color_ready)

    def _photo_color_ready(self,reply):
        if not self.isVisible():return
        self.color_button.setEnabled(True)
        if not reply.get('ok'):
            self.color_note.setText('Не удалось определить цвет по фото.');return
        result=reply['result']
        if result['signature']!=self._color_revision:
            self.color_note.setText('Вы изменили цвет во время расчёта. Ваше значение сохранено.');return
        measurement=result['measurement']
        if not measurement:
            self.color_note.setText('Недостаточно пригодных пикселей: проверьте маску, блики и тени.');return
        from app.infrastructure.core_color_hint import apply_photo_color
        apply_photo_color(self._preserved_attributes,measurement,force=True)
        self._applying_color=True
        combo=self.attribute_combos['Цвет'];value=measurement['label'];index=combo.findData(value)
        if index<0:combo.insertItem(combo.count()-1,value,value);index=combo.count()-2
        combo.setCurrentIndex(index);self.assumption_checks['Цвет'].setChecked(True)
        self._applying_color=False
        self._show_photo_color(measurement)

    def _show_photo_color(self,measurement):
        if not isinstance(measurement,dict) or not measurement.get('label'):return
        color=QColor(str(measurement.get('hex') or ''))
        if color.isValid():
            self.color_chip.setText('');self.color_chip.setStyleSheet(f'background:{color.name()};border:1px solid #8091a5;border-radius:4px;')
        self.color_chip.setToolTip('Образец основного тона по фотографии')
        self.color_note.setText('По фото: '+str(measurement['label']))
        components=measurement.get('components') or []
        details='; '.join(f"{row['label']}: {row['fraction']:.0%}" for row in components if isinstance(row,dict) and isinstance(row.get('fraction'),(int,float)))
        self.color_note.setToolTip('Доли анализируемых пикселей: '+details+'\n'+str(measurement.get('limitation') or 'Цвет по фото ориентировочный.'))

    def selected_facies(self) -> str:
        value = self.facies.currentData()
        return "" if value in (None, "", CUSTOM_VALUE) else str(value).strip()

    def _resolve_class(self, label: str, index: str | None = None) -> dict | None:
        if self._uses_primary_catalog:
            return resolve_catalog_class(label, index, self._catalog_classes)
        return resolve_facies_class(label, index)

    def _filter_facies(self, text: str) -> None:
        query = text.strip().casefold()
        for index in range(self.facies.count()):
            self.facies.view().setRowHidden(index, bool(query and query not in self.facies.itemText(index).casefold()))

    def _apply_suggestions(self, suggestions: dict) -> None:
        for field, value in suggestions.items():
            if field == "Название фации":
                if self.selected_facies():
                    continue
                index = self.facies.findData(value)
                resolved = self._resolve_class(str(value))
                if index < 0 and resolved:
                    index = self.facies.findData(resolved['code'])
                if index < 0:
                    for candidate in range(self.facies.count()):
                        if self.facies.itemText(candidate).casefold() == str(value).casefold():
                            index = candidate; break
                if index < 0:
                    self.facies.insertItem(self.facies.count()-1, str(value) + " · подсказка модели", str(value))
                    index = self.facies.count()-2
                if index >= 0:
                    self.facies.setCurrentIndex(index); self.facies_assumed.setChecked(True)
                continue
            combo = self.attribute_combos.get(field)
            if combo is None or combo.currentData():
                continue
            index = combo.findData(value)
            if index < 0:
                combo.insertItem(combo.count() - 1, str(value), str(value))
                index = combo.count() - 2
            combo.setCurrentIndex(index)
            self.assumption_checks[field].setChecked(True)

    def selected_lithology(self) -> str:
        """Compatibility alias for code that used the old dialog API."""
        return self.selected_facies()

    def _observations(self) -> dict[str, str]:
        values = {
            field: str(combo.currentData() or "")
            for field, combo in self.attribute_combos.items()
            if combo.currentData() not in {None, "", CUSTOM_VALUE}
        }
        values.update({
            "Гидродинамический режим": self.hydrodynamic_regime.text().strip(),
            "Фациальная ассоциация": self.vlm_association.text().strip(),
            "Обстановка седиментации": self.vlm_environment.text().strip(),
        })
        return values

    def _assumed_fields(self) -> list[str]:
        fields = [field for field, check in self.assumption_checks.items()
                  if check.isChecked() and self.attribute_combos[field].currentData() not in {None, "", CUSTOM_VALUE}]
        if self.facies_assumed.isChecked() and self.selected_facies():
            fields.append("Название фации")
        fields.extend(field for field, (edit, initial) in self._automatic_context.items() if edit.text().strip() == initial.strip())
        return fields

    def _compose(self):
        try:
            top, base = self.selected_depth_range()
        except ValueError:
            top, base = None, None
        metadata = self.facies.currentData(FACIES_METADATA_ROLE) or {}
        facies = str(metadata.get("Название фации") or self.selected_facies())
        return compose_description(self._observations(), top, base, facies_name=facies,
                                   assumptions=self._assumed_fields())

    def _refresh_description(self) -> None:
        draft = self._compose()
        automatic = self.template_mode.isChecked()
        self.vlm_description.setReadOnly(automatic)
        if automatic:
            if self.vlm_description.toPlainText().strip() != draft.text:
                self.description_confirmed.setChecked(False)
            self.vlm_description.blockSignals(True)
            self.vlm_description.setPlainText(draft.text)
            self.vlm_description.blockSignals(False)
        self.description_mode_note.setText(
            "Текст обновляется при выборе признаков. Для свободной правки снимите отметку выше."
            if automatic else "Ручной текст сохранится. Изменение признаков его не перезапишет."
        )
        unresolved = any(value == PHOTO_UNRESOLVED for value in self._observations().values())
        missing = ("Нет определения по фото: " if unresolved else "Не заполнены: ") + ", ".join(draft.missing) + "." if draft.missing else "Основные признаки заполнены."
        try:
            self.selected_depth_range()
        except ValueError as exc:
            missing = str(exc) + ". " + missing
        self.missing_fields.setText(missing + " Пустое поле не означает отсутствие признака.")
        hints = review_description(self._observations())
        self.review_notes.setPlainText("\n\n".join(hint.message for hint in hints)
                                      or "По заполненным признакам противоречий из набора правил не найдено. Это не подтверждает правильность определения.")
        self.preview_tabs.setTabText(1, f"Проверка ({len(hints)})" if hints else "Проверка и фото")
        for field, button in self._evidence_buttons.items():
            value = observation(self._observations().get(field))[0]
            entry = self._evidence.get(field)
            valid = valid_evidence(entry, value, self._preview_digest)
            reviewed = valid and entry.get("reviewed") is True and field not in self._assumed_fields()
            button.setText("Учебный фрагмент ✓" if reviewed else "Фрагмент сохранён" if valid else "Фрагмент признака…")
        if self._similar_signature is not None and self._similar_signature != self._query_signature():
            self.examples_status.setText("Признаки изменены. Обновите поиск примеров; прежние результаты не применяются к новым значениям.")
            self.facies_candidates.clear()

    def _description_inputs_changed(self, *_args) -> None:
        self.description_confirmed.setChecked(False)
        self._refresh_description()

    def _toggle_template_mode(self, automatic: bool) -> None:
        if automatic:
            self._manual_description = self.vlm_description.toPlainText()
        else:
            self.vlm_description.blockSignals(True)
            self.vlm_description.setPlainText(self._manual_description)
            self.vlm_description.blockSignals(False)
        self._description_inputs_changed()

    def _manual_description_changed(self) -> None:
        if not self.template_mode.isChecked():
            self._manual_description = self.vlm_description.toPlainText()
            self.description_confirmed.setChecked(False)

    def selected_attributes(self) -> dict[str, str]:
        selected_reference = self.facies.currentData(FACIES_METADATA_ROLE)
        resolved = self._resolve_class(self.selected_facies())
        values = dict(selected_reference) if isinstance(selected_reference, dict) else dict(resolved["metadata"]) if resolved else {}
        values.setdefault("Код фации", self.selected_facies())
        values.update(self._preserved_attributes)
        if '__grain_measurement' in values:values['__grain_measurement_stale']='1' if self._grain_measurement_stale else '0'
        # Catalogue lithotypes/criteria never become measured properties.
        for field in LITHOLOGY_ATTRIBUTE_OPTIONS:
            values.pop(field, None)
        values.update(self._observations())
        values["__vlm_facies_association"] = self.vlm_association.text().strip()
        values["__vlm_depositional_environment"] = self.vlm_environment.text().strip()
        values["__vlm_description"] = self.vlm_description.toPlainText().strip()
        values["__vlm_hydrodynamic_regime"] = self.hydrodynamic_regime.text().strip()
        values["Гидродинамический режим"] = values["__vlm_hydrodynamic_regime"]
        # Update all readers (including old project and Excel aliases), not
        # just geometry-only intervals. Clearing a field must clear its alias.
        values["Краткое описание"] = values["__vlm_description"]
        values["Фациальная ассоциация"] = values["__vlm_facies_association"]
        values["Обстановка седиментации"] = values["__vlm_depositional_environment"]
        values["__vlm_lithology"] = values.get("Название породы", "")
        values["__vlm_texture"] = values.get("Текстура", "")
        values["Название фации"] = str((selected_reference or {}).get("Название фации") or self.selected_facies() or "Не определена")
        values["__vlm_facies_name"] = values["Название фации"]
        values[ASSUMPTIONS_KEY] = json.dumps(self._assumed_fields(), ensure_ascii=False)
        if values.get('Цвет') and values['Цвет'] != PHOTO_UNRESOLVED:
            values['__color_expert_confirmed']='0' if 'Цвет' in self._assumed_fields() else '1'
        elif self._color_user_touched:
            values['__color_expert_confirmed']='1'
        values["__description_mode"] = "template" if self.template_mode.isChecked() else "manual"
        values["__description_composer"] = SCHEMA
        values["__description_manual_text"] = self._manual_description
        values["__description_template_text"] = self._compose().text
        values[EVIDENCE_KEY] = json.dumps(self._current_evidence(), ensure_ascii=False)
        values["__description_review"] = json.dumps([hint.to_dict() for hint in review_description(self._observations())], ensure_ascii=False)
        values["Источник описания"] = ("шаблон по выбранным признакам участка керна"
                                      if self.template_mode.isChecked() else "ручной текст карточки участка")
        self.settings.setValue("last/facies", self.selected_facies())
        saturation = str(values.get("Флюидонасыщение") or "")
        if saturation:
            self.settings.setValue("last/saturation", saturation)
        return values

    def _current_evidence(self):
        result = {}
        observations = self._observations()
        for field, entry in self._evidence.items():
            value = observation(observations.get(field))[0]
            if field in EVIDENCE_FIELDS and valid_evidence(entry, value, self._preview_digest):
                result[field] = dict(entry, reviewed=entry.get("reviewed") is True and field not in self._assumed_fields())
        return result

    def _edit_feature_evidence(self, field):
        value = observation(self._observations().get(field))[0]
        entry = self._current_evidence().get(field)
        dialog = FeatureEvidenceDialog(self._preview_pixmap, field, value, entry,
                    assumed=field in self._assumed_fields(), parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if dialog.result_entry is None:
                self._evidence.pop(field, None)
            else:
                self._evidence[field] = dialog.result_entry
            self.description_confirmed.setChecked(False)
            self._refresh_description()

    def _show_zoom(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Выбранный участок керна · колесо — масштаб, перетаскивание — перемещение")
        dialog.resize(800,700)
        layout = QVBoxLayout(dialog)
        view = EvidenceView(self._preview_pixmap)
        view.selection_mode = False
        view.setDragMode(view.DragMode.ScrollHandDrag)
        layout.addWidget(view)
        dialog.show(); view.fit_photo(); dialog.exec()

    def _build_review_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        self.preview_tabs.addTab(page, "Проверка и фото")
        quality_messages=[str(row.get('message','')) for row in self._quality_report.get('issues',[]) if isinstance(row,dict)]
        self.quality_summary=QLabel('\n'.join(quality_messages) or ('Фотографических предупреждений не найдено (эвристическая оценка).' if self._quality_report else 'Качество фото ещё не проверено. Проверку можно запустить из очереди проверки керна.'))
        self.quality_summary.setWordWrap(True);layout.addWidget(self.quality_summary)
        if self._quality_report.get('scope')=='interval' and not self._preview_pixmap.isNull():
            labels=QPushButton(f"Показать исключённые этикетки ({len(self._quality_report.get('label_boxes',[]))})…")
            labels.clicked.connect(self._show_quality_exclusions);layout.addWidget(labels)
        self.review_notes = QPlainTextEdit(); self.review_notes.setReadOnly(True)
        self.review_notes.setObjectName("descriptionReviewNotes")
        layout.addWidget(self.review_notes,1)
        note = QLabel("Сравнение трёх участков вдоль длинной оси изображения. Различия цвета и полос — повод проверить неоднородность, а не определение фации или слоистости.")
        note.setWordWrap(True); layout.addWidget(note)
        strips = QHBoxLayout()
        if not self._preview_pixmap.isNull():
            photo = self._preview_pixmap; side = min(photo.width(),photo.height())
            for index, fraction in enumerate((0.,.5,1.),1):
                offset = round((max(photo.width(),photo.height())-side)*fraction)
                tile = photo.copy(0 if photo.height()>=photo.width() else offset,
                                  offset if photo.height()>=photo.width() else 0, side,side)
                box=QVBoxLayout(); label=QLabel()
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setPixmap(tile.scaled(110,110,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
                box.addWidget(label); box.addWidget(QLabel(f"Участок {index}"));strips.addLayout(box)
        layout.addLayout(strips)
        zoom = QPushButton("Рассмотреть участок крупно…")
        zoom.setEnabled(not self._preview_pixmap.isNull()); zoom.clicked.connect(self._show_zoom)
        layout.addWidget(zoom)
        reminder = QLabel("Кнопки «Фрагмент признака…» у слоистости, текстуры и биотурбации позволяют выделить и отдельно подтвердить учебную область.")
        reminder.setWordWrap(True);layout.addWidget(reminder)

    def _show_quality_exclusions(self):
        from app.ui.dialogs.photo_quality_dialog import PhotoQualityDialog
        PhotoQualityDialog(self._preview_pixmap,self._quality_report,self).exec()

    def _editor_signature(self):
        return json.dumps([self._observations(),self._assumed_fields(),self.selected_facies(),
                           self.facies.currentIndex(),self.depth_from.text(),self.depth_to.text(),
                           self.vlm_description.toPlainText(),self.template_mode.isChecked(),
                           self.training_confirmed.isChecked(),self.description_confirmed.isChecked(),
                           self._current_evidence()],ensure_ascii=False,sort_keys=True)

    def has_unsaved_changes(self):
        return self._editor_signature()!=self._opening_signature

    def _build_examples_tab(self):
        page=QWidget();layout=QVBoxLayout(page);self.preview_tabs.addTab(page,"Примеры")
        self.examples_status=QLabel("Поиск использует только примеры, отдельно подтверждённые специалистом. Сходство изображений не является вероятностью правильного определения.")
        self.examples_status.setWordWrap(True);layout.addWidget(self.examples_status)
        self.search_examples=QPushButton("Найти похожие проверенные примеры")
        self.search_examples.setEnabled(not self._preview_pixmap.isNull())
        self.search_examples.clicked.connect(self._search_examples);layout.addWidget(self.search_examples)
        self.examples_list=QListWidget();self.examples_list.setMinimumHeight(100)
        self.examples_list.currentRowChanged.connect(self._show_example);layout.addWidget(self.examples_list,1)
        self.example_description=QPlainTextEdit();self.example_description.setReadOnly(True);layout.addWidget(self.example_description,1)
        self.facies_candidates=QPlainTextEdit();self.facies_candidates.setReadOnly(True)
        self.facies_candidates.setPlaceholderText("Варианты фаций появятся после накопления проверенных примеров с совпадением нескольких диагностических признаков.")
        layout.addWidget(self.facies_candidates,1)
        export=QPushButton("Подготовить учебные фрагменты…");export.clicked.connect(self._export_examples);layout.addWidget(export)

    def _query_signature(self):
        return json.dumps([self._observations(),self._assumed_fields()],sort_keys=True,ensure_ascii=False)

    def _search_examples(self):
        from app.infrastructure.attribute_tasks import submit
        self.search_examples.setEnabled(False);self.examples_status.setText("Ищу подтверждённые примеры в локальной базе…")
        self._similar_signature=self._query_signature()
        submit("similar", {"image_png":png_bytes(self._preview_pixmap),"exclude_id":self._example_id,
                           "attributes":self._observations(),"assumptions":self._assumed_fields(),
                           "query_signature":self._similar_signature}, self._examples_ready)

    def _examples_ready(self, reply):
        self.search_examples.setEnabled(True)
        if not reply["ok"]:
            self.examples_status.setText("Не удалось прочитать базу: "+reply["error"]);return
        result=reply['result']
        if result['query_signature']!=self._query_signature():
            self.examples_status.setText("Признаки изменились во время поиска. Запустите поиск ещё раз.");return
        self._similar_rows=result['similar'];self.examples_list.clear()
        for row in self._similar_rows:
            item=QListWidgetItem(f"{row['well']} · {row.get('facies_name') or 'фация не подтверждена'}")
            item.setIcon(QIcon(row['image_path']));self.examples_list.addItem(item)
        self.examples_status.setText(f"Проверенных описаний в базе: {result['reviewed']}. Похожих примеров: {len(self._similar_rows)}. Сходство по цвету изображения, не диагноз.")
        notes=[]
        for candidate in result['candidates']:
            notes.append(f"{candidate['facies']} · {candidate['name']}\nСовпало: {', '.join(candidate['matched'])}.\nОтличается: {', '.join(candidate['conflicts']) or 'нет заполненных различий'}.\nНе хватает: {', '.join(candidate['missing']) or 'основные признаки заполнены'}.\nЭто вариант для проверки, не назначенная фация.")
        self.facies_candidates.setPlainText("\n\n".join(notes) or "Недостаточно подтверждённых примеров фаций с совпадением нескольких признаков. По одной породе фация не назначается.")
        if self._similar_rows:self.examples_list.setCurrentRow(0)
        else:self.example_description.clear()

    def _show_example(self,index):
        self.example_description.setPlainText(self._similar_rows[index]['description'] if 0<=index<len(self._similar_rows) else "")

    def _export_examples(self):
        from datetime import datetime
        from pathlib import Path
        from app.infrastructure.attribute_tasks import submit
        folder=QFileDialog.getExistingDirectory(self,"Куда сохранить проверенные учебные фрагменты")
        if not folder:return
        destination=Path(folder)/("Признаки_керна_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        submit("export",{"destination":destination},self._export_ready)

    def _export_ready(self,reply):
        if not reply['ok']:
            QMessageBox.warning(self,"Учебные фрагменты",reply['error']);return
        report=reply['result']
        QMessageBox.information(self,"Учебные фрагменты",f"Подготовлено фрагментов: {report['examples']}.\nЭто подготовка данных, не выполненное обучение. Для обучения нужны несколько независимых скважин и разные проверенные классы каждого признака.")

    def _choose_custom_facies(self, index: int) -> None:
        if self.facies.itemData(index) != CUSTOM_VALUE:
            return
        code, accepted = QInputDialog.getText(self, "Своя фация", "Введите код фации:")
        code = code.strip()
        if not accepted or not code:
            self.facies.setCurrentIndex(0 if self.facies.count() > 1 else -1)
            return
        existing = self.facies.findData(code)
        if existing < 0:
            self.facies.insertItem(self.facies.count() - 1, code, code)
            existing = self.facies.count() - 2
        self.facies.setCurrentIndex(existing)

    def _update_facies_reference(self, _index: int = -1) -> None:
        metadata = self.facies.currentData(FACIES_METADATA_ROLE)
        if not isinstance(metadata, dict):
            self.facies_reference.setText("Фация не выбрана из справочника. Наблюдения можно сохранить без фации.")
            return
        rows = [
            ("Обстановка", metadata.get("Обстановка седиментации")),
            ("Ассоциация", metadata.get("Фациальная ассоциация")),
            ("Энергия", metadata.get("Энергия среды")),
            ("Режим", metadata.get("Гидродинамический режим")),
            ("Литотипы", metadata.get("Предполагаемые литотипы")),
        ]
        self.facies_reference.setText("Справочная характеристика, не измерение по фото:\n" + "\n".join(f"{title}: {value}" for title, value in rows if value))

    def _choose_custom_attribute(self, field_name: str, combo: QComboBox, index: int) -> None:
        if combo.itemData(index) != CUSTOM_VALUE:
            return
        value, accepted = QInputDialog.getText(self, f"{field_name}: свой вариант", "Введите значение:")
        value = value.strip()
        if not accepted or not value:
            combo.setCurrentIndex(0)
            return
        existing = combo.findData(value)
        if existing < 0:
            combo.insertItem(combo.count() - 1, value, value)
            existing = combo.count() - 2
        combo.setCurrentIndex(existing)

    def selected_depth_range(self) -> tuple[float | None, float | None]:
        values = []
        for edit, original_text, original in zip((self.depth_from, self.depth_to), self._initial_depth_texts, self._original_depths):
            values.append(self._parse_depth(str(original)) if edit.text() == original_text and original is not None
                          else self._parse_depth(edit.text()))
        depth_from, depth_to = values
        if (depth_from is None) != (depth_to is None):
            raise ValueError("Укажите обе границы интервала или очистите оба поля")
        if depth_from is not None and depth_to is not None and depth_to <= depth_from:
            raise ValueError("Конец слоя должен быть больше начала")
        return depth_from, depth_to

    @staticmethod
    def _parse_depth(value: str) -> float | None:
        value = str(value or "").strip().replace(",", ".")
        if not value:
            return None
        try:
            result = float(value)
        except ValueError as exc:
            raise ValueError("Глубина должна быть числом") from exc
        if not math.isfinite(result):
            raise ValueError("Глубина должна быть конечным числом")
        return result

    def accept(self) -> None:
        try:
            self.selected_depth_range()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте глубины", str(exc))
            return
        if self.training_confirmed.isChecked() and not self.selected_facies():
            QMessageBox.warning(self, "Не указана фация", "Для обучения выберите фацию. Карточку без фации можно сохранить как черновик.")
            return
        if self.training_confirmed.isChecked() and self.facies_assumed.isChecked():
            QMessageBox.warning(self, "Фация предположительная", "Предположительную фацию можно сохранить как черновик, но не как подтверждённую метку обучения.")
            return
        super().accept()

    @staticmethod
    def _format_depth(value: float) -> str:
        return f"{float(value):.3f}".rstrip("0").rstrip(".")

    def _open_photo_zoom(self):
        from app.ui.dialogs.feature_evidence_dialog import EvidenceView
        from app.ui.dialog_theme import apply_light_card
        dialog = QDialog(self); dialog.setWindowTitle("Фото породы · колесо — увеличение, мышь — перемещение")
        dialog.resize(850, 760); apply_light_card(dialog)
        layout = QVBoxLayout(dialog)
        view = EvidenceView(self._preview_pixmap, dialog)
        view.selection_mode = False
        view.setDragMode(view.DragMode.ScrollHandDrag)
        layout.addWidget(view)
        fit = QPushButton("Показать целиком"); fit.clicked.connect(view.fit_photo); layout.addWidget(fit)
        dialog.show(); view.fit_photo(); dialog.exec()

    def _request_delete(self) -> None:
        self.delete_requested.emit()
        self.reject()
