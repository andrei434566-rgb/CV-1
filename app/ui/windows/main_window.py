from __future__ import annotations

import json
import math
import re
import shutil
from collections import deque
from bisect import bisect_left
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import QDir, QPointF, QSettings, QThread, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QCursor, QPolygonF
from PySide6.QtWidgets import QApplication, QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QDockWidget, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QProgressDialog, QPushButton, QSizePolicy, QSpinBox, QStatusBar, QTabWidget, QToolBar, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from app.domain.facies_catalog import facies_metadata, resolve_facies_class
from app.domain.facies_palette import normalize_facies_palette
from app.domain.lithology import default_lithology_palette, normalize_lithology_palette
from app.domain.models import PhotoRecord
from app.infrastructure.io.las_parser import parse_las_file
from app.infrastructure.core_report_export import export_core_description_report
from app.infrastructure.cvat_coco_import import CvatImagesMissingError, import_cvat_coco_zip, import_cvat_coco_zips
from app.infrastructure.excel_core_description import (
    CoreInterval,
    create_automatic_interval_detections,
    create_depth_bound_detections,
    layers_for_photo,
    photo_interval_from_filename,
    read_description_workbook,
    suggest_photo_intervals_from_excel,
    workbook_import_summary,
)
from app.infrastructure.expert_corrections import (
    exact_correction, exact_interval_edit, expert_memory_digest, expert_memory_is_confirmed,
    reassign_project, related_text_corrections, save_correction, save_interval_edit,
)
from app.infrastructure.facies_mapping import facies_mapping_context, save_expert_facies_mapping
from app.infrastructure.facies_postprocess import UNRECOGNIZED_FACIES, has_complete_core_column_coverage, postprocess_detections
from app.infrastructure.image_loading import load_working_pixmap
from app.infrastructure.lithology_attribute_service import LithologyAttributeService
from app.infrastructure.local_chat_intents import hide_unsolicited_sources, needs_local_knowledge, needs_selected_image, quick_reply, wants_folder_search, wants_open_datasets, wants_photo_file, wants_sources
from app.infrastructure.local_search_worker import LocalSearchWorker
from app.infrastructure.local_theory_lookup import lookup_theory_answer
from app.infrastructure.local_vlm_client import LocalVlmClient
from app.infrastructure.local_llama_runtime import available_runtime
from app.infrastructure.vlm_training_data import expert_description_digest, expert_description_is_confirmed
from app.infrastructure.cv1_analysis import (
    PIPELINE_MODE, IntervalJob, apply_interval_description, create_core_column_description_detections,
    mark_interval_failure, write_interval_crop,
)
from app.infrastructure.ml.fine_tune_worker import FineTuneWorker
from app.infrastructure.ml.core_column_service import (
    CoreColumnRecognizer,
    CoreColumnTrainingWorker,
    MIN_COLUMN_TRAINING_PHOTOS,
    export_core_column_dataset,
    normalize_columns,
)
from app.infrastructure.ml.chat_photo_worker import ChatPhotoColumnWorker
from app.infrastructure.ml.yolo_model_service import SegmentationWorker
from app.infrastructure.pdf_photo_import import render_pdf_pages
from app.infrastructure.photo_caption_ocr import read_caption_metadata
from app.infrastructure.primary_facies_catalog import (
    SNAPSHOT_NAME, load_catalog_snapshot, read_primary_facies_workbook,
    resolve_catalog_class, save_catalog_snapshot,
)
from app.infrastructure.project_storage import MANIFEST_NAME, load_project, save_project
from app.infrastructure.training_dataset import MIN_TRAINING_SAMPLES, automatic_samples_count, export_training_dataset, unlabeled_manual_samples_count, verified_samples_count
from app.infrastructure.training_quality import review_queue, training_quality
from app.infrastructure.training_recommendation import TrainingRecommendation, recommend_training
from app.runtime_paths import application_root, bundled_root, user_data_root
from app.ui.dialogs.depth_range_dialog import DepthRangeDialog
from app.ui.dialogs.excel_report_inputs_dialog import ExcelReportInputsDialog
from app.ui.dialogs.core_columns_dialog import CoreColumnsDialog
from app.ui.dialogs.facies_dialog import FaciesDialog
from app.ui.dialogs.model_preparation_dialog import ModelPreparationDialog
from app.ui.dialogs.log_editor_dialog import LogEditorDialog
from app.ui.dialogs.lithology_palette_dialog import FaciesPaletteDialog, LithologyPaletteDialog
from app.ui.dialogs.local_vlm_settings_dialog import LocalVlmSettingsDialog
from app.ui.dialogs.training_center_dialog import TrainingCenterDialog
from app.ui.dialogs.open_datasets_dialog import OpenDatasetsDialog
from app.ui.dialogs.photo_interval_dialog import PhotoIntervalDialog
from app.ui.dialogs.photo_interval_mapping_dialog import PhotoIntervalMappingDialog
from app.ui.dialogs.well_depth_dialog import WellDepthDialog
from app.ui.widgets.workspace_canvas import StackColumnItem, WorkspaceCanvas, facies_color
from app.ui.widgets.assistant_panel import AssistantPanel


class MainWindow(QMainWindow):
    """Kern Analyzer board built with the PySide patterns of the original app."""

    TREE_KIND_ROLE = Qt.ItemDataRole.UserRole
    TREE_VALUE_ROLE = Qt.ItemDataRole.UserRole + 1
    TREE_INDEX_ROLE = Qt.ItemDataRole.UserRole + 2
    PHOTO_LOAD_BATCH_SIZE = 6

    def __init__(self):
        super().__init__()
        self._records: list[PhotoRecord] = []
        self._expert_session_id = uuid4().hex
        self._edit_autosave = QTimer(self)
        self._edit_unsaved = False
        self._edit_autosave.setSingleShot(True)
        self._edit_autosave.setInterval(1200)
        self._edit_autosave.timeout.connect(self._flush_edit_autosave)
        self._segmentation_thread: QThread | None = None
        self._segmentation_worker: SegmentationWorker | None = None
        self._segmentation_failed = False
        self._training_thread: QThread | None = None
        self._training_worker: FineTuneWorker | None = None
        self._column_training_thread: QThread | None = None
        self._column_training_worker: CoreColumnTrainingWorker | None = None
        self._column_training_succeeded_model: Path | None = None
        self._training_epoch_current = 0
        self._training_epoch_total = 0
        self._fine_tune_success_model: Path | None = None
        self._activity_dialog: QProgressDialog | None = None
        self._segmentation_updated_count = 0
        self._deleted_records: list[tuple[PhotoRecord, int, object]] = []
        self._queued_records: list[PhotoRecord] = []
        self._pending_photo_imports: deque[tuple[Path, str, float | None, float | None]] = deque()
        self._facies_dialogs: list[FaciesDialog] = []
        self._settings = QSettings("Kern Analyzer", "KernAnalyzer")
        self._primary_facies_catalog = self._load_global_primary_catalog()
        saved_facies_model = str(self._settings.value("facies/model_path", "") or "")
        self._selected_model_path: Path | None = Path(saved_facies_model) if saved_facies_model else None
        self._missing_model_notice: str | None = None
        self._gis_data: dict | None = None
        self._rigis_data: dict | None = None
        self._lithology_palette = self._read_lithology_palette()
        StackColumnItem.set_lithology_palette(self._lithology_palette)
        self._saved_facies_palette = self._read_facies_palette()
        self._facies_palette: list[dict[str, str]] = []
        StackColumnItem.set_facies_palette([])
        saved_column_model = str(self._settings.value("core_columns/model_path", "") or "")
        self._selected_column_model_path: Path | None = Path(saved_column_model) if saved_column_model else None
        self._confidence_threshold = self._read_confidence_threshold()
        self._segmentation_profile = self._read_segmentation_profile()
        self._well_logs: dict[str, dict[str, dict]] = {}
        self._empty_wells: list[str] = []
        self._well_order: list[str] = []
        self._well_depth_ranges: dict[str, tuple[float, float]] = {}
        self._well_depth_references: dict[str, str] = {}
        self._well_depth_settings: dict[str, dict] = {}
        self._well_intervals: dict[str, tuple[float, float]] = {}
        self._well_interpretations: dict[str, list[dict]] = {}
        self._well_report_inputs: dict[str, dict] = {}
        self._active_well_name = str(self._settings.value("last/active_well", "Скважина 1"))
        self._project_folder: Path | None = None
        self._project_title = "Новый проект"
        self._vlm_target: tuple[PhotoRecord, object | None] | None = None
        self._vlm_pending: tuple[str, str, object | None, str] | None = None
        self._vlm_pending_meta: dict | None = None
        self._vlm_chat_queue: deque[dict] = deque()
        self._vlm_dispatch_scheduled = False
        self._vlm_closing = False
        self._selected_description_job: IntervalJob | None = None
        self._vlm_history: list[dict[str, str]] = []
        self._analysis_queue: deque[IntervalJob] = deque()
        self._analysis_pending: tuple[str, IntervalJob | list[IntervalJob], Path | list[Path]] | None = None
        self._analysis_active = False
        self._analysis_total = self._analysis_done = self._analysis_success = 0
        self._analysis_geometry_issues: list[dict] = []
        self._chat_photo_path: Path | None = None
        self._chat_record_ids: dict[str, str] = {}
        self._chat_pending_photos: list[Path] = []
        self._chat_pending_excel_path: Path | None = None
        self._chat_document_path: Path | None = None
        self._chat_photo_columns: dict | None = None
        self._chat_photo_question = ""
        self._chat_photo_history: list[dict[str, str]] = []
        self._chat_photo_histories: dict[str, list[dict[str, str]]] = {}
        self._chat_photo_thread: QThread | None = None
        self._chat_photo_worker: ChatPhotoColumnWorker | None = None
        self._local_search_worker: LocalSearchWorker | None = None
        self._theory_import_thread = None
        self._theory_import_queue: list[str] = []

        self.setWindowTitle("Kern Analyzer")
        self.setMinimumSize(960, 640)
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("ЛКМ — переместить фото · зажатое колёсико — двигать доску · колёсико — масштаб")

        self.workspace = WorkspaceCanvas(self)
        self.workspace.setObjectName("mainWorkspace")
        self._connect_workspace(self.workspace, source_photos=True)
        self.sheets = QTabWidget(self)
        self.sheets.setObjectName("workspaceSheets")
        self.sheets.setDocumentMode(True)
        self.sheets.setTabPosition(QTabWidget.TabPosition.South)
        self.sheets.addTab(self.workspace, "\u041a\u0435\u0440\u043d \u0438 \u043e\u043f\u0438\u0441\u0430\u043d\u0438\u0435")
        # Пока это единственная рабочая область, нижняя вкладка только
        # дублирует очевидное содержимое и занимает полезное место.
        self.sheets.tabBar().hide()
        self._workspace_frame = QFrame(self)
        self._workspace_frame.setObjectName("workspaceFrame")
        workspace_layout = QVBoxLayout(self._workspace_frame)
        workspace_layout.setContentsMargins(9, 9, 9, 9)
        workspace_layout.addWidget(self.sheets)
        self.setCentralWidget(self._workspace_frame)
        self._build_ui()
        self._build_project_sidebar()
        self._build_visual_shell()
        self._show_remembered_model_status()

    def closeEvent(self, event) -> None:
        if self._theory_import_queue or (self._theory_import_thread is not None and self._theory_import_thread.isRunning()):
            self.assistant_panel.append_message("system", "Дождитесь окончания чтения документов перед закрытием приложения.")
            event.ignore()
            return
        for name in ("_segmentation_thread", "_training_thread", "_column_training_thread"):
            thread = getattr(self, name, None)
            if thread is not None and thread.isRunning():
                self.assistant_panel.append_message("system", "Дождитесь окончания обработки или обучения перед закрытием приложения.")
                event.ignore()
                return
        if self._chat_photo_thread is not None and self._chat_photo_thread.isRunning():
            self.assistant_panel.append_message("system", "Дождитесь завершения поиска столбиков перед закрытием приложения.")
            event.ignore()
            return
        if self._local_search_worker is not None and self._local_search_worker.isRunning():
            self.assistant_panel.append_message("system", "Дождитесь завершения поиска в папках перед закрытием приложения.")
            event.ignore()
            return
        if (self._edit_autosave.isActive() or self._edit_unsaved) and not self._flush_edit_autosave():
            event.ignore()
            return
        if hasattr(self, "_vlm_client"):
            self._vlm_closing = True
            self._analysis_active = False
            self._analysis_queue.clear()
            if self._analysis_pending is not None:
                crops = self._analysis_pending[2]
                for crop in crops if isinstance(crops, list) else [crops]:
                    crop.unlink(missing_ok=True)
                self._analysis_pending = None
            for request in self._vlm_chat_queue:
                self._cleanup_vlm_image(request.get("temporary_image"))
            self._vlm_chat_queue.clear()
            if self._vlm_pending_meta is not None:
                self._cleanup_vlm_image(self._vlm_pending_meta.get("temporary_image"))
                self._vlm_pending_meta = None
            self._vlm_client.stop()
        self._cleanup_vlm_image()
        super().closeEvent(event)

    def _connect_workspace(self, workspace: WorkspaceCanvas, source_photos: bool = False) -> None:
        """Connect the core-description workspace to the project data model."""
        workspace.facies_selected.connect(self._begin_polygon_edit)
        workspace.facies_context_requested.connect(self._open_facies_editor)
        workspace.facies_geometry_changed.connect(self._on_facies_geometry_changed)
        workspace.facies_created.connect(self._add_manual_contour)
        workspace.column_paint_requested.connect(self._paint_stack_column)
        workspace.interpretation_interval_created.connect(self._create_interpretation_interval)
        workspace.interpretation_interval_resized.connect(self._resize_interpretation_interval)
        workspace.interpretation_interval_context_requested.connect(self._open_interpretation_interval_menu)
        workspace.well_order_changed.connect(self._set_well_order_from_canvas)
        if source_photos:
            workspace.photo_deleted.connect(self._remove_photo)
            workspace.photo_selected.connect(self._select_photo_for_assistant)

    def _select_photo_for_assistant(self, record: PhotoRecord) -> None:
        self._vlm_target = (record, None)
        self._active_well_name = record.well_name
        self._update_header_context()

    def _build_ui(self) -> None:
        toolbar = QToolBar("Главная панель", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        title = QLabel("  KERN ANALYZER  ")
        title.setObjectName("appTitle")
        toolbar.addWidget(title)
        toolbar.addSeparator()
        toolbar.addWidget(
            self._create_toolbar_menu(
                "Файл",
                [
                    ("Новый проект", self.new_project, "Очистить рабочую область и начать новый проект", "Ctrl+N"),
                    ("Открыть проект…", self.open_project_dialog, "Открыть папку сохранённого проекта", "Ctrl+Shift+O"),
                    ("Сохранить проект", self.save_project, "Сохранить текущий проект", "Ctrl+S"),
                    ("Сохранить проект как…", self.save_project_as, "Сохранить проект в новой папке", "Ctrl+Shift+S"),
                    ("Загрузить фото/PDF…", self.open_images, "Выбрать изображения или PDF; каждая страница PDF станет отдельным фото", "Ctrl+O"),
                    ("Автоматически определить интервалы на фото…", self.detect_photo_intervals, "Выделить на фото устойчивые визуальные интервалы и заполнить строки без назначения фаций"),
                    ("Импорт разметки CVAT (COCO ZIP)…", self.import_cvat_coco, "Создать проект из фото, контуров и классов из выгрузки CVAT"),
                    ("Импорт нескольких CVAT ZIP…", self.import_cvat_coco_batch, "Последовательно объединить несколько COCO ZIP с защитой памяти и дубликатов"),
                    ("Импорт Excel + JPG для обучения…", self.import_excel_photo_batch, "Сопоставить интервалы описания Excel с подписями на фото керна"),
                    ("Импорт Excel + выбранные фото…", self.import_excel_photo_files, "Выбрать один или несколько снимков и сопоставить их с интервалами Excel"),
                    ("Заполнить паспорт итогового Excel…", self.configure_excel_report_inputs, "Внести поля специалиста для выбранной скважины"),
                    ("Сохранить послойное описание (Excel/Word/PDF)…", self.export_core_description_report, "Сформировать послойное описание по вкладке «Керн и описание»"),
                ],
                "Работа с файлами",
            )
        )
        toolbar.addWidget(
            self._create_toolbar_menu(
                "Столбики",
                [
                    ("Проверить границы на фото…", self.configure_core_columns, "Исправить и подтвердить обучающую разметку столбиков"),
                    ("Переходы между фациями…", self.review_visual_contacts, "Найти неровные визуальные контакты, проверить глубины и при желании создать черновики масок"),
                    ("Выбрать модель столбиков…", self.select_core_column_model, "Выбрать отдельную YOLO-модель класса core_column"),
                    ("Обучить модель столбиков…", self.start_core_column_training, "Обучить отдельный первый этап на подтверждённых границах"),
                ],
                "Отдельный этап 1: поиск и обучение столбиков керна",
            )
        )
        toolbar.addWidget(
            self._create_toolbar_menu(
                "Модель",
                [
                    ("Подготовка модели и перенос…", self.show_model_preparation, "Справочник фаций, покрытие классов, проверка и экспорт без подключения модели"),
                    ("Загрузить основной справочник фаций из Excel…", self.import_primary_facies_catalog, "Использовать классы из своего Excel для следующего обучения сегментации"),
                    ("Выбрать модель сегментации…", self.select_detection_model, "Выбрать файл модели .pt"),
                    ("Переразметить фации выбранной моделью…", self.resegment_facies_with_selected_model, "Заменить текущие фациальные маски новым прогнозом"),
                    ("История дообучений…", self.show_training_history, "Сравнить сохранённые модели по validation-метрикам"),
                    ("Проверить датасет перед обучением…", self.show_training_quality, "Проверить малые классы, дубликаты и размер масок"),
                    ("Порог уверенности масок…", self.select_confidence_threshold, "Ниже порог — больше масок для ручной проверки"),
                    ("Режим сегментации…", self.select_segmentation_profile, "Быстрый, стандартный или детальный режим обработки фото"),
                    ("Проверить сомнительные маски…", self.review_uncertain_masks, "Открыть очередь автоматических масок с низкой уверенностью"),
                    ("Подобрать атрибуты пород…", self.suggest_lithology_attributes, "Отдельно от фаций предложить цвет, зернистость и слоистость внутри масок"),
                    ("Настроить литологическую палетку…", self.configure_lithology_palette, "Свои названия пород, цвета, условные знаки и узоры"),
                    ("Настроить палетку фаций…", self.configure_facies_palette, "Классы и цвета из масок загруженных фотографий"),
                ],
                "Модель детекции и сегментации",
            )
        )
        self._new_contour_action = QAction("Новый контур", self)
        self._new_contour_action.setCheckable(True)
        self._new_contour_action.setToolTip("Нарисовать контур вручную на фотографии")
        self._new_contour_action.toggled.connect(self._set_new_contour_mode)
        toolbar.addAction(self._new_contour_action)
        add_vertex = QAction("＋ Точка", self)
        add_vertex.setToolTip("Добавить вершину в выбранный контур: нажмите кнопку, затем щёлкните по контуру")
        add_vertex.triggered.connect(self._prepare_polygon_vertex_insert)
        toolbar.addAction(add_vertex)
        delete_vertex = QAction("− Точка", self)
        delete_vertex.setToolTip("Удалить выделенную вершину контура (также работает клавиша Delete)")
        delete_vertex.triggered.connect(self._delete_selected_polygon_vertex)
        toolbar.addAction(delete_vertex)
        finish_polygon_edit = QAction("✓ Готово", self)
        finish_polygon_edit.setToolTip("Закончить редактирование контура (Esc)")
        finish_polygon_edit.triggered.connect(self._finish_polygon_edit)
        toolbar.addAction(finish_polygon_edit)
        back_to_photos = QAction("К фото", self)
        back_to_photos.setToolTip("Вернуться к исходным фотографиям с разметкой")
        back_to_photos.setShortcut("Home")
        back_to_photos.triggered.connect(self.return_to_photos)
        toolbar.addAction(back_to_photos)

        # Keep training controls on their own toolbar row.  The former layout
        # appended them to an already full row, so Qt moved them into the
        # overflow popup on ordinary laptop-sized windows.
        training_toolbar = QToolBar("Обучение фаций", self)
        training_toolbar.setObjectName("trainingToolbar")
        training_toolbar.setMovable(False)
        training_toolbar.setFloatable(False)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, training_toolbar)
        training_toolbar.addWidget(self._create_training_bar())
        self._register_shortcuts()

        self.setStyleSheet(
            """
            QMainWindow { background: #fbfcfe; }
            QTabWidget#workspaceSheets::pane { border: none; background: #fbfcfe; }
            QTabWidget#workspaceSheets::tab-bar { alignment: left; }
            QTabBar::tab { background: #eef1f7; color: #596176; border: 1px solid #d9deea; border-bottom: none; padding: 8px 18px; margin-right: 2px; min-width: 130px; }
            QTabBar::tab:selected { background: #ffffff; color: #5149ca; font-weight: 700; border-top: 2px solid #655be8; }
            QTabBar::tab:hover { background: #f7f7ff; color: #5149ca; }
            QToolBar#mainToolbar { background: #ffffff; border: none; spacing: 8px; padding: 9px 14px 5px 14px; }
            QToolBar#trainingToolbar { background: #ffffff; border: none; border-bottom: 1px solid #e4e8f0; spacing: 6px; padding: 3px 14px 8px 14px; }
            QLabel#appTitle { color: #35306f; font-weight: 800; letter-spacing: 1px; padding: 0 10px; }
            QToolButton#toolbarMenuButton { color: #262b3d; font-weight: 600; background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 7px 11px; }
            QToolButton#toolbarMenuButton:hover { background: #f0f1ff; border-color: #e4e3ff; color: #554dcc; }
            QToolBar QToolButton { color: #262b3d; font-weight: 600; background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 7px 11px; }
            QToolBar QToolButton:hover, QToolBar QToolButton:checked { background: #f0f1ff; border-color: #e4e3ff; color: #554dcc; }
            QFrame#trainingBar { background: #f5f7ff; border: 1px solid #dde2fb; border-radius: 8px; padding: 2px 4px; }
            QLabel#trainingCount { color: #4a5270; font-size: 12px; font-weight: 600; padding: 2px 5px; }
            QLabel#trainingProgressLabel { color: #554dcc; font-size: 12px; font-weight: 700; padding: 2px 3px; }
            QProgressBar#trainingProgress { min-width: 72px; max-width: 72px; max-height: 10px; border: 1px solid #cfcdf1; border-radius: 5px; background: #ebeafe; }
            QProgressBar#trainingProgress::chunk { background: #554dcc; border-radius: 4px; }
            QSpinBox#trainingEpochs { min-width: 46px; max-width: 64px; padding: 3px; border: 1px solid #cfd4e1; border-radius: 5px; background: #ffffff; color: #34394b; }
            QPushButton#trainButton { color: white; background: #554dcc; border: none; border-radius: 5px; padding: 6px 10px; font-weight: 700; }
            QPushButton#trainButton:hover { background: #433bb5; }
            QPushButton#trainButton:disabled { color: #8d95ad; background: #e2e6f1; }
            QMenu#toolbarDropdown { background: #ffffff; border: 1px solid #dde1eb; border-radius: 8px; padding: 5px; }
            QMenu#toolbarDropdown::item { color: #252a3b; padding: 9px 34px 9px 12px; border-radius: 5px; }
            QMenu#toolbarDropdown::item:selected { background: #efefff; color: #5149ca; }
            QDockWidget { color: #3a4054; font-weight: 700; background: #ffffff; }
            QDockWidget::title { background: #ffffff; padding: 10px 12px; border-bottom: 1px solid #e6e9f0; }
            QTreeWidget#projectTree { border: none; background: #fbfcfe; color: #454b60; padding: 6px; }
            QTreeWidget#projectTree::item { height: 27px; border-radius: 5px; padding: 2px 5px; }
            QTreeWidget#projectTree::item:selected { background: #e9e8ff; color: #4e46bd; }
            QStatusBar { background: #ffffff; border-top: 1px solid #e4e8f0; color: #7c8497; padding-left: 12px; }
            """
        )

    def _read_lithology_palette(self) -> list[dict[str, str]]:
        raw = self._settings.value("lithology/palette_json", "")
        if raw:
            try:
                palette = normalize_lithology_palette(json.loads(str(raw)))
                if palette:
                    return palette
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return default_lithology_palette()

    def _save_lithology_palette(self) -> None:
        self._settings.setValue(
            "lithology/palette_json",
            json.dumps(self._lithology_palette, ensure_ascii=False),
        )
        self._settings.sync()

    def configure_lithology_palette(self) -> None:
        dialog = LithologyPaletteDialog(self._lithology_palette, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        palette = dialog.palette()
        if not palette:
            return
        self._lithology_palette = palette
        self._save_lithology_palette()
        StackColumnItem.set_lithology_palette(palette)
        if self._records or self._empty_wells:
            self._show_stack()
        self.statusBar().showMessage(f"Литологическая палетка сохранена · пород: {len(palette)}")

    def _read_facies_palette(self) -> list[dict[str, str]]:
        raw = self._settings.value("facies/palette_json", "")
        if raw:
            try:
                return normalize_facies_palette(json.loads(str(raw)))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return []

    def _facies_palette_defaults(self) -> list[dict[str, str]]:
        rows = []
        seen: set[str] = set()
        for record in self._records:
            for detection in record.detections:
                label = str(detection.label or "").strip()
                key = label
                if not label or label == "Новый контур" or key in seen:
                    continue
                rows.append(
                    {
                        "key": label,
                        "name": label,
                        "symbol": label[:12],
                        "color": facies_color(label).name(QColor.NameFormat.HexRgb),
                        "pattern": "solid",
                    }
                )
                seen.add(key)
        return normalize_facies_palette(rows)

    def _sync_facies_palette(self) -> None:
        saved = {item["key"]: item for item in self._saved_facies_palette}
        merged = []
        current_keys: set[str] = set()
        for default in self._facies_palette_defaults():
            normalized_key = default["key"]
            current_keys.add(normalized_key)
            custom = saved.get(normalized_key)
            merged.append({**default, **(custom or {}), "key": default["key"]})
        # Keep explicitly added user classes available for manual reassignment,
        # even when the current batch has no prediction of that class yet.
        merged.extend(item for key, item in saved.items() if key not in current_keys)
        self._facies_palette = normalize_facies_palette(merged)
        StackColumnItem.set_facies_palette(self._facies_palette)

    def configure_facies_palette(self) -> None:
        defaults = self._facies_palette_defaults()
        if not defaults:
            QMessageBox.information(
                self,
                "Палетка фаций",
                "Сначала загрузите и распознайте фотографии: список фаций собирается из масок в памяти.",
            )
            return
        self._sync_facies_palette()
        dialog = FaciesPaletteDialog(self._facies_palette, defaults, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        self._saved_facies_palette = dialog.palette()
        self._settings.setValue(
            "facies/palette_json",
            json.dumps(self._saved_facies_palette, ensure_ascii=False),
        )
        self._settings.sync()
        self._sync_facies_palette()
        self._show_stack()
        self.statusBar().showMessage(f"Палетка фаций сохранена · классов из фотографий: {len(self._facies_palette)}")

    def _create_training_bar(self) -> QFrame:
        bar = QFrame(self)
        bar.setObjectName("trainingBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(5, 2, 3, 2)
        layout.setSpacing(5)
        self._training_count_label = QLabel(bar)
        self._training_count_label.setObjectName("trainingCount")
        self._training_epochs_label = QLabel("Макс. эпох:", bar)
        self._training_epochs_label.setObjectName("trainingCount")
        self._training_epochs = QSpinBox(bar)
        self._training_epochs.setObjectName("trainingEpochs")
        self._training_epochs.setRange(1, 500)
        self._training_epochs.setValue(int(self._settings.value("training/epochs", 20)))
        self._training_epochs.setToolTip("Верхний предел эпох; обучение может остановиться раньше по контрольной выборке")
        self._training_epochs.valueChanged.connect(lambda value: self._settings.setValue("training/epochs", value))
        self._training_auto_epochs = QCheckBox("авто", bar)
        self._training_auto_epochs.setChecked(str(self._settings.value("training/auto_epochs", "true")).casefold() != "false")
        self._training_auto_epochs.setToolTip("Автоматически применять рассчитанный безопасный максимум эпох")
        self._training_auto_epochs.toggled.connect(self._on_training_auto_epochs_toggled)
        self._training_recommendation_label = QLabel(bar)
        self._training_recommendation_label.setObjectName("trainingCount")
        self._training_progress_label = QLabel("⏳ Обучение…", bar)
        self._training_progress_label.setObjectName("trainingProgressLabel")
        self._training_progress = QProgressBar(bar)
        self._training_progress.setObjectName("trainingProgress")
        self._training_progress.setRange(0, 0)
        self._training_progress.setTextVisible(False)
        self._training_button = QPushButton("Дообучить", bar)
        self._training_button.setObjectName("trainButton")
        self._training_button.clicked.connect(self.start_fine_tune)
        layout.addWidget(self._training_count_label)
        layout.addWidget(self._training_epochs_label)
        layout.addWidget(self._training_epochs)
        layout.addWidget(self._training_auto_epochs)
        layout.addWidget(self._training_recommendation_label)
        layout.addWidget(self._training_progress_label)
        layout.addWidget(self._training_progress)
        layout.addWidget(self._training_button)
        self._refresh_training_bar()
        return bar

    def _show_activity(self, title: str, label: str, total: int) -> None:
        if self._activity_dialog is None:
            self._activity_dialog = QProgressDialog(self)
            self._activity_dialog.setCancelButton(None)
            self._activity_dialog.setAutoClose(False)
            self._activity_dialog.setAutoReset(False)
            self._activity_dialog.setWindowModality(Qt.WindowModality.NonModal)
            self._activity_dialog.setMinimumDuration(0)
        self._activity_dialog.setWindowTitle(title)
        self._activity_dialog.setLabelText(label)
        self._activity_dialog.setRange(0, max(1, total))
        self._activity_dialog.setValue(0)
        self._activity_dialog.show()

    def _update_activity(self, label: str, current: int, total: int, *, busy: bool = False) -> None:
        if self._activity_dialog is None:
            return
        self._activity_dialog.setLabelText(label)
        if busy:
            # The VLM runs one column at a time and cannot report token-level
            # progress. Show an active indicator instead of a misleading 0%.
            self._activity_dialog.setRange(0, 0)
            return
        self._activity_dialog.setRange(0, max(1, total))
        self._activity_dialog.setValue(max(0, min(current, max(1, total))))

    def _close_activity(self) -> None:
        if self._activity_dialog is not None:
            self._activity_dialog.close()

    def _training_examples_count(self) -> int:
        return verified_samples_count(self._records)

    def _automatic_examples_count(self) -> int:
        return automatic_samples_count(self._records)

    def _unlabeled_manual_examples_count(self) -> int:
        return unlabeled_manual_samples_count(self._records)

    def _refresh_training_bar(self) -> None:
        if not hasattr(self, "_training_count_label"):
            return
        count = self._training_examples_count()
        quality = training_quality(self._records)
        automatic_count = self._automatic_examples_count()
        unlabeled_count = self._unlabeled_manual_examples_count()
        needed = max(0, MIN_TRAINING_SAMPLES - count)
        self._training_count_label.setText(f"Для обучения: {count} · новые: {unlabeled_count} · авто: {automatic_count}")
        self._training_count_label.setToolTip(quality.summary)
        recommendation = self._current_training_recommendation(quality)
        self._training_recommendation_label.setText(
            f"ориентир {recommendation.recommended_epochs} · стоп {recommendation.patience}"
        )
        self._training_recommendation_label.setToolTip(recommendation.details)
        if self._training_auto_epochs.isChecked():
            self._training_epochs.setValue(recommendation.max_epochs)
        is_busy = self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None
        is_training = self._training_thread is not None
        self._training_progress_label.setVisible(is_training)
        self._training_progress.setVisible(is_training)
        if is_training:
            total = max(1, self._training_epoch_total or self._training_epochs.value())
            current = max(0, min(total, self._training_epoch_current))
            self._training_progress_label.setText(
                f"⏳ Эпоха {current} из {total} · осталось {total - current}"
            )
            self._training_progress.setRange(0, total)
            self._training_progress.setValue(current)
        self._training_auto_epochs.setEnabled(not is_busy)
        self._training_epochs.setEnabled(not is_busy and not self._training_auto_epochs.isChecked())
        self._training_button.setEnabled(count >= MIN_TRAINING_SAMPLES and not is_busy)
        if is_busy:
            self._training_button.setToolTip("Дождитесь завершения текущей операции")
        elif needed:
            self._training_button.setToolTip(f"Для дообучения добавьте ещё {needed} вручную проверенных слоёв")
        else:
            self._training_button.setToolTip("Собрать разметки и дообучить копию выбранной модели")

    @staticmethod
    def _recommended_training_epochs(examples: int) -> int:
        """Legacy total-only estimate retained for integrations using this API."""
        if examples < 20:
            return 80
        if examples < 50:
            return 60
        if examples < 100:
            return 45
        if examples < 250:
            return 35
        return 25

    def _on_training_auto_epochs_toggled(self, enabled: bool) -> None:
        self._settings.setValue("training/auto_epochs", bool(enabled))
        self._refresh_training_bar()

    def _current_training_recommendation(self, quality=None) -> TrainingRecommendation:
        return recommend_training(
            self._records,
            known_model_classes=self._known_model_facies(),
            quality=quality,
        )

    def _known_model_facies(self) -> set[str] | None:
        """Read class names cheaply, without loading a YOLO checkpoint in the UI."""
        model_path = self._resolve_model_path()
        if model_path is None:
            return None
        catalog_path = model_path.parent / "facies_catalog.json"
        try:
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return {str(name).strip() for name in catalog if str(name).strip()} if isinstance(catalog, dict) else None

    def _build_project_sidebar(self) -> None:
        sidebar = QDockWidget("ПРОЕКТЫ", self)
        sidebar.setObjectName("projectSidebar")
        self._project_sidebar = sidebar
        sidebar.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        sidebar.setMinimumWidth(230)
        sidebar.setMaximumWidth(330)
        self._sidebar_content = QWidget(sidebar)
        self._sidebar_layout = QVBoxLayout(self._sidebar_content)
        self._sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.project_tree = QTreeWidget(self._sidebar_content)
        self.project_tree.setObjectName("projectTree")
        self.project_tree.setHeaderHidden(True)
        self.project_tree.itemClicked.connect(self._select_tree_context)
        self.project_tree.itemActivated.connect(self._activate_project_tree_item)
        self.project_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.project_tree.customContextMenuRequested.connect(self._open_project_tree_context_menu)
        self._sidebar_layout.addWidget(self.project_tree)
        sidebar.setWidget(self._sidebar_content)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, sidebar)
        self._refresh_project_tree()

    def _build_visual_shell(self) -> None:
        """Present the existing editor in the compact three-pane layout."""
        old_toolbar = self.findChild(QToolBar, "mainToolbar")
        training_toolbar = self.findChild(QToolBar, "trainingToolbar")
        tools_menu = QMenu("Настройки", self)
        tools_menu.setObjectName("shellSettingsMenu")
        contour_actions: list[QAction] = []
        if old_toolbar is not None:
            contour_actions = [
                action for action in old_toolbar.actions()
                if action.text().strip() in {"Новый контур", "＋ Точка", "− Точка", "✓ Готово", "К фото"}
            ]
            old_toolbar.hide()
        if training_toolbar is not None:
            training_toolbar.hide()
            self.removeToolBarBreak(training_toolbar)

        header = QToolBar("Навигация", self)
        header.setObjectName("shellHeader")
        header.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, header)
        brand = QLabel("▏ KERN  ANALYZER")
        brand.setObjectName("shellBrand")
        header.addWidget(brand)
        divider = QLabel("  │  ")
        divider.setObjectName("shellDivider")
        header.addWidget(divider)
        self._header_project = QLabel("Новый проект  /  ")
        self._header_project.setObjectName("shellBreadcrumb")
        header.addWidget(self._header_project)
        self._header_well = QComboBox()
        self._header_well.setObjectName("shellWell")
        self._header_well.activated.connect(self._header_well_changed)
        header.addWidget(self._header_well)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        header.addWidget(spacer)
        self._save_indicator = QLabel("●  Локальный проект")
        self._save_indicator.setObjectName("shellSaveIndicator")
        header.addWidget(self._save_indicator)
        upload = QPushButton("＋ Загрузить фото")
        upload.setObjectName("shellUploadPhoto")
        upload.setToolTip("Загрузить фотографии керна или PDF (Ctrl+O)")
        upload.clicked.connect(self.open_images)
        header.addWidget(upload)
        model_button = QPushButton("◈ Модель сегментации")
        model_button.setObjectName("shellLoadModel")
        model_button.setToolTip("Подключить обученную YOLO-модель сегментации фаций (.pt)")
        model_button.clicked.connect(self.select_detection_model)
        header.addWidget(model_button)
        save = QPushButton("Сохранить")
        save.setObjectName("shellSave")
        save.clicked.connect(self.save_project)
        header.addWidget(save)
        settings = QToolButton()
        settings.setObjectName("shellTools")
        settings.setText("⚙")
        settings.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        settings.setMenu(tools_menu)
        tools_menu.addAction("Новый проект", self.new_project)
        tools_menu.addAction("Открыть проект…", self.open_project_dialog)
        tools_menu.addAction("Загрузить фото/PDF…", self.open_images)
        tools_menu.addAction("Выбрать модель сегментации…", self.select_detection_model)
        tools_menu.addAction("Проверить границы керна…", self.configure_core_columns)
        tools_menu.addAction("Сохранить проект как…", self.save_project_as)
        if contour_actions:
            edit_menu = tools_menu.addMenu("Правка разметки")
            edit_menu.addActions(contour_actions)
        tools_menu.addSeparator()
        self._assistant_visibility_action = tools_menu.addAction("Показать ИИ-ассистента")
        self._assistant_visibility_action.setCheckable(True)
        self._assistant_visibility_action.setChecked(True)
        self._assistant_visibility_action.toggled.connect(self._set_assistant_visible)
        tools_menu.addAction("Настроить помощника и папки…", self.configure_local_vlm)
        tools_menu.addAction("Учебные материалы из открытых источников…", self.show_open_datasets)
        header.addWidget(settings)

        nav_layout = self._sidebar_layout
        nav_layout.setContentsMargins(8, 14, 8, 8)
        nav_layout.setSpacing(4)
        self._nav_buttons: dict[str, QPushButton] = {}
        for title, callback in (
            ("▱   Проект", lambda: self.project_tree.setFocus()),
            ("▧   Фото керна", self.return_to_photos),
            ("▢   Разметка", self._select_markup_view),
            ("⌁   Столбик", self._select_column_view),
            ("▤   Отчёт", self.export_core_description_report),
        ):
            button = QPushButton(title)
            button.setObjectName("shellNav")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, name=title, handler=callback: self._navigate(name, handler))
            nav_layout.insertWidget(nav_layout.count() - 1, button)
            self._nav_buttons[title] = button
        nav_layout.insertWidget(nav_layout.count() - 1, QLabel("СКВАЖИНЫ И ФОТО"))
        nav_layout.setStretch(nav_layout.count() - 1, 1)
        self._project_sidebar.setTitleBarWidget(QWidget())
        self._project_sidebar.setMinimumWidth(170)
        self._project_sidebar.setMaximumWidth(260)
        self._nav_buttons["▧   Фото керна"].setChecked(True)

        self.assistant_panel = AssistantPanel(self)
        self.assistant_panel.send_requested.connect(self._send_vlm_chat)
        self.assistant_panel.correction_requested.connect(self._correct_assistant_answer)
        self.assistant_panel.files_requested.connect(self._attach_chat_files)
        self.assistant_panel.files_dropped.connect(self._handle_chat_files)
        self.assistant_panel.found_photo_requested.connect(lambda path: self._handle_chat_files([path]))
        self.assistant_panel.photo_clear_requested.connect(self._clear_chat_photo)
        self.assistant_panel.photo_analysis_requested.connect(self._describe_chat_photo)
        self.assistant_panel.expand_requested.connect(self._toggle_assistant_width)
        assistant_dock = QDockWidget("ИИ-ассистент", self)
        self._assistant_dock = assistant_dock
        assistant_dock.setObjectName("assistantDock")
        assistant_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        assistant_dock.setTitleBarWidget(QWidget())
        assistant_dock.setWidget(self.assistant_panel)
        assistant_dock.setMinimumWidth(310)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, assistant_dock)
        self.resizeDocks([self._project_sidebar, assistant_dock], [190, 360], Qt.Orientation.Horizontal)

        self._vlm_client = LocalVlmClient(self)
        self._vlm_client.reply.connect(self._on_vlm_reply)
        self._vlm_client.state_changed.connect(self._on_vlm_state_changed)

        self.workspace.setProperty("clean_background", True)
        self.workspace.viewport().update()
        self.setStyleSheet(self.styleSheet() + """
            QToolBar#shellHeader { background: #f6fbff; border-bottom: 1px solid #dbeafb; padding: 7px 14px; spacing: 12px; }
            QLabel#shellBrand { color: #086bd7; font-size: 16px; font-weight: 800; letter-spacing: 2px; padding: 5px 10px; }
            QLabel#shellDivider { color: #d5e2f3; }
            QLabel#shellBreadcrumb { color: #103d81; font-weight: 600; }
            QComboBox#shellWell { color: #103d81; background: transparent; border: none; min-width: 120px; font-weight: 700; }
            QLabel#shellSaveIndicator { color: #04bc69; }
            QPushButton#shellUploadPhoto { background: #087ce8; color: white; border: none; border-radius: 10px; padding: 9px 16px; font-weight: 700; }
            QPushButton#shellUploadPhoto:hover { background: #0668c4; }
            QPushButton#shellLoadModel { background: #edf5ff; color: #0a65bb; border: 1px solid #a6d3fb; border-radius: 10px; padding: 9px 13px; font-weight: 700; }
            QPushButton#shellLoadModel:hover { background: #dcecff; border-color: #087ce8; }
            QComboBox#shellWell { background: #ffffff; border: 1px solid #c8dff5; border-radius: 8px; padding: 7px 24px 7px 10px; min-width: 120px; color: #103d81; }
            QComboBox#shellWell:hover, QComboBox#shellWell:focus { border-color: #087ce8; }
            QMenu#shellSettingsMenu { background: #ffffff; color: #153e77; border: 1px solid #d4e5f6; border-radius: 9px; padding: 7px; }
            QMenu#shellSettingsMenu::item { padding: 8px 22px; border-radius: 6px; }
            QMenu#shellSettingsMenu::item:selected { background: #e6f3ff; color: #076bd0; }
            QPushButton#shellSave { background: #e5f4ff; color: #0878df; border: 1px solid #1397f7; border-radius: 10px; padding: 9px 16px; font-weight: 700; }
            QToolButton#shellTools { color: #0b3b85; background: transparent; border: none; font-size: 19px; padding: 4px 9px; }
            QDockWidget#projectSidebar, QDockWidget#assistantDock { background: #f7fbff; border: none; }
            QFrame#workspaceFrame { background: white; border: 1px solid #dceafa; border-radius: 12px; }
            QGraphicsView#mainWorkspace { background: white; border: none; }
            QPushButton#shellNav { text-align: left; color: #0d3980; background: transparent; border: none; border-radius: 9px; padding: 12px 10px; font-size: 13px; }
            QPushButton#shellNav:checked, QPushButton#shellNav:hover { color: white; background: #087ce8; font-weight: 700; }
            QFrame#assistantPanel { background: #f8fbff; border-left: 1px solid #dceafa; }
            QLabel#assistantTitle { color: #092d6f; font-size: 19px; font-weight: 800; }
            QLabel#assistantStatus { color: #04bd66; font-size: 12px; }
            QLabel#assistantContext { color: #264a82; background: #edf5ff; border: 1px solid #dceafb; border-radius: 12px; padding: 12px; }
            QScrollArea#assistantTranscript { background: transparent; border: none; }
            QWidget#assistantMessages { background: transparent; }
            QMainWindow::separator { background: #dceafa; width: 6px; }
            QMainWindow::separator:hover { background: #85bfff; }
            QPlainTextEdit#assistantComposer { color: #1a3e75; background: white; border: 1px solid #d8e8fb; border-radius: 12px; padding: 10px; }
            QPushButton#assistantAction { color: #1167c2; background: #e9f4ff; border: 1px solid #d6eaff; border-radius: 8px; padding: 8px; }
            QPushButton#assistantSend { color: white; background: #087ce8; border: none; border-radius: 8px; padding: 9px 12px; font-weight: 700; }
            QTreeWidget#projectTree { background: #f7fbff; color: #234c83; }
            QTreeWidget#projectTree::item:selected { background: #dceeff; color: #0b66c2; }
        """)
        self._update_header_context()

    def _navigate(self, name: str, handler) -> None:
        for title, button in self._nav_buttons.items():
            button.setChecked(title == name)
        handler()

    def _set_assistant_visible(self, visible: bool) -> None:
        self._assistant_dock.setVisible(visible)
        self._assistant_visibility_action.setText(
            "Скрыть ИИ-ассистента" if visible else "Показать ИИ-ассистента"
        )
        if visible:
            self.resizeDocks([self._assistant_dock], [360], Qt.Orientation.Horizontal)

    def _toggle_assistant_width(self) -> None:
        target = min(760, max(520, self.width() * 55 // 100)) if self._assistant_dock.width() < 500 else 360
        self.resizeDocks([self._assistant_dock], [target], Qt.Orientation.Horizontal)

    def _select_markup_view(self) -> None:
        self.return_to_photos()
        self.statusBar().showMessage("Выберите контур на фото. Новый контур: ⚙ → Правка разметки")

    def _select_column_view(self) -> None:
        self._show_stack()
        self.workspace.fit_content()

    def _header_well_changed(self, index: int) -> None:
        value = self._header_well.itemText(index)
        if value:
            self._active_well_name = value
            self._update_assistant_context()

    def _update_header_context(self) -> None:
        if not hasattr(self, "_header_well"):
            return
        self._header_project.setText(f"{self._project_title}  /  ")
        self._header_well.blockSignals(True)
        self._header_well.clear()
        wells = self._well_names()
        self._header_well.addItems(wells or [self._active_well_name])
        self._header_well.setCurrentText(self._active_well_name)
        self._header_well.blockSignals(False)
        self._update_assistant_context()

    def _update_assistant_context(self) -> None:
        if not hasattr(self, "assistant_panel"):
            return
        target = self._vlm_target
        suffix = ""
        if target is not None and any(record is target[0] for record in self._records):
            suffix = f"\nФото: {Path(target[0].path).name}"
            if target[1] is not None:
                suffix += f" · фация: {target[1].label}"
        self.assistant_panel.context.setText(
            f"Проект: {self._project_title}\nСкважина: {self._active_well_name}{suffix}"
        )

    def _local_vlm_config(self) -> dict:
        root = application_root().parent / "vlm_core_v1"
        snapshot_root = root / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots"
        snapshots = sorted(path for path in snapshot_root.glob("*") if (path / "config.json").is_file())
        defaults = {
            "python": root / ".venv" / "Scripts" / "python.exe",
            "base": str(snapshots[-1]) if snapshots else "",
            "adapter": "",
            "folders": [str(root / "data" / "knowledge")] if (root / "data" / "knowledge").is_dir() else [],
        }
        raw_folders = self._settings.value("vlm/knowledge_folders_json", "")
        try:
            folders = json.loads(str(raw_folders)) if raw_folders else defaults["folders"]
        except (ValueError, TypeError):
            folders = defaults["folders"]
        return {
            "python": str(self._settings.value("vlm/python", "") or defaults["python"]),
            "base": str(self._settings.value("vlm/base", "") or defaults["base"]),
            "adapter": str(self._settings.value("vlm/adapter", "") or defaults["adapter"]),
            "theory_adapter": str(self._settings.value("vlm/theory_adapter", "") or ""),
            "model_mode": str(self._settings.value("vlm/model_mode", "base") or "base"),
            "folders": [str(folder) for folder in folders if isinstance(folder, str)],
        }

    def _knowledge_folders(self, question: str = "", *, include_archive: bool = False) -> list[str]:
        """Use the project, selected photo and chosen folders for offline search."""
        folders: list[str] = []
        seen: set[str] = set()

        def add(value: str | Path | None) -> None:
            if value is None:
                return
            folder = str(value)
            key = folder.casefold().rstrip("\\/")
            if key and key not in seen:
                seen.add(key)
                folders.append(folder)

        # The user's imported theory remains available after a restart and
        # takes priority over unrelated folder searches.
        theory_index = user_data_root() / "chat_materials" / "theory_index"
        if theory_index.is_dir():
            add(theory_index)
        archive_search = include_archive or bool(re.search(r"\b(?:скваж\w*|месторожд\w*)\b", question.casefold()))
        if archive_search:
            archives = [Path(drive.absoluteFilePath()) / "Керн фото" for drive in QDir.drives()]
            available_archives = [archive for archive in archives if archive.is_dir()]
            for archive in available_archives:
                add(archive)
            if not available_archives:
                add(Path("E:/Керн фото"))
        else:
            add(bundled_root() / "app" / "resources" / "theory")
            add(application_root().parent / "vlm_core_v1" / "data" / "knowledge")
        if self._chat_document_path is not None:
            add(self._chat_document_path.parent)
        add(self._project_folder)
        if self._vlm_target is not None:
            add(Path(self._vlm_target[0].path).parent)
        if self._chat_photo_path is not None:
            add(self._chat_photo_path.parent)
        from app.infrastructure.open_dataset_library import prepared_knowledge_folders
        for folder in prepared_knowledge_folders(user_data_root() / "open_datasets"):
            add(folder)
        for folder in self._local_vlm_config()["folders"]:
            add(folder)
        add(bundled_root() / "app" / "resources" / "theory")
        add(application_root().parent / "vlm_core_v1" / "data" / "knowledge")
        return folders[:12]

    def configure_local_vlm(self) -> None:
        if self._analysis_active or self._vlm_pending is not None:
            self.assistant_panel.append_message("system", "Дождитесь завершения текущего ответа перед сменой модели.")
            return
        dialog = LocalVlmSettingsDialog(self._local_vlm_config(), self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        config = dialog.values()
        self._settings.setValue("vlm/python", config["python"])
        self._settings.setValue("vlm/base", config["base"])
        self._settings.setValue("vlm/adapter", config["adapter"])
        self._settings.setValue("vlm/theory_adapter", config["theory_adapter"])
        self._settings.setValue("vlm/model_mode", config["model_mode"])
        self._settings.setValue("vlm/knowledge_folders_json", json.dumps(config["folders"], ensure_ascii=False))
        self._settings.sync()
        self._vlm_client.stop()
        self.assistant_panel.append_message("system", "Локальные настройки сохранены. Папок знаний: " + str(len(config["folders"])))
        for record in self._records:
            for detection in record.detections:
                if (
                    detection.attributes.get("__core_column_description") == "1"
                    and detection.attributes.get("__interpretation_status") == "error"
                ):
                    detection.attributes["__interpretation_status"] = "waiting"
                    detection.attributes["Статус интерпретации"] = "ожидает описания столбика локальной VLM"
        if self._has_waiting_cv1_descriptions():
            QTimer.singleShot(0, self._start_cv1_descriptions)

    def show_training_center(self) -> None:
        dialog = TrainingCenterDialog(self)
        dialog.exec()

    def show_open_datasets(self, *, source_id: str | None = None, files: list[Path] | None = None) -> None:
        dialog = OpenDatasetsDialog(self, source_id=source_id, files=files)
        dialog.materials_ready.connect(self._open_dataset_prepared)
        dialog.exec()

    def _open_dataset_prepared(self, result: dict) -> None:
        if result.get("operation") == "reviewed_pairs":
            self.assistant_panel.append_message(
                "system", f"Подключены проверенные описания: {result['counts']['train']} для обучения, "
                f"{result['counts']['holdout']} оставлены для контроля по другим скважинам."
            )
            return
        knowledge = Path(str(result.get("knowledge_dir") or "__missing__"))
        if knowledge.is_dir():
            folders = list(self._local_vlm_config()["folders"])
            if str(knowledge) not in folders:
                folders.append(str(knowledge))
                self._settings.setValue("vlm/knowledge_folders_json", json.dumps(folders, ensure_ascii=False))
                self._settings.sync()
        self.assistant_panel.append_message(
            "system", "Учебные материалы подготовлены. Исходные классы и описания сохранены; "
            "проверьте их перед обучением. Документы источника доступны для локального поиска."
            if knowledge.is_dir() else
            "Учебные материалы подготовлены по классам источника. Результат и обучение доступны в окне учебных материалов."
        )

    def _attach_dataset_archives(self, paths: list[Path]) -> None:
        from app.infrastructure.open_dataset_library import identify_archive
        from app.infrastructure.open_dataset_sources import list_sources
        grouped: dict[str | None, list[Path]] = {}
        for path in paths:
            grouped.setdefault(identify_archive(path), []).append(path)
        for identifier, archives in grouped.items():
            if identifier is None:
                sources = list_sources()
                value, accepted = QInputDialog.getItem(
                    self, "Учебный архив", "Выберите источник добавленного архива:",
                    [source.name_ru for source in sources], 0, False,
                )
                if not accepted:
                    continue
                identifier = next(source.id for source in sources if source.name_ru == value)
            self.show_open_datasets(source_id=identifier, files=archives)

    def _load_global_primary_catalog(self) -> dict | None:
        path = str(self._settings.value("facies/primary_catalog_snapshot", "") or "")
        return load_catalog_snapshot(Path(path)) if path else None

    def _resolve_training_class(self, label: str, index: str | None = None) -> dict | None:
        if self._primary_facies_catalog is not None:
            return resolve_catalog_class(label, index, self._primary_facies_catalog["classes"])
        return resolve_facies_class(label, index)

    def import_primary_facies_catalog(self, *, filename_override: Path | None = None) -> bool:
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Справочник фаций", "Дождитесь завершения обработки или обучения.")
            return False
        if filename_override is None:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Основной справочник фаций для сегментации", "", "Excel (*.xlsx *.xlsm)",
            )
        else:
            filename = str(filename_override)
        if not filename:
            return False
        try:
            catalog = read_primary_facies_workbook(Path(filename))
            snapshot = user_data_root() / "facies_catalogs" / f"{catalog['source_sha256']}.json"
            save_catalog_snapshot(snapshot, catalog)
            if self._project_folder is not None:
                save_catalog_snapshot(self._project_folder / SNAPSHOT_NAME, catalog)
        except Exception as exc:
            QMessageBox.warning(self, "Справочник не загружен", str(exc))
            return False
        self._primary_facies_catalog = catalog
        self._settings.setValue("facies/primary_catalog_snapshot", str(snapshot))
        self._settings.sync()
        self._refresh_training_bar()
        unresolved = sum(
            1 for record in self._records for detection in record.detections
            if detection.training_ready and not self._resolve_training_class(
                detection.label, (detection.attributes or {}).get("Индекс фации")
            )
        )
        QMessageBox.information(
            self, "Основной справочник подключён",
            f"Источник: {catalog['source_name']}\nКлассов: {len(catalog['classes'])}.\n"
            f"Проверенных слоёв вне нового справочника: {unresolved}.\n\n"
            "Этот Excel задаёт названия фаций для дальнейшей работы. "
            "Границы слоёв на фото нужно проверить отдельно.",
        )
        return True

    def _vlm_context(self) -> str:
        context = f"Проект: {self._project_title}; скважина: {self._active_well_name}. "
        if self._vlm_target is not None:
            record, detection = self._vlm_target
            if any(item is record for item in self._records):
                context += f"Выбранное фото: {Path(record.path).name}. "
                if any(item is detection for item in record.detections):
                    mapping = facies_mapping_context(record, detection)
                    if mapping:
                        context += mapping + " "
                    else:
                        context += f"Текущая метка интервала (требует проверки): {detection.label}. "
                    remembered = exact_interval_edit(record, detection, project=self._expert_project_key(), well=record.well_name)
                    if remembered:
                        attributes = remembered.get("attributes") or {}
                        description = str(attributes.get("__vlm_description") or attributes.get("Краткое описание") or "").strip()
                        if description:
                            context += "Проверенное специалистом описание этого же интервала: " + description[:700] + ". "
                    if detection.depth_from is not None and detection.depth_to is not None:
                        context += f"Интервал фации: {detection.depth_from:g}–{detection.depth_to:g} м. "
        rows = []
        for record in self._records_for_well(self._active_well_name)[:8]:
            for layer in record.detections[:12]:
                if layer.depth_from is not None and layer.depth_to is not None:
                    rows.append(f"{layer.depth_from:g}–{layer.depth_to:g} м: {layer.label}")
        if rows:
            context += "Слои проекта: " + "; ".join(rows[:35])
        return context[:2300]

    def _vlm_image_path(self, *, crop_facies: bool) -> str:
        if self._vlm_target is None:
            return ""
        record, detection = self._vlm_target
        if not any(item is record for item in self._records) or record.pixmap.isNull():
            return ""
        image = record.pixmap.toImage()
        if crop_facies and any(item is detection for item in record.detections) and detection.polygon:
            rect = QPolygonF(detection.polygon).boundingRect().toAlignedRect().intersected(image.rect())
            if not rect.isEmpty():
                image = image.copy(rect)
        temporary = user_data_root() / "vlm_temp"
        temporary.mkdir(parents=True, exist_ok=True)
        path = temporary / f"selected_core_{uuid4().hex}.png"
        if crop_facies and detection is not None and detection.attributes.get("__geometry_only") == "1":
            return str(write_interval_crop(record, IntervalJob.create(record, detection), path))
        if not image.save(str(path), "PNG"):
            raise RuntimeError("Не удалось подготовить выбранное фото для локальной VLM")
        return str(path)

    @staticmethod
    def _cleanup_vlm_image(path: str | Path | None = None) -> None:
        try:
            if path:
                Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    def _attach_chat_photo(self) -> None:
        self._attach_chat_files()

    def _attach_chat_files(self) -> None:
        """One chat entry point for photos, depth descriptions and local reading material."""
        start = str(self._chat_photo_path.parent) if self._chat_photo_path else ""
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Добавить материалы в чат", start,
            "Фото, модели, Excel и документы (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp *.pt *.xlsx *.xlsm *.xls *.csv *.tsv *.txt *.md *.pdf *.docx *.json *.jsonl *.zip);;Все файлы (*)",
        )
        if not filenames:
            return
        self._handle_chat_files(filenames)

    def _handle_chat_files(self, filenames: list[str]) -> None:
        image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        excel_suffixes = {".xlsx", ".xlsm", ".xls"}
        document_suffixes = {".csv", ".tsv", ".txt", ".md", ".pdf", ".docx", ".json", ".jsonl"}
        paths = [Path(name) for name in filenames]
        images = [path for path in paths if path.suffix.casefold() in image_suffixes]
        excels = [path for path in paths if path.suffix.casefold() in excel_suffixes]
        documents = [path for path in paths if path.suffix.casefold() in document_suffixes]
        archives = [path for path in paths if path.suffix.casefold() == ".zip"]
        models = [path for path in paths if path.suffix.casefold() == ".pt"]
        for path in models[:1]:
            self.select_detection_model(filename_override=path)
        if len(models) > 1:
            self.assistant_panel.append_message("system", "Для границ используется один best.pt. Добавьте нужный файл отдельно.")
        if documents:
            self._queue_chat_theory_documents(documents)
        for path in excels + archives:
            self._add_chat_document(path)
        if len(excels) > 1:
            self.assistant_panel.append_message("system", f"Таблиц добавлено: {len(excels)}. Они доступны для вопросов в чате; сопоставление с проектом запускается отдельно.")
        for path in paths:
            if path not in images + excels + documents + archives + models:
                self.assistant_panel.append_message("system", f"Тип файла «{path.name}» пока не поддерживается.")
        if images:
            try:
                self.assistant_panel.set_photo(images[-1])
            except Exception as exc:
                QMessageBox.warning(self, "Фото не открыто", str(exc))
                return
            self._chat_photo_path = images[-1]
            self._chat_pending_photos = list(images)
            self._chat_photo_columns = None
            self._chat_photo_history = list(self._chat_photo_histories.get(str(images[-1].resolve()), []))
            if len(images) > 1:
                self.assistant_panel.append_message("system", f"Добавлено фото: {len(images)}. В чате показано последнее.")
        if excels:
            self._chat_pending_excel_path = excels[0]
        if images:
            self.assistant_panel.append_message(
                "system", "Фото прикреплено к чату. Задайте вопрос о нём. Для обработки проекта напишите «обработай фото» или «собери колонку».",
            )

    def _add_chat_document(self, source: Path) -> None:
        try:
            destination_dir = user_data_root() / "chat_materials"
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / f"{uuid4().hex[:8]}_{source.name}"
            shutil.copy2(source, destination)
            config = self._local_vlm_config()
            folders = list(config["folders"])
            if str(destination_dir) not in folders:
                folders.insert(0, str(destination_dir))
                self._settings.setValue("vlm/knowledge_folders_json", json.dumps(folders, ensure_ascii=False))
                self._settings.sync()
            self._chat_document_path = destination
            self.assistant_panel.append_message("user", f"📎 {source.name}")
            if source.suffix.casefold() == ".zip":
                self.assistant_panel.append_message("system", f"Добавлен архив «{source.name}»: доступен поиск имён файлов. Для чтения документов распакуйте архив и добавьте их в чат.")
            else:
                self.assistant_panel.append_message("system", f"Добавлена таблица «{source.name}»: помощник может искать текст в первых пяти листах. Для сопоставления с фото используйте обработку проекта.")
        except OSError as exc:
            self.assistant_panel.append_message("system", f"Не удалось добавить «{source.name}»: {exc}")

    def _queue_chat_theory_documents(self, paths: list[Path]) -> None:
        self._theory_import_queue.extend(str(path) for path in paths)
        self.assistant_panel.append_message("system", f"Читаю документы: {len(paths)}. Это может занять время; продолжайте работу.")
        self._start_next_theory_import()

    def _start_next_theory_import(self) -> None:
        if self._theory_import_thread is not None or not self._theory_import_queue:
            return
        from app.infrastructure.theory_chat_import import TheoryImportThread
        batch, self._theory_import_queue = self._theory_import_queue, []
        thread = TheoryImportThread(batch, user_data_root() / "chat_materials", self)
        self._theory_import_thread = thread
        thread.document_result.connect(self._theory_import_finished_document)
        thread.finished.connect(self._theory_import_finished_batch)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _theory_import_finished_document(self, result: dict) -> None:
        name = str(result.get("name") or "документ")
        if not result.get("ok"):
            self.assistant_panel.append_message("system", f"Не удалось прочитать «{name}»: {result.get('error', 'неизвестная ошибка')}")
            return
        self._chat_document_path = Path(str(result["original"]))
        self.assistant_panel.append_message("user", f"📎 {name}")
        message = (f"Текст «{name}» сохранён локально: фрагментов {result['chunks']}. "
                   "Можно задавать вопросы; помощник будет искать в документе. "
                   "Веса модели при добавлении файла не изменяются.")
        if result.get("warnings"):
            message += " " + " ".join(result["warnings"])
        self.assistant_panel.append_message("system", message)

    def _theory_import_finished_batch(self) -> None:
        self._theory_import_thread = None
        self._start_next_theory_import()

    def _clear_chat_photo(self) -> None:
        self._chat_photo_path = None
        self._chat_pending_photos = []
        self._chat_photo_columns = None
        self._chat_photo_question = ""
        self._chat_photo_history.clear()
        self.assistant_panel.clear_photo()

    def _pair_chat_photo_with_excel(self) -> None:
        if self._chat_photo_path is None:
            self.assistant_panel.append_message("system", "Сначала прикрепите фото к чату.")
            return
        if self._vlm_pending is not None or self._chat_photo_thread is not None:
            self.assistant_panel.append_message("system", "Дождитесь завершения анализа фото.")
            return
        self.import_excel_photo_batch(photo_paths_override=[self._chat_photo_path])

    def _describe_chat_photo(self) -> None:
        if self._chat_photo_path is None:
            self.assistant_panel.append_message("system", "Сначала добавьте фото кнопкой «Добавить файл».")
            return
        question = "Опиши загруженное фото керна на русском: видимые породы, цвет, зернистость, текстуры, слоистость и контакты. Не придумывай глубины и невидимые признаки."
        self.assistant_panel.append_message("user", f"Опиши фото: {self._chat_photo_path.name}")
        self._analyze_chat_photo(question)

    def _analyze_chat_photo(self, question: str) -> None:
        if self._chat_photo_path is None:
            return
        self._chat_photo_question = question
        self._send_vlm_request(
            "photo_chat", question, image_override=str(self._chat_photo_path),
            extra_context=(
                "Границы интервалов определяет отдельная модель YOLO. Не придумывай координаты, глубины или число физических столбиков. "
                "Текущий запрос относится к видимым признакам фото."
            ),
        )
        return

    def _import_chat_analysis_photos(self, paths: list[Path]) -> None:
        """Queue chat photos through the same bounded importer as project photos."""
        queued_paths = {str(item[0].resolve()) for item in self._pending_photo_imports}
        for path in paths:
            resolved = str(path.resolve())
            if resolved in queued_paths:
                continue
            top, base = self._photo_interval_from_path(path)
            self._pending_photo_imports.append((path, self._active_well_name, top, base))
            queued_paths.add(resolved)
        if self._pending_photo_imports:
            self.assistant_panel.append_message(
                "system", f"Поставлено в очередь фото: {len(paths)}. Загружаю и обрабатываю небольшими партиями."
            )
            self._load_next_photo_batch()

    def _record_for_chat_photo(self, path: Path) -> PhotoRecord | None:
        resolved = path.resolve()
        identifier = self._chat_record_ids.get(str(resolved))
        return next((record for record in self._records if record.identifier == identifier or Path(record.path).resolve() == resolved), None)

    def _analyze_chat_photo_legacy(self, question: str) -> None:
        """Compatibility helper for old projects; not the CV-1 chat entry path."""
        if self._chat_photo_columns and self._chat_photo_columns.get("path") == str(self._chat_photo_path):
            self._send_vlm_request(
                "photo_chat", question, image_override=str(self._chat_photo_path),
                extra_context=self._chat_photo_context(self._chat_photo_columns),
            )
            return
        self.assistant_panel.set_busy(True)
        self.assistant_panel.status.setText("●  Ищу столбики на фото…")
        self._chat_photo_thread = QThread(self)
        self._chat_photo_worker = ChatPhotoColumnWorker(self._chat_photo_path, self._selected_column_model_path)
        self._chat_photo_worker.moveToThread(self._chat_photo_thread)
        self._chat_photo_thread.started.connect(self._chat_photo_worker.run)
        self._chat_photo_worker.succeeded.connect(self._on_chat_photo_columns)
        self._chat_photo_worker.failed.connect(self._on_chat_photo_columns_failed)
        self._chat_photo_worker.finished.connect(self._chat_photo_thread.quit)
        self._chat_photo_thread.finished.connect(self._on_chat_photo_worker_finished)
        self._chat_photo_thread.finished.connect(self._chat_photo_worker.deleteLater)
        self._chat_photo_thread.finished.connect(self._chat_photo_thread.deleteLater)
        self._chat_photo_thread.start()

    @staticmethod
    def _chat_photo_context(result: dict) -> str:
        columns = result.get("columns") or []
        return (
            "Это отдельное фото, загруженное в чат; глубины и метки фаций для него не заданы. "
            f"Локальный поиск столбиков предварительно выделил {len(columns)} колонок "
            f"(метод: {result.get('detector', 'неизвестно')}). "
            "Не считай предварительные границы подтверждёнными геологом."
        )

    def _on_chat_photo_columns(self, result: dict) -> None:
        if self._chat_photo_path is None or result.get("path") != str(self._chat_photo_path):
            return
        self._chat_photo_columns = result
        columns = result["columns"]
        self.assistant_panel.show_photo_columns(columns, result["width"], result["height"])
        if columns:
            details = "\n".join(
                f"{number}. x {box['left']:.0f}–{box['right']:.0f}, y {box['top']:.0f}–{box['bottom']:.0f} пикс."
                for number, box in enumerate(columns[:12], start=1)
            )
            if len(columns) > 12:
                details += f"\n…ещё {len(columns) - 12} рамок показаны на фото."
            self.assistant_panel.append_message(
                "assistant",
                f"Предварительно найдено столбиков: {len(columns)} ({result['detector']}).\n{details}\nГолубые рамки показаны на фото; проверьте их перед использованием.",
            )
        else:
            self.assistant_panel.append_message("assistant", "Столбики автоматически не выделены. Опишу видимые признаки на фото целиком.")
        self._send_vlm_request(
            "photo_chat", self._chat_photo_question, image_override=str(self._chat_photo_path),
            extra_context=self._chat_photo_context(result),
        )

    def _on_chat_photo_columns_failed(self, message: str) -> None:
        self.assistant_panel.append_message("system", "Поиск столбиков не завершён: " + message)
        if self._chat_photo_path is not None:
            self._send_vlm_request(
                "photo_chat", self._chat_photo_question, image_override=str(self._chat_photo_path),
                extra_context="Это отдельное фото, загруженное в чат. Границы столбиков не определены; не придумывай их.",
            )
        else:
            self.assistant_panel.set_busy(False)

    def _on_chat_photo_worker_finished(self) -> None:
        self._chat_photo_thread = None
        self._chat_photo_worker = None

    def _send_vlm_request(self, operation: str, question: str = "", *, image_override: str = "", extra_context: str = "") -> None:
        if operation == "photo_chat" and image_override:
            remembered = exact_correction(
                question, operation="photo_chat", image_path=image_override,
                project=self._expert_project_key(), well=self._active_well_name,
            )
            if remembered:
                self._chat_photo_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": remembered},
                ])
                self._chat_photo_history = self._chat_photo_history[-12:]
                self.assistant_panel.append_message(
                    "assistant", "Проверено специалистом:\n" + remembered,
                    correction={"operation": "photo_chat", "question": question,
                                "answer": remembered, "image_path": image_override},
                )
                self.assistant_panel.set_busy(False)
                return
        config = self._local_vlm_config()
        folders = self._knowledge_folders(question)
        temporary_image = ""
        try:
            use_image = operation in {"describe", "photo_chat"} or (operation == "chat" and needs_selected_image(question))
            has_imported_theory = (user_data_root() / "chat_materials" / "theory_index").is_dir()
            use_knowledge = (operation == "search" or
                             (operation in {"chat", "photo_chat"} and (needs_local_knowledge(question) or wants_sources(question))) or
                             (operation == "chat" and (self._chat_document_path is not None or has_imported_theory)))
            image = image_override or (self._vlm_image_path(crop_facies=operation == "describe") if use_image else "")
            temporary_image = image if image and not image_override else ""
            if operation == "describe" and not image:
                raise ValueError("Нет изображения выбранной фации для описания.")
            context = extra_context if operation == "photo_chat" else (
                self._vlm_context() if operation == "describe" or use_image or use_knowledge or "проект" in question.casefold() or "скважин" in question.casefold() else ""
            )
            payload = {
                "operation": operation, "question": question,
                "context": context,
                "image": image, "history": (self._chat_photo_history if operation == "photo_chat" else self._vlm_history)[-6:],
                "folders": folders, "use_knowledge": use_knowledge,
                "unavailable_folders": [folder for folder in folders if not Path(folder).is_dir()],
                "show_sources": wants_sources(question),
                "expert_examples": related_text_corrections(
                    question, project=self._expert_project_key(), well=self._active_well_name,
                ) if operation == "chat" and not use_image else [],
                "base": config["base"], "adapter": config["adapter"],
                "theory_adapter": config["theory_adapter"],
                "model_mode": config["model_mode"],
            }
            target = self._vlm_target[1] if self._vlm_target is not None else None
            selected_job = (
                IntervalJob.create(self._vlm_target[0], target)
                if operation == "describe" and target is not None and target.attributes.get("__geometry_only") == "1"
                else None
            )
        except Exception as exc:
            self._cleanup_vlm_image(temporary_image)
            self.assistant_panel.append_message("system", str(exc))
            return
        self._vlm_chat_queue.append({
            "python": config["python"], "payload": payload, "target": target,
            "selected_job": selected_job, "temporary_image": temporary_image,
            "photo_path": str(image_override) if operation == "photo_chat" else "",
        })
        if self._vlm_pending is not None or self._analysis_pending is not None or self._segmentation_thread is not None:
            self.assistant_panel.append_message("system", "Вопрос в очереди. Отвечу после текущего фрагмента обработки.")
        self._refresh_chat_status()
        self._schedule_vlm_dispatch()

    def _on_vlm_state_changed(self, text: str) -> None:
        if self._vlm_pending is not None:
            self.assistant_panel.status.setText(text)
        else:
            self._refresh_chat_status()

    def _refresh_chat_status(self) -> None:
        if self._vlm_pending is not None:
            text = "●  Готовлю ответ. Можно отправить следующий вопрос"
        elif self._vlm_chat_queue:
            text = f"●  В очереди вопросов: {len(self._vlm_chat_queue)}"
        elif self._analysis_active or self._segmentation_thread is not None:
            text = "●  Обработка фото идёт отдельно · чат доступен"
        else:
            text = "●  Готов к работе"
        self.assistant_panel.status.setText(text)

    def _schedule_vlm_dispatch(self) -> None:
        if self._vlm_closing or self._vlm_dispatch_scheduled:
            return
        self._vlm_dispatch_scheduled = True
        QTimer.singleShot(0, self._drain_vlm_requests)

    def _drain_vlm_requests(self) -> None:
        self._vlm_dispatch_scheduled = False
        if self._vlm_closing or self._vlm_pending is not None or self._analysis_pending is not None:
            return
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            return
        if self._vlm_chat_queue:
            request = self._vlm_chat_queue.popleft()
            payload = request["payload"]
            if payload["operation"] == "chat":
                payload["history"] = list(self._vlm_history[-6:])
            elif payload["operation"] == "photo_chat" and request.get("photo_path"):
                history_key = str(Path(request["photo_path"]).resolve())
                payload["history"] = list(self._chat_photo_histories.get(history_key, [])[-6:])
            try:
                request_id = self._vlm_client.send(request["python"], request["payload"])
            except Exception as exc:
                self._cleanup_vlm_image(request.get("temporary_image"))
                self.assistant_panel.append_message("system", str(exc))
                self._schedule_vlm_dispatch()
                return
            self._vlm_pending = (request_id, payload["operation"], request["target"], payload["question"])
            self._vlm_pending_meta = request
            self._selected_description_job = request["selected_job"]
        elif self._analysis_active:
            self._advance_cv1_description()
        elif self._queued_records:
            records, self._queued_records = self._queued_records, []
            self.run_segmentation(records)
        self._refresh_chat_status()

    def _send_vlm_chat(self, question: str) -> None:
        self.assistant_panel.append_message("user", question)
        process_project = bool(re.match(
            r"\s*(?:обработай|обработать|проанализируй все|собери (?:(?:един\w*|общ\w*)\s+)?(?:колонк\w*|столбик\w*)|добавь .*в проект)",
            question.casefold(),
        ))
        if process_project:
            paths = list(self._chat_pending_photos or ([self._chat_photo_path] if self._chat_photo_path else []))
            if paths:
                if self._chat_pending_excel_path is not None:
                    self.import_excel_photo_batch(photo_paths_override=paths, excel_path_override=self._chat_pending_excel_path)
                else:
                    self._import_chat_analysis_photos(paths)
            else:
                self.run_segmentation()
            return
        if wants_open_datasets(question):
            self.show_open_datasets()
            return
        if wants_photo_file(question):
            self._search_local_folders(question, display_question=False)
            return
        photo_request = needs_selected_image(question)
        expert_answer = (
            exact_correction(question, project=self._expert_project_key(), well=self._active_well_name)
            if not photo_request else None
        )
        immediate = expert_answer or quick_reply(question, has_history=bool(self._vlm_history))
        if immediate is None and not photo_request:
            cards = bundled_root() / "app" / "resources" / "theory" / "theory_cards_v5.json"
            immediate = lookup_theory_answer(question, cards, include_sources=wants_sources(question))
            if immediate is None and wants_sources(question) and len(question.split()) <= 5:
                previous_question = next((turn["content"] for turn in reversed(self._vlm_history) if turn["role"] == "user"), "")
                referenced = lookup_theory_answer(previous_question, cards, include_sources=True) if previous_question else None
                if referenced and "\n\nИсточники:\n" in referenced:
                    immediate = "Источники:\n" + referenced.split("\n\nИсточники:\n", 1)[1]
        if immediate is not None:
            self.assistant_panel.append_message(
                "assistant", ("Проверено специалистом:\n" if expert_answer else "") + immediate,
                correction={"operation": "chat", "question": question, "answer": immediate},
            )
            self._vlm_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": immediate},
            ])
            self._vlm_history = self._vlm_history[-12:]
            return
        visual_search = bool(re.search(r"\bна (?:этом |данном )?(?:фото|снимке|изображении)\b", question.casefold()))
        if wants_folder_search(question) and not visual_search:
            self._search_local_folders(question, display_question=False)
            return
        if self._chat_photo_path is not None and photo_request:
            self._analyze_chat_photo(question)
            return
        self._send_vlm_request("chat", question)

    def _expert_project_key(self) -> str:
        return str(self._project_folder.resolve()) if self._project_folder else f"unsaved:{self._expert_session_id}"

    def _correct_assistant_answer(self, data: dict) -> None:
        """Accept only an explicit edit, never treat raw model output as truth."""
        question = str(data.get("question") or "").strip()
        answer = str(data.get("answer") or "").strip()
        operation = str(data.get("operation") or "chat")
        if not question or not answer or operation not in {"chat", "photo_chat"}:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Поправка специалиста")
        dialog.resize(650, 460)
        layout = QVBoxLayout(dialog)
        question_label = QLabel("Вопрос: " + question, dialog)
        question_label.setWordWrap(True)
        layout.addWidget(question_label)
        layout.addWidget(QLabel("Исправьте ответ модели и сохраните проверенный вариант:", dialog))
        editor = QPlainTextEdit(dialog)
        editor.setPlainText(answer)
        layout.addWidget(editor, 1)
        general = QCheckBox(
            "Общее правило для всех проектов · добавить в текстовое обучение",
            dialog,
        )
        general.setVisible(operation == "chat")
        if operation == "chat":
            scope_note = QLabel("Без флажка ответ будет учитываться только для этой скважины проекта.", dialog)
            scope_note.setWordWrap(True)
            layout.addWidget(scope_note)
        layout.addWidget(general)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        corrected = editor.toPlainText().strip()
        try:
            save_correction(
                question, answer, corrected, operation=operation,
                image_path=str(data.get("image_path") or ""),
                project=self._expert_project_key(), well=self._active_well_name,
                general=general.isChecked() if operation == "chat" else False,
            )
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Поправка не сохранена", str(exc))
            return
        history = self._chat_photo_history if operation == "photo_chat" else self._vlm_history
        for turn in reversed(history):
            if turn.get("role") == "assistant" and turn.get("content") == answer:
                turn["content"] = corrected
                break
        self.assistant_panel.append_message(
            "system",
            "Поправка специалиста сохранена. При повторе вопроса будет использован проверенный ответ. "
            "Для изменения весов Qwen соберите поправки в центре обучения и запустите дообучение.\n\n" + corrected,
        )

    def _search_local_folders(self, question: str, *, display_question: bool = True) -> None:
        if display_question:
            self.assistant_panel.append_message("user", "Поиск: " + question)
        if self._local_search_worker is not None and self._local_search_worker.isRunning():
            self.assistant_panel.append_message("system", "Поиск уже выполняется.")
            return
        photo_only = wants_photo_file(question)
        folders = self._knowledge_folders(question, include_archive=True)
        available = [folder for folder in folders if Path(folder).is_dir()]
        missing = [folder for folder in folders if not Path(folder).is_dir()]
        known_photos = [(record.path, record.well_name) for record in self._records] if photo_only else []
        if not available and not known_photos:
            self.assistant_panel.append_message("assistant", "Нет доступных локальных папок для поиска. Подключите их через «Папки…».")
            return
        self.assistant_panel.set_busy(True)
        self.assistant_panel.status.setText("●  Ищу в локальных папках…")
        worker = LocalSearchWorker(
            question, available, self, photo_only=photo_only, known_photos=known_photos,
        )
        self._local_search_worker = worker
        worker.results_ready.connect(lambda result: self._on_local_search_results(question, result, available, missing))
        worker.search_failed.connect(lambda message: self.assistant_panel.append_message("system", message))
        worker.finished.connect(self._on_local_search_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _on_local_search_results(self, question: str, result: dict, available: list[str], missing: list[str]) -> None:
        hits = list(result.get("hits") or [])
        stats = dict(result.get("stats") or {})
        if wants_photo_file(question):
            if hits:
                note = f"Показано найденных фото: {len(hits)}. Каждый путь ведёт к существующему локальному файлу."
                if stats.get("truncated"):
                    note += " Поиск достиг лимита файлов; результаты могут быть неполными."
                if missing:
                    note += " Недоступны папки: " + ", ".join(missing[:2]) + "."
                self.assistant_panel.append_photo_matches(hits, note)
            else:
                message = "Фото по этому запросу не найдено среди доступных файлов."
                if missing:
                    message += " Недоступны папки: " + ", ".join(missing[:2]) + ". Проверьте их в «Папки…»."
                if stats.get("truncated"):
                    message += " Поиск достиг лимита файлов и мог пропустить снимки."
                self.assistant_panel.append_message("assistant", message)
            return
        if hits:
            details = "\n\n".join(
                f"[{index}] {Path(hit['path']).parent.name}/{Path(hit['path']).name}\n{hit['excerpt']}"
                + (f"\n{hit['path']}" if wants_sources(question) else "")
                for index, hit in enumerate(hits, 1)
            )
            message = "Найдено в локальных материалах:\n" + details
        else:
            message = "По этому запросу в просмотренных файлах совпадений не найдено. Проверьте, где лежат описания, через «Папки…»."
        message += f"\n\nПросмотрено документов: {stats.get('documents_read', 0)}; файлов и записей архивов: {stats.get('entries_seen', 0)}."
        if stats.get("truncated"):
            message += " Поиск достиг лимита большого архива и мог пропустить часть файлов."
        if missing:
            message += "\n\nНедоступные папки не просмотрены: " + ", ".join(missing[:3])
        self.assistant_panel.append_message("assistant", message)

    def _on_local_search_finished(self) -> None:
        self._local_search_worker = None
        self.assistant_panel.set_busy(False)

    def _describe_selected_facies(self) -> None:
        if self._vlm_target is None or self._vlm_target[1] is None:
            self.assistant_panel.append_message("system", "Сначала выберите фацию на фото или в дереве проекта.")
            return
        self.assistant_panel.append_message("user", "Описать выбранную фацию по фото")
        self._send_vlm_request("describe")

    def _on_vlm_reply(self, result: dict) -> None:
        try:
            self._handle_vlm_reply(result)
        finally:
            self._refresh_chat_status()
            self._schedule_vlm_dispatch()

    def _handle_vlm_reply(self, result: dict) -> None:
        if self._analysis_pending is not None and str(result.get("id")) == self._analysis_pending[0]:
            self._on_cv1_interval_reply(result)
            return
        pending = self._vlm_pending
        if pending is None or str(result.get("id")) != pending[0]:
            return
        self._vlm_pending = None
        meta, self._vlm_pending_meta = self._vlm_pending_meta or {}, None
        photo_path = str(meta.get("photo_path") or "")
        selected_job, self._selected_description_job = self._selected_description_job, None
        self._cleanup_vlm_image(meta.get("temporary_image"))
        if not result.get("ok"):
            self.assistant_panel.append_message("system", str(result.get("error") or "Локальная VLM не ответила"))
            return
        operation, target = pending[1], pending[2]
        hits = list(result.get("hits") or [])
        if operation == "search":
            if not hits:
                self.assistant_panel.append_message("assistant", "Совпадений в выбранных локальных папках не найдено.")
            else:
                self.assistant_panel.append_message("assistant", "Найдено в локальных файлах:\n" + "\n\n".join(
                    f"[{i}] {hit['path'] if wants_sources(pending[3]) else Path(hit['path']).name}\n{hit['excerpt']}"
                    for i, hit in enumerate(hits, 1)
                ))
            return
        answer = str(result.get("answer") or "").strip()
        incomplete = bool(result.get("incomplete"))
        if operation in {"chat", "photo_chat"} and not wants_sources(pending[3]):
            answer = hide_unsolicited_sources(answer)
        if not answer and operation in {"chat", "photo_chat"}:
            message = (
                "Модель не смогла завершить описание фото. Попробуйте указать один конкретный признак."
                if incomplete else
                "Локальная VLM завершила запрос без текста. Попробуйте короткий вопрос или уменьшите изображение."
            )
            self.assistant_panel.append_message("system", message)
            return
        if operation in {"chat", "photo_chat"}:
            history_key = str(Path(photo_path).resolve()) if photo_path else ""
            history = list(self._chat_photo_histories.get(history_key, [])) if operation == "photo_chat" else self._vlm_history
            history.extend([
                {"role": "user", "content": pending[3]},
                {"role": "assistant", "content": answer},
            ])
            if operation == "photo_chat":
                self._chat_photo_histories[history_key] = history[-12:]
                if self._chat_photo_path is not None and str(self._chat_photo_path.resolve()) == history_key:
                    self._chat_photo_history = history[-12:]
            else:
                self._vlm_history = history[-12:]
        if operation == "describe":
            fields = result.get("fields")
            found = next(((record, layer) for record in self._records for layer in record.detections if layer is target), None)
            if not isinstance(fields, dict) or not str(fields.get("description") or "").strip() or found is None:
                self.assistant_panel.append_message("system", "VLM не вернула корректные поля. Ответ оставлен для ручной проверки:\n" + answer)
                return
            record, layer = found
            if selected_job is not None:
                try:
                    apply_interval_description(record, selected_job, result)
                except ValueError as exc:
                    mark_interval_failure(selected_job, str(exc))
                    self.assistant_panel.append_message("system", str(exc))
                    return
                self._refresh_facies_views(record)
                self.assistant_panel.append_message("assistant", "Черновик описания привязан к выбранному полигону. Проверьте его перед экспортом:\n" + str(layer.attributes.get("__vlm_description") or ""))
                return
            layer.attributes.update({
                "__vlm_facies_association": str(fields.get("facies_association") or ""),
                "__vlm_depositional_environment": str(fields.get("depositional_environment") or ""),
                "__vlm_description": str(fields.get("description") or ""),
            })
            if self._project_folder is not None:
                self._save_current_project(self._project_folder)
            self.assistant_panel.append_message("assistant", "Поля VLM заполнены для выбранной фации. Проверьте перед экспортом:\n" + answer)
            return
        source_note = "\n\nНайденные источники: " + "; ".join(
            f"[{i}] {hit['path']}" for i, hit in enumerate(hits, 1)
        ) if hits and wants_sources(pending[3]) else ""
        review_note = "\n\nЧерновик по фото: границы столбиков и геологические признаки проверьте на оригинале в полном разрешении." if operation == "photo_chat" else ""
        photo_label = (
            f"Описание фото «{Path(photo_path).name}»:\n"
            if operation == "photo_chat" and photo_path else ""
        )
        completion_note = (
            "\n\nМодель остановилась до завершения ответа; показаны только законченные предложения. "
            "Для подробностей уточните интересующий признак."
            if incomplete else ""
        )
        correction = None
        if operation == "photo_chat" and photo_path:
            correction = {
                "operation": "photo_chat", "question": pending[3], "answer": answer,
                "image_path": photo_path,
            }
        elif operation == "chat" and not needs_selected_image(pending[3]):
            correction = {"operation": "chat", "question": pending[3], "answer": answer}
        self.assistant_panel.append_message(
            "assistant", photo_label + answer + completion_note + source_note + review_note,
            correction=correction,
        )

    def _create_toolbar_menu(self, title: str, actions, tooltip: str | None = None) -> QToolButton:
        """The QToolBar + QToolButton + QMenu construction used in Kern Analyzer."""
        menu = QMenu(title, self)
        menu.setObjectName("toolbarDropdown")
        for entry in actions:
            action_title, handler, action_tip = entry[:3]
            action = QAction(action_title, self)
            action.triggered.connect(handler)
            action.setToolTip(action_tip)
            if len(entry) > 3:
                action.setShortcut(entry[3])
                self.addAction(action)
            menu.addAction(action)

        button = QToolButton(self)
        button.setObjectName("toolbarMenuButton")
        button.setText(f"{title} ▾")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(menu)
        button.setToolTip(tooltip or title)
        return button

    def _register_shortcuts(self) -> None:
        """Keyboard controls stay available without adding extra top-panel menus."""
        for shortcut, handler in (
            ("Ctrl+Z", self.undo_delete),
            ("Delete", self.workspace.delete_selected_photos),
            ("Ctrl+0", self.workspace.fit_content),
        ):
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            self.addAction(action)

    def return_to_photos(self) -> None:
        self.sheets.setCurrentWidget(self.workspace)
        if self.workspace.focus_photos():
            self.statusBar().showMessage("Возврат к исходным фотографиям и разметке")
        else:
            self.statusBar().showMessage("Сначала загрузите фотографии")

    def _projects_root(self) -> Path:
        return user_data_root() / "projects"

    def _trained_models_root(self) -> Path:
        """Published models live here instead of inside a training-run tree."""
        return user_data_root() / "models" / "trained"

    def _core_column_models_root(self) -> Path:
        return user_data_root() / "models" / "core_columns"

    def _pdf_pages_root(self) -> Path:
        return self._projects_root() / "pdf_pages"

    @staticmethod
    def _safe_folder_name(value: str) -> str:
        name = "".join("_" if char in '<>:"/\\|?*' else char for char in value).strip(". ")
        return name or "untitled_project"

    def _refresh_project_tree(self) -> None:
        if not hasattr(self, "project_tree"):
            return
        self.project_tree.clear()
        opened = QTreeWidgetItem([f"Открыт: {self._project_title}"])
        opened.setExpanded(True)
        self.project_tree.addTopLevelItem(opened)
        well_nodes: dict[str, QTreeWidgetItem] = {}
        for well_name in self._well_names():
            well_node = QTreeWidgetItem([f"\u0421\u041a\u0412\u0410\u0416\u0418\u041d\u0410 \u00b7 {well_name}"])
            well_node.setData(0, self.TREE_KIND_ROLE, "well")
            well_node.setData(0, self.TREE_VALUE_ROLE, well_name)
            well_node.setExpanded(True)
            opened.addChild(well_node)
            well_nodes[well_name] = well_node
        for record in self._records:
            child = QTreeWidgetItem([Path(record.path).name])
            child.setData(0, self.TREE_KIND_ROLE, "photo")
            child.setData(0, self.TREE_VALUE_ROLE, record.identifier)
            child.setToolTip(0, record.path)
            well_nodes.setdefault(record.well_name, opened).addChild(child)
            for display_index, detection in enumerate(StackColumnItem._sort_by_reading_order(record.detections), start=1):
                source_index = next(
                    index for index, source_detection in enumerate(record.detections) if source_detection is detection
                )
                interval = ""
                if detection.depth_from is not None or detection.depth_to is not None:
                    top = "?" if detection.depth_from is None else f"{detection.depth_from:g}"
                    base = "?" if detection.depth_to is None else f"{detection.depth_to:g}"
                    interval = f" · {top}–{base} м"
                if detection.attributes.get("__core_column_description") == "1":
                    try:
                        column_index = int(detection.attributes.get("__core_column_index", str(display_index - 1))) + 1
                    except (TypeError, ValueError):
                        column_index = display_index
                    item_title = f"Столбик {column_index}: {detection.label}{interval}"
                else:
                    item_title = f"Слой {display_index}: {detection.label}{interval}"
                layer = QTreeWidgetItem([item_title])
                layer.setData(0, self.TREE_KIND_ROLE, "layer")
                layer.setData(0, self.TREE_VALUE_ROLE, record.identifier)
                layer.setData(0, self.TREE_INDEX_ROLE, source_index)
                child.addChild(layer)
            child.setExpanded(True)
        for well_name in ():
            well = QTreeWidgetItem([f"{well_name} · без керна"])
            well.setData(0, self.TREE_KIND_ROLE, "well")
            well.setData(0, self.TREE_VALUE_ROLE, well_name)
            opened.addChild(well)

        saved_root = QTreeWidgetItem(["Сохранённые проекты"])
        saved_root.setExpanded(True)
        self.project_tree.addTopLevelItem(saved_root)
        root = self._projects_root()
        if root.is_dir():
            for folder in sorted((item for item in root.iterdir() if (item / MANIFEST_NAME).is_file()), key=lambda item: item.name.casefold()):
                item = QTreeWidgetItem([folder.name.removesuffix(".kern_analyzer")])
                item.setData(0, self.TREE_KIND_ROLE, "project")
                item.setData(0, self.TREE_VALUE_ROLE, str(folder))
                item.setToolTip(0, str(folder))
                saved_root.addChild(item)
        self._update_header_context()

    def _activate_project_tree_item(self, item: QTreeWidgetItem, _column: int) -> None:
        kind = item.data(0, self.TREE_KIND_ROLE)
        value = item.data(0, self.TREE_VALUE_ROLE)
        if kind == "project" and value:
            self._open_project(Path(str(value)))
        elif kind == "well" and value:
            self._active_well_name = str(value)
            self._update_header_context()
        elif kind == "photo" and value:
            self.workspace.focus_photo(str(value))
            record = next((record for record in self._records if record.identifier == str(value)), None)
            if record is not None:
                self._vlm_target = (record, None)
                self._active_well_name = record.well_name
                self._update_header_context()
        elif kind == "layer" and value is not None:
            layer = self._tree_layer(str(value), item.data(0, self.TREE_INDEX_ROLE))
            if layer:
                record, detection = layer
                self.workspace.focus_facies(record, detection)
                self._vlm_target = (record, detection)
                self._active_well_name = record.well_name
                self._update_header_context()

    def _select_tree_context(self, item: QTreeWidgetItem, _column: int) -> None:
        kind = item.data(0, self.TREE_KIND_ROLE)
        value = item.data(0, self.TREE_VALUE_ROLE)
        if kind == "well" and value:
            self._active_well_name = str(value)
            self._vlm_target = None
        elif kind in {"photo", "layer"} and value:
            record = next((record for record in self._records if record.identifier == str(value)), None)
            if record is not None:
                detection = None
                if kind == "layer":
                    found = self._tree_layer(str(value), item.data(0, self.TREE_INDEX_ROLE))
                    detection = found[1] if found else None
                self._vlm_target = (record, detection)
                self._active_well_name = record.well_name
        self._update_header_context()

    def _open_project_tree_context_menu(self, position) -> None:
        item = self.project_tree.itemAt(position)
        if item is None:
            return
        if item.data(0, self.TREE_KIND_ROLE) == "well":
            well_name = str(item.data(0, self.TREE_VALUE_ROLE) or "")
            if well_name:
                menu = QMenu(self)
                edit_passport = menu.addAction("Заполнить паспорт итогового Excel…")
                edit_passport.triggered.connect(lambda _checked=False, name=well_name: self._edit_excel_report_inputs(name))
                menu.exec(self.project_tree.viewport().mapToGlobal(position))
            return
        if item.data(0, self.TREE_KIND_ROLE) != "layer":
            return
        value = item.data(0, self.TREE_VALUE_ROLE)
        layer = self._tree_layer(str(value), item.data(0, self.TREE_INDEX_ROLE))
        if layer is None:
            return
        record, detection = layer
        menu = QMenu(self)
        edit = menu.addAction("Редактировать параметры фации")
        edit.triggered.connect(lambda: self._open_facies_editor(record, detection))
        menu.addSeparator()
        move_up = menu.addAction("↑ Переместить слой выше")
        move_up.setEnabled(self._can_move_layer(record, detection, -1))
        move_up.triggered.connect(lambda: self._move_layer(record, detection, -1))
        move_down = menu.addAction("↓ Переместить слой ниже")
        move_down.setEnabled(self._can_move_layer(record, detection, 1))
        move_down.triggered.connect(lambda: self._move_layer(record, detection, 1))
        menu.exec(self.project_tree.viewport().mapToGlobal(position))

    def _tree_layer(self, identifier: str, index) -> tuple[PhotoRecord, object] | None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        record = next((item for item in self._records if item.identifier == identifier), None)
        if record is None or not 0 <= index < len(record.detections):
            return None
        return record, record.detections[index]

    def new_project(self) -> None:
        if self._analysis_active or self._segmentation_thread is not None or self._vlm_pending is not None:
            self.assistant_panel.append_message("system", "Дождитесь завершения текущего анализа перед сменой проекта.")
            return
        if (self._edit_autosave.isActive() or self._edit_unsaved) and not self._flush_edit_autosave():
            return
        if self._records:
            answer = QMessageBox.question(
                self,
                "Новый проект",
                "Создать новый проект? Несохранённые изменения текущего проекта будут потеряны.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.workspace.clear_workspace()
        self._records = []
        self._deleted_records.clear()
        self._pending_photo_imports.clear()
        self._project_folder = None
        self._expert_session_id = uuid4().hex
        self._primary_facies_catalog = self._load_global_primary_catalog()
        # The last explicitly chosen YOLO belongs to the application, not to
        # the project being cleared. Keep it for the next photos as well.
        self._project_title = "Новый проект"
        self._gis_data = None
        self._rigis_data = None
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order.clear()
        self._well_depth_ranges.clear()
        self._well_depth_settings.clear()
        self._well_depth_references.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._well_report_inputs.clear()
        self._vlm_target = None
        self._vlm_history.clear()
        self._chat_pending_excel_path = None
        self._chat_document_path = None
        self._chat_photo_path = None
        self._chat_record_ids.clear()
        self._chat_pending_photos = []
        self.assistant_panel.clear_photo()
        self._active_well_name = "Скважина 1"
        if hasattr(self, "_save_indicator"):
            self._save_indicator.setText("●  Локальный проект")
        self.setWindowTitle("Kern Analyzer — Новый проект")
        self._refresh_project_tree()
        self._refresh_training_bar()
        model = self._resolve_model_path()
        model_note = f" · модель границ: {model.name}" if model else ""
        self.statusBar().showMessage("Создан новый проект" + model_note)
        if model is None:
            self._show_remembered_model_status()

    def open_project_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Открыть проект Kern Analyzer", str(self._projects_root()))
        if folder:
            self._open_project(Path(folder))

    def _open_project(self, folder: Path) -> None:
        if self._analysis_active or self._vlm_pending is not None or self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Подождите", "Дождитесь окончания сегментации перед открытием проекта.")
            return
        if (self._edit_autosave.isActive() or self._edit_unsaved) and not self._flush_edit_autosave():
            return
        try:
            title, records, positions, missing, wells = load_project(folder)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось открыть проект", str(exc))
            return
        self.workspace.clear_workspace()
        self._records = records
        self._primary_facies_catalog = load_catalog_snapshot(folder / SNAPSHOT_NAME) or self._load_global_primary_catalog()
        self._vlm_target = None
        self._vlm_history.clear()
        self._chat_pending_excel_path = None
        self._chat_document_path = None
        self._chat_photo_path = None
        self._chat_record_ids.clear()
        self._chat_pending_photos = []
        self.assistant_panel.clear_photo()
        for record in records:
            if record.analysis_mode == PIPELINE_MODE:
                if record.core_columns:
                    self._update_depth_segments_for_columns(record)
                    self._ensure_core_column_description_rows(record)
                continue
            for detection in record.detections:
                resolved = self._resolve_training_class(
                    detection.label, (detection.attributes or {}).get("Индекс фации")
                )
                if resolved:
                    for key, value in resolved["metadata"].items():
                        detection.attributes.setdefault(key, value)
            if record.core_columns and not record.detections:
                # Older projects can have detected columns but no facies rows.
                # Show their entire physical core as explicitly unresolved.
                record.detections = postprocess_detections(record, [])
        self._deleted_records.clear()
        self._project_folder = folder
        self._edit_unsaved = False
        self._edit_autosave.stop()
        self._project_title = title
        # Keep the explicitly selected global model. Opening a project never
        # promotes an old training run implicitly.
        self._gis_data = None
        self._rigis_data = None
        self._well_logs.clear()
        recorded_wells = {record.well_name for record in records}
        self._well_order = [
            str(item["name"])
            for item in sorted(wells, key=lambda item: int(item.get("order", 1_000_000)))
            if str(item.get("name") or "")
        ]
        self._empty_wells = [str(item["name"]) for item in wells if str(item["name"]) not in recorded_wells]
        self._well_intervals = {
            str(item["name"]): (float((item.get("core_interval") or item["interval"])[0]), float((item.get("core_interval") or item["interval"])[1]))
            for item in wells
            if isinstance(item.get("core_interval") or item.get("interval"), list) and len(item.get("core_interval") or item.get("interval")) == 2
        }
        self._well_depth_ranges = {
            str(item["name"]): (float(item["well_depth_range"][0]), float(item["well_depth_range"][1]))
            for item in wells
            if isinstance(item.get("well_depth_range"), list) and len(item["well_depth_range"]) == 2
        }
        self._well_depth_references = {
            str(item["name"]): str(item.get("depth_reference"))
            for item in wells
            if item.get("depth_reference")
        }
        self._well_depth_settings = {
            str(item["name"]): dict(item["depth_settings"])
            for item in wells
            if isinstance(item.get("depth_settings"), dict)
        }
        self._well_interpretations = {
            str(item["name"]): [dict(interval) for interval in item.get("interpretations", []) if isinstance(interval, dict)]
            for item in wells
            if item.get("interpretations")
        }
        self._well_report_inputs = {
            str(item["name"]): dict(item["excel_report_inputs"])
            for item in wells
            if isinstance(item.get("excel_report_inputs"), dict)
        }
        self._active_well_name = records[0].well_name if records else "Скважина 1"
        for record in records:
            self.workspace.restore_photo(record, positions.get(record.identifier, QPointF()))
        if records:
            self._show_stack()
        self.setWindowTitle(f"Kern Analyzer — {title}")
        self._refresh_project_tree()
        self._refresh_training_bar()
        message = f"Открыт проект: {title} · фото: {len(records)}"
        model = self._resolve_model_path()
        if model is not None:
            message += f" · модель границ: {model.name}"
        elif self._selected_model_path is not None:
            message += " · сохранённый best.pt недоступен"
            self._show_remembered_model_status()
        if missing:
            message += f" · не найдены: {len(missing)}"
        self.statusBar().showMessage(message)
        if self._has_waiting_cv1_descriptions():
            QTimer.singleShot(0, self._start_cv1_descriptions)

    def save_project(self) -> None:
        if self._project_folder is None:
            self.save_project_as()
            return
        self._save_current_project(self._project_folder)

    def save_project_as(self) -> None:
        title, accepted = QInputDialog.getText(self, "Сохранить проект", "Название проекта:", text=self._project_title)
        if not accepted or not title.strip():
            return
        safe_title = "".join("_" if char in '<>:"/\\|?*' else char for char in title.strip()).strip(". ")
        if not safe_title:
            QMessageBox.warning(self, "Название проекта", "Укажите допустимое название.")
            return
        folder = self._projects_root() / f"{safe_title}.kern_analyzer"
        if folder.exists() and (folder / MANIFEST_NAME).is_file():
            answer = QMessageBox.question(
                self,
                "Перезаписать проект",
                f"Проект «{safe_title}» уже существует. Перезаписать его?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self._project_title = title.strip()
        self._save_current_project(folder)

    def _save_current_project(self, folder: Path) -> bool:
        previous_feedback_key = self._expert_project_key()
        try:
            save_project(folder, self._project_title, self._records, self.workspace.photo_positions(), self._project_well_metadata())
            if self._primary_facies_catalog is not None:
                save_catalog_snapshot(folder / SNAPSHOT_NAME, self._primary_facies_catalog)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось сохранить проект", str(exc))
            return False
        self._project_folder = folder
        self._edit_unsaved = False
        self._edit_autosave.stop()
        feedback_warning = ""
        try:
            reassign_project(previous_feedback_key, self._expert_project_key())
        except OSError as exc:
            feedback_warning = f" · память поправок не перенесена: {exc}"
        self.setWindowTitle(f"Kern Analyzer — {self._project_title}")
        if hasattr(self, "_save_indicator"):
            self._save_indicator.setText("●  Только что сохранено")
        self._refresh_project_tree()
        self.statusBar().showMessage(f"Проект сохранён: {folder.name}{feedback_warning}")
        return True

    def _schedule_edit_autosave(self) -> None:
        if not self._records:
            return
        self._edit_unsaved = True
        if hasattr(self, "_save_indicator"):
            self._save_indicator.setText("●  Сохранение изменений…")
        self._edit_autosave.start()

    def _flush_edit_autosave(self) -> bool:
        self._edit_autosave.stop()
        if not self._records:
            self._edit_unsaved = False
            return True
        folder = self._project_folder
        if folder is None:
            folder = self._projects_root() / f"Черновик-{self._expert_session_id}.kern_analyzer"
        saved = self._save_current_project(folder)
        if not saved and hasattr(self, "_save_indicator"):
            self._save_indicator.setText("●  Изменения не сохранены")
        return saved

    def open_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Загрузить фото или PDF",
            "",
            "Изображения и PDF (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf);;Изображения (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;PDF (*.pdf);;Все файлы (*)",
        )
        if not paths:
            return

        known_wells = self._well_names()
        suggested_well = self._active_well_name if self._active_well_name in known_wells else f"Скважина {len(known_wells) + 1}"
        well_name, accepted = QInputDialog.getText(
            self,
            "Скважина для фотографий",
            "Название скважины:",
            text=suggested_well,
        )
        if not accepted or not well_name.strip():
            return
        well_name = well_name.strip()
        self._active_well_name = well_name
        self._settings.setValue("last/active_well", well_name)

        image_paths: list[Path] = []
        pdf_errors: list[str] = []
        for selected_path in map(Path, paths):
            if selected_path.suffix.casefold() != ".pdf":
                image_paths.append(selected_path)
                continue
            try:
                image_paths.extend(render_pdf_pages(selected_path, self._pdf_pages_root()))
            except Exception as exc:
                pdf_errors.append(f"{selected_path.name}: {exc}")

        for image_path in image_paths:
            photo_depth_from, photo_depth_to = self._photo_interval_from_path(image_path)
            self._pending_photo_imports.append((image_path, well_name, photo_depth_from, photo_depth_to))

        if not image_paths:
            detail = "\n".join(pdf_errors)
            QMessageBox.warning(self, "Kern Analyzer", f"Не удалось открыть выбранные изображения.\n{detail}".strip())
            return
        message = f"Поставлено в очередь фото: {len(image_paths)} · загружаю партиями по {self.PHOTO_LOAD_BATCH_SIZE}…"
        if pdf_errors:
            message += f" · PDF с ошибкой: {len(pdf_errors)}"
        self.statusBar().showMessage(message)
        self._load_next_photo_batch()

    def _load_next_photo_batch(self) -> None:
        """Read and analyse a bounded batch so a large folder stays responsive."""
        if self._segmentation_thread is not None or not self._pending_photo_imports:
            return
        records: list[PhotoRecord] = []
        new_records: list[PhotoRecord] = []
        failed_paths: list[str] = []
        while self._pending_photo_imports and len(records) < self.PHOTO_LOAD_BATCH_SIZE:
            image_path, well_name, photo_depth_from, photo_depth_to = self._pending_photo_imports.popleft()
            existing = self._record_for_chat_photo(image_path)
            if existing is not None:
                if not any(item.identifier == existing.identifier for item in records):
                    records.append(existing)
                continue
            try:
                pixmap = load_working_pixmap(image_path)
            except Exception:
                failed_paths.append(image_path.name)
                continue
            if pixmap.isNull():
                failed_paths.append(image_path.name)
                continue
            record = PhotoRecord(
                identifier=str(uuid4()), path=str(image_path), pixmap=pixmap, well_name=well_name,
                photo_depth_from=photo_depth_from, photo_depth_to=photo_depth_to,
                analysis_mode=PIPELINE_MODE,
            )
            self._records.append(record)
            self._chat_record_ids[str(image_path.resolve())] = record.identifier
            records.append(record)
            new_records.append(record)
        if failed_paths:
            names = ", ".join(failed_paths[:5])
            suffix = f" и ещё {len(failed_paths) - 5}" if len(failed_paths) > 5 else ""
            self.assistant_panel.append_message(
                "system", f"Не удалось прочитать фото ({len(failed_paths)}): {names}{suffix}. Остальные файлы продолжат обрабатываться."
            )
        if not records:
            self.statusBar().showMessage("Не удалось прочитать фото из очереди.")
            return
        if new_records:
            self.workspace.add_photos(new_records)
        self._vlm_target = (records[-1], None)
        self._update_header_context()
        self._deleted_records.clear()
        self._refresh_project_tree()
        self._refresh_training_bar()
        if self._resolve_model_path() is None:
            columns_found = unresolved_intervals = column_rows_created = 0
            for record in records:
                record.core_columns = self._suggest_core_columns(record)
                self._update_depth_segments_for_columns(record)
                if record.analysis_mode == PIPELINE_MODE:
                    column_rows_created += self._ensure_core_column_description_rows(record)
                else:
                    self._fill_unclassified_columns(record)
                columns_found += len(record.core_columns)
                unresolved_intervals += sum(item.label == UNRECOGNIZED_FACIES for item in record.detections)
                self.workspace.update_photo_detections(record)
            if self._well_names():
                self._show_stack()
            self._refresh_project_tree()
            self._refresh_training_bar()
            self.statusBar().showMessage(
                f"Загружено {len(records)} фото · столбиков: {columns_found} · "
                f"интервалов для проверки: {unresolved_intervals} · "
                f"VLM опишет целые столбики без масок: {column_rows_created}"
            )
            if self._pending_photo_imports:
                QTimer.singleShot(25, self._load_next_photo_batch)
            elif self._has_waiting_cv1_descriptions():
                self._start_cv1_descriptions()
            return
        self.statusBar().showMessage(
            f"Загружено {len(records)} фото · в очереди: {len(self._pending_photo_imports)} · запускаю сегментацию…"
        )
        self.run_segmentation(records)

    def import_cvat_coco(self) -> None:
        """Create a clean Kern Analyzer project from a human-marked CVAT COCO ZIP."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Импорт CVAT", "Дождитесь окончания текущей операции.")
            return
        archive_path, _ = QFileDialog.getOpenFileName(
            self,
            "Импорт разметки CVAT",
            "",
            "CVAT COCO ZIP (*.zip);;Все файлы (*)",
        )
        if not archive_path:
            return
        if self._records:
            answer = QMessageBox.question(
                self,
                "Импорт CVAT",
                "Импорт создаст новый проект и заменит несохранённые данные в рабочей области. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = self._projects_root() / f"cvat_{Path(archive_path).stem}_{stamp}"
        try:
            records, summary = import_cvat_coco_zip(Path(archive_path), project_folder / "images")
        except CvatImagesMissingError as exc:
            images_folder = QFileDialog.getExistingDirectory(
                self,
                "В ZIP нет изображений — выберите папку исходной задачи CVAT",
                str(Path(archive_path).parent),
            )
            if not images_folder:
                QMessageBox.information(self, "Импорт CVAT", str(exc))
                return
            try:
                records, summary = import_cvat_coco_zip(
                    Path(archive_path), project_folder / "images", Path(images_folder),
                )
            except Exception as retry_exc:
                QMessageBox.critical(self, "Импорт CVAT", str(retry_exc))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Импорт CVAT", str(exc))
            return

        report_path = project_folder / "cvat_import_report.txt"
        report_path.write_text(
            "\n".join([
                f"Архив: {archive_path}",
                f"COCO JSON: {summary.get('annotation_file', '')}",
                f"Фото импортировано: {summary['images']}",
                f"Контуры импортировано: {summary['contours']}",
                f"Классы: {summary['classes']}",
                f"Изображения не найдены: {summary.get('missing_images', 0)}",
                f"Изображения не открылись: {summary.get('unreadable_images', 0)}",
                f"Аннотации пропущены: {summary.get('skipped_annotations', 0)}",
                "",
            ]),
            encoding="utf-8",
        )

        well_name, accepted = QInputDialog.getText(
            self,
            "Импорт CVAT",
            "Название скважины для импортированных фото:",
            text="Скважина 1",
        )
        if not accepted or not well_name.strip():
            return
        for record in records:
            record.well_name = well_name.strip()

        self.workspace.clear_workspace()
        self._records = records
        self._deleted_records.clear()
        self._project_folder = project_folder
        self._project_title = f"CVAT · {Path(archive_path).stem}"
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order = [well_name.strip()]
        self._well_depth_ranges.clear()
        self._well_depth_references.clear()
        self._well_depth_settings.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._well_report_inputs.clear()
        self._active_well_name = well_name.strip()
        self.workspace.add_photos(records)
        self._save_current_project(project_folder)
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        QMessageBox.information(
            self,
            "Импорт CVAT завершён",
            (
                f"Фото: {summary['images']}\n"
                f"Контуры: {summary['contours']}\n"
                f"Классы: {summary['classes']}\n"
                f"Пропущено аннотаций: {summary.get('skipped_annotations', 0)}\n\n"
                "Контуры уже отмечены как проверенные. Откройте слой и заполните 16 параметров описания.\n\n"
                f"Диагностика: {report_path}"
            ),
        )

    def import_cvat_coco_batch(self) -> None:
        """Sequentially import several COCO ZIPs; never unpack the full batch in memory."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Импорт CVAT", "Дождитесь окончания текущей операции.")
            return
        archive_paths, _ = QFileDialog.getOpenFileNames(
            self, "Импорт нескольких разметок CVAT", "", "CVAT COCO ZIP (*.zip);;Все файлы (*)",
        )
        if not archive_paths:
            return
        if self._records:
            answer = QMessageBox.question(
                self, "Импорт CVAT", "Импорт создаст новый проект и заменит текущую рабочую область. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        paths = [Path(path) for path in archive_paths]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = self._projects_root() / f"cvat_batch_{stamp}"

        def progress(current: int, total: int, label: str) -> None:
            self._update_activity(label, current, total)
            QApplication.processEvents()

        self._show_activity("Импорт CVAT", f"Подготовка {len(paths)} ZIP…", len(paths))
        try:
            records, summary = import_cvat_coco_zips(paths, project_folder / "images", progress=progress)
        except CvatImagesMissingError as exc:
            self._close_activity()
            images_folder = QFileDialog.getExistingDirectory(
                self, "В ZIP нет изображений — выберите общую папку исходных фото CVAT", str(paths[0].parent),
            )
            if not images_folder:
                QMessageBox.information(self, "Импорт CVAT", str(exc))
                return
            self._show_activity("Импорт CVAT", f"Подготовка {len(paths)} ZIP…", len(paths))
            try:
                records, summary = import_cvat_coco_zips(
                    paths, project_folder / "images", Path(images_folder), progress,
                )
            except Exception as retry_exc:
                QMessageBox.critical(self, "Импорт CVAT", str(retry_exc))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Импорт CVAT", str(exc))
            return
        finally:
            self._close_activity()

        well_name, accepted = QInputDialog.getText(
            self, "Импорт CVAT", "Название скважины для импортированных фото:", text="Скважина 1",
        )
        if not accepted or not well_name.strip():
            return
        for record in records:
            record.well_name = well_name.strip()
        report_path = project_folder / "cvat_batch_import_report.txt"
        report_path.write_text(
            "\n".join([
                *[f"Архив: {path}" for path in paths],
                f"Фото импортировано: {summary['images']}",
                f"Контуры импортировано: {summary.get('contours', 0)}",
                f"Дубликаты фото: {summary.get('duplicate_images', 0)}",
                f"Пропущено аннотаций: {summary.get('skipped_annotations', 0)}",
                "",
            ]), encoding="utf-8",
        )
        self.workspace.clear_workspace()
        self._records = records
        self._deleted_records.clear()
        self._project_folder = project_folder
        self._project_title = f"CVAT · пакет {len(paths)} ZIP"
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order = [well_name.strip()]
        self._well_depth_ranges.clear()
        self._well_depth_references.clear()
        self._well_depth_settings.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._well_report_inputs.clear()
        self._active_well_name = well_name.strip()
        # ``add_photos`` remains the only long-lived image holder. Archives
        # have already been closed, and photos are on disk in per-job folders.
        self.workspace.add_photos(records)
        self._save_current_project(project_folder)
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        QMessageBox.information(
            self, "Пакетный импорт CVAT завершён",
            f"ZIP: {len(paths)}\nФото: {summary['images']}\nКонтуры: {summary.get('contours', 0)}\n"
            f"Дубликаты: {summary.get('duplicate_images', 0)}\n\nДиагностика: {report_path}",
        )

    def import_excel_photo_files(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Выберите фотографии для сопоставления с Excel", "",
            "Фото керна (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)",
        )
        if filenames:
            self.import_excel_photo_batch(photo_paths_override=[Path(name) for name in filenames])

    def import_excel_photo_batch(self, *, photo_paths_override: list[Path] | None = None,
                                 excel_path_override: Path | None = None) -> None:
        """Build reviewed training candidates from a depth-description Excel and photos."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Импорт Excel + JPG", "Дождитесь окончания текущей операции.")
            return
        if excel_path_override is None:
            excel_path, _ = QFileDialog.getOpenFileName(
                self, "Выберите послойное описание", "",
                "Excel (*.xlsx *.xlsm);;Все файлы (*)",
            )
        else:
            excel_path = str(excel_path_override)
        if not excel_path:
            return
        if photo_paths_override is None:
            photos_folder = QFileDialog.getExistingDirectory(
                self, "Выберите папку с фотографиями керна", str(Path(excel_path).parent),
            )
            if not photos_folder:
                return
            photo_paths = sorted(
                [path for path in Path(photos_folder).rglob("*")
                 if path.is_file() and path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}],
                key=lambda path: path.name.casefold(),
            )
        else:
            photo_paths = sorted(photo_paths_override, key=lambda path: path.name.casefold())
        if self._records:
            answer = QMessageBox.question(
                self,
                "Импорт Excel + JPG",
                "Импорт создаст новый проект и заменит текущую рабочую область. Продолжить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            layers, workbook_issues = read_description_workbook(Path(excel_path))
        except Exception as exc:
            QMessageBox.critical(self, "Импорт Excel + JPG", str(exc))
            return
        if not photo_paths:
            QMessageBox.warning(self, "Импорт Excel + фото", "В выбранной папке не найдены фотографии.")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_folder = self._projects_root() / f"excel_jpg_{Path(excel_path).stem}_{stamp}"
        images_folder = project_folder / "images"
        records: list[PhotoRecord] = []
        wells_in_excel = sorted({layer.well for layer in layers}, key=str.casefold)
        default_well = wells_in_excel[0] if len(wells_in_excel) == 1 else ""
        mapping_rows = suggest_photo_intervals_from_excel([
            (path, *self._detect_photo_interval(path, default_well))
            for path in photo_paths
        ], layers)
        import_summary = workbook_import_summary(layers)
        interval_titles = "; ".join(
            item.title for item in import_summary["core_intervals"][:6]
        )
        if len(import_summary["core_intervals"]) > 6:
            interval_titles += f"; ещё {len(import_summary['core_intervals']) - 6}"
        context_lines = [
            f"Месторождение: {', '.join(import_summary['fields']) or 'не найдено'}",
            f"Скважины: {', '.join(import_summary['wells'])}",
            f"Интервалы керна из Excel: {interval_titles or 'отдельно не заданы'}",
            (
                f"Слои фаций: {import_summary['facies_layers']}; "
                f"с литологическим описанием/параметрами: {import_summary['described_layers']}. "
                "Границы на фото будут вычислены пропорционально реальной мощности каждого слоя."
            ),
        ]
        mapping_dialog = PhotoIntervalMappingDialog(
            mapping_rows,
            self,
            context_text="\n".join(context_lines),
        )
        while mapping_dialog.exec() == mapping_dialog.DialogCode.Accepted:
            try:
                photo_mappings = mapping_dialog.mappings()
                break
            except ValueError as exc:
                QMessageBox.warning(self, "Проверка интервалов", str(exc))
        else:
            return

        report: list[str] = [
            f"Excel: {excel_path}",
            f"Фото найдено: {len(photo_paths)}",
            "Сопоставление выполнено только по подтверждённым интервалам (имена файлов не используются как идентификатор).",
            "",
        ]
        report.extend(f"[Excel] {issue.source}: {issue.message}" for issue in workbook_issues)
        for number, (photo_path, photo_interval) in enumerate(photo_mappings, start=1):
            try:
                report.append(f"[Интервал] {photo_path.name}: {photo_interval.top:g}–{photo_interval.base:g} м (подтверждено пользователем)")
                matching_layers = layers_for_photo(
                    layers,
                    photo_interval.well,
                    photo_interval.top,
                    photo_interval.base,
                )
                if not matching_layers:
                    report.append(
                        f"[Пропуск] {photo_path.name}: в Excel нет пересекающегося интервала фации."
                    )
                    continue
                output_path = images_folder / f"{number:05d}_{photo_path.name}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(photo_path, output_path)
                from PySide6.QtGui import QPixmap

                pixmap = load_working_pixmap(output_path)
                if pixmap.isNull():
                    report.append(f"[Пропуск] {photo_path.name}: JPEG не удалось открыть.")
                    continue
                detections, image_issues, depth_segments = create_depth_bound_detections(
                    output_path,
                    pixmap,
                    photo_interval.top,
                    photo_interval.base,
                    matching_layers,
                )
                report.extend(f"[Проверка] {issue.source}: {issue.message}" for issue in image_issues)
                if not detections:
                    continue
                report.extend(
                    f"[Слой] {photo_path.name}: {item.label} · "
                    f"{item.depth_from:g}–{item.depth_to:g} м · "
                    f"мощность {item.depth_to - item.depth_from:g} м · требуется проверка"
                    for item in detections
                    if item.depth_from is not None and item.depth_to is not None
                )
                records.append(
                    PhotoRecord(
                        identifier=str(uuid4()),
                        path=str(output_path),
                        pixmap=pixmap,
                        detections=detections,
                        well_name=photo_interval.well,
                        photo_depth_from=photo_interval.top,
                        photo_depth_to=photo_interval.base,
                        depth_segments=depth_segments,
                        core_columns=[
                            {key: value for key, value in segment.items() if key in {"left", "top", "right", "bottom"}}
                            for segment in depth_segments
                        ],
                    )
                )
            except Exception as exc:
                report.append(f"[Пропуск] {photo_path.name}: {exc}")

        project_folder.mkdir(parents=True, exist_ok=True)
        report_path = project_folder / "excel_jpg_import_report.txt"
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        if not records:
            QMessageBox.warning(
                self,
                "Импорт Excel + JPG",
                f"Не удалось создать ни одной разметки. Проверьте отчёт:\n{report_path}",
            )
            return

        self.workspace.clear_workspace()
        self._records = records
        self._deleted_records.clear()
        self._project_folder = project_folder
        self._project_title = f"Excel + JPG · {Path(excel_path).stem}"
        self._well_logs.clear()
        self._empty_wells.clear()
        self._well_order = sorted({record.well_name for record in records})
        self._well_depth_ranges.clear()
        self._well_depth_references.clear()
        self._well_depth_settings.clear()
        self._well_intervals.clear()
        self._well_interpretations.clear()
        self._well_report_inputs.clear()
        for well_name in self._well_order:
            well_layers = layers_for_photo(layers, well_name, -1.0e12, 1.0e12)
            if not well_layers:
                continue
            core_tops = [
                item.core_top if item.core_top is not None else item.top
                for item in well_layers
            ]
            core_bases = [
                item.core_base if item.core_base is not None else item.base
                for item in well_layers
            ]
            field_names = list(dict.fromkeys(item.field_name for item in well_layers if item.field_name))
            self._well_intervals[well_name] = (min(core_tops), max(core_bases))
            self._well_depth_ranges[well_name] = (
                min([item.top for item in well_layers] + core_tops),
                max([item.base for item in well_layers] + core_bases),
            )
            self._well_depth_references[well_name] = "MD"
            self._well_depth_settings[well_name] = {
                "coordinate_system": "MD",
                "unit": "m",
                "datum": "RKB / KB",
                "datum_elevation": None,
                "source_curve": "",
                "source": "Excel",
                "field": "; ".join(field_names),
            }
        self._active_well_name = records[0].well_name
        self.workspace.add_photos(records)
        self._save_current_project(project_folder)
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        QMessageBox.information(
            self,
            "Импорт Excel + JPG завершён",
            (
                f"Сопоставлено фото: {len(records)} из {len(photo_paths)}\n"
                f"Наложено слоёв фаций: {sum(len(record.detections) for record in records)}\n"
                "Все автоматически спроецированные границы требуют проверки и подтверждения перед обучением.\n\n"
                f"Диагностика сохранена в:\n{report_path}"
            ),
        )

    def _photo_interval_for_excel_import(self, photo_path: Path, default_well: str) -> tuple[CoreInterval | None, str]:
        """Use filename first, then the printed caption, then an explicit user interval."""
        try:
            return photo_interval_from_filename(photo_path), "имя файла"
        except ValueError:
            pass
        try:
            caption = read_caption_metadata(photo_path)
            return CoreInterval(caption.well or default_well, caption.top, caption.base), "OCR подписи"
        except Exception as exc:
            ocr_reason = str(exc)
        dialog = PhotoIntervalDialog(photo_path.name, well=default_well, parent=self)
        while dialog.exec() == dialog.DialogCode.Accepted:
            try:
                well, top, base = dialog.values()
                return CoreInterval(well, top, base), "введено вручную"
            except ValueError as exc:
                QMessageBox.warning(self, "Интервал фотографии", str(exc))
        self.statusBar().showMessage(f"Пропущено фото без интервала: {photo_path.name} · OCR: {ocr_reason}")
        return None, "не задан"

    def _detect_photo_interval(self, photo_path: Path, default_well: str) -> tuple[CoreInterval | None, str]:
        """Non-interactive prefill for the batch mapping table.

        The resulting value is always displayed for human confirmation; no
        filename or OCR guess can silently bind masks to an Excel layer.
        """
        try:
            interval = photo_interval_from_filename(photo_path)
            return CoreInterval(interval.well or default_well, interval.top, interval.base), "интервал найден в имени — подтвердите"
        except ValueError:
            pass
        try:
            caption = read_caption_metadata(photo_path)
            return CoreInterval(caption.well or default_well, caption.top, caption.base), "OCR подписи — подтвердите"
        except Exception as exc:
            return None, f"интервал не найден: {exc}"

    def _remove_photo(self, identifier: str, position) -> None:
        for index, record in enumerate(self._records):
            if record.identifier != identifier:
                continue
            self._deleted_records.append((record, index, position))
            del self._records[index]
            if self._well_names():
                self._show_stack()
            self._refresh_project_tree()
            self._refresh_training_bar()
            self.statusBar().showMessage("Фото удалено · Ctrl+Z — вернуть")
            return

    def undo_delete(self) -> None:
        if not self._deleted_records:
            self.statusBar().showMessage("Нет удалённых фотографий для восстановления")
            return
        record, index, position = self._deleted_records.pop()
        self._records.insert(index, record)
        self.workspace.restore_photo(record, position)
        if self._well_names():
            self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        self.statusBar().showMessage("Фото восстановлено")

    def run_segmentation(self, records: list[PhotoRecord] | None = None) -> None:
        if self._vlm_closing:
            return
        records = list({id(record): record for record in (records or self._records)}.values())
        if not records:
            QMessageBox.information(self, "Kern Analyzer", "Сначала загрузите фотографии через File → Загрузить фотографии.")
            return
        if self._analysis_active:
            self._queued_records.extend(records)
            self.statusBar().showMessage("Фото добавлены в очередь после описания текущих интервалов…")
            return
        if self._vlm_pending is not None or self._chat_photo_thread is not None or self._training_thread is not None:
            self._queued_records.extend(records)
            self.assistant_panel.append_message("system", "Фото поставлены в очередь обработки после текущего ответа или обучения.")
            self._schedule_vlm_dispatch()
            return
        if self._column_training_thread is not None:
            QMessageBox.information(self, "Распознавание", "Дождитесь окончания обучения модели столбиков.")
            return
        if self._segmentation_thread is not None:
            self._queued_records.extend(records)
            self.statusBar().showMessage("Фото добавлены в очередь сегментации…")
            return

        model_path = self._resolve_model_path()
        if model_path is None:
            self._show_remembered_model_status()
            column_rows_created = 0
            for record in records:
                record.analysis_mode = PIPELINE_MODE
                if not record.core_columns_verified:
                    record.core_columns = self._suggest_core_columns(record)
                self._update_depth_segments_for_columns(record)
                column_rows_created += self._ensure_core_column_description_rows(record)
                self.workspace.update_photo_detections(record)
            self._show_stack()
            self._refresh_project_tree()
            self._refresh_training_bar()
            if self._project_folder is not None:
                self._save_current_project(self._project_folder)
            missing = sum(not record.core_columns for record in records)
            self.statusBar().showMessage(
                f"Фото подготовлены · без найденных столбиков: {missing} · "
                f"локальная VLM опишет целые столбики без масок: {column_rows_created}"
            )
            if self._has_waiting_cv1_descriptions():
                self._start_cv1_descriptions()
            return

        # Sequential GPU stages avoid keeping Qwen and YOLO resident together.
        self._vlm_client.stop()
        self._segmentation_failed = False
        self._analysis_geometry_issues = []
        self._segmentation_updated_count = 0
        self._show_activity("Сегментация", f"Подготовка {len(records)} фото…", len(records))
        self._refresh_training_bar()
        self._segmentation_thread = QThread(self)
        self._segmentation_worker = SegmentationWorker(
            model_path,
            [
                (
                    record.path,
                    record.pixmap.width(),
                    record.pixmap.height(),
                    record.core_columns or None,
                    record.core_columns_verified,
                )
                for record in records
            ],
            self._confidence_threshold,
            self._segmentation_options()["image_size"],
            self._segmentation_options()["max_detections"],
            self._resolve_core_column_model_path(),
            geometry_only=True,
        )
        self._segmentation_worker.moveToThread(self._segmentation_thread)
        self._segmentation_thread.started.connect(self._segmentation_worker.run)
        self._segmentation_worker.progress_changed.connect(self._show_segmentation_progress)
        self._segmentation_worker.image_ready.connect(self._apply_image_detections)
        self._segmentation_worker.geometry_issues_ready.connect(self._on_geometry_issues)
        self._segmentation_worker.failed.connect(self._segmentation_error)
        self._segmentation_worker.finished.connect(self._segmentation_thread.quit)
        self._segmentation_thread.finished.connect(self._on_segmentation_finished)
        self._segmentation_thread.finished.connect(self._segmentation_worker.deleteLater)
        self._segmentation_thread.finished.connect(self._segmentation_thread.deleteLater)
        self._segmentation_thread.start()

    def detect_photo_intervals(self) -> None:
        """Create editable visual interval rows when no description is available."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Автоинтервалы", "Дождитесь окончания текущей операции.")
            return
        if not self._records:
            QMessageBox.information(self, "Автоинтервалы", "Сначала добавьте фотографии керна.")
            return
        answer = QMessageBox.question(
            self,
            "Автоматически определить интервалы",
            "Текущие маски на фото будут заменены визуальными интервалами без назначенной фации. "
            "Глубины будут заполнены, если они указаны в имени фотографии. Продолжить?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        total = len(self._records)
        created = 0
        failed: list[str] = []
        self._show_activity("Определение интервалов", f"Подготовка {total} фото…", total)
        for index, record in enumerate(self._records, start=1):
            self._update_activity(f"Интервалы {index}/{total}: {Path(record.path).name}", index, total)
            try:
                photo_top, photo_base = self._record_photo_interval(record)
                detections, segments = create_automatic_interval_detections(
                    Path(record.path), record.pixmap, photo_top, photo_base,
                )
                record.detections = detections
                record.depth_segments = segments
                created += len(detections)
                self.workspace.update_photo_detections(record)
            except Exception as exc:
                failed.append(f"{Path(record.path).name}: {exc}")
        self._close_activity()
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        detail = "" if not failed else f"\n\nНе обработано: {len(failed)}. Первое: {failed[0]}"
        QMessageBox.information(
            self,
            "Автоинтервалы готовы",
            f"Создано интервалов: {created}.\nНазначьте фацию в карточке нужного интервала, чтобы добавить его в обучение.{detail}",
        )

    def review_visual_contacts(self) -> None:
        """Read-only contact preview; only explicit confirmation replaces masks."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Переходы фаций", "Дождитесь окончания текущей операции.")
            return
        if not self._records:
            QMessageBox.information(self, "Переходы фаций", "Сначала загрузите фотографии керна.")
            return
        from app.ui.dialogs.visual_contact_review_dialog import VisualContactReviewDialog

        current = self._vlm_target[0] if self._vlm_target is not None else None
        dialog = VisualContactReviewDialog(
            self._records, self._record_photo_interval, self, selected_record=current
        )
        dialog.apply_requested.connect(self._apply_visual_contact_drafts)
        dialog.exec()

    def _apply_visual_contact_drafts(self, record: PhotoRecord, drafts: list, segments: list) -> None:
        """Install unverified contact polygons only after the dialog's warning."""
        if not any(item is record for item in self._records):
            return
        for detection in drafts:
            detection.training_ready = False
        record.detections = drafts
        record.depth_segments = segments
        self.workspace.update_photo_detections(record)
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        self.statusBar().showMessage(
            f"Черновые переходы: {len(drafts)} интервалов. Проверьте фации и границы вручную."
        )

    def _resolve_model_path(self) -> Path | None:
        # Preparation and annotation work without weights. A bundled/older
        # model is not necessarily the user's model or even a facies model.
        path = self._selected_model_path
        return path if path is not None and path.is_file() else None

    def _show_remembered_model_status(self) -> None:
        """Restore the last explicit choice without loading checkpoint weights."""
        path = self._selected_model_path
        if path is None:
            return
        if path.is_file():
            self._missing_model_notice = None
            self.statusBar().showMessage(f"Модель границ запомнена: {path.name}")
            return
        key = str(path)
        if self._missing_model_notice == key:
            return
        self._missing_model_notice = key
        message = (
            f"Последняя модель границ «{path.name}» не найдена: {path}. "
            "Добавьте best.pt в чат или выберите файл в настройках модели."
        )
        self.assistant_panel.append_message("system", message)
        self.statusBar().showMessage(message)

    def _resolve_core_column_model_path(self) -> Path | None:
        """Return only a dedicated stage-1 model; never confuse it with facies."""
        root = bundled_root()
        candidates: list[Path | None] = [
            self._selected_column_model_path,
            root / "models" / "core_columns" / "best.pt",
        ]
        trained = sorted(
            self._core_column_models_root().glob("*/best.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(trained)
        return next((path for path in candidates if path is not None and path.is_file()), None)

    def select_core_column_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать модель столбиков",
            str(self._selected_column_model_path.parent) if self._selected_column_model_path else "",
            "Модели YOLO (*.pt);;Все файлы (*)",
        )
        if not file_path:
            return
        try:
            recognizer = CoreColumnRecognizer(Path(file_path))
        except Exception as exc:
            QMessageBox.critical(self, "Модель столбиков", str(exc))
            return
        if recognizer.model is None:
            QMessageBox.warning(
                self,
                "Неверная модель",
                "В модели нет отдельного класса core_column. Выберите модель, обученную в модуле «Столбики».",
            )
            return
        self._selected_column_model_path = Path(file_path)
        self._settings.setValue("core_columns/model_path", str(self._selected_column_model_path))
        self.statusBar().showMessage(f"Модель столбиков выбрана: {self._selected_column_model_path.name}")

    def start_core_column_training(self) -> None:
        if self._analysis_active or self._vlm_pending is not None:
            self.assistant_panel.append_message("system", "Дождитесь завершения анализа перед обучением.")
            return
        """Train stage 1 only from explicitly confirmed column rectangles."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Обучение столбиков", "Дождитесь окончания текущей операции.")
            return
        confirmed = sum(1 for record in self._records if record.core_columns_verified and record.core_columns)
        if confirmed < MIN_COLUMN_TRAINING_PHOTOS:
            QMessageBox.information(
                self,
                "Мало разметки",
                f"Проверьте границы столбиков минимум на {MIN_COLUMN_TRAINING_PHOTOS} фото. Сейчас подтверждено: {confirmed}.",
            )
            return
        base_model = self._resolve_core_column_model_path() or self._resolve_model_path()
        if base_model is None:
            QMessageBox.critical(self, "Нет базовой модели", "Не найден файл .pt, с которого можно начать обучение.")
            return
        epochs, accepted = QInputDialog.getInt(
            self,
            "Обучение столбиков",
            f"Подтверждённых фото: {confirmed}. Число эпох:",
            30, 1, 500, 1,
        )
        if not accepted:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_folder = self._project_folder or (self._projects_root() / self._safe_folder_name(self._project_title))
        training_root = base_folder / "training" / "core_columns"
        try:
            dataset = export_core_column_dataset(self._records, training_root / "datasets" / stamp)
        except Exception as exc:
            QMessageBox.critical(self, "Датасет столбиков", str(exc))
            return
        self._column_training_thread = QThread(self)
        self._column_training_succeeded_model = None
        self._column_training_worker = CoreColumnTrainingWorker(
            base_model,
            Path(dataset["data_yaml"]),
            training_root / "runs" / stamp,
            self._core_column_models_root() / stamp,
            epochs,
        )
        self._column_training_worker.moveToThread(self._column_training_thread)
        self._column_training_thread.started.connect(self._column_training_worker.run)
        self._column_training_worker.progress.connect(self.statusBar().showMessage)
        self._column_training_worker.epoch_progress.connect(self._on_column_training_epoch)
        self._column_training_worker.succeeded.connect(self._on_column_training_succeeded)
        self._column_training_worker.failed.connect(self._on_column_training_failed)
        self._column_training_worker.finished.connect(self._column_training_thread.quit)
        self._column_training_thread.finished.connect(self._on_column_training_finished)
        self._column_training_thread.finished.connect(self._column_training_worker.deleteLater)
        self._column_training_thread.finished.connect(self._column_training_thread.deleteLater)
        self._show_activity("Обучение столбиков", f"Подготовлено фото: {confirmed}", epochs)
        self._refresh_training_bar()
        self._column_training_thread.start()

    def _on_column_training_epoch(self, current: int, total: int) -> None:
        label = f"Обучение столбиков: эпоха {current} из {total}"
        self.statusBar().showMessage(label)
        self._update_activity(label, current, total)

    def _on_column_training_succeeded(self, model_path: str) -> None:
        self._selected_column_model_path = Path(model_path)
        self._column_training_succeeded_model = Path(model_path)
        self._settings.setValue("core_columns/model_path", model_path)
        self.statusBar().showMessage(f"Модель столбиков готова: {model_path}")

    def _on_column_training_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка обучения столбиков", message)

    def _on_column_training_finished(self) -> None:
        trained = self._column_training_succeeded_model is not None and self._column_training_succeeded_model.is_file()
        self._column_training_thread = None
        self._schedule_vlm_dispatch()
        self._column_training_worker = None
        self._column_training_succeeded_model = None
        self._close_activity()
        self._refresh_training_bar()
        if trained:
            unverified = [record for record in self._records if not record.core_columns_verified]
            if unverified:
                self.statusBar().showMessage("Модель столбиков готова · повторяю этап 1 для непроверенных фото…")
                self.run_segmentation(unverified)

    def _best_project_model(self, project_folder: Path | None) -> Path | None:
        """Different control splits are not a valid automatic model ranking."""
        return None

    def select_detection_model(self, *, filename_override: Path | None = None) -> None:
        if self._analysis_active or self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Модель фаций", "Дождитесь окончания текущей операции.")
            return
        if filename_override is not None:
            file_path = str(filename_override)
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Выбрать модель границ керна",
                str(self._selected_model_path.parent) if self._selected_model_path else "",
                "Модели YOLO (*.pt);;Все файлы (*)",
            )
        if not file_path:
            return
        candidate = Path(file_path).expanduser().resolve()
        if not candidate.is_file():
            QMessageBox.warning(self, "Модель сегментации", "Файл модели .pt не найден.")
            return
        try:
            from ultralytics import YOLO
            loaded = YOLO(str(candidate))
            if getattr(loaded, "task", None) != "segment":
                raise ValueError("Выбранная .pt не является YOLO-моделью сегментации (segment).")
            del loaded
        except Exception as exc:
            QMessageBox.warning(self, "Модель сегментации", f"Не удалось открыть .pt:\n{exc}")
            return
        self._selected_model_path = candidate
        self._settings.setValue("facies/model_path", str(self._selected_model_path))
        self._settings.sync()
        self._missing_model_notice = None
        self._refresh_training_bar()
        self.statusBar().showMessage(f"Модель границ запомнена: {self._selected_model_path.name}")
        if self._records:
            self.resegment_facies_with_selected_model()

    def resegment_facies_with_selected_model(self) -> None:
        """Replace all facies masks after an explicit model change/re-run."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Переразметка фаций", "Дождитесь окончания текущей операции.")
            return
        if not self._records:
            QMessageBox.information(self, "Переразметка фаций", "В проекте нет фотографий.")
            return
        model_path = self._resolve_model_path()
        if model_path is None:
            QMessageBox.critical(self, "Модель не найдена", "Сначала выберите фациальную модель .pt.")
            return
        answer = QMessageBox.question(
            self,
            "Переразметить фации?",
            (
                f"Все текущие фациальные маски будут заменены прогнозом модели «{model_path.name}».\n"
                "Подтверждённые границы столбиков керна сохранятся.\n\n"
                "Ручные исправления фаций также будут заменены. Продолжить?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage(f"Модель {model_path.name} выбрана · переразметка не запущена")
            return
        self.run_segmentation(self._records)

    def show_training_history(self) -> None:
        """Show locally saved validation metrics without depending on a server."""
        root = self._trained_models_root() / self._safe_folder_name(self._project_title)
        items: list[tuple[float, str]] = []
        for info_path in root.glob("*/training_info.json"):
            try:
                import json

                info = json.loads(info_path.read_text(encoding="utf-8"))
                metrics = dict(info.get("metrics") or {})
            except (OSError, ValueError, TypeError):
                continue
            m_ap = metrics.get("mAP50-95")
            comparison = dict(info.get("comparison") or {})
            delta = comparison.get("mAP50-95")
            delta_text = "" if delta is None else f" · к исходной: {float(delta):+.1%}"
            metric_text = "нет validation-метрик" if m_ap is None else (
                f"mAP50-95: {float(m_ap):.1%} · "
                f"precision: {float(metrics.get('precision', 0)):.1%} · "
                f"recall: {float(metrics.get('recall', 0)):.1%}{delta_text}"
            )
            sample_counts = dict(info.get("dataset_summary") or {}).get("class_counts", {})
            counts_text = " · ".join(f"{name}: {count}" for name, count in sample_counts.items())
            recommendation = info.get("recommended_epochs")
            completed_epochs = info.get("completed_epochs")
            max_epochs = info.get("max_epochs", info.get("epochs"))
            patience = info.get("early_stopping_patience")
            suffix = f"\n  фации: {counts_text}" if counts_text else ""
            if recommendation:
                suffix += f" · рекомендовано эпох: {recommendation}"
            if completed_epochs:
                suffix += f" · выполнено: {completed_epochs} из {max_epochs}"
            if patience:
                suffix += f" · автостоп: {patience}"
            items.append((float(m_ap) if m_ap is not None else -1.0, f"{info_path.parent.name}: {metric_text}{suffix}"))
        if not items:
            QMessageBox.information(self, "История дообучений", "В этом проекте ещё нет завершённых дообучений с метриками.")
            return
        lines = [text for _, text in sorted(items, reverse=True)]
        QMessageBox.information(self, "История дообучений", "\n".join(lines))

    def show_training_quality(self) -> None:
        self.show_model_preparation()

    def show_model_preparation(self) -> None:
        dialog = ModelPreparationDialog(self._records, self._resolve_model_path(), self, primary_catalog=self._primary_facies_catalog)
        dialog.check_requested.connect(lambda: self._check_model_dataset(dialog))
        dialog.blueprint_requested.connect(self.export_model_blueprint)
        dialog.dataset_requested.connect(self.export_model_dataset)
        dialog.review_requested.connect(lambda: self._review_preparation_issue(dialog))
        dialog.exec()

    def _check_model_dataset(self, dialog) -> None:
        from app.infrastructure.training_dataset import inspect_training_dataset
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            report = inspect_training_dataset(self._records, primary_catalog=self._primary_facies_catalog)
            dialog.show_check(report)
        except Exception as exc:
            QMessageBox.warning(self, "Проверка выборки", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _review_preparation_issue(self, dialog) -> None:
        candidate = next(((record, detection) for record in self._records for detection in record.detections
                          if not self._resolve_training_class(detection.label, detection.attributes.get("Индекс фации"))), None)
        if candidate is None:
            candidate = next(iter(review_queue(self._records)), None)
        if candidate is None:
            QMessageBox.information(self, "Проверка слоёв", "Не найдено непроверенных или несопоставленных фаций. Проверьте выборку перед экспортом.")
            return
        dialog.accept()
        self.workspace.focus_facies(*candidate)
        self._open_facies_editor(*candidate)

    def export_model_blueprint(self) -> None:
        from app.infrastructure.model_blueprint import export_model_blueprint
        parent = QFileDialog.getExistingDirectory(self, "Куда сохранить заготовку для другого компьютера?")
        if not parent:
            return
        output = Path(parent) / f"facies_blueprint_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        try:
            export_model_blueprint(output)
        except Exception as exc:
            QMessageBox.warning(self, "Экспорт заготовки", str(exc))
            return
        QMessageBox.information(self, "Заготовка сохранена", f"Справочник и схема обучения:\n{output}\n\nВеса модели не входят в заготовку. Подключите свою .pt на рабочем компьютере.")

    def export_model_dataset(self) -> None:
        if self._segmentation_thread or self._training_thread or self._column_training_thread:
            QMessageBox.information(self, "Экспорт", "Дождитесь завершения текущей операции.")
            return
        parent = QFileDialog.getExistingDirectory(self, "Куда сохранить обучающую выборку?")
        if not parent:
            return
        output = Path(parent) / f"facies_dataset_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            dataset = export_training_dataset(self._records, output, primary_catalog=self._primary_facies_catalog)
        except Exception as exc:
            QMessageBox.warning(self, "Датасет не подготовлен", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        QMessageBox.information(self, "Датасет сохранён", f"Примеров: {dataset['sample_count']}\n{output}\n\nПеренесите папку целиком. Для экспорта модель и GPU не требуются.")

    def select_confidence_threshold(self) -> None:
        """Let an interpreter trade precision for recall without code changes."""
        value, accepted = QInputDialog.getDouble(
            self,
            "Порог уверенности масок",
            "Показывать маски с уверенностью от 0,01 до 0,99:\n"
            "меньше значение — больше кандидатов, больше ручной проверки.",
            self._confidence_threshold,
            0.01,
            0.99,
            2,
        )
        if not accepted:
            return
        self._confidence_threshold = round(float(value), 2)
        self._settings.setValue("segmentation/confidence_threshold", self._confidence_threshold)
        self.statusBar().showMessage(
            f"Порог масок: {self._confidence_threshold:.0%}. Запустите сегментацию повторно для применения."
        )

    def select_segmentation_profile(self) -> None:
        profiles = {
            "Быстро · 512 px": "fast",
            "Стандарт · 640 px": "standard",
            "Детально · 1024 px": "detailed",
        }
        current = next((index for index, value in enumerate(profiles.values()) if value == self._segmentation_profile), 1)
        selected, accepted = QInputDialog.getItem(
            self,
            "Режим сегментации",
            "Детальный режим точнее на небольших слоях, но требует больше GPU-памяти и времени:",
            list(profiles),
            current,
            False,
        )
        if not accepted:
            return
        self._segmentation_profile = profiles[selected]
        self._settings.setValue("segmentation/profile", self._segmentation_profile)
        settings = self._segmentation_options()
        self._confidence_threshold = settings["confidence"]
        self._settings.setValue("segmentation/confidence_threshold", self._confidence_threshold)
        self.statusBar().showMessage(
            f"Режим сегментации: {selected} · порог {settings['confidence']:.0%} · до {settings['max_detections']} масок. "
            "Запустите сегментацию повторно для применения."
        )

    def _read_segmentation_profile(self) -> str:
        value = str(self._settings.value("segmentation/profile", "standard"))
        return value if value in {"fast", "standard", "detailed"} else "standard"

    def _segmentation_image_size(self) -> int:
        return int(self._segmentation_options()["image_size"])

    def _segmentation_options(self) -> dict[str, int | float]:
        """One predictable quality/speed preset, including detection density."""
        profiles = {
            "fast": {"image_size": 512, "confidence": 0.60, "max_detections": 300},
            "standard": {"image_size": 640, "confidence": 0.50, "max_detections": 1000},
            "detailed": {"image_size": 1024, "confidence": 0.35, "max_detections": 2000},
        }
        return profiles[self._segmentation_profile]

    def configure_core_columns(self) -> None:
        """Persist interpreter-corrected core bounds and re-run only that photo."""
        if self._segmentation_thread is not None or self._training_thread is not None or self._column_training_thread is not None:
            QMessageBox.information(self, "Колонки керна", "Дождитесь завершения текущей обработки.")
            return
        if not self._records:
            QMessageBox.information(self, "Колонки керна", "Сначала добавьте хотя бы одну фотографию.")
            return
        choices = [f"{index + 1}. {Path(record.path).name}" for index, record in enumerate(self._records)]
        selected, accepted = QInputDialog.getItem(self, "Колонки керна", "Выберите фотографию:", choices, 0, False)
        if not accepted:
            return
        record = self._records[choices.index(selected)]
        columns = record.core_columns or self._suggest_core_columns(record)
        dialog = CoreColumnsDialog(columns, (record.pixmap.width(), record.pixmap.height()), self)
        if dialog.exec() != CoreColumnsDialog.DialogCode.Accepted:
            return
        corrected = dialog.columns()
        if not corrected:
            QMessageBox.warning(self, "Колонки керна", "Укажите хотя бы одну корректную колонку (право > лево, низ > верх).")
            return
        record.core_columns = corrected
        record.core_columns_verified = True
        self._update_depth_segments_for_columns(record)
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        self.statusBar().showMessage("Границы колонок сохранены · повторно сегментирую это фото…")
        self.run_segmentation([record])

    def _suggest_core_columns(self, record: PhotoRecord) -> list[dict[str, float]]:
        """Populate the editor from the automatic detector; user remains final authority."""
        try:
            return CoreColumnRecognizer(self._resolve_core_column_model_path()).recognize_path(
                record.path,
                (record.pixmap.width(), record.pixmap.height()),
            )
        except (ImportError, OSError, ValueError):
            return []

    def _ensure_core_column_description_rows(self, record: PhotoRecord) -> int:
        """Represent detected physical columns when no facies masks exist."""
        if record.analysis_mode != PIPELINE_MODE or not record.core_columns:
            return 0
        new_rows = create_core_column_description_detections(record, record.detections)
        record.detections.extend(new_rows)
        return len(new_rows)

    def _has_waiting_cv1_descriptions(self) -> bool:
        return any(
            record.analysis_mode == PIPELINE_MODE
            and detection.attributes.get("__geometry_only") == "1"
            and detection.attributes.get("__interpretation_status", "waiting") == "waiting"
            and not detection.training_ready
            and detection.attributes.get("__manual_interval_edited") != "1"
            for record in self._records for detection in record.detections
        )

    def _fill_unclassified_columns(self, record: PhotoRecord) -> None:
        """Build a complete editable facies tape even before a model is loaded."""
        if record.analysis_mode == PIPELINE_MODE:
            return
        if not record.core_columns:
            return
        self._update_depth_segments_for_columns(record)
        if any(item.label not in {UNRECOGNIZED_FACIES, "Новый контур"} for item in record.detections):
            record.detections = postprocess_detections(record, record.detections)
            return
        photo_top, photo_base = self._record_photo_interval(record)
        try:
            visual_intervals, _ = create_automatic_interval_detections(
                Path(record.path), record.pixmap, photo_top, photo_base,
                core_columns=record.core_columns, label=UNRECOGNIZED_FACIES,
            )
        except (OSError, ValueError):
            visual_intervals = []
        manual = [item for item in record.detections if item.label == "Новый контур"]
        record.detections = postprocess_detections(record, [*manual, *visual_intervals])

    def review_uncertain_masks(self) -> None:
        """Open the highest-value automatic candidate for human verification."""
        queue = review_queue(self._records)
        if not queue:
            QMessageBox.information(
                self,
                "Проверка масок",
                "Нет непроверенных автоматических масок. Запустите сегментацию или добавьте новые фото.",
            )
            return
        record, detection = queue[0]
        self.workspace.focus_facies(record, detection)
        self._open_facies_editor(record, detection)
        QMessageBox.information(
            self,
            "Очередь ручной проверки",
            f"Открыта самая сомнительная маска. Всего в очереди: {len(queue)}.\n"
            "После сохранения откройте этот пункт снова — будет показана следующая.",
        )

    def suggest_lithology_attributes(self) -> None:
        """Run the independent attribute recognizer without replacing facies."""
        if not self._records:
            QMessageBox.information(self, "Атрибуты пород", "Сначала добавьте фото и получите маски фаций.")
            return
        answer = QMessageBox.question(
            self,
            "Подобрать атрибуты пород",
            "Будут рассчитаны визуальные подсказки «Цвет», «Зернистость» и «Слоистость». "
            "Они сохраняются отдельно от наблюдений. В карточке слоя можно принять нужные значения.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        service = LithologyAttributeService()
        changed = 0
        for record in self._records:
            for detection in record.detections:
                suggestions = service.suggest(record, detection)
                if suggestions:
                    detection.attributes["__attribute_suggestions"] = json.dumps(suggestions, ensure_ascii=False)
                    changed += len(suggestions)
            self.workspace.update_photo_detections(record)
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        QMessageBox.information(
            self,
            "Атрибуты пород",
            f"Добавлено рекомендаций: {changed}. Фации и их границы не изменялись.",
        )

    def _read_confidence_threshold(self) -> float:
        try:
            value = float(self._settings.value("segmentation/confidence_threshold", 0.50))
        except (TypeError, ValueError):
            value = 0.50
        return max(0.01, min(0.99, value))

    def start_fine_tune(self) -> None:
        """Export reviewed contours and fine-tune a separate copy of the model."""
        if self._analysis_active or self._vlm_pending is not None:
            self.assistant_panel.append_message("system", "Дождитесь завершения анализа перед обучением.")
            return
        if self._segmentation_thread is not None:
            QMessageBox.information(self, "Дообучение", "Сначала дождитесь окончания сегментации.")
            return
        if self._training_thread is not None:
            return
        examples = self._training_examples_count()
        if examples < MIN_TRAINING_SAMPLES:
            QMessageBox.information(
                self,
                "Дообучение",
                f"Нужно минимум {MIN_TRAINING_SAMPLES} вручную проверенных слоёв. Сейчас: {examples}.",
            )
            return
        quality = training_quality(self._records)
        if (
            quality.underrepresented
            or quality.severe_imbalance
            or quality.unnamed_manual
            or quality.missing_depth
            or quality.duplicate_masks
            or quality.too_small_masks
            or quality.too_large_masks
        ):
            answer = QMessageBox.question(
                self,
                "Проверка качества разметки",
                "Перед дообучением обнаружены пункты, которые могут ухудшить качество модели:\n\n"
                f"{quality.summary}\n\n"
                "Естественный дисбаланс фаций внутри скважины не блокирует обучение.\n\n"
                "Продолжить всё равно?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        model_path = self._resolve_model_path()
        if model_path is None:
            QMessageBox.critical(
                self,
                "Модель не найдена",
                "Выберите исходную модель .pt через меню «Модель» перед дообучением.",
            )
            return
        device, device_label = FineTuneWorker._best_device()
        if device == "cpu":
            QMessageBox.critical(
                self,
                "Нужна GPU для дообучения",
                "CUDA-видеокарта не доступна в этой версии приложения. Дообучение на CPU отключено, "
                "чтобы не перегружать компьютер. Установите/соберите версию с CUDA PyTorch и обновите драйвер NVIDIA.",
            )
            return
        recommendation = self._current_training_recommendation(quality)
        epochs = self._training_epochs.value()
        recommended_epochs = recommendation.recommended_epochs
        answer = QMessageBox.question(
            self,
            "Запустить дообучение?",
            (
                f"Будет создан датасет из {examples} вручную проверенных слоёв и обучена "
                f"новая копия модели (до {epochs} эпох, рабочий ориентир: {recommended_epochs}).\n"
                f"При отсутствии улучшения {recommendation.patience} эпох подряд обучение остановится раньше.\n"
                f"Надёжность расчёта: {recommendation.confidence}.\n"
                f"Устройство: {device_label}. Исходный файл модели не изменится."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # A second-level timestamp can collide after a failed/retried launch.
        # Make every dataset/run/output directory unambiguously unique.
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        if self._project_folder is None:
            project_folder = self._projects_root() / f"{self._safe_folder_name(self._project_title)}_{stamp}.kern_analyzer"
            self._save_current_project(project_folder)
        elif self._project_folder is not None:
            # Copy externally selected photos into the project before the
            # dataset is exported, so this exact training session remains
            # reproducible after a file is moved or deleted outside Kern Analyzer.
            self._save_current_project(self._project_folder)
        if self._project_folder is None:
            return
        base_folder = self._project_folder
        training_root = base_folder / "training"
        published_dir = self._trained_models_root() / self._safe_folder_name(self._project_title) / stamp
        try:
            dataset = export_training_dataset(self._records, training_root / "datasets" / stamp,
                                              primary_catalog=self._primary_facies_catalog)
        except Exception as exc:
            QMessageBox.critical(self, "Не удалось подготовить датасет", str(exc))
            return

        self._training_thread = QThread(self)
        self._training_worker = FineTuneWorker(
            model_path,
            Path(dataset["data_yaml"]),
            training_root / "runs" / stamp,
            published_dir,
            epochs=epochs,
            dataset_summary={
                "split_strategy": dataset.get("split_strategy"),
                "source_digest": dataset.get("source_digest"),
                "val_class_counts": dataset.get("val_class_counts"),
                "unsupported_validation_classes": dataset.get("unsupported_validation_classes"),
                "sample_count": dataset["sample_count"],
                "class_counts": dataset["class_counts"],
                "source_photo_count": recommendation.source_photos,
                "well_count": recommendation.wells,
                "train_photo_count": dataset["train_photo_count"],
                "val_photo_count": dataset["val_photo_count"],
                "validation_grouped_by_photo": dataset["validation_grouped_by_photo"],
                "recommendation_confidence": recommendation.confidence,
                "recommendation_reasons": list(recommendation.reasons),
                "quality": quality.summary,
            },
            recommended_epochs=recommended_epochs,
            early_stopping_patience=recommendation.patience,
        )
        self._training_epoch_current = 0
        self._training_epoch_total = epochs
        self._fine_tune_success_model = None
        self._show_activity("Дообучение модели", f"Подготовка: 0 из {epochs} эпох…", epochs)
        self._training_worker.moveToThread(self._training_thread)
        self._training_thread.started.connect(self._training_worker.run)
        self._training_worker.progress.connect(self.statusBar().showMessage)
        self._training_worker.epoch_progress.connect(self._on_training_epoch_progress)
        self._training_worker.succeeded.connect(self._on_fine_tune_succeeded)
        self._training_worker.failed.connect(self._on_fine_tune_failed)
        self._training_worker.finished.connect(self._training_thread.quit)
        self._training_thread.finished.connect(self._on_fine_tune_finished)
        self._training_thread.finished.connect(self._training_worker.deleteLater)
        self._training_thread.finished.connect(self._training_thread.deleteLater)
        self._refresh_training_bar()
        self.statusBar().showMessage(
            f"Подготовлено {dataset['sample_count']} разметок · до {epochs} эпох · "
            f"автостоп {recommendation.patience}…"
        )
        self._training_thread.start()

    def _on_fine_tune_succeeded(self, model_path: str) -> None:
        self._close_activity()
        self._fine_tune_success_model = Path(model_path)
        self.statusBar().showMessage(f"Дообучение завершено · кандидат сохранён: {model_path}")
        QMessageBox.information(
            self,
            "Дообучение завершено",
            "Кандидат модели, справочник и результаты проверки сохранены:\n"
            f"{Path(model_path).parent}\n\nРучная разметка сохранена. Посмотрите training_info.json и историю обучения перед выбором модели.",
        )

    def _on_fine_tune_failed(self, message: str) -> None:
        self._close_activity()
        self._fine_tune_success_model = None
        QMessageBox.critical(self, "Ошибка дообучения", message)

    def _on_training_epoch_progress(self, current: int, total: int) -> None:
        if self._training_thread is None:
            return
        self._training_epoch_current = max(0, int(current))
        self._training_epoch_total = max(1, int(total))
        self._refresh_training_bar()
        self._update_activity(f"Дообучение: эпоха {self._training_epoch_current} из {self._training_epoch_total}", self._training_epoch_current, self._training_epoch_total)
        self.statusBar().showMessage(
            f"Дообучение: эпоха {self._training_epoch_current} из {self._training_epoch_total} "
            f"· осталось {max(0, self._training_epoch_total - self._training_epoch_current)}"
        )

    def _on_fine_tune_finished(self) -> None:
        self._close_activity()
        completed_model = self._fine_tune_success_model
        self._training_thread = None
        self._schedule_vlm_dispatch()
        self._training_worker = None
        self._training_epoch_current = 0
        self._training_epoch_total = 0
        self._refresh_training_bar()
        if completed_model is not None:
            self.statusBar().showMessage("Кандидат сохранён. Выберите модель через меню «Модель» после проверки результатов; разметка не изменена.")

    def import_las_file(self) -> None:
        self._import_log_file("gis")

    def import_rigis_file(self) -> None:
        self._import_log_file("rigis")

    def import_many_las_files(self) -> None:
        self._import_many_log_files("gis")

    def import_many_rigis_files(self) -> None:
        self._import_many_log_files("rigis")

    def _import_log_file(self, kind: str) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Загрузить LAS для {title} (можно выбрать несколько)",
            "",
            "LAS-файлы (*.las *.LAS);;Все файлы (*)",
        )
        if not paths:
            return
        self._import_log_paths(kind, paths)

    def _import_many_log_files(self, kind: str) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Загрузить несколько LAS для {title}",
            "",
            "LAS-файлы (*.las *.LAS);;Все файлы (*)",
        )
        if paths:
            self._import_log_paths(kind, paths)

    def _import_log_paths(self, kind: str, paths: list[str]) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        loaded, failures = 0, []
        for file_path in paths:
            auto_matched_well = self._well_for_log_file(file_path)
            well_name = auto_matched_well or self._choose_well(f"{title}: выбор скважины")
            if well_name is None:
                continue
            try:
                log_data = parse_las_file(file_path)
            except Exception as exc:
                failures.append(f"{Path(file_path).name}: {exc}")
                continue
            self._set_curve_display_names(log_data)
            self._set_well_log(well_name, kind, log_data, merge=True)
            if well_name not in self._well_depth_ranges:
                self._ask_for_well_depth_setup(log_data, well_name)
            if self._records_for_well(well_name) and well_name not in self._well_intervals:
                self._ask_for_core_interval(title, log_data, well_name)
            loaded += 1
        self._show_stack()
        message = f"Загружено {title}: {loaded}"
        if failures:
            message += f" · ошибок: {len(failures)}"
        self.statusBar().showMessage(message)

    def add_manual_gis_curve(self) -> None:
        well_name = self._choose_well("Новая ГИС-кривая: выбор скважины")
        if well_name is None:
            return
        log_data = self._well_log(well_name, "gis")
        if not log_data:
            QMessageBox.information(self, "Новая ГИС-кривая", "Сначала загрузите LAS для этой скважины.")
            return
        name, accepted = QInputDialog.getText(self, "Новая ГИС-кривая", "Мнемоника кривой:", text="USER")
        name = name.strip().upper()
        if not accepted or not name:
            return
        existing = {str(curve.get("mnemonic") or "") for curve in log_data.get("curves", [])}
        if name in existing:
            QMessageBox.warning(self, "Новая ГИС-кривая", "Кривая с такой мнемоникой уже существует.")
            return
        log_data.setdefault("curves", []).append({"index": len(log_data.get("curves", [])), "mnemonic": name, "unit": "", "description": "Вручную созданная кривая"})
        log_data["curves"][-1]["visible"] = True
        log_data.setdefault("values", {})[name] = [0.0] * len(log_data.get("depths", []))
        self._set_well_log(well_name, "gis", log_data)
        self._show_stack()
        self.statusBar().showMessage(f"Добавлена кривая {name} · откройте «Настроить ГИС…», чтобы вписать значения")

    def _build_gis_preview(self) -> dict | None:
        return self._build_log_preview(self._gis_data)

    def _build_rigis_preview(self) -> dict | None:
        return self._build_log_preview(self._rigis_data)

    def _well_names(self) -> list[str]:
        names = [record.well_name or "Скважина 1" for record in self._records]
        available = list(dict.fromkeys([*names, *self._empty_wells]))
        ordered = [name for name in self._well_order if name in available]
        return [*ordered, *(name for name in available if name not in ordered)]

    def _project_well_metadata(self) -> list[dict]:
        return [
            {
                "name": name,
                "order": index,
                **({"core_interval": list(self._well_intervals[name])} if name in self._well_intervals else {}),
                **({"well_depth_range": list(self._well_depth_ranges[name])} if name in self._well_depth_ranges else {}),
                **({"depth_reference": self._well_depth_references[name]} if name in self._well_depth_references else {}),
                **({"depth_settings": self._well_depth_settings[name]} if name in self._well_depth_settings else {}),
                **({"interpretations": self._well_interpretations[name]} if name in self._well_interpretations else {}),
                **({"excel_report_inputs": self._well_report_inputs[name]} if name in self._well_report_inputs else {}),
            }
            for index, name in enumerate(self._well_names())
        ]

    def create_empty_well(self) -> None:
        default = f"Скважина {len(self._well_names()) + 1}"
        name, accepted = QInputDialog.getText(self, "Новая скважина", "Название скважины:", text=default)
        name = name.strip()
        if not accepted or not name:
            return
        if name in self._well_names():
            QMessageBox.warning(self, "Новая скважина", "Скважина с таким названием уже существует.")
            return
        self._empty_wells.append(name)
        self._active_well_name = name
        self._settings.setValue("last/active_well", name)
        self._refresh_project_tree()
        self._show_stack()
        self.configure_well_depth()
        self.statusBar().showMessage(f"Создана скважина без керна: {name} · можно загрузить ГИС или РИГИС")

    def _records_for_well(self, well_name: str) -> list[PhotoRecord]:
        return [record for record in self._records if (record.well_name or "Скважина 1") == well_name]

    def _choose_well(self, title: str) -> str | None:
        wells = self._well_names()
        if not wells:
            QMessageBox.information(self, title, "Сначала загрузите фотографии керна.")
            return None
        if len(wells) == 1:
            return wells[0]
        current = wells.index(self._active_well_name) if self._active_well_name in wells else 0
        well, accepted = QInputDialog.getItem(self, title, "Скважина:", wells, current, False)
        return str(well) if accepted and well else None

    @staticmethod
    def _file_match_key(file_path: str) -> str:
        """Normalise file names while ignoring common core/log suffixes and separators."""
        stem = Path(file_path).stem.casefold().replace("ё", "е")
        words = [word for word in re.split(r"[_\W]+", stem, flags=re.UNICODE) if word]
        ignored = {"core", "ker", "kern", "photo", "img", "image", "gis", "rigis", "las", "log", "фото", "керн", "гис", "ригис"}
        words = [word for word in words if word not in ignored]
        # A final image sequence such as ``well-12-core-01`` is ignored,
        # while the well number in ``well-12`` remains significant.
        while len(words) > 2 and words[-1].isdigit():
            words.pop()
        return "_".join(words)

    def _well_for_log_file(self, log_path: str) -> str | None:
        """Return one well only when its photo name gives an unambiguous file-name match."""
        log_key = self._file_match_key(log_path)
        if not log_key:
            return None
        scores: dict[str, int] = {}
        for well_name in self._well_names():
            candidates = [self._file_match_key(record.path) for record in self._records_for_well(well_name)]
            well_key = self._file_match_key(well_name)
            score = 0
            for key in candidates:
                if key == log_key:
                    score = max(score, 100)
                elif key and (key.startswith(log_key + "_") or log_key.startswith(key + "_")):
                    score = max(score, 80)
            if well_key == log_key:
                score = max(score, 95)
            if score:
                scores[well_name] = score
        if not scores:
            return None
        highest = max(scores.values())
        matches = [well_name for well_name, score in scores.items() if score == highest]
        return matches[0] if len(matches) == 1 else None

    def _well_log(self, well_name: str, kind: str) -> dict | None:
        return self._well_logs.get(well_name, {}).get(kind)

    def _set_well_log(self, well_name: str, kind: str, log_data: dict, merge: bool = False) -> None:
        if merge and self._well_log(well_name, kind):
            log_data = self._merge_log_data(self._well_log(well_name, kind), log_data)
        self._well_logs.setdefault(well_name, {})[kind] = log_data
        self._active_well_name = well_name
        self._settings.setValue("last/active_well", well_name)
        # Keep the established single-well fields in sync for compatibility.
        if kind == "gis":
            self._gis_data = log_data
        else:
            self._rigis_data = log_data

    @staticmethod
    def _set_curve_display_names(log_data: dict) -> None:
        """Use the LAS file name as the user-facing curve title in the tablet."""
        source_file = str(log_data.get("file_name") or Path(str(log_data.get("source_path") or "curve")).name)
        source_stem = Path(source_file).stem or "curve"
        depth_curve = str(log_data.get("depth_curve") or "")
        data_curves = [curve for curve in log_data.get("curves", []) if str(curve.get("mnemonic") or "") != depth_curve]
        for curve in data_curves:
            mnemonic = str(curve.get("mnemonic") or "")
            curve["source_file"] = source_file
            curve["display_name"] = source_stem if len(data_curves) == 1 else f"{source_stem} · {mnemonic}"

    @staticmethod
    def _merge_log_data(base: dict, incoming: dict) -> dict:
        """Append curves from another LAS file, resampling them to the first log's depth grid."""
        merged = {
            **base,
            "curves": [dict(curve) for curve in base.get("curves", [])],
            "depths": list(base.get("depths", []) or []),
            "values": {name: list(values or []) for name, values in (base.get("values", {}) or {}).items()},
        }
        target_depths = merged["depths"]
        incoming_depths = [float(value) if value is not None else None for value in incoming.get("depths", [])]
        source_depth_curve = incoming.get("depth_curve")
        existing_names = {str(curve.get("mnemonic") or "") for curve in merged["curves"]}
        for curve in incoming.get("curves", []):
            source_name = str(curve.get("mnemonic") or "")
            if not source_name or source_name == source_depth_curve:
                continue
            name = source_name
            suffix = 2
            while name in existing_names:
                name = f"{source_name}_{suffix}"
                suffix += 1
            source_values = list((incoming.get("values", {}) or {}).get(source_name, []) or [])
            curve_meta = {**curve, "index": len(merged["curves"]), "mnemonic": name, "visible": True}
            if name != source_name:
                curve_meta["display_name"] = f"{curve.get('display_name') or source_name} ({suffix - 1})"
            merged["curves"].append(curve_meta)
            merged["values"][name] = MainWindow._resample_curve(incoming_depths, source_values, target_depths)
            existing_names.add(name)
        return merged

    @staticmethod
    def _resample_curve(source_depths: list[float | None], source_values: list, target_depths: list) -> list[float | None]:
        pairs = sorted((depth, value) for depth, value in zip(source_depths, source_values) if depth is not None and value is not None)
        if not pairs:
            return [None] * len(target_depths)
        xs = [pair[0] for pair in pairs]
        ys = [float(pair[1]) for pair in pairs]
        result = []
        for target in target_depths:
            if target is None:
                result.append(None)
                continue
            index = bisect_left(xs, float(target))
            if index == 0:
                result.append(ys[0])
            elif index >= len(xs):
                result.append(ys[-1])
            else:
                x1, x2, y1, y2 = xs[index - 1], xs[index], ys[index - 1], ys[index]
                ratio = (float(target) - x1) / max(x2 - x1, 1e-9)
                result.append(y1 + ratio * (y2 - y1))
        return result

    @staticmethod
    def _build_log_preview(log_data: dict | None) -> dict | None:
        if not log_data:
            return None
        depth_curve = log_data.get("depth_curve")
        curve_metas = [
            curve
            for curve in log_data.get("curves", [])
            if curve.get("mnemonic") and curve.get("mnemonic") != depth_curve and curve.get("visible", True)
        ]
        depths = list(log_data.get("depths", []) or [])
        tracks = []
        for curve_meta in curve_metas:
            curve_name = str(curve_meta["mnemonic"])
            values = list((log_data.get("values", {}) or {}).get(curve_name, []) or [])
            valid = [float(value) for value in values if value is not None]
            if not valid:
                continue
            lower, upper = min(valid), max(valid)
            span = max(upper - lower, 1e-9)
            step = max(1, len(depths) // 700)
            points = [
                {"depth": float(depths[index]), "normalized": (float(value) - lower) / span}
                for index, value in enumerate(values)
                if index < len(depths) and index % step == 0 and value is not None and depths[index] is not None
            ]
            tracks.append({"curve": curve_meta.get("display_name") or curve_name, "points": points, "min": lower, "max": upper})
        valid_depths = [float(depth) for depth in depths if depth is not None]
        preview = {"well_name": log_data.get("well_name"), "tracks": tracks, "depth_unit": log_data.get("depth_unit") or "m"}
        if len(valid_depths) >= 2:
            preview["depth_range"] = (min(valid_depths), max(valid_depths))
        return preview

    def _build_well_log_preview(self, well_name: str, kind: str) -> dict | None:
        preview = self._build_log_preview(self._well_log(well_name, kind))
        depth_settings = self._well_depth_settings.get(well_name, {})
        if preview is None and well_name in self._well_depth_ranges:
            preview = {"well_name": well_name, "tracks": [], "depth_unit": depth_settings.get("unit") or "m"}
        if preview is not None and well_name in self._well_depth_ranges:
            preview = {
                **preview,
                "depth_range": self._well_depth_ranges[well_name],
                "depth_reference": self._well_depth_references.get(well_name, "MD"),
                "depth_unit": depth_settings.get("unit") or preview.get("depth_unit") or "m",
                "depth_datum": depth_settings.get("datum") or "",
            }
        return preview

    def _ordered_detections(self, well_name: str | None = None):
        records = self._records_for_well(well_name) if well_name else self._records
        return [detection for _, detection in StackColumnItem.ordered_layer_entries(records)]

    def _can_move_layer(self, record: PhotoRecord, detection, direction: int) -> bool:
        entries = StackColumnItem.ordered_layer_entries(self._records_for_well(record.well_name))
        current = next((index for index, (_, item) in enumerate(entries) if item is detection), -1)
        target = current + (-1 if direction < 0 else 1)
        return current >= 0 and 0 <= target < len(entries)

    def _move_layer(self, record: PhotoRecord, detection, direction: int) -> None:
        """Move a facies layer in the continuous well column and keep it filled."""

        entries = StackColumnItem.ordered_layer_entries(self._records_for_well(record.well_name))
        current = next((index for index, (_, item) in enumerate(entries) if item is detection), -1)
        target = current + (-1 if direction < 0 else 1)
        if current < 0 or not 0 <= target < len(entries):
            self.statusBar().showMessage("Слой уже находится на границе собранного столбика")
            return

        spans: dict[int, float] = {}
        complete_depths = True
        depth_top = float("inf")
        for _, item in entries:
            if item.depth_from is None or item.depth_to is None:
                complete_depths = False
                break
            top, base = sorted((float(item.depth_from), float(item.depth_to)))
            spans[id(item)] = max(0.0, base - top)
            depth_top = min(depth_top, top)

        entries[current], entries[target] = entries[target], entries[current]
        for stack_order, (_, item) in enumerate(entries):
            item.stack_order = stack_order

        # In a depth-bound assembled core, vertical placement follows depth.
        # Re-sequence the moved layers with their own thicknesses so their
        # intervals touch and the column can never acquire an empty gap.
        if complete_depths and spans and depth_top != float("inf"):
            cursor = depth_top
            for _, item in entries:
                item.depth_from = cursor
                item.depth_to = cursor + spans[id(item)]
                cursor = item.depth_to

        self._refresh_facies_views(record)
        movement = "выше" if direction < 0 else "ниже"
        self.statusBar().showMessage(f"Слой {detection.label} перемещён {movement} · столбик остался непрерывным")

    def _ask_for_well_depth_setup(self, log_data: dict, well_name: str) -> None:
        depths = [float(value) for value in log_data.get("depths", []) if value is not None]
        if len(depths) < 2:
            return
        references = [str(curve.get("mnemonic") or "") for curve in log_data.get("curves", []) if str(curve.get("mnemonic") or "")]
        selected = str(log_data.get("depth_curve") or "MD")
        dialog = WellDepthDialog(min(depths), max(depths), references, selected, self, self._well_depth_settings.get(well_name))
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base, reference, settings = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "\u0413\u043b\u0443\u0431\u0438\u043d\u0430 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b", str(exc))
            return
        self._well_depth_ranges[well_name] = (top, base)
        self._well_depth_references[well_name] = reference
        self._well_depth_settings[well_name] = settings
        self._apply_depth_curve(log_data, settings.get("source_curve") or "")

    def configure_well_depth(self) -> None:
        well_name = self._choose_well("\u0413\u043b\u0443\u0431\u0438\u043d\u0430 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b")
        if well_name is None:
            return
        log_data = self._well_log(well_name, "gis") or self._well_log(well_name, "rigis")
        depths = [float(value) for value in (log_data or {}).get("depths", []) if value is not None]
        top, base = self._well_depth_ranges.get(well_name, (min(depths), max(depths)) if len(depths) >= 2 else (0.0, 1000.0))
        references = [str(curve.get("mnemonic") or "") for curve in (log_data or {}).get("curves", []) if str(curve.get("mnemonic") or "")]
        selected = self._well_depth_references.get(well_name, str((log_data or {}).get("depth_curve") or "MD"))
        dialog = WellDepthDialog(top, base, references, selected, self, self._well_depth_settings.get(well_name))
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base, reference, settings = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "\u0413\u043b\u0443\u0431\u0438\u043d\u0430 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b", str(exc))
            return
        self._well_depth_ranges[well_name] = (top, base)
        self._well_depth_references[well_name] = reference
        self._well_depth_settings[well_name] = settings
        source_curve = settings.get("source_curve") or ""
        for kind in ("gis", "rigis"):
            candidate = self._well_log(well_name, kind)
            if candidate:
                self._apply_depth_curve(candidate, source_curve)
        self._show_stack()

    @staticmethod
    def _apply_depth_curve(log_data: dict, curve_name: str) -> None:
        """Use an explicitly selected LAS depth curve; never infer an MD/TVD conversion."""
        curve_name = str(curve_name or "").strip()
        values = list((log_data.get("values") or {}).get(curve_name, []) or [])
        valid = [value for value in values if value is not None]
        if len(valid) < 2:
            return
        log_data["depths"] = values
        log_data["depth_curve"] = curve_name

    def configure_core_interval(self) -> None:
        well_name = self._choose_well("\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b \u043a\u0435\u0440\u043d\u0430")
        if well_name is None:
            return
        fallback = self._well_depth_ranges.get(well_name, (0.0, 1.0))
        top, base = self._well_intervals.get(well_name, fallback)
        dialog = DepthRangeDialog(top, base, self)
        dialog.setWindowTitle("\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b \u043a\u0435\u0440\u043d\u0430 \u0432 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u0435")
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base = dialog.values()
            if top is None or base is None or top >= base:
                raise ValueError("\u041d\u0438\u0437 \u043a\u0435\u0440\u043d\u0430 \u0434\u043e\u043b\u0436\u0435\u043d \u0431\u044b\u0442\u044c \u0431\u043e\u043b\u044c\u0448\u0435 \u0432\u0435\u0440\u0445\u0430")
        except ValueError as exc:
            QMessageBox.warning(self, "\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b \u043a\u0435\u0440\u043d\u0430", str(exc))
            return
        self._apply_core_interval(top, base, well_name)
        self._show_stack()

    def _current_core_interval(self, log_data: dict, well_name: str | None = None) -> tuple[float, float]:
        if well_name and well_name in self._well_intervals:
            return self._well_intervals[well_name]
        depths = [
            value
            for detection in self._ordered_detections(well_name)
            for value in (detection.depth_from, detection.depth_to)
            if value is not None
        ]
        if len(depths) >= 2:
            return min(depths), max(depths)
        log_depths = [float(value) for value in log_data.get("depths", []) if value is not None]
        return (min(log_depths), max(log_depths)) if log_depths else (0.0, 1.0)

    def _ask_for_core_interval(self, title: str, log_data: dict, well_name: str | None = None) -> None:
        existing_depths = [
            value
            for detection in self._ordered_detections(well_name)
            for value in (detection.depth_from, detection.depth_to)
            if value is not None
        ]
        if well_name not in self._well_intervals and len(existing_depths) < 2:
            top, base = None, None
        else:
            top, base = self._current_core_interval(log_data, well_name)
        dialog = DepthRangeDialog(top, base, self)
        dialog.setWindowTitle(f"Интервал {'керна' if self._ordered_detections(well_name) else 'скважины'} для {title}")
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            top, base = dialog.values()
            if top is None or base is None or top >= base:
                raise ValueError("Низ интервала должен быть больше верха")
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте интервал", str(exc))
            return
        self._apply_core_interval(top, base, well_name)

    def _apply_core_interval(self, top: float, base: float, well_name: str | None = None) -> None:
        ordered = self._ordered_detections(well_name)
        if not ordered:
            if well_name:
                self._well_intervals[well_name] = (float(top), float(base))
            return
        heights = []
        for detection in ordered:
            ys = [point.y() for point in detection.polygon]
            heights.append(max(1.0, max(ys) - min(ys)) if ys else 1.0)
        total = sum(heights)
        cursor = float(top)
        for index, (detection, height) in enumerate(zip(ordered, heights)):
            next_depth = float(base) if index == len(ordered) - 1 else cursor + (base - top) * height / total
            detection.depth_from = cursor
            detection.depth_to = next_depth
            cursor = next_depth

    def _show_stack(self) -> None:
        # Older projects can contain physical columns but no facies masks, or
        # masks for only some columns.  Fill missing intervals before assembling
        # the tape; keep existing reviewed contours and mark gaps unresolved.
        for record in self._records:
            if record.analysis_mode == PIPELINE_MODE:
                # Display the actual masks and gaps; never redraw geometry as
                # a side effect of opening the assembled column.
                continue
            if record.core_columns and not record.core_columns_verified:
                cleaned = normalize_columns(
                    record.core_columns, (record.pixmap.width(), record.pixmap.height()),
                    suppress_narrow_shadows=True,
                )
                compound_tray = (
                    len(cleaned) <= 2
                    and any(
                        column["right"] - column["left"] > record.pixmap.width() * 0.45
                        for column in cleaned
                    )
                )
                rebuilt_from_photo = False
                if compound_tray:
                    proposed = self._suggest_core_columns(record)
                    if len(proposed) >= 3:
                        cleaned = proposed
                        rebuilt_from_photo = True
                if cleaned and (len(cleaned) < len(record.core_columns) or rebuilt_from_photo):
                    record.core_columns = cleaned
                    self._update_depth_segments_for_columns(record)
                    # Old automatic facies bands from the removed thin lane
                    # must not reappear as slices of the assembled core.  Keep
                    # every reviewed/manual contour exactly as the geologist
                    # drew it; uncertain gaps are filled by postprocessing.
                    retained = []
                    for detection in record.detections:
                        if detection.training_ready or detection.attributes.get("__manual_contour") == "1":
                            retained.append(detection)
                            continue
                        if rebuilt_from_photo:
                            # The old single tray mask cannot be trusted as a
                            # facies prediction for five newly separated cores.
                            continue
                        bounds = QPolygonF(detection.polygon).boundingRect()
                        if any(
                            max(0.0, min(bounds.right(), column["right"]) - max(bounds.left(), column["left"]))
                            >= 0.56 * (column["right"] - column["left"])
                            and bounds.bottom() > column["top"]
                            and bounds.top() < column["bottom"]
                            for column in cleaned
                        ):
                            retained.append(detection)
                    record.detections = postprocess_detections(record, retained)
                    self.workspace.update_photo_detections(record)
            if not record.core_columns or has_complete_core_column_coverage(record):
                continue
            if any(item.label != "Новый контур" for item in record.detections):
                record.detections = postprocess_detections(record, record.detections)
            else:
                self._fill_unclassified_columns(record)
            self.workspace.update_photo_detections(record)
        # The facies legend follows the model classes that are actually present
        # on the currently loaded photos, while retaining saved visual overrides.
        self._sync_facies_palette()
        wells = {well_name: self._records_for_well(well_name) for well_name in self._well_names()}
        if not wells:
            return
        well_logs = {
            well_name: (
                self._build_well_log_preview(well_name, "gis"),
                self._build_well_log_preview(well_name, "rigis"),
            )
            for well_name in wells
        }
        self.workspace.show_correlation(
            wells,
            well_logs,
            well_interpretations=self._well_interpretations,
            spacing=180.0,
            vertical_scale=1.0,
            connect_layers=False,
            description_only=True,
            place_source_photos=True,
        )

    def show_correlation_profile(self) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "Корреляция", "Для корреляции создайте или загрузите минимум две скважины.")
            return
        self._show_stack()
        self.sheets.setCurrentWidget(self.correlation_workspace)
        self.correlation_workspace.fit_content()
        self.statusBar().showMessage("Корреляция построена · пунктиром соединены совпадающие литологические интервалы")

    def change_well_order(self) -> None:
        wells = self._well_names()
        if len(wells) < 2:
            QMessageBox.information(self, "\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d", "\u0414\u043e\u0431\u0430\u0432\u044c\u0442\u0435 \u0445\u043e\u0442\u044f \u0431\u044b \u0434\u0432\u0435 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b.")
            return
        well_name, accepted = QInputDialog.getItem(self, "\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d", "\u041f\u0435\u0440\u0435\u043c\u0435\u0441\u0442\u0438\u0442\u044c \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u0443:", wells, 0, False)
        if not accepted or not well_name:
            return
        targets = ["\u0412 \u043d\u0430\u0447\u0430\u043b\u043e", *[name for name in wells if name != well_name]]
        target, accepted = QInputDialog.getItem(self, "\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d", "\u0420\u0430\u0441\u043f\u043e\u043b\u043e\u0436\u0438\u0442\u044c:", targets, 0, False)
        if not accepted:
            return
        new_order = [name for name in wells if name != well_name]
        insert_at = 0 if target == targets[0] else new_order.index(target) + 1
        new_order.insert(insert_at, str(well_name))
        self._well_order = new_order
        self._show_stack()
        self.statusBar().showMessage("\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d \u043e\u0431\u043d\u043e\u0432\u043b\u0451\u043d")

    def _set_well_order_from_canvas(self, ordered_wells) -> None:
        requested = [str(name) for name in ordered_wells if str(name)]
        available = self._well_names()
        if not requested or set(requested) - set(available):
            return
        updated = [*requested, *(name for name in available if name not in requested)]
        if updated == available:
            return
        self._well_order = updated
        self._show_stack()
        self.statusBar().showMessage("\u041f\u043e\u0440\u044f\u0434\u043e\u043a \u0441\u043a\u0432\u0430\u0436\u0438\u043d \u0438\u0437\u043c\u0435\u043d\u0451\u043d \u043f\u0435\u0440\u0435\u0442\u0430\u0441\u043a\u0438\u0432\u0430\u043d\u0438\u0435\u043c")

    def add_correlation_line_from_layer(self) -> None:
        choices: list[tuple[str, str, float, str]] = []
        for well_name in self._well_names():
            for index, detection in enumerate(self._ordered_detections(well_name), start=1):
                depths = [value for value in (detection.depth_from, detection.depth_to) if value is not None]
                if not depths:
                    continue
                depth = sum(float(value) for value in depths) / len(depths)
                label = str(detection.label or "\u0421\u043b\u043e\u0439")
                choices.append((f"{well_name} \u00b7 {label} {index} \u00b7 {depth:g} \u043c", well_name, depth, label))
        if not choices:
            QMessageBox.information(self, "\u041b\u0438\u043d\u0438\u044f \u043f\u043e \u0441\u043b\u043e\u044e", "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u0434\u0430\u0439\u0442\u0435 \u0433\u043b\u0443\u0431\u0438\u043d\u044b \u0441\u043b\u043e\u0451\u0432.")
            return
        titles = [choice[0] for choice in choices]
        selected, accepted = QInputDialog.getItem(self, "\u041b\u0438\u043d\u0438\u044f \u043f\u043e \u0441\u043b\u043e\u044e", "\u0412\u044b\u0431\u0435\u0440\u0438\u0442\u0435 \u0441\u043b\u043e\u0439-\u043e\u0440\u0438\u0435\u043d\u0442\u0438\u0440:", titles, 0, False)
        if not accepted or not selected:
            return
        _, well_name, depth, label = choices[titles.index(selected)]
        name, accepted = QInputDialog.getText(self, "\u041b\u0438\u043d\u0438\u044f \u043f\u043e \u0441\u043b\u043e\u044e", "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435 \u043b\u0438\u043d\u0438\u0438:", text=label)
        if not accepted or not name.strip():
            return
        self._show_stack()
        if self.correlation_workspace.add_straight_layer_correlation_curve(name.strip(), well_name, depth):
            self.sheets.setCurrentWidget(self.correlation_workspace)
            self.statusBar().showMessage("\u041b\u0438\u043d\u0438\u044f \u043f\u0440\u0438\u0432\u044f\u0437\u0430\u043d\u0430 \u043a \u0441\u043b\u043e\u044e \u0438 \u043f\u043e\u0441\u0442\u0440\u043e\u0435\u043d\u0430 \u043e\u0434\u043d\u043e\u0439 \u043f\u0440\u044f\u043c\u043e\u0439 \u0447\u0435\u0440\u0435\u0437 \u0432\u0441\u0435 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b")

    def add_manual_correlation_curve(self) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "Корреляция", "Для линии нужны минимум две скважины.")
            return
        self._show_stack()
        name, accepted = QInputDialog.getText(self, "Новая линия корреляции", "Название линии:", text="Новый пласт")
        if not accepted or not name.strip():
            return
        if self.correlation_workspace.add_manual_correlation_curve(name.strip()):
            self.statusBar().showMessage(f"Добавлена линия корреляции: {name.strip()} · перетащите её на нужной скважине")

    def configure_correlation(self) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "Корреляция", "Сначала добавьте минимум две скважины.")
            return
        spacing, accepted = QInputDialog.getDouble(
            self,
            "Корреляция: расстояние",
            "Расстояние между скважинами, px:",
            self._correlation_spacing,
            20.0,
            5000.0,
            0,
        )
        if not accepted:
            return
        scale, accepted = QInputDialog.getDouble(
            self,
            "Корреляция: шаг по глубине",
            "Вертикальный масштаб (1 = исходный):",
            self._correlation_vertical_scale,
            0.2,
            8.0,
            2,
        )
        if not accepted:
            return
        self._correlation_spacing = spacing
        self._correlation_vertical_scale = scale
        self._show_stack()
        self.correlation_workspace.fit_content()

    def _open_correlation_curve_menu(self, curve: dict, well_name: str) -> None:
        menu = QMenu(self)
        rename = menu.addAction("Изменить название…")
        recolor = menu.addAction("Изменить цвет…")
        thickness = menu.addAction("Толщина линии…")
        style_menu = menu.addMenu("Тип линии")
        styles = {
            "Сплошная": "solid",
            "Пунктир": "dash",
            "Точки": "dot",
            "Штрих-точка": "dash_dot",
        }
        for title, value in styles.items():
            action = style_menu.addAction(title)
            action.setCheckable(True)
            action.setChecked(curve.get("style") == value)
            action.triggered.connect(lambda _=False, selected=value: self._set_curve_value(curve, "style", selected))
        menu.addSeparator()
        up = menu.addAction(f"Сдвинуть выше на скважине «{well_name}»")
        down = menu.addAction(f"Сдвинуть ниже на скважине «{well_name}»")
        reset = menu.addAction(f"Сбросить сдвиг на «{well_name}»")
        break_line = menu.addAction(f"Разорвать на скважине «{well_name}»")
        restore_menu = menu.addMenu("Восстановить на скважине")
        for broken_well in curve.get("breaks", []):
            action = restore_menu.addAction(str(broken_well))
            action.triggered.connect(lambda _=False, selected=str(broken_well): self._set_curve_break(curve, selected, False))

        rename.triggered.connect(lambda: self._rename_curve(curve))
        recolor.triggered.connect(lambda: self._recolor_curve(curve))
        thickness.triggered.connect(lambda: self._change_curve_thickness(curve))
        up.triggered.connect(lambda: self._shift_curve_in_well(curve, well_name, -10.0))
        down.triggered.connect(lambda: self._shift_curve_in_well(curve, well_name, 10.0))
        reset.triggered.connect(lambda: self._set_curve_well_offset(curve, well_name, 0.0))
        break_line.triggered.connect(lambda: self._set_curve_break(curve, well_name, True))
        menu.exec(QCursor.pos())

    def _set_curve_value(self, curve: dict, key: str, value) -> None:
        curve[key] = value
        self._show_stack()

    def _rename_curve(self, curve: dict) -> None:
        name, accepted = QInputDialog.getText(self, "Название корреляции", "Название:", text=str(curve.get("name") or ""))
        if accepted and name.strip():
            self._set_curve_value(curve, "name", name.strip())

    def _recolor_curve(self, curve: dict) -> None:
        color = QColorDialog.getColor(QColor(str(curve.get("color") or "#734cbe")), self, "Цвет корреляционной линии")
        if color.isValid():
            self._set_curve_value(curve, "color", color.name(QColor.NameFormat.HexRgb))

    def _change_curve_thickness(self, curve: dict) -> None:
        width, accepted = QInputDialog.getDouble(
            self,
            "Толщина корреляционной линии",
            "Толщина, px:",
            float(curve.get("width", 1.8)),
            0.5,
            20.0,
            1,
        )
        if accepted:
            self._set_curve_value(curve, "width", width)

    def _shift_curve_in_well(self, curve: dict, well_name: str, delta: float) -> None:
        offsets = curve.setdefault("offsets", {})
        self._set_curve_well_offset(curve, well_name, float(offsets.get(well_name, 0.0)) + delta)

    def _set_curve_well_offset(self, curve: dict, well_name: str, offset: float) -> None:
        curve.setdefault("offsets", {})[well_name] = offset
        self._show_stack()

    def _set_curve_break(self, curve: dict, well_name: str, broken: bool) -> None:
        breaks = [str(name) for name in curve.get("breaks", [])]
        if broken and well_name not in breaks:
            breaks.append(well_name)
        if not broken:
            breaks = [name for name in breaks if name != well_name]
        curve["breaks"] = breaks
        self._show_stack()

    def _paint_stack_column(self, record: PhotoRecord, detection, column: str) -> None:
        if column == "facies":
            self._sync_facies_palette()
            menu = QMenu(self)
            current_key = str(detection.label or "").strip().casefold()
            palette_actions = []
            for item in self._facies_palette:
                action = menu.addAction(f"{item['symbol']}  ·  {item['name']}")
                action.setCheckable(True)
                action.setChecked(item["key"].casefold() == current_key)
                palette_actions.append((action, item))
            menu.addSeparator()
            full_editor = menu.addAction("Открыть полные параметры интервала…")
            configure_action = menu.addAction("Настроить и сохранить палетку фаций…")
            selected = menu.exec(QCursor.pos())
            if selected is None:
                return
            if selected is configure_action:
                self.configure_facies_palette()
                return
            if selected is full_editor:
                self._open_facies_editor(record, detection)
                return
            chosen = next((item for action, item in palette_actions if selected is action), None)
            if chosen is None:
                return
            detection.label = chosen["key"]
            for field_name in (
                "Код фации",
                "Индекс фации",
                "Название фации",
                "Энергия среды",
                "Гидродинамический режим",
            ):
                detection.attributes.pop(field_name, None)
            resolved = self._resolve_training_class(detection.label)
            if resolved:
                detection.attributes.update(resolved["metadata"])
            detection.training_ready = resolved is not None
            self._refresh_facies_views(record)
            self.statusBar().showMessage(
                f"Фация интервала: {chosen['name']} · разметка отмечена для дообучения"
            )
            return

        if column == "lithology":
            menu = QMenu(self)
            menu.setTitle("Литология")
            current_name = str(detection.attributes.get("Название породы") or "").strip().casefold()
            palette_actions = []
            for item in self._lithology_palette:
                action = menu.addAction(f"{item['symbol']}  ·  {item['name']}")
                action.setCheckable(True)
                action.setChecked(item["name"].casefold() == current_name)
                palette_actions.append((action, item))
            menu.addSeparator()
            custom_color_action = menu.addAction("Свой цвет для этого интервала…")
            configure_action = menu.addAction("Настроить и сохранить палетку…")
            selected = menu.exec(QCursor.pos())
            if selected is None:
                return
            if selected is configure_action:
                self.configure_lithology_palette()
                return
            if selected is custom_color_action:
                initial = QColor(str(detection.attributes.get("Цвет литологии") or "#d8b45b"))
                color = QColorDialog.getColor(initial, self, "Свой цвет литологии")
                if not color.isValid():
                    return
                detection.attributes["Цвет литологии"] = color.name(QColor.NameFormat.HexRgb)
                self._refresh_facies_views(record)
                self.statusBar().showMessage("Литология: цвет интервала изменён")
                return
            chosen = next((item for action, item in palette_actions if selected is action), None)
            if chosen is None:
                return
            detection.attributes["Название породы"] = chosen["name"]
            detection.attributes.pop("Цвет литологии", None)
            self._settings.setValue("last/lithology", chosen["name"])
            self._refresh_facies_views(record)
            self.statusBar().showMessage(
                f"Литология интервала: {chosen['name']} ({chosen['symbol']})"
            )
            return

        field, title, fallback = "Цвет насыщения", "Кисть насыщения", "#e0b653"
        initial = QColor(str(detection.attributes.get(field) or fallback))
        color = QColorDialog.getColor(initial, self, title)
        if not color.isValid():
            return
        value = color.name(QColor.NameFormat.HexRgb)
        detection.attributes[field] = value
        saturation = str(detection.attributes.get("Флюидонасыщение") or "")
        if saturation:
            self._settings.setValue(f"saturation_colors/{saturation}", value)
        self._refresh_facies_views(record)
        self.statusBar().showMessage(f"{title}: цвет интервала изменён")

    def _create_interpretation_interval(self, well_name: str, kind: str, depth_from: float, depth_to: float) -> None:
        title = {"lithology": "Литология", "facies": "Фации"}.get(kind, "Насыщение")
        if kind in {"lithology", "facies"}:
            if kind == "facies":
                self._sync_facies_palette()
            palette = self._lithology_palette if kind == "lithology" else self._facies_palette
            if not palette:
                QMessageBox.information(self, title, "Для этой дорожки пока нет доступных классов.")
                return
            titles = [f"{item['symbol']} · {item['name']}" for item in palette]
            setting_key = "last/lithology" if kind == "lithology" else "last/facies_track"
            last_key = str(self._settings.value(setting_key, "") or "")
            keys = [str(item.get("key") or item["name"]) for item in palette]
            initial = keys.index(last_key) if last_key in keys else 0
            selected_title, accepted = QInputDialog.getItem(
                self,
                title,
                "Элемент из сохранённой палетки:",
                titles,
                initial,
                False,
            )
            if not accepted or not selected_title:
                return
            item = palette[titles.index(selected_title)]
            item_key = str(item.get("key") or item["name"])
            self._settings.setValue(setting_key, item_key)
            self._well_interpretations.setdefault(well_name, []).append(
                {
                    "kind": kind,
                    "key": item_key,
                    "name": item["name"],
                    "symbol": item["symbol"],
                    "color": item["color"],
                    "pattern": item["pattern"],
                    "depth_from": depth_from,
                    "depth_to": depth_to,
                }
            )
            self._show_stack()
            self.statusBar().showMessage(f"Добавлен интервал «{item['name']}»: {depth_from:g}–{depth_to:g} м")
            return

        default_name = "Новый литологический интервал" if kind == "lithology" else "Новый интервал насыщения"
        name, accepted = QInputDialog.getText(self, title, "Название интервала:", text=default_name)
        if not accepted or not name.strip():
            return
        fallback = "#d8b45b" if kind == "lithology" else "#e0b653"
        color = QColorDialog.getColor(QColor(fallback), self, f"Цвет: {title}")
        if not color.isValid():
            return
        self._well_interpretations.setdefault(well_name, []).append(
            {
                "kind": kind,
                "name": name.strip(),
                "color": color.name(QColor.NameFormat.HexRgb),
                "depth_from": depth_from,
                "depth_to": depth_to,
            }
        )
        self._show_stack()
        self.statusBar().showMessage(f"Добавлен интервал «{name.strip()}»: {depth_from:g}–{depth_to:g} м")

    def _resize_interpretation_interval(self, well_name: str, index: int, depth_from: float, depth_to: float) -> None:
        intervals = self._well_interpretations.get(well_name, [])
        if not 0 <= index < len(intervals):
            return
        intervals[index]["depth_from"] = depth_from
        intervals[index]["depth_to"] = depth_to
        self._show_stack()
        self.statusBar().showMessage(f"Интервал изменён: {depth_from:g}–{depth_to:g} м")

    def _open_interpretation_interval_menu(self, well_name: str, index: int) -> None:
        intervals = self._well_interpretations.get(well_name, [])
        if not 0 <= index < len(intervals):
            return
        interval = intervals[index]
        menu = QMenu(self)
        palette = None
        if str(interval.get("kind")) in {"lithology", "facies"}:
            palette = menu.addAction("Выбрать из палетки…")
            menu.addSeparator()
        rename = menu.addAction("Изменить название…")
        recolor = menu.addAction("Изменить цвет…")
        delete = menu.addAction("Удалить интервал")
        rename.triggered.connect(lambda: self._rename_interpretation_interval(well_name, index))
        recolor.triggered.connect(lambda: self._recolor_interpretation_interval(well_name, index))
        delete.triggered.connect(lambda: self._delete_interpretation_interval(well_name, index))
        if palette is not None:
            palette.triggered.connect(lambda: self._set_interpretation_palette(well_name, index))
        menu.exec(QCursor.pos())

    def _set_interpretation_palette(self, well_name: str, index: int) -> None:
        interval = self._well_interpretations[well_name][index]
        kind = str(interval.get("kind") or "")
        if kind == "facies":
            self._sync_facies_palette()
        palette = self._lithology_palette if kind == "lithology" else self._facies_palette
        if not palette:
            QMessageBox.information(self, "Палетка", "Для этой дорожки пока нет доступных классов.")
            return
        titles = [f"{item['symbol']} · {item['name']}" for item in palette]
        keys = [str(item.get("key") or item["name"]) for item in palette]
        current = str(interval.get("key") or interval.get("name") or "")
        initial = keys.index(current) if current in keys else 0
        selected_title, accepted = QInputDialog.getItem(
            self,
            "Литология" if kind == "lithology" else "Фации",
            "Элемент из сохранённой палетки:",
            titles,
            initial,
            False,
        )
        if not accepted or not selected_title:
            return
        item = palette[titles.index(selected_title)]
        interval.update(
            key=str(item.get("key") or item["name"]),
            name=item["name"],
            symbol=item["symbol"],
            color=item["color"],
            pattern=item["pattern"],
        )
        self._show_stack()

    def _rename_interpretation_interval(self, well_name: str, index: int) -> None:
        interval = self._well_interpretations[well_name][index]
        name, accepted = QInputDialog.getText(self, "Название интервала", "Название:", text=str(interval.get("name") or ""))
        if accepted and name.strip():
            interval["name"] = name.strip()
            self._show_stack()

    def _recolor_interpretation_interval(self, well_name: str, index: int) -> None:
        interval = self._well_interpretations[well_name][index]
        color = QColorDialog.getColor(QColor(str(interval.get("color") or "#8f71d2")), self, "Цвет интервала")
        if color.isValid():
            interval["color"] = color.name(QColor.NameFormat.HexRgb)
            self._show_stack()

    def _delete_interpretation_interval(self, well_name: str, index: int) -> None:
        intervals = self._well_interpretations.get(well_name, [])
        if not 0 <= index < len(intervals):
            return
        del intervals[index]
        if not intervals:
            self._well_interpretations.pop(well_name, None)
        self._show_stack()
        self.statusBar().showMessage("Интерпретационный интервал удалён")

    def export_presentation_png(self) -> None:
        default_name = f"{self._project_title or 'kern_analyzer'}_presentation.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт для презентации",
            str((self._project_folder or self._projects_root()) / default_name),
            "PNG высокого качества (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            canvas = self.sheets.currentWidget()
            canvas.export_png(path, long_side=6000)
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт PNG", str(exc))
            return
        self.statusBar().showMessage(f"Экспортировано для презентации: {Path(path).name} · до 6000 px по длинной стороне")

    def configure_excel_report_inputs(self) -> None:
        well_name = self._choose_well("Паспорт для итогового Excel")
        if well_name is None:
            return
        self._edit_excel_report_inputs(well_name)

    def _edit_excel_report_inputs(self, well_name: str) -> None:
        dialog = ExcelReportInputsDialog(well_name, self._well_report_inputs.get(well_name), self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        values = dialog.result_values()
        if values:
            self._well_report_inputs[well_name] = values
        else:
            self._well_report_inputs.pop(well_name, None)
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        else:
            self.statusBar().showMessage(f"Паспорт Excel заполнен: {well_name} · сохраните проект (Ctrl+S)")

    def _core_description_report_wells(self) -> list[dict]:
        """Capture the exact visual and editable data of the core-description sheet."""
        self._show_stack()
        columns = {item.well_name: item for item in self.workspace._correlation_items if item.well_name}
        wells: list[dict] = []
        for well_name in self._well_names():
            column = columns.get(well_name)
            if column is None:
                continue
            region = column.sceneBoundingRect()
            for record in self._records_for_well(well_name):
                photo_item = self.workspace._items_by_identifier.get(record.identifier)
                if photo_item is not None:
                    region = region.united(photo_item.sceneBoundingRect())
            image = self.workspace.render_scene_region(region.adjusted(-12, -12, 12, 12), long_side=7200)
            layers = []
            for record in self._records_for_well(well_name):
                for detection in StackColumnItem._sort_by_reading_order(record.detections):
                    attributes = dict(detection.attributes or {})
                    layers.append(
                        {
                            "label": detection.label,
                            "depth_from": detection.depth_from,
                            "depth_to": detection.depth_to,
                            "attributes": attributes,
                            "vlm": {
                                "facies_association": attributes.get("__vlm_facies_association", ""),
                                "depositional_environment": attributes.get("__vlm_depositional_environment", ""),
                                "description": attributes.get("__vlm_description", ""),
                            },
                        }
                    )
            wells.append(
                {
                    "name": well_name,
                    "image": image,
                    "depth_range": self._well_depth_ranges.get(well_name),
                    "depth_reference": self._well_depth_references.get(well_name, "MD"),
                    "depth_settings": dict(self._well_depth_settings.get(well_name, {})),
                    "core_interval": self._well_intervals.get(well_name),
                    "excel_report_inputs": dict(self._well_report_inputs.get(well_name, {})),
                    "layers": layers,
                }
            )
        return wells

    def export_core_description_report(self) -> None:
        if not self._well_names():
            QMessageBox.information(self, "Отчёт", "Сначала добавьте хотя бы одну скважину или фотографии керна.")
            return
        missing_columns = [Path(record.path).name for record in self._records if not record.core_columns]
        unresolved = sum(
            detection.label in {"", "Новый контур", UNRECOGNIZED_FACIES}
            for record in self._records if record.core_columns
            for detection in postprocess_detections(record, record.detections)
        )
        if missing_columns or unresolved:
            details = []
            if missing_columns:
                details.append(f"На {len(missing_columns)} фото не найдены столбики керна; проверьте их границы.")
            if unresolved:
                details.append(f"Интервалов без установленной фации: {unresolved}; назначьте им фации.")
            QMessageBox.warning(self, "Разрез ещё не заполнен", "\n".join(details))
            return
        default = (self._project_folder or self._projects_root()) / f"{self._project_title or 'kern_analyzer'}_послойное_описание.xlsx"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить послойное седиментологическое описание",
            str(default),
            "Книга Excel (*.xlsx);;Документ Word (*.docx);;PDF (*.pdf)",
        )
        if not path:
            return
        suffix = Path(path).suffix.casefold()
        if suffix == ".pdf" or "PDF" in selected_filter:
            report_format = "pdf"
        elif suffix == ".xlsx" or "Excel" in selected_filter:
            report_format = "xlsx"
        else:
            report_format = "docx"
        if not suffix:
            path += f".{report_format}"
        try:
            wells = self._core_description_report_wells()
            export_core_description_report(path, self._project_title, wells, report_format)
        except Exception as exc:
            QMessageBox.critical(self, "Экспорт отчёта", str(exc))
            return
        self.statusBar().showMessage(f"Отчёт сохранён: {Path(path).name} · скважин: {len(wells)}")

    def edit_gis(self) -> None:
        self._edit_log("gis")

    def edit_rigis(self) -> None:
        self._edit_log("rigis")

    def _edit_log(self, kind: str) -> None:
        title = "ГИС" if kind == "gis" else "РИГИС"
        well_name = self._choose_well(f"{title}: выбор скважины")
        if well_name is None:
            return
        log_data = self._well_log(well_name, kind)
        if not log_data:
            QMessageBox.information(self, title, f"Сначала загрузите LAS-файл для {title} этой скважины.")
            return
        top, base = self._current_core_interval(log_data, well_name)
        dialog = LogEditorDialog(log_data, top, base, f"Настройка {title} · {well_name}", self, settings_key=kind)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            updated_data, top, base = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте данные", str(exc))
            return
        self._set_well_log(well_name, kind, updated_data)
        self._apply_core_interval(top, base, well_name)
        self._show_stack()
        self.statusBar().showMessage(f"{title} обновлён и привязан к интервалу {top:g}–{base:g} м")

    def create_demo_las(self) -> None:
        """Create editable synthetic LAS curves for immediately testing the stack layout."""
        well_name = self._choose_well("Демо-ГИС: выбор скважины")
        if well_name is None:
            return
        ordered = self._ordered_detections(well_name)
        if not ordered:
            QMessageBox.information(self, "Демо-ГИС", "Сначала загрузите фото и дождитесь сегментации.")
            return
        start_depth = 1000.0
        for index, detection in enumerate(ordered):
            if detection.depth_from is None:
                detection.depth_from = start_depth + index
            if detection.depth_to is None:
                detection.depth_to = start_depth + index + 1
        end_depth = max(detection.depth_to or start_depth for detection in ordered)
        step = 0.02
        depths = [round(start_depth + index * step, 4) for index in range(int((end_depth - start_depth) / step) + 1)]
        values = {
            "GR": [65 + 30 * math.sin((depth - start_depth) * 2.6) for depth in depths],
            "RT": [20 + 15 * math.cos((depth - start_depth) * 1.7) + 5 * math.sin((depth - start_depth) * 6) for depth in depths],
            "RHOB": [2.35 + 0.18 * math.sin((depth - start_depth) * 3.1) for depth in depths],
            "NPHI": [0.24 + 0.09 * math.cos((depth - start_depth) * 2.2) for depth in depths],
        }
        self._set_well_log(well_name, "gis", {
            "file_name": "demo_las.las",
            "well_name": "Демо-скважина",
            "depth_curve": "DEPT",
            "depth_unit": "M",
            "curves": [
                {"mnemonic": "DEPT", "unit": "M"},
                {"mnemonic": "GR", "unit": "API"},
                {"mnemonic": "RT", "unit": "OHMM"},
                {"mnemonic": "RHOB", "unit": "G/C3"},
                {"mnemonic": "NPHI", "unit": "V/V"},
            ],
            "depths": depths,
            "values": {"DEPT": depths, **values},
        })
        for record in self._records_for_well(well_name):
            self.workspace.update_photo_detections(record)
        self._show_stack()
        self.statusBar().showMessage("Созданы демо-данные LAS · двойной клик по «Привязке» редактирует интервалы")

    def _show_segmentation_progress(self, current: int, total: int, file_name: str) -> None:
        label = f"Сегментация {current}/{total}: {file_name}"
        self.statusBar().showMessage(label)
        self._update_activity(label, current, total)

    def _apply_image_detections(self, image_path: str, detections, core_columns) -> None:
        record = next((item for item in self._records if item.path == image_path), None)
        if record is None:
            return
        record.analysis_mode = PIPELINE_MODE
        record.core_columns = [dict(column) for column in core_columns]
        self._update_depth_segments_for_columns(record)
        preserved = [
            item for item in record.detections
            if item.training_ready or item.attributes.get("__manual_contour") == "1"
        ]
        record.detections = postprocess_detections(record, [*preserved, *detections])
        self._ensure_core_column_description_rows(record)
        for detection in detections:
            self._sync_depth_range_from_polygon(record, detection)
        self.workspace.update_photo_detections(record)
        self._segmentation_updated_count += 1
        if self._segmentation_updated_count % 10 == 0:
            self._refresh_project_tree()

    def _segmentation_error(self, message: str) -> None:
        self._segmentation_failed = True
        QMessageBox.critical(self, "Ошибка сегментации", message)

    def _on_segmentation_finished(self) -> None:
        self._segmentation_thread = None
        self._schedule_vlm_dispatch()
        self._segmentation_worker = None
        self._refresh_project_tree()
        self._refresh_training_bar()
        if self._queued_records:
            if self._vlm_chat_queue:
                self._schedule_vlm_dispatch()
                return
            queued_records, self._queued_records = self._queued_records, []
            self.run_segmentation(queued_records)
            return
        if self._pending_photo_imports and not self._segmentation_failed:
            self._load_next_photo_batch()
            return
        self._close_activity()
        if not self._segmentation_failed:
            self._show_stack()
            if self._project_folder is not None:
                self._save_current_project(self._project_folder)
            count = sum(len(record.detections) for record in self._records)
            unresolved = sum(
                detection.label == UNRECOGNIZED_FACIES
                for record in self._records for detection in record.detections
            )
            missing = sum(not record.core_columns for record in self._records)
            assembly = (
                "общий столбик справа от фотографий создан"
                if missing < len(self._records) else "столбики не найдены, проверьте границы керна"
            )
            self.statusBar().showMessage(
                f"Сегментация завершена · объектов для описания: {count}, требуют уточнения: {unresolved}, "
                f"фото без столбиков: {missing} · {assembly}"
            )
            self._start_cv1_descriptions()

    def _on_geometry_issues(self, image_path: str, issues: list[dict]) -> None:
        record = next((item for item in self._records if item.path == image_path), None)
        if record is not None:
            record.analysis_issues = [dict(issue) for issue in issues]
        self._analysis_geometry_issues.extend({"photo": Path(image_path).name, **issue} for issue in issues)

    def _start_cv1_descriptions(self) -> None:
        if self._vlm_closing:
            return
        """Start the text stage only after the YOLO worker released its GPU memory."""
        if self._analysis_active:
            return
        self._analysis_geometry_issues = [
            {"photo": Path(record.path).name, **issue}
            for record in self._records if record.analysis_mode == PIPELINE_MODE
            for issue in record.analysis_issues
        ]
        self._analysis_queue.clear()
        for record in self._records:
            if record.analysis_mode != PIPELINE_MODE:
                continue
            for detection in record.detections:
                if (
                    detection.attributes.get("__geometry_only") == "1"
                    and detection.attributes.get("__interpretation_status", "waiting") == "waiting"
                    and not detection.training_ready
                ):
                    try:
                        self._analysis_queue.append(IntervalJob.create(record, detection))
                    except ValueError as exc:
                        detection.attributes["Статус интерпретации"] = str(exc)
        self._analysis_total = len(self._analysis_queue)
        self._analysis_done = self._analysis_success = 0
        if not self._analysis_queue:
            self.assistant_panel.append_message("system", "Геометрия обработана; новых интервалов для описания нет. Проверьте найденные границы и пропуски.")
            return
        self._analysis_active = True
        self._refresh_chat_status()
        self._show_activity("Описание керна", "Подготовка локальной модели…", self._analysis_total)
        QTimer.singleShot(0, self._advance_cv1_description)

    def _advance_cv1_description(self) -> None:
        if self._vlm_closing or not self._analysis_active or self._analysis_pending is not None:
            return
        if self._vlm_pending is not None or self._vlm_chat_queue:
            self._schedule_vlm_dispatch()
            return
        config = self._local_vlm_config()
        if available_runtime(config) is not None:
            self._advance_turbo_descriptions(config)
            return
        while self._analysis_queue:
            job = self._analysis_queue.popleft()
            record = next((item for item in self._records if item.identifier == job.record_id), None)
            crop = user_data_root() / "vlm_temp" / f"interval_{uuid4().hex}.png"
            try:
                if record is None or not job.is_current(record):
                    self._analysis_done += 1
                    continue
                write_interval_crop(record, job, crop)
                config = self._local_vlm_config()
                if not Path(config["python"]).is_file() or not (Path(config["base"]) / "config.json").is_file():
                    raise RuntimeError("Укажите локальную базовую Qwen и Python в настройках ИИ. Интернет для анализа не используется.")
                whole_column = job.detection.attributes.get("__core_column_description") == "1"
                if whole_column:
                    column_number = int(job.detection.attributes.get("__core_column_index", "0")) + 1
                    context = (
                        f"Скважина: {record.well_name}. Показан целый предварительно найденный столбик керна №{column_number}, "
                        "а не отдельный слой или фациальный полигон. Опиши видимые признаки по всей его длине; "
                        "если породы меняются, укажи эту изменчивость в кратком описании. "
                        "Не считай рамку столбика границей фации и не придумывай недостающие слои. "
                    )
                else:
                    context = f"Скважина: {record.well_name}. Показан один полигон YOLO; белый фон вне породы игнорируй. "
                if job.detection.depth_from is not None and job.detection.depth_to is not None:
                    context += f"Заданный интервал: {job.detection.depth_from:g}–{job.detection.depth_to:g} м. "
                request_id = self._vlm_client.send(config["python"], {
                    "operation": "describe", "image": str(crop), "context": context,
                    "whole_column": whole_column,
                    "base": config["base"], "adapter": config["adapter"],
                    "theory_adapter": config["theory_adapter"], "model_mode": config["model_mode"],
                    "folders": [], "use_knowledge": False,
                })
                self._analysis_pending = (request_id, job, crop)
                subject = "столбик" if whole_column else "интервал"
                label = f"Описываю {subject} {self._analysis_done + 1}/{self._analysis_total} · {Path(record.path).name}"
                self._refresh_chat_status()
                self._update_activity(label, self._analysis_done, self._analysis_total, busy=True)
                return
            except Exception as exc:
                crop.unlink(missing_ok=True)
                if record is not None:
                    mark_interval_failure(job, str(exc))
                self._analysis_done += 1
                # A missing runtime affects every interval; do not loop through
                # the same failing configuration or launch repeated processes.
                if isinstance(exc, RuntimeError):
                    for remaining in self._analysis_queue:
                        mark_interval_failure(remaining, str(exc))
                    self._analysis_done += len(self._analysis_queue)
                    self._analysis_queue.clear()
                    self.assistant_panel.append_message("system", str(exc))
        self._finish_cv1_descriptions()

    def _advance_turbo_descriptions(self, config: dict) -> None:
        """Decode several full descriptions on one shared model, with stable job bindings."""
        if not self._analysis_queue:
            self._finish_cv1_descriptions()
            return
        first = self._analysis_queue.popleft()
        candidates = [first]
        while self._analysis_queue and len(candidates) < 4:
            candidates.append(self._analysis_queue.popleft())
        jobs, crops, items = [], [], []
        for job in candidates:
            record = next((row for row in self._records if row.identifier == job.record_id), None)
            crop = user_data_root() / "vlm_temp" / f"interval_{uuid4().hex}.png"
            try:
                if record is None or not job.is_current(record):
                    self._analysis_done += 1
                    continue
                write_interval_crop(record, job, crop)
                whole_column = job.detection.attributes.get("__core_column_description") == "1"
                context = (
                    f"Скважина: {record.well_name}. Показан целый столбик; опиши изменения по всей длине. "
                    if whole_column else f"Скважина: {record.well_name}. Показан один полигон YOLO; белый фон игнорируй. "
                )
                if job.detection.depth_from is not None and job.detection.depth_to is not None:
                    context += f"Заданный интервал: {job.detection.depth_from:g}–{job.detection.depth_to:g} м."
                jobs.append(job)
                crops.append(crop)
                items.append({"id": str(len(jobs)), "image": str(crop), "context": context, "whole_column": whole_column})
            except Exception as exc:
                crop.unlink(missing_ok=True)
                mark_interval_failure(job, str(exc))
                self._analysis_done += 1
        if not jobs:
            if not self._analysis_queue:
                self._finish_cv1_descriptions()
            else:
                self._schedule_vlm_dispatch()
            return
        try:
            request_id = self._vlm_client.send(config["python"], {
                **config, "operation": "describe_batch", "items": items, "folders": [], "use_knowledge": False,
            })
        except Exception as exc:
            for job, crop in zip(jobs, crops):
                crop.unlink(missing_ok=True)
                mark_interval_failure(job, str(exc))
            self._analysis_done += len(jobs)
            self._schedule_vlm_dispatch()
            return
        self._analysis_pending = (request_id, jobs, crops)
        label = f"Подробные описания {self._analysis_done + 1}–{self._analysis_done + len(jobs)}/{self._analysis_total}"
        self._update_activity(label, self._analysis_done, self._analysis_total, busy=True)
        self._refresh_chat_status()

    def _apply_turbo_batch_reply(self, pending: tuple, result: dict) -> None:
        _, jobs, crops = pending
        replies = result.get("items") or {}
        for index, (job, crop) in enumerate(zip(jobs, crops), start=1):
            crop.unlink(missing_ok=True)
            record = next((row for row in self._records if row.identifier == job.record_id), None)
            response = replies.get(str(index)) if result.get("ok") else None
            if not isinstance(response, dict):
                response = {"ok": False, "error": result.get("error") or "Не получено описание данного интервала."}
            if record is not None:
                try:
                    apply_interval_description(record, job, response)
                    self._analysis_success += 1
                except ValueError as exc:
                    if job.is_current(record):
                        mark_interval_failure(job, str(exc))
                self.workspace.update_photo_detections(record)
            self._analysis_done += 1
        if not result.get("ok"):
            error = str(result.get("error") or "Локальная модель не ответила.")
            for remaining in self._analysis_queue:
                mark_interval_failure(remaining, error)
            self._analysis_done += len(self._analysis_queue)
            self._analysis_queue.clear()
            self.assistant_panel.append_message("system", error)
        self._schedule_vlm_dispatch()

    def _on_cv1_interval_reply(self, result: dict) -> None:
        pending = self._analysis_pending
        if pending is None:
            return
        self._analysis_pending = None
        if isinstance(pending[1], list):
            self._apply_turbo_batch_reply(pending, result)
            return
        _, job, crop = pending
        crop.unlink(missing_ok=True)
        record = next((item for item in self._records if item.identifier == job.record_id), None)
        if record is not None:
            try:
                apply_interval_description(record, job, result)
                self._analysis_success += 1
            except ValueError as exc:
                if job.is_current(record):
                    mark_interval_failure(job, str(exc))
            self.workspace.update_photo_detections(record)
        self._analysis_done += 1
        if not result.get("ok"):
            error = str(result.get("error") or "Локальная модель не ответила.")
            for remaining in self._analysis_queue:
                mark_interval_failure(remaining, error)
            self._analysis_done += len(self._analysis_queue)
            self._analysis_queue.clear()
            self.assistant_panel.append_message("system", error)
        self._schedule_vlm_dispatch()

    def _finish_cv1_descriptions(self) -> None:
        self._analysis_active = False
        self._refresh_chat_status()
        self._close_activity()
        self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        if self._project_folder is not None:
            self._save_current_project(self._project_folder)
        whole_columns = sum(
            detection.attributes.get("__core_column_description") == "1"
            and detection.attributes.get("__interpretation_status") == "draft"
            for record in self._records for detection in record.detections
        )
        message = (
            f"Анализ завершён. Описано объектов: {self._analysis_success} из {self._analysis_total}. "
            "Геометрия взята из масок YOLO или рамок найденных столбиков; описания и фации — черновики локальной VLM. "
            "Проверьте их в карточках."
        )
        if whole_columns:
            message += f" Целиком описано столбиков без фациальных масок: {whole_columns}."
        if self._analysis_geometry_issues:
            message += f" Замечаний к покрытию и пересечениям: {len(self._analysis_geometry_issues)}. Пропуски не заполнены выдуманными фациями."
            message += "\n" + "\n".join(f"{issue['photo']}: {issue.get('message', '')}" for issue in self._analysis_geometry_issues[:3])
        self.assistant_panel.append_message("assistant", message)
        self.statusBar().showMessage(message)
        self._schedule_vlm_dispatch()

    def _update_depth_segments_for_columns(self, record: PhotoRecord) -> None:
        """Calibrate the continuous stage-1 tape when a photo interval is known."""
        columns = [
            {key: float(item[key]) for key in ("left", "top", "right", "bottom")}
            for item in record.core_columns
            if all(key in item for key in ("left", "top", "right", "bottom"))
        ]
        if not columns:
            record.depth_segments = []
            return
        photo_top, photo_base = self._record_photo_interval(record)
        if photo_top is None or photo_base is None or photo_base <= photo_top:
            record.depth_segments = columns
            return
        total_height = sum(max(1.0, item["bottom"] - item["top"]) for item in columns)
        cursor = float(photo_top)
        segments: list[dict[str, float]] = []
        for index, column in enumerate(columns):
            span = (float(photo_base) - float(photo_top)) * max(1.0, column["bottom"] - column["top"]) / total_height
            depth_to = float(photo_base) if index == len(columns) - 1 else cursor + span
            segments.append({**column, "depth_from": cursor, "depth_to": depth_to})
            cursor = depth_to
        record.depth_segments = segments

    def _open_facies_editor(self, record: PhotoRecord, detection) -> None:
        self._vlm_target = (record, detection)
        self._active_well_name = record.well_name
        self._update_header_context()
        # Correlation editing is deliberately disabled while Kern Analyzer focuses
        # on core description and creation of a training dataset.
        if False:
            menu = QMenu(self)
            align = menu.addAction("\u041d\u043e\u0432\u0430\u044f \u043f\u0440\u044f\u043c\u0430\u044f \u043f\u043e \u0441\u043b\u043e\u044e")
            attach_menu = menu.addMenu("\u041f\u0440\u0438\u0432\u044f\u0437\u0430\u0442\u044c \u043a \u0441\u043e\u0437\u0434\u0430\u043d\u043d\u043e\u0439 \u043b\u0438\u043d\u0438\u0438")
            curve_actions = {}
            for curve in self.correlation_workspace.correlation_curves():
                action = attach_menu.addAction(str(curve.get("name") or curve.get("label") or "\u041b\u0438\u043d\u0438\u044f"))
                curve_actions[action] = curve
            attach_menu.setEnabled(bool(curve_actions))
            edit = menu.addAction("\u0420\u0435\u0434\u0430\u043a\u0442\u0438\u0440\u043e\u0432\u0430\u0442\u044c \u043b\u0438\u0442\u043e\u043b\u043e\u0433\u0438\u044e")
            selected = menu.exec(QCursor.pos())
            if selected is align:
                self._align_straight_to_layer(record, detection)
                return
            if selected in curve_actions:
                self._align_layer_to_existing_curve(record, detection, curve_actions[selected])
                return
            if selected is not edit:
                return
        for current_dialog in list(self._facies_dialogs):
            current_dialog.close()
        self.workspace.end_polygon_edit()
        dialog = FaciesDialog(
            detection.label,
            detection.confidence,
            detection.attributes,
            detection.depth_from,
            detection.depth_to,
            self,
            training_ready=detection.training_ready,
            primary_catalog=self._primary_facies_catalog,
            description_confirmed=expert_memory_is_confirmed(detection) or expert_description_is_confirmed(detection),
        )
        dialog.setModal(False)
        dialog.accepted.connect(lambda: self._save_lithology(record, detection, dialog))
        dialog.delete_requested.connect(lambda: self._delete_facies(record, detection))
        dialog.move_up_requested.connect(lambda: self._move_layer(record, detection, -1))
        dialog.move_down_requested.connect(lambda: self._move_layer(record, detection, 1))
        dialog.finished.connect(lambda _: self._forget_facies_dialog(dialog))
        self._facies_dialogs.append(dialog)
        dialog.show()
        self.statusBar().showMessage("Выберите фацию/параметры или переместите слой выше/ниже в колонке")

    def _align_layer_to_existing_curve(self, record: PhotoRecord, detection, curve: dict) -> None:
        depths = [float(value) for value in (detection.depth_from, detection.depth_to) if value is not None]
        if not depths:
            QMessageBox.information(self, "\u041f\u0440\u0438\u0432\u044f\u0437\u043a\u0430 \u043a \u043b\u0438\u043d\u0438\u0438", "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u0434\u0430\u0439\u0442\u0435 \u0433\u043b\u0443\u0431\u0438\u043d\u0443 \u0441\u043b\u043e\u044f.")
            return
        if self.correlation_workspace.align_curve_to_layer(curve, record.well_name, sum(depths) / len(depths)):
            self.statusBar().showMessage("\u0421\u043b\u043e\u0439 \u043f\u0440\u0438\u0432\u044f\u0437\u0430\u043d \u043a \u0432\u044b\u0431\u0440\u0430\u043d\u043d\u043e\u0439 \u043a\u043e\u0440\u0440\u0435\u043b\u044f\u0446\u0438\u043e\u043d\u043d\u043e\u0439 \u043b\u0438\u043d\u0438\u0438")

    def _align_straight_to_layer(self, record: PhotoRecord, detection) -> None:
        if len(self._well_names()) < 2:
            QMessageBox.information(self, "\u0412\u044b\u0440\u0430\u0432\u043d\u0438\u0432\u0430\u043d\u0438\u0435", "\u0414\u043b\u044f \u043b\u0438\u043d\u0438\u0438 \u043d\u0443\u0436\u043d\u044b \u043c\u0438\u043d\u0438\u043c\u0443\u043c \u0434\u0432\u0435 \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u044b.")
            return
        depths = [float(value) for value in (detection.depth_from, detection.depth_to) if value is not None]
        if not depths:
            QMessageBox.information(self, "\u0412\u044b\u0440\u0430\u0432\u043d\u0438\u0432\u0430\u043d\u0438\u0435", "\u0421\u043d\u0430\u0447\u0430\u043b\u0430 \u0437\u0430\u0434\u0430\u0439\u0442\u0435 \u0433\u043b\u0443\u0431\u0438\u043d\u0443 \u0441\u043b\u043e\u044f.")
            return
        depth = sum(depths) / len(depths)
        self._show_stack()
        name = f"{detection.label} \u00b7 {depth:g} \u043c"
        if self.correlation_workspace.add_straight_layer_correlation_curve(name, record.well_name, depth):
            self.statusBar().showMessage("\u0421\u043b\u043e\u0439 \u0432\u044b\u0440\u043e\u0432\u043d\u0435\u043d \u043e\u0434\u043d\u043e\u0439 \u043f\u0440\u044f\u043c\u043e\u0439 \u043f\u043e \u0432\u0441\u0435\u043c \u0441\u043a\u0432\u0430\u0436\u0438\u043d\u0430\u043c")

    def _begin_polygon_edit(self, record: PhotoRecord, detection) -> None:
        self._vlm_target = (record, detection)
        self._active_well_name = record.well_name
        self._update_header_context()
        for current_dialog in list(self._facies_dialogs):
            current_dialog.close()
        if self.workspace.begin_polygon_edit(record, detection):
            self.statusBar().showMessage(
                "Контур: перетаскивайте точки · «＋ Точка» или Shift+клик добавляет · «− Точка»/Delete удаляет · ✓ Готово завершает"
            )

    def _prepare_polygon_vertex_insert(self) -> None:
        if self.workspace.prepare_polygon_vertex_insert():
            self.statusBar().showMessage("Добавление вершины: щёлкните в нужном месте на контуре · Esc — отменить")
            return
        QMessageBox.information(self, "Добавить точку", "Сначала щёлкните по контуру, который нужно редактировать.")

    def _delete_selected_polygon_vertex(self) -> None:
        if self.workspace.delete_selected_polygon_vertex():
            self.statusBar().showMessage("Вершина удалена")
            return
        QMessageBox.information(
            self,
            "Удалить точку",
            "Сначала щёлкните по вершине контура. У полигона всегда должно остаться минимум три точки.",
        )

    def _finish_polygon_edit(self) -> None:
        self.workspace.end_polygon_edit()
        if self._vlm_target is not None:
            record, _ = self._vlm_target
            if any(item is record for item in self._records) and record.core_columns:
                record.detections = postprocess_detections(record, record.detections)
                self._refresh_facies_views(record)
        self.statusBar().showMessage("Редактирование контура завершено")

    def _set_new_contour_mode(self, active: bool) -> None:
        if active and not self._records:
            self._new_contour_action.setChecked(False)
            self.statusBar().showMessage("Сначала загрузите фотографию")
            return
        self.workspace.set_new_contour_mode(active)
        if active:
            self.statusBar().showMessage(
                "Новый контур: ставьте точки левой кнопкой на фото · Enter или двойной клик — завершить · Esc — отменить"
            )

    def _add_manual_contour(self, record: PhotoRecord, detection) -> None:
        # It was drawn by the user, so it should be visible in the counter
        # immediately.  It becomes a trainable example once a facies code is
        # selected; an unnamed YOLO class would be invalid training data.
        detection.training_ready = False
        detection.attributes["__manual_contour"] = "1"
        record.detections.append(detection)
        self.workspace.set_new_contour_mode(False)
        self._new_contour_action.setChecked(False)
        self._refresh_facies_views(record)
        self.workspace.begin_polygon_edit(record, detection)
        self.statusBar().showMessage("Новый контур добавлен в счётчик «новые» · назначьте фацию правой кнопкой мыши")

    def _save_lithology(self, record: PhotoRecord, detection, dialog: FaciesDialog) -> None:
        try:
            depth_from, depth_to = dialog.selected_depth_range()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте глубины", str(exc))
            return
        selected_label = dialog.selected_facies()
        if not selected_label:
            QMessageBox.warning(self, "Не указана фация", "Введите или выберите основную метку фации, например DWCh.")
            return
        selected_attributes = dialog.selected_attributes()
        if dialog.training_confirmed.isChecked() and not self._resolve_training_class(selected_label, selected_attributes.get("Индекс фации")):
            QMessageBox.warning(self, "Уточните фацию", "Для обучения финальной модели выберите точную фацию из справочника. Свою метку можно сохранить без отметки об обучении.")
            return
        manually_drawn = detection.label == "Новый контур" or detection.attributes.get("__manual_contour") == "1"
        detection.label = selected_label
        detection.attributes = selected_attributes
        detection.attributes["__manual_interval_edited"] = "1"
        if manually_drawn:
            detection.attributes["__manual_contour"] = "1"
        detection.depth_from = depth_from
        detection.depth_to = depth_to
        geometry_updated = self._sync_polygon_to_depth_range(record, detection)
        detection.training_ready = dialog.training_confirmed.isChecked()
        detection.attributes.pop("__expert_description_confirmed", None)
        detection.attributes.pop("__expert_description_sha256", None)
        detection.attributes["__expert_memory_confirmed"] = "0"
        detection.attributes.pop("__expert_memory_sha256", None)
        if detection.attributes.get("__geometry_only") == "1":
            if dialog.description_confirmed.isChecked() and detection.training_ready and detection.attributes.get("__vlm_description", "").strip():
                detection.attributes["__interpretation_status"] = "reviewed"
                detection.attributes["Статус интерпретации"] = "описание, фация и границы подтверждены специалистом"
                detection.attributes["__expert_description_confirmed"] = "1"
                detection.attributes["__expert_description_sha256"] = expert_description_digest(detection)
            else:
                detection.attributes["__interpretation_status"] = "draft"
                detection.attributes["Статус интерпретации"] = "черновик, требуется проверка геолога"
        if dialog.description_confirmed.isChecked() and detection.attributes.get("__vlm_description", "").strip():
            detection.attributes["__expert_memory_confirmed"] = "1"
            detection.attributes["__expert_memory_sha256"] = expert_memory_digest(detection)
        memory_errors = []
        try:
            save_expert_facies_mapping(record, detection, confirmed=True)
        except (OSError, ValueError, TypeError) as exc:
            memory_errors.append("привязка фации: " + str(exc))
        try:
            save_interval_edit(record, detection, project=self._expert_project_key(), well=record.well_name)
        except (OSError, ValueError, TypeError) as exc:
            memory_errors.append("журнал поправок: " + str(exc))
        if memory_errors:
            QMessageBox.warning(self, "Память правок", "Не всё удалось запомнить: " + "; ".join(memory_errors) + ". Проверьте доступ к фото и сохраните проект.")
        if record.core_columns:
            record.detections = postprocess_detections(record, record.detections)
        self._refresh_facies_views(record)
        suffix = " · подсветка контура обновлена" if geometry_updated else ""
        self.statusBar().showMessage(f"Параметры фации сохранены: {detection.label}{suffix}")

    def _open_depth_binding(self, record: PhotoRecord, detection) -> None:
        dialog = DepthRangeDialog(detection.depth_from, detection.depth_to, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        try:
            detection.depth_from, detection.depth_to = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "Проверьте глубины", str(exc))
            return
        geometry_updated = self._sync_polygon_to_depth_range(record, detection)
        self._refresh_facies_views(record)
        suffix = " · подсветка контура обновлена" if geometry_updated else ""
        self.statusBar().showMessage(f"Привязка слоя обновлена{suffix}")

    def _delete_facies(self, record: PhotoRecord, detection) -> None:
        record.detections = [item for item in record.detections if item is not detection]
        if record.core_columns:
            record.detections = postprocess_detections(record, record.detections)
        self._refresh_facies_views(record)
        self.statusBar().showMessage("Подсветка удалена")

    def _forget_facies_dialog(self, dialog: FaciesDialog) -> None:
        if dialog in self._facies_dialogs:
            self._facies_dialogs.remove(dialog)

    def _on_facies_geometry_changed(self, record: PhotoRecord, detection) -> None:
        depth_updated = self._sync_depth_range_from_polygon(record, detection)
        # Editing geometry alone does not confirm the model's class choice.
        # An already reviewed sample retains its explicit confirmation.
        self._refresh_facies_views(record)
        suffix = " · интервал глубин обновлён" if depth_updated else ""
        if detection.training_ready:
            suffix += " · разметка добавлена в обучение"
        self.statusBar().showMessage(f"Контур фации обновлён{suffix}")

    @staticmethod
    def _photo_interval_from_path(path: Path) -> tuple[float | None, float | None]:
        try:
            interval = photo_interval_from_filename(path)
        except ValueError:
            return None, None
        return interval.top, interval.base

    def _record_photo_interval(self, record: PhotoRecord) -> tuple[float | None, float | None]:
        if record.photo_depth_from is not None and record.photo_depth_to is not None:
            return record.photo_depth_from, record.photo_depth_to
        return self._photo_interval_from_path(Path(record.path))

    @staticmethod
    def _depth_segment_for_polygon(record: PhotoRecord, detection) -> dict[str, float] | None:
        """Return the calibration column that owns most of a polygon."""
        if not record.depth_segments:
            return None
        bounds = QPolygonF(detection.polygon).boundingRect()
        center_x = bounds.center().x()
        center_y = bounds.center().y()

        def score(segment: dict[str, float]) -> tuple[float, float]:
            left, top = segment["left"], segment["top"]
            right, bottom = segment["right"], segment["bottom"]
            overlap = max(0.0, min(bounds.right(), right) - max(bounds.left(), left)) * max(0.0, min(bounds.bottom(), bottom) - max(bounds.top(), top))
            contains_center = left <= center_x <= right and top <= center_y <= bottom
            return (1.0 if contains_center else 0.0, overlap)

        return max(record.depth_segments, key=score, default=None)

    def _depth_mapping_for_polygon(self, record: PhotoRecord, detection) -> dict[str, float] | None:
        segment = self._depth_segment_for_polygon(record, detection)
        if segment is not None:
            return segment
        photo_top, photo_base = self._record_photo_interval(record)
        if photo_top is None or photo_base is None:
            return None
        return {
            "left": 0.0, "top": 0.0, "right": float(record.pixmap.width()), "bottom": float(record.pixmap.height()),
            "depth_from": photo_top, "depth_to": photo_base,
        }

    def _sync_depth_range_from_polygon(self, record: PhotoRecord, detection) -> bool:
        """Map the edited vertical polygon span back to measured depth."""
        segment = self._depth_mapping_for_polygon(record, detection)
        if segment is None or not all(key in segment for key in ("depth_from", "depth_to")):
            return False
        bounds = QPolygonF(detection.polygon).boundingRect()
        pixel_top, pixel_base = segment["top"], segment["bottom"]
        depth_top, depth_base = segment["depth_from"], segment["depth_to"]
        top_y = max(pixel_top, min(pixel_base, bounds.top()))
        base_y = max(pixel_top, min(pixel_base, bounds.bottom()))
        if base_y - top_y <= 1e-6 or depth_base <= depth_top:
            return False
        depth_span = depth_base - depth_top
        pixel_span = pixel_base - pixel_top
        detection.depth_from = round(depth_top + (top_y - pixel_top) / pixel_span * depth_span, 3)
        detection.depth_to = round(depth_top + (base_y - pixel_top) / pixel_span * depth_span, 3)
        return True

    def _sync_polygon_to_depth_range(self, record: PhotoRecord, detection) -> bool:
        """Resize an existing polygon vertically to the edited depth range."""
        segment = self._depth_mapping_for_polygon(record, detection)
        depth_from, depth_to = detection.depth_from, detection.depth_to
        if (
            segment is None or depth_from is None or depth_to is None
            or not all(key in segment for key in ("depth_from", "depth_to"))
            or depth_to <= depth_from or not detection.polygon
        ):
            return False
        bounds = QPolygonF(detection.polygon).boundingRect()
        old_top, old_base = bounds.top(), bounds.bottom()
        if old_base - old_top <= 1e-6:
            return False
        pixel_top, pixel_base = segment["top"], segment["bottom"]
        depth_top, depth_base = segment["depth_from"], segment["depth_to"]
        if depth_from < depth_top - 1e-6 or depth_to > depth_base + 1e-6 or depth_base <= depth_top:
            return False
        scale = (pixel_base - pixel_top) / (depth_base - depth_top)
        target_top = pixel_top + (depth_from - depth_top) * scale
        target_base = pixel_top + (depth_to - depth_top) * scale
        if target_base - target_top <= 1e-6:
            return False
        target_scale = (target_base - target_top) / (old_base - old_top)
        record_width = record.pixmap.width()
        detection.polygon = [
            QPointF(
                max(0.0, min(float(record_width), point.x())),
                max(0.0, min(float(record.pixmap.height()), target_top + (point.y() - old_top) * target_scale)),
            )
            for point in detection.polygon
        ]
        return True

    def _refresh_facies_views(self, record: PhotoRecord) -> None:
        self.workspace.update_photo_detections(record)
        if self._well_names():
            self._show_stack()
        self._refresh_project_tree()
        self._refresh_training_bar()
        self._schedule_edit_autosave()
