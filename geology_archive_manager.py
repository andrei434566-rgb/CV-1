"""One-window entry point for local geological archive preparation."""
from __future__ import annotations
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (QApplication,QFileDialog,QHBoxLayout,QLabel,
    QMainWindow,QMessageBox,QPlainTextEdit,QPushButton,QVBoxLayout,QWidget)

BASE=Path(__file__).resolve().parent

class ScanWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    done = Signal()

    def __init__(self, root, destination):
        super().__init__()
        self.root, self.destination = root, destination

    @Slot()
    def run(self):
        try:
            from index_geology_archive import scan
            self.finished.emit(scan(self.root, self.destination))
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.done.emit()


class HybridWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    done = Signal()

    def __init__(self, photo, model, output, vlm_model):
        super().__init__()
        self.photo, self.model, self.output, self.vlm_model = photo, model, output, vlm_model

    @Slot()
    def run(self):
        try:
            from cv1_hybrid_analysis import analyze
            self.finished.emit(analyze(self.photo, self.model, self.output, self.vlm_model))
        except Exception as exc:
            self.failed.emit("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        finally:
            self.done.emit()

class VlmPhotoWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)
    done = Signal()

    def __init__(self, photo, output, model):
        super().__init__()
        self.photo, self.output, self.model = photo, output, model

    @Slot()
    def run(self):
        try:
            from cv1_vlm_photo import analyze
            self.finished.emit(analyze(self.photo, self.output, self.model))
        except Exception as exc:
            self.failed.emit("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        finally:
            self.done.emit()


class ArchiveManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV-1 — Геологическая база и обучение ИИ | диагностика v5")
        self.resize(850,500)
        self.root=None
        self.scan_thread=None
        self.hybrid_thread=None
        self.open_review_after_scan=False
        self.action_buttons=[]
        self.output_dir=BASE/"_cv1_local_indexes"
        self.path_label=QLabel("Папка не выбрана")
        self.status=QPlainTextEdit()
        self.status.setReadOnly(True)
        body=QWidget()
        layout=QVBoxLayout(body)
        layout.addWidget(QLabel("Импорт скважин, фотографий, Excel и теоретических материалов"))
        layout.addWidget(self.path_label)
        self.output_label=QLabel("Результаты: "+str(self.output_dir))
        layout.addWidget(self.output_label)
        row=QHBoxLayout()
        for title,fn in [
            ("Выбрать базу",self.choose),
            ("Папка результатов",self.choose_output),
            ("Сканировать",self.scan),
            ("Диагностика",self.diagnose),
            ("Проверить фото ↔ Excel",self.review),
            ("Экспорт для VLM",self.export),
            ("База знаний Excel",self.export_knowledge),
            ("Проверить CVAT YOLO",self.inspect_cvat),
            ("Импорт CVAT YOLO",self.import_cvat),
            ("YOLO + VLM: фото",self.hybrid_photo),
            ("VLM: анализ керна",self.vlm_photo),
            ("Привязка глубин",self.depth_mapping),
        ]:
            button=QPushButton(title)
            button.clicked.connect(fn)
            row.addWidget(button)
            if title in ("Сканировать","Проверить фото ↔ Excel"):
                self.action_buttons.append(button)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.setCentralWidget(body)

    def choose(self):
        directory=QFileDialog.getExistingDirectory(self,"Выбрать корневую папку геологической базы")
        if directory:
            self.root=Path(directory)
            self.path_label.setText(str(self.root))
            self.status.appendPlainText("Выбрана база: "+str(self.root))

    def choose_output(self):
        directory=QFileDialog.getExistingDirectory(self,"Выбрать локальную папку результатов",str(BASE))
        if directory:
            self.output_dir=Path(directory)
            self.output_label.setText("Результаты: "+str(self.output_dir))

    def index_dir(self):
        import hashlib
        key=hashlib.sha256(str(self.root).casefold().encode("utf-8")).hexdigest()[:12]
        return self.output_dir/("archive_"+key)

    def _ready(self):
        if self.root is None:
            QMessageBox.information(self,"Нет папки","Сначала выберите корневую папку.")
            return False
        return True

    def scan(self, open_review=False):
        if not self._ready():
            return
        if self.scan_thread is not None:
            self.status.appendPlainText("Индексация уже выполняется. Дождитесь завершения.")
            return
        self.open_review_after_scan = open_review
        self.status.appendPlainText("Сканирование запущено в фоновом потоке. Окно остаётся доступным.")
        for button in self.action_buttons:
            button.setEnabled(False)
        self.scan_thread=QThread(self)
        self.scan_worker=ScanWorker(self.root,self.index_dir())
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.finished.connect(self._scan_complete)
        self.scan_worker.failed.connect(self._scan_failed)
        self.scan_worker.done.connect(self.scan_thread.quit)
        self.scan_worker.done.connect(self.scan_worker.deleteLater)
        self.scan_thread.finished.connect(self._scan_cleanup)
        self.scan_thread.start()

    @Slot(dict)
    def _scan_complete(self, result):
        self.status.appendPlainText("Индексация завершена:\\n"+json.dumps(result,ensure_ascii=False,indent=2))
        if self.open_review_after_scan:
            self._show_review()

    @Slot(str)
    def _scan_failed(self, message):
        self.status.appendPlainText("Ошибка индексации: "+message)
        QMessageBox.critical(self,"Ошибка индексации",message)

    @Slot()
    def _scan_cleanup(self):
        thread=self.scan_thread
        self.scan_thread=None
        self.scan_worker=None
        self.open_review_after_scan=False
        for button in self.action_buttons:
            button.setEnabled(True)
        if thread is not None:
            thread.deleteLater()

    def _show_review(self):
        path=self.index_dir()/"archive_index.json"
        if not path.exists():
            QMessageBox.warning(self,"Нет индекса","Сначала выполните сканирование.")
            return
        from review_geology_pairs import ReviewWindow
        self.review_window=ReviewWindow(path)
        self.review_window.show()

    def diagnose(self):
        if not self._ready(): return
        from diagnose_geology import save_report
        try:
            dest, report = save_report(self.root, self.index_dir())
            idx=report.get("index",{})
            sheets=report.get("excel",[])
            total=sum(x.get("facies_rows",0) for x in sheets)
            empty=[x for x in sheets if x.get("facies_rows",0)==0]
            uv=idx.get("uv_named_still_in_index","нет индекса")
            lines=[
                "=== CV-1 | ПРОВЕРКА НА ЭКРАНЕ ===",
                "Версия диагностики: v4 / 21.10.2026",
                "Индекс: "+str(idx.get("path","не найден")),
                "Фото в индексе: "+str(idx.get("pairs","?")),
                "UV по названию в индексе: "+str(uv),
                "Проверено Excel: "+str(len(sheets)),
                "Прочитано строк фаций: "+str(total),
                "Excel без распознанных фаций: "+str(len(empty)),
                "",
                "ПРИМЕРЫ EXCEL:",
            ]
            for sheet in sheets[:8]:
                lines.append(Path(sheet["file"]).name+" | фаций: "+str(sheet.get("facies_rows",0)))
                if sheet.get("error"): lines.append("ОШИБКА: "+str(sheet["error"]))
                for row in sheet.get("examples",[])[:2]:
                    lines.append("  "+str(row.get("top"))+"–"+str(row.get("bottom"))+
                        " | "+str(row.get("facies_index"))+" | "+str(row.get("facies")))
            lines.extend(["", "ПРИМЕРЫ UV В ИНДЕКСЕ:"])
            lines.extend(Path(p).name for p in idx.get("uv_examples", [])[:8])
            lines.append("ПРОВЕРКА UV-ФИЛЬТРА: "+str(report.get("filename_uv_checks",{})))
            lines.extend(["","ПУТИ ИСПОЛНЯЕМОГО КОДА:"])
            lines.extend(str(k)+": "+str(v) for k,v in report.get("runtime",{}).items())
            lines.append("=== КОНЕЦ ПРОВЕРКИ ===")
            self.status.setPlainText("\n".join(lines))
            self.status.verticalScrollBar().setValue(0)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка диагностики", str(exc))

    def review(self):
        self.scan(open_review=True)

    def depth_mapping(self):
        if not self._ready(): return
        from geology_depth_mapping import generate
        try:
            result=generate(self.index_dir())
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
            if result.get("status")=="template_created":
                QMessageBox.information(self,"Привязка глубин",
                    "Создан lane_annotations.json. Укажите координаты рядов и их интервалы; повторное нажатие построит карту глубин.")
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка привязки глубин",str(exc))

    def vlm_photo(self):
        if self.hybrid_thread is not None:
            QMessageBox.information(self, "VLM", "Дождитесь завершения анализа.")
            return
        photo, _ = QFileDialog.getOpenFileName(
            self, "Фотография для VLM", str(BASE),
            "Изображения (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)")
        if not photo:
            return
        from PySide6.QtWidgets import QInputDialog
        model, ok = QInputDialog.getText(
            self, "Локальная VLM", "Название установленной модели Ollama:",
            text="qwen2.5vl:7b")
        if not ok or not model.strip():
            return
        import hashlib
        key = hashlib.sha256(str(Path(photo).resolve()).encode("utf-8")).hexdigest()[:12]
        destination = self.output_dir / "vlm_photos" / key
        self.hybrid_thread = QThread(self)
        self.hybrid_worker = VlmPhotoWorker(photo, destination, model.strip())
        self.hybrid_worker.moveToThread(self.hybrid_thread)
        self.hybrid_thread.started.connect(self.hybrid_worker.run)
        self.hybrid_worker.finished.connect(self._vlm_photo_complete)
        self.hybrid_worker.failed.connect(self._hybrid_failed)
        self.hybrid_worker.done.connect(self.hybrid_thread.quit)
        self.hybrid_worker.done.connect(self.hybrid_worker.deleteLater)
        self.hybrid_thread.finished.connect(self._hybrid_cleanup)
        self.status.appendPlainText("VLM анализирует фотографию без YOLO...")
        self.hybrid_thread.start()

    @Slot(dict)
    def _vlm_photo_complete(self, result):
        self.status.appendPlainText(
            "=== VLM: геологическое описание ===\\n" + result["description"] +
            "\\nФайл: " + result["description_file"])

    def hybrid_photo(self):
        if self.hybrid_thread is not None:
            QMessageBox.information(self, "Анализ", "Дождитесь окончания текущего анализа.")
            return
        photo, _ = QFileDialog.getOpenFileName(
            self, "Фотография керна", str(BASE),
            "Изображения (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)")
        if not photo:
            return
        model, _ = QFileDialog.getOpenFileName(
            self, "Модель сегментации best.pt", str(BASE), "Модель YOLO (*.pt)")
        if not model:
            return
        from PySide6.QtWidgets import QInputDialog
        vlm, ok = QInputDialog.getText(
            self, "Локальная VLM", "Имя модели в Ollama (пусто = только YOLO):",
            text="")
        if not ok:
            return
        import hashlib
        key = hashlib.sha256(str(Path(photo).resolve()).encode("utf-8")).hexdigest()[:12]
        destination = self.output_dir / "hybrid" / key
        self.hybrid_thread = QThread(self)
        self.hybrid_worker = HybridWorker(photo, model, destination, vlm.strip())
        self.hybrid_worker.moveToThread(self.hybrid_thread)
        self.hybrid_thread.started.connect(self.hybrid_worker.run)
        self.hybrid_worker.finished.connect(self._hybrid_complete)
        self.hybrid_worker.failed.connect(self._hybrid_failed)
        self.hybrid_worker.done.connect(self.hybrid_thread.quit)
        self.hybrid_worker.done.connect(self.hybrid_worker.deleteLater)
        self.hybrid_thread.finished.connect(self._hybrid_cleanup)
        self.status.appendPlainText("Анализ фотографии запущен...")
        self.hybrid_thread.start()

    @Slot(dict)
    def _hybrid_complete(self, result):
        self.status.appendPlainText("=== YOLO + VLM ===\\n" + json.dumps(result, ensure_ascii=False, indent=2))

    @Slot(str)
    def _hybrid_failed(self, error):
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / ("hybrid_error_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")
        details = error.strip() or "Ошибка без текста. Проверьте консоль Python."
        log_file.write_text(details, encoding="utf-8")
        self.status.appendPlainText("Ошибка YOLO + VLM:\\n" + details + "\\nЛог: " + str(log_file))
        QMessageBox.critical(self, "Ошибка YOLO + VLM", details[-2500:] + "\\n\\nПолный лог: " + str(log_file))

    @Slot()
    def _hybrid_cleanup(self):
        thread = self.hybrid_thread
        self.hybrid_thread = None
        self.hybrid_worker = None
        if thread is not None:
            thread.deleteLater()

    def import_cvat(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать экспорт CVAT YOLO Segmentation 1.1", str(BASE),
            "CVAT ZIP (*.zip);;Все файлы (*)")
        if not path:
            directory = QFileDialog.getExistingDirectory(
                self, "Или выбрать распакованную папку CVAT", str(BASE))
            if not directory:
                return
            path = directory
        from cvat_yolo_segmentation_import import audit
        try:
            destination = self.output_dir / "cvat_audit"
            result = audit(path, destination)
            self.status.appendPlainText("Проверка CVAT:\\n" + json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка CVAT", str(exc))

    def inspect_cvat(self):
        source, _ = QFileDialog.getOpenFileName(
            self, "Выберите экспорт CVAT YOLO Segmentation 1.1", str(BASE),
            "CVAT ZIP (*.zip)")
        if not source:
            source = QFileDialog.getExistingDirectory(
                self, "Или выберите распакованную папку CVAT")
        if not source:
            return
        try:
            from inspect_cvat_yolo import inspect
            result = inspect(source)
            self.status.appendPlainText("=== CVAT YOLO Segmentation ===\\n" +
                json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка проверки CVAT", str(exc))

    def export_knowledge(self):
        if not self._ready(): return
        from export_geology_vlm import export_text_knowledge
        try:
            result=export_text_knowledge(self.index_dir())
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка экспорта знаний",str(exc))

    def export(self):
        if not self._ready(): return
        from export_geology_vlm import export
        try:
            result=export(self.index_dir())
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка экспорта",str(exc))

if __name__=="__main__":
    app=QApplication(sys.argv)
    window=ArchiveManager()
    window.show()
    sys.exit(app.exec())
