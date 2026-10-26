"""One-window entry point for local geological archive preparation."""
from __future__ import annotations
import json
import sys
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


class ArchiveManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV-1 — Геологическая база и обучение ИИ | диагностика v5")
        self.resize(850,500)
        self.root=None
        self.scan_thread=None
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
