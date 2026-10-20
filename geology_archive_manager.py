"""One-window entry point for local geological archive preparation."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from PySide6.QtWidgets import (QApplication,QFileDialog,QHBoxLayout,QLabel,
    QMainWindow,QMessageBox,QPlainTextEdit,QPushButton,QVBoxLayout,QWidget)

BASE=Path(__file__).resolve().parent

class ArchiveManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV-1 — Геологическая база и обучение ИИ")
        self.resize(850,500)
        self.root=None
        self.path_label=QLabel("Папка не выбрана")
        self.status=QPlainTextEdit()
        self.status.setReadOnly(True)
        body=QWidget()
        layout=QVBoxLayout(body)
        layout.addWidget(QLabel("Импорт скважин, фотографий, Excel и теоретических материалов"))
        layout.addWidget(self.path_label)
        row=QHBoxLayout()
        for title,fn in [
            ("Выбрать базу",self.choose),
            ("Сканировать",self.scan),
            ("Проверить фото ↔ Excel",self.review),
            ("Экспорт для VLM",self.export),
            ("Привязка глубин",self.depth_mapping),
        ]:
            button=QPushButton(title)
            button.clicked.connect(fn)
            row.addWidget(button)
        layout.addLayout(row)
        layout.addWidget(self.status)
        self.setCentralWidget(body)

    def choose(self):
        directory=QFileDialog.getExistingDirectory(self,"Выбрать корневую папку геологической базы")
        if directory:
            self.root=Path(directory)
            self.path_label.setText(str(self.root))
            self.status.appendPlainText("Выбрана база: "+str(self.root))

    def _ready(self):
        if self.root is None:
            QMessageBox.information(self,"Нет папки","Сначала выберите корневую папку.")
            return False
        return True

    def scan(self):
        if not self._ready(): return
        from index_geology_archive import scan
        try:
            result=scan(self.root,self.root/"_index_cv1")
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка индексации",str(exc))

    def review(self):
        if not self._ready(): return
        path=self.root/"_index_cv1"/"archive_index.json"
        if not path.exists():
            QMessageBox.information(self,"Нет индекса","Сначала нажмите «Сканировать».")
            return
        from review_geology_pairs import ReviewWindow
        self.review_window=ReviewWindow(path)
        self.review_window.show()

    def depth_mapping(self):
        if not self._ready(): return
        from geology_depth_mapping import generate
        try:
            result=generate(self.root/"_index_cv1")
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
            if result.get("status")=="template_created":
                QMessageBox.information(self,"Привязка глубин",
                    "Создан lane_annotations.json. Укажите координаты рядов и их интервалы; повторное нажатие построит карту глубин.")
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка привязки глубин",str(exc))

    def export(self):
        if not self._ready(): return
        from export_geology_vlm import export
        try:
            result=export(self.root/"_index_cv1")
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка экспорта",str(exc))

if __name__=="__main__":
    app=QApplication(sys.argv)
    window=ArchiveManager()
    window.show()
    sys.exit(app.exec())
