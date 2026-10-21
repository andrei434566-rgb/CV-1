"""One-window entry point for local geological archive preparation."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from PySide6.QtWidgets import (QApplication,QFileDialog,QHBoxLayout,QLabel,
    QMainWindow,QMessageBox,QPlainTextEdit,QPushButton,QVBoxLayout,QWidget)

BASE=Path(__file__).resolve().parent

class ArchiveManager(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV-1 — Геологическая база и обучение ИИ | диагностика v4")
        self.resize(850,500)
        self.root=None
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

    def scan(self):
        if not self._ready(): return
        from index_geology_archive import scan
        try:
            result=scan(self.root,self.index_dir())
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка индексации",str(exc))

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
        if not self._ready(): return
        path=self.index_dir()/"archive_index.json"
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
            result=generate(self.index_dir())
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
            result=export(self.index_dir())
            self.status.appendPlainText(json.dumps(result,ensure_ascii=False,indent=2))
        except Exception as exc:
            QMessageBox.critical(self,"Ошибка экспорта",str(exc))

if __name__=="__main__":
    app=QApplication(sys.argv)
    window=ArchiveManager()
    window.show()
    sys.exit(app.exec())
