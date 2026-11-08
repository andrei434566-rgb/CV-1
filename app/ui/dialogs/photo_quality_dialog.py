"""Inspect photographic warnings and the exact candidate label exclusions."""
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor,QPen
from PySide6.QtWidgets import QCheckBox,QDialog,QHBoxLayout,QLabel,QPushButton,QVBoxLayout
from app.ui.dialogs.feature_evidence_dialog import EvidenceView

REVIEW_STYLE='''
QDialog { background: #f7faff; }
QLabel,QCheckBox { color: #163660; }
QLineEdit,QComboBox,QPlainTextEdit,QTableWidget { background: white; color: #163660; border: 1px solid #ccdaed; border-radius: 4px; padding: 4px; }
QHeaderView::section { background: #eaf2fc; color: #163660; padding: 5px; border: 1px solid #d7e3f2; }
QPushButton { color: #164c85; background: white; border: 1px solid #c8daf0; border-radius: 5px; padding: 6px 10px; }
QPushButton:hover { background: #eaf4ff; }
'''


class PhotoQualityDialog(QDialog):
    def __init__(self,pixmap,report,parent=None):
        super().__init__(parent);self.resize(850,760);self.setWindowTitle('Качество фото и исключённые этикетки')
        self.setStyleSheet(REVIEW_STYLE)
        layout=QVBoxLayout(self)
        issues=[str(row.get('message','')) for row in report.get('issues',[]) if isinstance(row,dict)]
        note=QLabel('\n'.join(issues) or 'Фотографических предупреждений по эвристикам не найдено. Это не подтверждает видимость всех геологических признаков.')
        note.setWordWrap(True);layout.addWidget(note)
        self.view=EvidenceView(pixmap);self.view.selection_mode=False
        self.view.setDragMode(self.view.DragMode.ScrollHandDrag);layout.addWidget(self.view,1)
        self.boxes=[];pen=QPen(QColor('#e38a09'),2);pen.setCosmetic(True)
        for box in report.get('label_boxes',[]):
            l,t,r,b=box
            self.boxes.append(self.view.scene_data.addRect(QRectF(l*pixmap.width(),t*pixmap.height(),(r-l)*pixmap.width(),(b-t)*pixmap.height()),pen))
        info=QLabel(f'Предполагаемых этикеток: {len(self.boxes)}. Эти области исключены из оценки цвета и контрастных полос. Исходное фото и контуры сохранены. Распознавание этикеток эвристическое — проверьте рамки.')
        info.setWordWrap(True);layout.addWidget(info)
        controls=QHBoxLayout()
        toggle=QCheckBox('Показать области исключения');toggle.setChecked(True)
        toggle.toggled.connect(lambda active:[item.setVisible(active) for item in self.boxes]);controls.addWidget(toggle)
        fit=QPushButton('Показать целиком');fit.clicked.connect(self.view.fit_photo);controls.addWidget(fit)
        close=QPushButton('Закрыть');close.clicked.connect(self.close);controls.addWidget(close);layout.addLayout(controls)

    def showEvent(self,event):
        super().showEvent(event);self.view.fit_photo()
