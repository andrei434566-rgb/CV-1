"""Zoom, pan and human-selected feature evidence inside a masked core crop."""
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout)

from app.infrastructure.attribute_examples import pixel_digest
from app.infrastructure.supervised_learning import REVIEW_BASIS


EVIDENCE_GUIDANCE = {
    'Контакт': (
        'Выделите одну границу и породу по обе её стороны. Исключите этикетки, линейку и трещины вне контакта.',
        'Обе стороны контакта видны. Тип подтверждён описанием специалиста.',
    ),
    'Степень цементации': (
        'Выделите породу, к которой относится подтверждённая степень цементации.',
        'Метка подтверждена описанием образца специалистом или испытанием.',
    ),
}


class EvidenceView(QGraphicsView):
    selection_changed = Signal()
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.photo = pixmap
        self.scene_data = QGraphicsScene(self)
        self.setScene(self.scene_data)
        self.scene_data.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.rectangle = self.scene_data.addRect(QRectF(), QPen(QColor("#008be8"), 2), QColor(0,139,232,35))
        self.rectangle.setZValue(2)
        self.start = None
        self.selection_mode = True
        self.setMinimumSize(420,320)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def fit_photo(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        if .01 < self.transform().m11()*factor < 30:
            self.scale(factor,factor)
        event.accept()

    def mousePressEvent(self,event):
        if self.selection_mode and event.button()==Qt.MouseButton.LeftButton:
            self.start=self.mapToScene(event.pos()); self.rectangle.setRect(QRectF())
            self.selection_changed.emit(); event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        if self.start is not None:
            rect=QRectF(self.start,self.mapToScene(event.pos())).normalized().intersected(self.sceneRect())
            self.rectangle.setRect(rect); self.selection_changed.emit(); event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if self.start is not None:
            self.mouseMoveEvent(event); self.start=None; event.accept(); return
        super().mouseReleaseEvent(event)

    def normalized_roi(self):
        rect=self.rectangle.rect()
        if min(rect.width(),rect.height())<16:
            return None
        return [rect.left()/self.photo.width(),rect.top()/self.photo.height(),
                rect.right()/self.photo.width(),rect.bottom()/self.photo.height()]


class FeatureEvidenceDialog(QDialog):
    def __init__(self, pixmap, field, value, entry=None, *, assumed=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Фрагмент: " + field)
        self.resize(780,650)
        self.field,self.value=field,value
        layout=QVBoxLayout(self)
        guidance, confirmation = EVIDENCE_GUIDANCE.get(field, (
            'Выделите прямоугольником породу с этим признаком, исключая этикетки и линейки.',
            'Фрагмент и признак проверены — использовать как учебный пример',
        ))
        note=QLabel(f"{field}: {value or 'значение ещё не выбрано'}. {guidance} Колесо — увеличение.")
        note.setWordWrap(True); layout.addWidget(note)
        self.view=EvidenceView(pixmap); layout.addWidget(self.view,1)
        if entry:
            l,t,r,b=entry['roi']; self.view.rectangle.setRect(QRectF(l*pixmap.width(),t*pixmap.height(),(r-l)*pixmap.width(),(b-t)*pixmap.height()))
        controls=QHBoxLayout()
        selection=QCheckBox("Выделять фрагмент"); selection.setChecked(True)
        def change_mode(active):
            self.view.selection_mode=active
            self.view.setDragMode(QGraphicsView.DragMode.NoDrag if active else QGraphicsView.DragMode.ScrollHandDrag)
        selection.toggled.connect(change_mode); controls.addWidget(selection)
        fit=QPushButton("Показать целиком"); fit.clicked.connect(self.view.fit_photo);controls.addWidget(fit)
        reset=QPushButton("Убрать выделение");reset.clicked.connect(lambda:self.view.rectangle.setRect(QRectF()));controls.addWidget(reset)
        layout.addLayout(controls)
        self.reviewed=QCheckBox(confirmation)
        self.reviewed.setChecked(bool(entry and entry.get('reviewed') is True and not assumed))
        self.reviewed.setEnabled(bool(value) and not assumed)
        self.reviewed.setToolTip("Предположительное значение не становится подтверждённой учебной меткой.")
        self.view.selection_changed.connect(lambda:self.reviewed.setChecked(False))
        reset.clicked.connect(lambda:self.reviewed.setChecked(False))
        layout.addWidget(self.reviewed)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить фрагмент")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.result_entry=None

    def showEvent(self,event):
        super().showEvent(event);self.view.fit_photo()

    def accept(self):
        roi=self.view.normalized_roi()
        if roi is None:
            if self.view.rectangle.rect().isEmpty():
                self.result_entry=None;super().accept();return
            QMessageBox.warning(self,"Маленький фрагмент","Выделите область размером хотя бы 16×16 пикселей исходного фрагмента.");return
        if not self.value:
            QMessageBox.warning(self,"Не выбран признак","Сначала выберите значение признака в карточке.");return
        self.result_entry={"roi":roi,"value":self.value,"image_sha256":pixel_digest(self.view.photo),
                           "reviewed":self.reviewed.isChecked(),"source":"expert_selected_roi"}
        if self.reviewed.isChecked() and self.field in REVIEW_BASIS:
            self.result_entry['review_basis'] = REVIEW_BASIS[self.field]
        super().accept()
