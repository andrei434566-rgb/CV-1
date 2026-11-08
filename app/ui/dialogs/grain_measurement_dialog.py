"""Measure in original pixmap coordinates with explicit physical calibration."""
import math
from PySide6.QtCore import Qt,QLineF,QPointF,QRectF,Signal
from PySide6.QtGui import QColor,QPen
from PySide6.QtWidgets import (QCheckBox,QDialog,QDialogButtonBox,QDoubleSpinBox,QGraphicsScene,
    QGraphicsView,QHBoxLayout,QLabel,QMessageBox,QPushButton,QTableWidget,QTableWidgetItem,QVBoxLayout)
from app.infrastructure.grain_measurement import assess_measurements,line_length
from app.infrastructure.attribute_examples import pixel_digest


class GrainMeasureView(QGraphicsView):
    line_drawn=Signal(object)
    def __init__(self,pixmap,parent=None):
        super().__init__(parent);self.photo=pixmap;self.mode='scale';self.start=None;self.pending=None
        self.data=QGraphicsScene(self);self.setScene(self.data);self.data.addPixmap(pixmap);self.setSceneRect(QRectF(pixmap.rect()))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse);self.setMinimumSize(450,300)
        self.lines=[]
    def fit_photo(self):self.fitInView(self.sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)
    def wheelEvent(self,event):
        factor=1.2 if event.angleDelta().y()>0 else 1/1.2
        if .005<self.transform().m11()*factor<40:self.scale(factor,factor)
        event.accept()
    def _point(self,event):
        p=self.mapToScene(event.pos());return QPointF(max(0,min(self.photo.width(),p.x())),max(0,min(self.photo.height(),p.y())))
    def set_mode(self,mode):
        self.mode=mode;self.start=None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if mode=='pan' else QGraphicsView.DragMode.NoDrag)
    def mousePressEvent(self,event):
        if self.mode!='pan' and event.button()==Qt.MouseButton.LeftButton:
            self.start=self._point(event);pen=QPen(QColor('#007de0' if self.mode=='scale' else '#13a15f'),2);pen.setCosmetic(True)
            self.pending=self.data.addLine(QLineF(self.start,self.start),pen);self.pending.setZValue(3);event.accept();return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.start is not None and self.pending:
            self.pending.setLine(QLineF(self.start,self._point(event)));event.accept();return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if self.start is not None:
            end=self._point(event);coords=[self.start.x(),self.start.y(),end.x(),end.y()]
            self.data.removeItem(self.pending);self.pending=None;self.start=None
            if math.hypot(coords[2]-coords[0],coords[3]-coords[1])>0:self.line_drawn.emit(coords)
            event.accept();return
        super().mouseReleaseEvent(event)
    def draw_lines(self,reference,grains):
        for item in self.lines:self.data.removeItem(item)
        self.lines=[]
        for line,color in [(reference,'#007de0')]+[(g,'#13a15f') for g in grains]:
            if not line:continue
            pen=QPen(QColor(color),2);pen.setCosmetic(True)
            item=self.data.addLine(QLineF(*line),pen);item.setZValue(2);self.lines.append(item)


class GrainMeasurementDialog(QDialog):
    def __init__(self,pixmap,entry=None,parent=None):
        super().__init__(parent)
        from app.ui.dialog_theme import apply_light_card
        apply_light_card(self);self.setWindowTitle('Зернистость · измерение по масштабу');self.resize(1050,780)
        self.sha=pixel_digest(pixmap);self.size=[pixmap.width(),pixmap.height()];self.reference=None;self.grains=[];self.result_entry=None
        layout=QVBoxLayout(self)
        note=QLabel('1. Синий отрезок — известная длина линейки или ширина образца в мм. '
                    '2. Зелёные отрезки — одинаково выбранные поперечные размеры отдельных песчаных зёрен. '
                    'Колесо увеличивает фото, но не добавляет исходных пикселей.')
        note.setWordWrap(True);layout.addWidget(note)
        row=QHBoxLayout();layout.addLayout(row,1)
        self.view=GrainMeasureView(pixmap);row.addWidget(self.view,3)
        side=QVBoxLayout();row.addLayout(side,2)
        side.addWidget(QLabel('Длина синего отрезка, мм:'))
        self.reference_mm=QDoubleSpinBox();self.reference_mm.setRange(.001,100000);self.reference_mm.setDecimals(4);self.reference_mm.setValue(10.);side.addWidget(self.reference_mm)
        for text,mode in [('Задать масштаб','scale'),('Измерять зёрна','grain'),('Перемещать фото','pan')]:
            button=QPushButton(text);button.clicked.connect(lambda checked=False,m=mode:self.view.set_mode(m));side.addWidget(button)
        fit=QPushButton('Фото целиком');fit.clicked.connect(self.view.fit_photo);side.addWidget(fit)
        self.same_plane=QCheckBox('Масштаб и порода в одной плоскости')
        self.same_plane.setToolTip('Без заметной перспективы. Размер из другой плоскости, например далёкой линейки, не задаёт масштаб зёрен.')
        self.visible=QCheckBox('Границы отдельных зёрен видны')
        self.visible.setToolTip('Линии слоистости, трещины, поры и цемент не являются границами отдельных зёрен.')
        self.sand=QCheckBox('Измеряю песчаные обломки')
        for checkbox in (self.same_plane,self.visible,self.sand):side.addWidget(checkbox);checkbox.toggled.connect(self.update_measurement)
        self.table=QTableWidget(0,3);self.table.setHorizontalHeaderLabels(['Пиксели','Размер, мм','Разрешение']);self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        side.addWidget(self.table,1)
        delete=QPushButton('Убрать выбранное измерение');delete.clicked.connect(self.remove_measurement);side.addWidget(delete)
        self.status=QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
        limits=QLabel('Уэнтуорт: тонкозернистый 0,0625–0,125; мелкозернистый 0,125–0,25; '
                     'среднезернистый 0,25–0,5; крупнозернистый 0,5–1 мм. Нижняя граница включена, верхняя исключена. '
                     'Это оценка видимых двумерных сечений, а не лабораторный D50.')
        limits.setWordWrap(True);layout.addWidget(limits)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Save|QDialogButtonBox.StandardButton.Cancel)
        self.save=buttons.button(QDialogButtonBox.StandardButton.Save);self.save.setText('Сохранить измерения')
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText('Отмена')
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.view.line_drawn.connect(self.add_line);self.reference_mm.valueChanged.connect(self.update_measurement)
        if isinstance(entry,dict) and entry.get('image_sha256')==self.sha and entry.get('image_size')==self.size:
            try:
                reference=entry['reference_line'];line_length(reference,self.size)
                self.reference=reference;self.reference_mm.setValue(float(entry['reference_mm']))
                self.grains=[r['line'] for r in entry.get('grain_lines',[]) if line_length(r['line'],self.size)>0][:500]
                self.same_plane.setChecked(entry.get('same_plane_confirmed') is True)
                self.visible.setChecked(entry.get('grains_visible_confirmed') is True)
                self.sand.setChecked(entry.get('sand_context_confirmed') is True)
            except (ValueError,TypeError,KeyError):self.reference=None;self.grains=[]
        self.update_measurement()
    def showEvent(self,event):super().showEvent(event);self.view.fit_photo()
    def add_line(self,line):
        if self.view.mode=='scale':self.reference=line
        elif len(self.grains)<500:self.grains.append(line)
        self.update_measurement()
    def remove_measurement(self):
        i=self.table.currentRow()
        if 0<=i<len(self.grains):self.grains.pop(i);self.update_measurement()
    def update_measurement(self,*_):
        self.view.draw_lines(self.reference,self.grains);self.measurement=None;self.table.setRowCount(0)
        try:
            self.measurement=assess_measurements(self.reference,self.reference_mm.value(),self.grains,self.size,
                same_plane=self.same_plane.isChecked(),grains_visible=self.visible.isChecked(),
                sand_context=self.sand.isChecked(),image_sha256=self.sha)
            m=self.measurement
            for r in m['grain_lines']:
                i=self.table.rowCount();self.table.insertRow(i)
                for c,value in enumerate((f"{r['pixels']:.1f}",f"{r['apparent_mm']:.4f}",'повтор' if r['duplicate'] else 'достаточно' if r['resolved'] else 'мало пикселей')):
                    self.table.setItem(i,c,QTableWidgetItem(value))
            text=f"Масштаб {m['mm_per_pixel']:.5g} мм/пикс.; минимальный принимаемый размер {m['minimum_resolved_grain_mm']:.4g} мм. "
            if m.get('median_apparent_mm') is not None:text+=f"Медиана видимых размеров {m['median_apparent_mm']:.4g} мм. "
            self.status.setText(text+(m['label']+'. ' if m['label'] else '')+m['reason'])
            self.save.setEnabled(True);self.save.setText('Перенести предположение в карточку' if m['label'] else 'Сохранить измерения без назначения зернистости')
        except (ValueError,TypeError) as e:self.status.setText(str(e));self.save.setEnabled(False)
    def accept(self):
        self.update_measurement()
        if self.measurement is None:return
        self.result_entry=self.measurement;super().accept()
