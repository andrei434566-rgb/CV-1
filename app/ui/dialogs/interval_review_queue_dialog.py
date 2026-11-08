"""One navigable queue for photographic, geometric and descriptive review."""
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView,QComboBox,QDialog,QHeaderView,QHBoxLayout,
    QLabel,QLineEdit,QPlainTextEdit,QPushButton,QTableWidget,QTableWidgetItem,QVBoxLayout)
from app.infrastructure.interval_review_queue import acknowledge,build_interval_review_queue
from app.ui.dialogs.photo_quality_dialog import REVIEW_STYLE


class IntervalReviewQueueDialog(QDialog):
    def __init__(self, records_provider, open_item, changed, audit_photos, parent=None,*,readiness=None):
        super().__init__(parent)
        self.setStyleSheet(REVIEW_STYLE)
        self.setWindowTitle('Очередь проверки керна')
        self.resize(1120,750)
        self.provider,self.open_callback,self.changed,self.audit_photos=records_provider,open_item,changed,audit_photos
        self._visible=[];self._deferred=set();self._last_open=None
        layout=QVBoxLayout(self)
        note=QLabel('Сначала показаны наиболее важные замечания. «Замечания рассмотрены» относится только к текущему состоянию и не подтверждает фацию или обучение.')
        note.setWordWrap(True);layout.addWidget(note)
        controls=QHBoxLayout()
        self.filter=QLineEdit();self.filter.setPlaceholderText('Скважина, файл или причина проверки…')
        self.mode=QComboBox();self.mode.addItems(['Ожидают проверки','Отложенные','Рассмотренные','Все'])
        self.filter.textChanged.connect(self.refresh);self.mode.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.filter,1);controls.addWidget(self.mode)
        if readiness:
            summary=QPushButton('Сводка проекта…');summary.clicked.connect(readiness);controls.addWidget(summary)
        audit=QPushButton('Проверить качество фото');audit.clicked.connect(lambda:self.audit_photos(force=True));controls.addWidget(audit)
        layout.addLayout(controls)
        self.status=QLabel();layout.addWidget(self.status)
        self.table=QTableWidget(0,5)
        self.table.setHorizontalHeaderLabels(['Приоритет','Скважина','Фото','Интервал','Причины проверки'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setSectionResizeMode(4,QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(0,100);self.table.setColumnWidth(1,120);self.table.setColumnWidth(2,200);self.table.setColumnWidth(3,140)
        self.table.itemSelectionChanged.connect(self._selected)
        self.table.cellDoubleClicked.connect(lambda *_:self.open_selected())
        layout.addWidget(self.table,1)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(160);layout.addWidget(self.details)
        actions=QHBoxLayout()
        for title,handler in [('Открыть',self.open_selected),('Следующий',self.open_next),
                              ('Замечания рассмотрены',self.mark_reviewed),('Отложить',self.defer)]:
            button=QPushButton(title);button.clicked.connect(handler);actions.addWidget(button)
        close=QPushButton('Закрыть');close.clicked.connect(self.close);actions.addWidget(close);layout.addLayout(actions)
        self.timer=QTimer(self);self.timer.setInterval(5000);self.timer.timeout.connect(self.refresh);self.timer.start()
        self.refresh()

    def current(self):
        index=self.table.currentRow()
        return self._visible[index] if 0<=index<len(self._visible) else None

    def refresh(self,*_args):
        previous=self.current();previous_key=previous.key if previous else None
        previous_index=max(0,self.table.currentRow())
        items=build_interval_review_queue(self.provider())
        query=self.filter.text().strip().casefold();mode=self.mode.currentIndex();visible=[]
        for item in items:
            deferred=(item.key,item.fingerprint) in self._deferred
            if mode==0 and (item.acknowledged or deferred):continue
            if mode==1 and not deferred:continue
            if mode==2 and not item.acknowledged:continue
            text=' '.join([item.record.well_name,Path(item.record.path).name,*[r['message'] for r in item.reasons]])
            if query and query not in text.casefold():continue
            visible.append(item)
        self._visible=visible[:2000]
        self.table.blockSignals(True);self.table.setRowCount(len(self._visible))
        for row,item in enumerate(self._visible):
            reasons='; '.join(r['message'] for r in item.reasons)
            values=[{1:'Уточнить',2:'Проверить',3:'В первую очередь'}[item.severity],item.record.well_name,
                    Path(item.record.path).name,item.interval,reasons]
            for col,value in enumerate(values):
                cell=QTableWidgetItem(value);cell.setToolTip(value);self.table.setItem(row,col,cell)
        self.table.blockSignals(False)
        selected=next((i for i,item in enumerate(self._visible) if item.key==previous_key),min(previous_index,max(0,len(self._visible)-1)))
        if self._visible:self.table.selectRow(selected)
        else:self.details.clear()
        pending=sum(not item.acknowledged for item in items)
        self.status.setText(f'Замечаний по объектам: {len(items)} · не рассмотрено: {pending} · показано: {len(self._visible)}. Отложение действует до закрытия окна.')
        self._selected()

    def _selected(self):
        item=self.current()
        if item:self.details.setPlainText('\n\n'.join(r['message'] for r in item.reasons))

    def open_selected(self):
        item=self.current()
        if item and self.open_callback(item) is not False:self._last_open=item.key

    def open_next(self):
        if not self._visible:return
        index=next((i+1 for i,item in enumerate(self._visible) if item.key==self._last_open),max(0,self.table.currentRow()))
        self.table.selectRow(index%len(self._visible));self.open_selected()

    def mark_reviewed(self):
        item=self.current()
        if item:
            # Revalidate references and current state, including changes while
            # the model or editor was working. Never acknowledge a stale row.
            current=next((row for row in build_interval_review_queue(self.provider()) if row.key==item.key),None)
            if current and current.fingerprint==item.fingerprint:
                acknowledge(current);self.changed()
        self.refresh()

    def defer(self):
        item=self.current()
        if item:self._deferred.add((item.key,item.fingerprint))
        self.refresh()

    def closeEvent(self,event):
        self.timer.stop();self._deferred.clear();super().closeEvent(event)

    def showEvent(self,event):
        self.timer.start();self.refresh();super().showEvent(event)
