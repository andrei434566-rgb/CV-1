"""Read-only project summary; incomplete data stay exportable."""
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QAbstractItemView,QComboBox,QDialog,QHeaderView,QHBoxLayout,
    QLabel,QLineEdit,QPlainTextEdit,QPushButton,QTableWidget,QTableWidgetItem,QTabWidget,QVBoxLayout,QWidget)
from app.infrastructure.description_template import REVIEW_FIELDS
from app.ui.dialogs.photo_quality_dialog import REVIEW_STYLE


class ProjectReadinessDialog(QDialog):
    def __init__(self,provider,open_target,export_report,parent=None):
        super().__init__(parent)
        self.provider,self.open_target,self.export_report=provider,open_target,export_report
        self.setWindowTitle('Сводка проекта и проверка разреза');self.resize(1120,800);self.setStyleSheet(REVIEW_STYLE)
        self._audit=None;self._targets={};self._visible=[]
        self._selected_issue_key=None
        layout=QVBoxLayout(self)
        note=QLabel('Счётчики показывают заполнение строк отчёта и подтверждение описаний. Разбиение по долблениям учитывается. Черновой отчёт можно выгрузить при наличии замечаний.')
        note.setWordWrap(True);layout.addWidget(note)
        self.overall=QLabel();self.overall.setWordWrap(True);self.overall.setStyleSheet('font-weight:600; padding:8px; background:#eaf2fc;');layout.addWidget(self.overall)
        self.dirty=QLabel();self.dirty.setWordWrap(True);self.dirty.setStyleSheet('color:#985516;');layout.addWidget(self.dirty)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        summary=QWidget();sl=QVBoxLayout(summary);self.tabs.addTab(summary,'Заполнение')
        self.wells=QTableWidget(0,7)
        self.wells.setHorizontalHeaderLabels(['Скважина','Строки','С глубинами','Есть текст','Подтв. текст','Фация: всего / предп.','Длина интервалов'])
        self.wells.setColumnWidth(5,165)
        self.wells.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.wells.horizontalHeader().setSectionResizeMode(6,QHeaderView.ResizeMode.Stretch)
        self.wells.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);sl.addWidget(self.wells,1)
        self.fields=QTableWidget(0,4);self.fields.setHorizontalHeaderLabels(['Основное поле','Заполнено','Предположительно','Пусто'])
        self.fields.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.fields.setColumnWidth(1,130);self.fields.setColumnWidth(2,160)
        self.fields.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);sl.addWidget(self.fields,1)
        field_note=QLabel('Длина интервалов считается без двойного учёта перекрытий. Это длина привязанных данных, а не вынос керна. Предположение считается заполненным значением и показано отдельно.')
        field_note.setWordWrap(True);sl.addWidget(field_note)
        issues=QWidget();il=QVBoxLayout(issues);self.tabs.addTab(issues,'Замечания')
        controls=QHBoxLayout()
        self.well_filter=QComboBox();self.well_filter.addItem('Все скважины',-1)
        self.query=QLineEdit();self.query.setPlaceholderText('Поиск по замечаниям…')
        self.well_filter.currentIndexChanged.connect(self._render_issues);self.query.textChanged.connect(self._render_issues)
        controls.addWidget(self.well_filter);controls.addWidget(self.query,1);il.addLayout(controls)
        self.issue_count=QLabel();il.addWidget(self.issue_count)
        self.issue_table=QTableWidget(0,4);self.issue_table.setHorizontalHeaderLabels(['Приоритет','Скважина','Строки Excel','Замечание'])
        self.issue_table.horizontalHeader().setSectionResizeMode(3,QHeaderView.ResizeMode.Stretch)
        self.issue_table.setColumnWidth(0,110);self.issue_table.setColumnWidth(1,150)
        self.issue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.issue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.issue_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection);self.issue_table.setWordWrap(False)
        self.issue_table.itemSelectionChanged.connect(self._selected)
        self.issue_table.cellDoubleClicked.connect(lambda *_:self.open_selected());il.addWidget(self.issue_table,1)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);self.details.setMaximumHeight(130);il.addWidget(self.details)
        self.target_selector=QComboBox();il.addWidget(self.target_selector)
        actions=QHBoxLayout()
        for title,handler in [('Открыть участок',self.open_selected),('Обновить',self.refresh),('Выгрузить отчёт…',export_report),('Закрыть',self.close)]:
            button=QPushButton(title);button.clicked.connect(handler);actions.addWidget(button)
        layout.addLayout(actions)
        self.timer=QTimer(self);self.timer.setInterval(5000);self.timer.timeout.connect(self.refresh)
        self.refresh()

    def refresh(self):
        self._audit,self._targets,dirty=self.provider()
        totals=self._audit.totals
        self.overall.setText(f"Скважин: {totals['wells']} · строк отчёта: {totals['rows']} · глубины: {totals['valid_depths']} · есть текст: {totals['with_text']} · подтверждён текст: {totals['confirmed_descriptions']}\n"
            f"Основные поля: {totals['filled_fields']} / {totals['field_slots']} (предположений: {totals['assumed_fields']}) · пропусков между данными: {totals['gaps']} · перекрытий: {totals['overlaps']}")
        self.dirty.setText(f'В открытых карточках есть несохранённые правки: {dirty}. Сохраните их, чтобы новые значения попали в отчёт.' if dirty else '')
        self.wells.setRowCount(len(self._audit.wells))
        for i,well in enumerate(self._audit.wells):
            values=[well['name'],well['rows'],well['valid_depths'],well['with_text'],well['confirmed_descriptions'],
                    f"{well['named_facies']} / {well['assumed_facies']}",f"{well['mapped_length']:g} {well['unit']} · {well['depth_reference']}"]
            for j,value in enumerate(values):self.wells.setItem(i,j,QTableWidgetItem(str(value)))
        self.fields.setRowCount(len(REVIEW_FIELDS))
        for i,field in enumerate(REVIEW_FIELDS):
            filled=sum(well['fields'][field]['filled'] for well in self._audit.wells)
            assumed=sum(well['fields'][field]['assumed'] for well in self._audit.wells)
            for j,value in enumerate([field,f'{filled} / {totals["rows"]}',assumed,totals['rows']-filled]):self.fields.setItem(i,j,QTableWidgetItem(str(value)))
        previous=self.well_filter.currentData();self.well_filter.blockSignals(True)
        self.well_filter.clear();self.well_filter.addItem('Все скважины',-1)
        for i,well in enumerate(self._audit.wells):self.well_filter.addItem(well['name'],i)
        index=self.well_filter.findData(previous);self.well_filter.setCurrentIndex(max(0,index));self.well_filter.blockSignals(False)
        self.tabs.setTabText(1,f'Замечания ({len(self._audit.issues)})');self._render_issues()

    def current(self):
        index=self.issue_table.currentRow()
        return self._visible[index] if 0<=index<len(self._visible) else None

    def _render_issues(self,*_args):
        if self._audit is None:return
        previous=self.current();key=previous.key if previous else None
        query=self.query.text().strip().casefold();well=self.well_filter.currentData()
        matching=[issue for issue in self._audit.issues if (well in (-1,None) or issue.well_index==well)
                  and (not query or query in (issue.well+' '+issue.message).casefold())]
        self._visible=matching[:2000]
        self.issue_count.setText(f'Показано замечаний: {len(self._visible)} / {len(matching)}. Номера строк соответствуют шаблону Excel.')
        self.issue_table.blockSignals(True);self.issue_table.setRowCount(len(self._visible))
        for i,issue in enumerate(self._visible):
            values=[{1:'Уточнить',2:'Проверить',3:'Исправить'}[issue.severity],issue.well,
                    ', '.join(str(row+4) for row in issue.rows) or '—',issue.message]
            for j,value in enumerate(values):
                cell=QTableWidgetItem(value);cell.setToolTip(value);self.issue_table.setItem(i,j,cell)
        self.issue_table.blockSignals(False)
        if self._visible:self.issue_table.selectRow(next((i for i,row in enumerate(self._visible) if row.key==key),0))
        else:self.details.clear()
        self._selected()

    def _selected(self):
        issue=self.current()
        previous=self.target_selector.currentData() if issue and self._selected_issue_key==issue.key else None
        self._selected_issue_key=issue.key if issue else None
        self.target_selector.clear()
        if issue:
            self.details.setPlainText(issue.message)
            sources=list(issue.source_ids)
            if issue.code=='description_unconfirmed':
                sources.sort(key=lambda source:bool(getattr(self._targets.get(source),'description_confirmed',False)))
            for source in sources:
                target=self._targets.get(source)
                if target:
                    detection=target.detection
                    depths=f'{detection.depth_from:g}–{detection.depth_to:g}' if detection and detection.depth_from is not None and detection.depth_to is not None else 'без привязки'
                    self.target_selector.addItem(f'{Path(target.record.path).name} · {depths}',source)
            index=self.target_selector.findData(previous)
            if index>=0:self.target_selector.setCurrentIndex(index)

    def open_selected(self):
        issue=self.current()
        if issue is None:return
        live,live_targets,_=self.provider()
        if not any(row.key==issue.key for row in live.issues):
            self.refresh();self.dirty.setText('Данные изменились. Сводка обновлена.');return
        selected=self.target_selector.currentData()
        for source in ([selected] if selected else issue.source_ids):
            target=self._targets.get(source)
            current=live_targets.get(source)
            if target and current and target.record is current.record and target.detection is current.detection:
                self.open_target(target);return
        self.dirty.setText('Для этого замечания нет отдельного участка. Проверьте параметры скважины или загрузку фото.')

    def showEvent(self,event):
        super().showEvent(event);self.timer.start();self.refresh()

    def closeEvent(self,event):
        self.timer.stop();super().closeEvent(event)
