"""Non-generative classifier and local reference settings (legacy class name)."""
from pathlib import Path
import json
import sys
from PySide6.QtWidgets import (
    QDialog,QDialogButtonBox,QFileDialog,QFormLayout,QHBoxLayout,QLabel,QLineEdit,
    QListWidget,QMessageBox,QPushButton,QVBoxLayout,QWidget,QComboBox,
)
from app.infrastructure.local_vision import read_model_card

class LocalVlmSettingsDialog(QDialog):
    def __init__(self,config,parent=None):
        super().__init__(parent)
        from app.ui.dialog_theme import apply_light_card
        apply_light_card(self)
        self.setWindowTitle('Распознавание пород и локальный справочник');self.resize(720,520)
        layout=QVBoxLayout(self)
        note=QLabel('CV-1 Vision: локальный классификатор пород и поиск по документам. '
                    'Генеративные модели и обращения к серверу отключены.')
        note.setWordWrap(True);layout.addWidget(note)
        form=QFormLayout();layout.addLayout(form)
        self.python=QLineEdit(str(config.get('python','')))
        self.python.hide()
        if not getattr(sys, 'frozen', False):
            self.python=self._path_row(form,'Python:',str(config.get('python','')),file=True)
        self.base=self._path_row(form,'Папка классификатора:',str(config.get('base','')))
        self.model_choices=QComboBox();self.model_choices.addItem('Выбрать установленный вариант…','')
        from app.runtime_paths import user_data_root,bundled_root
        seen=set()
        for root in dict.fromkeys([user_data_root(),bundled_root()]):
            parent=root/'models/vision'
            try:choices=json.loads((parent/'candidates.json').read_text(encoding='utf8'))['models']
            except (OSError,ValueError,KeyError):continue
            for row in choices:
                path=(parent/str(row.get('folder',''))).resolve()
                if not path.is_relative_to(parent.resolve()) or str(path) in seen:continue
                try:read_model_card(path)
                except (OSError,ValueError):continue
                seen.add(str(path));self.model_choices.addItem(str(row.get('name') or path.name),str(path))
        if self.model_choices.count()>1:
            current=self.model_choices.findData(str(Path(str(config.get('base') or '.')).resolve()))
            if current>=0:self.model_choices.setCurrentIndex(current)
            form.addRow('Вариант распознавания:',self.model_choices)
            self.model_choices.activated.connect(lambda i:self.base.setText(self.model_choices.itemData(i)) if i else None)
        self.scope=QLabel()
        self.scope.setWordWrap(True);layout.addWidget(self.scope)
        self.base.textChanged.connect(self._update_scope);self._update_scope()
        layout.addWidget(QLabel('Папки с теорией и описаниями для локального поиска:'))
        self.folders=QListWidget()
        for folder in config.get('folders') or []:self.folders.addItem(str(folder))
        layout.addWidget(self.folders,1)
        row=QHBoxLayout();add=QPushButton('Добавить папку…');add.clicked.connect(self._add_folder);row.addWidget(add)
        remove=QPushButton('Убрать выбранную');remove.clicked.connect(lambda:self.folders.takeItem(self.folders.currentRow()));row.addWidget(remove)
        row.addStretch();layout.addLayout(row)
        actions=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        actions.button(QDialogButtonBox.Save).setText('Сохранить');actions.button(QDialogButtonBox.Cancel).setText('Отмена')
        actions.accepted.connect(self.accept);actions.rejected.connect(self.reject);layout.addWidget(actions)
    def _update_scope(self):
        from app.domain.rock_types import NATIVE_NAMES
        try:
            card=read_model_card(self.base.text().strip())
            names=[NATIVE_NAMES.get(c,{'limestone':'Известняк','shale':'Глинистая порода','garbage':'Фон / не керн'}.get(c,c)) for c in card['classes']]
            text='Обученные группы: '+', '.join(names)+'.'
            if card.get('experimental'):text+=' Исследовательский вариант; сравните с рабочим на ваших скважинах.'
            if card.get('noncommercial_only'):text+=' Только некоммерческие исследования.'
        except (OSError,ValueError):text='Выберите папку классификатора, чтобы увидеть его классы.'
        self.scope.setText(text+' Фации, зернистость и биотурбация требуют отдельной подтверждённой разметки.')

    def _path_row(self,form,label,value,file=False):
        container=QWidget();row=QHBoxLayout(container);row.setContentsMargins(0,0,0,0)
        edit=QLineEdit(value);row.addWidget(edit,1);button=QPushButton('Выбрать…')
        button.clicked.connect(lambda:self._browse(edit,file));row.addWidget(button);form.addRow(label,container);return edit
    def _browse(self,edit,file):
        value=(QFileDialog.getOpenFileName(self,'Выбрать Python',edit.text(),'Python (python.exe);;Все файлы (*)')[0]
               if file else QFileDialog.getExistingDirectory(self,'Папка модели с config.json и model.pt',edit.text()))
        if value:edit.setText(value)
    def _add_folder(self):
        value=QFileDialog.getExistingDirectory(self,'Папка с проверенными материалами')
        if value and all(self.folders.item(i).text()!=value for i in range(self.folders.count())):self.folders.addItem(value)
    def accept(self):
        try:
            if not Path(self.python.text().strip()).is_file():raise ValueError('Выберите существующий Python.')
            folder=Path(self.base.text().strip())
            if (folder/'config.json').is_file():read_model_card(folder)
            elif self.base.text().strip() and folder.exists():raise ValueError('В папке нет модели CV-1 Vision.')
        except (ValueError,OSError) as exc:
            QMessageBox.warning(self,'Настройки распознавания',str(exc));return
        super().accept()
    def values(self):
        folder=self.base.text().strip()
        return {'python':self.python.text().strip(),'base':folder if (Path(folder)/'config.json').is_file() else '',
                'model_mode':'vision','engine':'vision',
                'folders':[self.folders.item(i).text() for i in range(self.folders.count())]}
