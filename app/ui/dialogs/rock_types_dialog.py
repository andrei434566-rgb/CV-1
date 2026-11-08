"""Readable evidence and current model coverage for the user's ten rock classes."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidget, QListView, QPlainTextEdit, QPushButton, QVBoxLayout
from app.domain.rock_types import ROCK_PROFILES, SOURCES, training_status, rock_profile


class RockTypesDialog(QDialog):
    def __init__(self,parent=None,selected='',card=None):
        super().__init__(parent)
        from app.ui.dialog_theme import apply_light_card
        apply_light_card(self);self.setWindowTitle('Признаки пород · что распознаёт модель');self.resize(860,590)
        self.card=card or {};self.show_sources=False
        layout=QVBoxLayout(self)
        intro=QLabel('10 типов из вашего справочника. Характерные признаки помогают проверке; текст теории не заменяет размеченные фотографии.')
        intro.setWordWrap(True);layout.addWidget(intro)
        row=QHBoxLayout();layout.addLayout(row,1)
        self.classes=QListWidget();self.classes.setMinimumWidth(250)
        self.classes.setWordWrap(True);self.classes.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.classes.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.classes.setResizeMode(QListView.ResizeMode.Adjust)
        for p in ROCK_PROFILES:self.classes.addItem(p['name'])
        row.addWidget(self.classes,2)
        self.details=QPlainTextEdit();self.details.setReadOnly(True);row.addWidget(self.details,3)
        source=QPushButton('Показать источники');source.setCheckable(True)
        source.toggled.connect(self._sources);layout.addWidget(source)
        close=QPushButton('Закрыть');close.clicked.connect(self.accept);layout.addWidget(close)
        self.classes.currentRowChanged.connect(self._update)
        selected=(rock_profile(selected) or {}).get('name',selected)
        self.classes.setCurrentRow(next((i for i,p in enumerate(ROCK_PROFILES) if p['name']==selected),0))
    def _sources(self,checked):
        self.show_sources=checked;self._update(self.classes.currentRow())
    def _update(self,index):
        if not 0<=index<len(ROCK_PROFILES):return
        p=ROCK_PROFILES[index]
        parts=[p['name'],training_status(p,self.card), 'Видимые признаки\n'+p['visible'],
               'Что проверить\n'+p['check'],'Учебные данные\n'+p['training']]
        if self.card.get('noncommercial_only'):parts.append('Подключённая модель: только некоммерческие исследования (CC BY-NC 4.0 в учебных данных).')
        if self.show_sources:parts.append('Источники\n'+'\n\n'.join(title+'\n'+url for title,url in (SOURCES[k] for k in p['sources'])))
        self.details.setPlainText('\n\n'.join(parts))
