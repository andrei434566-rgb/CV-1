"""Illustrative source-labelled photo thumbnails next to actual CNN scores."""
from functools import lru_cache
import math, json
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout
from app.runtime_paths import bundled_root
from app.infrastructure.vision_models import LABELS, LABELS_RU
from app.domain.rock_types import NATIVE_NAMES
ROCK_NAMES = {
    code: name for code, name in (dict(zip(LABELS, LABELS_RU)) | NATIVE_NAMES).items()
    if code not in {"garbage", "background"}
}


@lru_cache(maxsize=16)
def reference_thumbnail(code):
    if code not in ROCK_NAMES:return QPixmap()
    return QPixmap(str(bundled_root()/'app/resources/lithology_examples'/f'{code}.jpg'))


@lru_cache(maxsize=16)
def reference_credit(code):
    try:
        data=json.loads((bundled_root()/'app/resources/lithology_examples/ATTRIBUTION.json').read_text(encoding='utf8'))
        entry=data['examples'][code]
        return entry.get('credit') or data['author']+' · '+data['license']
    except (ValueError,OSError,KeyError):return 'Источник примера не указан'


class LithologyPredictionGallery(QFrame):
    def __init__(self,predictions,parent=None):
        super().__init__(parent)
        self.setObjectName('lithologyPredictionGallery')
        self.setStyleSheet('QFrame#lithologyPredictionGallery {background:#ffffff;border:0;} '
                          'QFrame#lithologyPredictionCard {background:#f4f7fc;border:1px solid #dce5f0;border-radius:6px;} '
                          'QLabel {background:transparent;color:#163660;} QLabel#lithologyPredictionScore {font-size:16px;font-weight:700;}')
        layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        grid=QGridLayout();grid.setContentsMargins(0,0,0,0);grid.setSpacing(5);layout.addLayout(grid)
        rows=[];seen=set()
        for row in predictions if isinstance(predictions,list) else []:
            if not isinstance(row,dict):continue
            code=row.get('class');score=row.get('score')
            if code not in ROCK_NAMES or code in seen or isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:continue
            seen.add(code);rows.append((code,float(score)))
        rows.sort(key=lambda row:-row[1])
        total=len(rows);rows=rows[:4]
        for i,(code,score) in enumerate(rows):
            card=QFrame(self);card.setObjectName('lithologyPredictionCard');card.setProperty('classCode',code)
            row=QHBoxLayout(card);row.setContentsMargins(5,5,5,5);row.setSpacing(7)
            image=QLabel(card);image.setObjectName('lithologyPredictionImage');image.setFixedSize(56,56)
            image.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap=reference_thumbnail(code)
            if pixmap.isNull():image.setText('Нет\nпримера')
            else:image.setPixmap(pixmap.scaled(56,56,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
            image.setToolTip('Иллюстративное фото класса. '+reference_credit(code)+'. '
                             'Это не найденный ближайший аналог текущего интервала.')
            row.addWidget(image)
            text=QVBoxLayout();text.setSpacing(1)
            name=QLabel(ROCK_NAMES[code]);name.setWordWrap(True);name.setMaximumWidth(126)
            text.addWidget(name)
            percentage=QLabel(f'{score:.0%}');percentage.setObjectName('lithologyPredictionScore');text.addWidget(percentage)
            row.addLayout(text,1);grid.addWidget(card,i//2,i%2)
        if not rows:
            layout.addWidget(QLabel('Оценки пород недоступны.'))
            return
        note=QLabel('Фото — примеры классов. Проценты — оценки модели для выбранного участка.')
        note.setWordWrap(True);note.setMaximumWidth(420);note.setStyleSheet('font-size:11px;color:#596b83;');layout.addWidget(note)
        if total>4:
            extra=QLabel(f'Показаны 4 наиболее вероятные группы из {total}. Необученные классы не получают фиктивные проценты.')
            extra.setWordWrap(True);extra.setMaximumWidth(420);layout.addWidget(extra)
        attribution=QLabel('Фото: '+'; '.join(dict.fromkeys(reference_credit(c) for c,_ in rows)))
        attribution.setWordWrap(True);attribution.setMaximumWidth(420)
        attribution.setStyleSheet('font-size:10px;color:#687a92;');layout.addWidget(attribution)
