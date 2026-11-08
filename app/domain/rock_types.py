"""Requested rock vocabulary and observable evidence; never labels inferred from theory."""
from __future__ import annotations

ROCK_CLASSES = ('sandstone', 'siltstone', 'mudrock', 'crystalline',
                'carbonaceous_siltstone', 'carbonate_other', 'breccia_other', 'background')
ROCK_NAMES_RU = ('Песчаник', 'Алевролит', 'Аргиллит / глинистая порода',
                 'Магматические / метаморфические / железистые породы',
                 'Углистый алевролит · вне целевого списка',
                 'Карбонатная порода · вне целевого списка',
                 'Брекчия / обломки · вне целевого списка', 'Фон / не керн')
COAL_CLASSES = ROCK_CLASSES + ('coal',)
COAL_NAMES_RU = ROCK_NAMES_RU + ('Уголь · экспериментальный класс',)
NATIVE_NAMES = dict(zip(COAL_CLASSES, COAL_NAMES_RU))
TARGET_NAMES = {'sandstone': 'Песчаник', 'siltstone': 'Алевролит', 'mudrock': 'Аргиллит', 'coal': 'Уголь'}
SPECIAL_ARGILLITE_NAME='Битуминозный и окремненный аргилит'
SPECIAL_ARGILLITE_RULE='argillite AND (bituminous OR silicified)'

SOURCES = {
    'tpu': ('ТПУ · описание керна и терригенных пород', 'https://portal.tpu.ru/SHARED/n/NEDOLIVKO/disc1/Tab2/Practicum_Gl2.pdf'),
    'bgs_silt': ('BGS · Siltstone', 'https://webapps.bgs.ac.uk/bgsrcs/rcs_details.cfm?code=SLST'),
    'bgs_coal': ('BGS · Bituminous coal', 'https://webapps.bgs.ac.uk/bgsrcs/rcs_details.cfm?code=BITCOL'),
    'usgs_organic': ('USGS · Oil shale', 'https://www.usgs.gov/centers/central-energy-resources-science-center/science/oil-shale'),
    'nps': ('NPS · Basement', 'https://home.nps.gov/subjects/geology/gri-glossary-of-geologic-terms.htm'),
}

# Concise paraphrases; grain boundaries depend on the agreed classification.
# No colour-to-lithology lookup or automatic source-description relabelling.
ROCK_PROFILES = (
    dict(name='Гравелит', native=None, sources=['tpu'],
         visible='Различимые гравийные обломки, их форма, окатанность и распределение в более мелкой матрице.',
         check='Нужен масштаб для размера обломков. Не смешивать с брекчией и крупногалечным конгломератом.',
         training='Нужны подтверждённые фото гравелита с масштабом.'),
    dict(name='Песчаник', native='sandstone', sources=['tpu'],
         visible='Песчаные зёрна и их сортировка; возможны массивные и слоистые разновидности.',
         check='Цвет бывает разным. Точная зернистость, состав и цемент требуют достаточного увеличения или анализа.',
         training='Прямой класс sandstone в открытых базах.'),
    dict(name='Переслаивание с преобладанием песчаной фракции', native=None, sources=['tpu'],
         visible='Последовательность песчаных и более тонких прослоев; суммарная мощность песчаных слоёв преобладает.',
         check='Нужны границы слоёв, порядок сборки столбиков и мощности. Одна контрастная полоса этого не доказывает.',
         training='Нужна интервальная разметка переслаивания. Смесь прогнозов отдельных окон не является готовым классом.'),
    dict(name='Переслаивание с преобладанием алеврито-глинистой фракции', native=None, sources=['tpu'],
         visible='Чередование песчаных и тонкозернистых слоёв с преобладанием мощности алеврито-глинистых интервалов.',
         check='Сравнивать мощность по оси керна, а не площадь нескольких соседних столбиков или число обрезок.',
         training='Нужна интервальная разметка с подтверждёнными литологиями и мощностями.'),
    dict(name='Алевролит', native='siltstone', sources=['tpu', 'bgs_silt'],
         visible='Очень мелкая обломочная структура; часто различимы тонкие слойки.',
         check='С тонким песчаником и аргиллитом может быть неразличим на обзорном фото. Нужны крупный план и масштаб.',
         training='Прямые классы siltstone; углистый алевролит выделен отдельно.'),
    dict(name='Аргиллит', native='mudrock', sources=['tpu'],
         visible='Тонкозернистая уплотнённая глинистая порода; возможна плитчатость и тонкая слоистость.',
         check='Общий класс mudrock шире аргиллита. Долю глины и отличие от тонкого алевролита уточняют на образце.',
         training='Кандидат из mudstone/shale. Название требует проверки по принятому справочнику.'),
    dict(name='Битуминозный и окремненный аргилит', native=None, sources=['usgs_organic'],
         visible='Могут наблюдаться тёмная окраска, плотность, блеск и раковистые поверхности скола.',
         check='Общая категория: аргиллит битуминозный ИЛИ окремнённый, в том числе с обоими признаками. Каждый признак проверяется отдельно; чёрный цвет их не устанавливает.',
         training='Подходят фото подтверждённого битуминозного аргиллита, окремнённого аргиллита или аргиллита с обоими признаками. Битуминозность и окремнение сохраняются отдельно. Битуминозный известняк в эту категорию не входит.'),
    dict(name='Углистый аргилит', native=None, sources=['tpu', 'usgs_organic'],
         visible='Глинистая матрица с углефицированным растительным материалом или углистыми прослоями.',
         check='Не равен углю, чёрному сланцу или углистому алевролиту. Требуется подтверждение матрицы и органического вещества.',
         training='Нужны прямые экспертные метки. Carbonaceous siltstone сюда не переносится.'),
    dict(name='Уголь', native='coal', sources=['bgs_coal'],
         visible='Для каменного угля типичны чёрный цвет, матовые и блестящие полосы; характер скола зависит от разновидности.',
         check='Тёмный аргиллит и мокрая порода могут выглядеть похоже. Учебные угольные текстуры отличаются от фото керна.',
         training='Отдельная исходная разметка coal; пригодность на керне проверяется отдельно.'),
    dict(name='Породы фундамента', native='crystalline', sources=['nps'],
         visible='Для кристаллических разновидностей: срастание кристаллов, минеральная неоднородность, гнейсовая полосчатость.',
         check='Фундамент — положение в разрезе. Фото гранита или гнейса само по себе не подтверждает это положение.',
         training='Есть широкая группа магматических, метаморфических и железистых пород. Она включает BIF и не является отдельным классом фундамента. Положение в разрезе задаёт специалист.'),
)


def rock_training_label(value):
    """Merge explicit reviewed argillite variants under the user-defined OR class.

    Only whole label matches are allowed. No arbitrary prose or colour becomes
    a subtype. The umbrella category does not assert both properties are true.
    """
    label=str(value).strip()
    text=' '.join(label.casefold().replace('ё','е').replace('аргилит','аргиллит').split())
    category=' '.join(SPECIAL_ARGILLITE_NAME.casefold().replace('аргилит','аргиллит').split())
    variants={
        'битуминозный аргиллит':(True,None),'аргиллит битуминозный':(True,None),
        'окремненный аргиллит':(None,True),'аргиллит окремненный':(None,True),
        'битуминозный окремненный аргиллит':(True,True),
        'окремненный битуминозный аргиллит':(True,True),
    }
    if text==category:properties=(None,None)
    elif text in variants:properties=variants[text]
    else:return label,None
    return SPECIAL_ARGILLITE_NAME,{'argillite':True,'bituminous':properties[0],
                                   'silicified':properties[1],'category_rule':SPECIAL_ARGILLITE_RULE}


def rock_profile(name):
    normalized=' '.join(str(name).casefold().replace('ё', 'е').replace('аргилит', 'аргиллит').split())
    return next((dict(p) for p in ROCK_PROFILES
                 if ' '.join(p['name'].casefold().replace('ё', 'е').replace('аргилит', 'аргиллит').split()) == normalized), None)


def training_status(profile, card):
    native=profile['native']; classes=card.get('classes', []) if isinstance(card, dict) else []
    if native=='coal' and native in classes and not card.get('coal_core_validated',False):
        return 'Экспериментальный класс · автоматическое назначение отключено'
    if native=='crystalline' and native in classes:return 'Есть смежная группа; отдельный класс фундамента не обучен'
    if native and native in classes:
        return 'Есть в подключённом классификаторе' + (' · требуется геологическое уточнение' if native in {'mudrock','crystalline'} else '')
    if profile['name']=='Песчаник' and 'sandstone' in classes:return 'Есть в подключённом классификаторе'
    return 'Отдельного обученного класса пока нет'


def rock_reference_answer(question,show_sources=False):
    """Explicit reference lookup for definition/comparison questions, no generation."""
    import re
    text=str(question).casefold().replace('ё','е').replace('аргилит','аргиллит')
    if not any(w in text for w in ('что такое','отлич','признак','расскажи','определение','сравни','характерист','какие породы')):return ''
    patterns=[(6,r'битуминозн\w*.*аргиллит\w*|окремнен\w*.*аргиллит\w*'),
              (7,r'углист\w*\s+аргиллит\w*'),(0,r'гравелит\w*'),(1,r'песчаник\w*'),
              (4,r'алевролит\w*'),(5,r'аргиллит\w*'),(8,r'уголь|угля|угле|угли'),(9,r'фундамент\w*')]
    found=[];occupied=[]
    for index,pattern in patterns:
        for match in re.finditer(r'(?<![а-я])(?:'+pattern+r')(?![а-я])',text):
            a,b=match.span()
            if any(a<d and b>c for c,d in occupied):continue
            occupied.append((a,b));found.append((a,index))
    if not found:return ''
    indices=list(dict.fromkeys(i for _,i in sorted(found)))[:3]
    paragraphs=[]
    for i in indices:
        p=ROCK_PROFILES[i]
        paragraphs.append(p['name']+'\n'+p['visible']+'\n'+p['check'])
    if show_sources:
        keys=list(dict.fromkeys(k for i in indices for k in ROCK_PROFILES[i]['sources']))
        paragraphs.append('Источники:\n'+'\n'.join(title+': '+url for title,url in (SOURCES[k] for k in keys)))
    return '\n\n'.join(paragraphs)
