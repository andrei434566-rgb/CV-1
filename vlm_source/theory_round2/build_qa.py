"""Original short Russian QA, manually checked against the cited readable sources."""
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATE = '2026-10-25'
SOURCES = {
 'bgs_rcs': {
  'title':'British Geological Survey: The BGS Rock Classification Scheme',
  'url':'https://www.bgs.ac.uk/technologies/bgs-rock-classification-scheme/',
  'locator':'Principles; Volume 3, classification of sediments and sedimentary rocks',
  'supports':'Описательная основа названий, корневые термины и уточнения, различие названия породы и предположения о происхождении.',
  'access':'Read full public HTML principles, not the linked pay/download material.',
 },
 'iodp402_methods': {
  'title':'IODP Expedition 402 methods',
  'url':'https://publications.iodp.org/proceedings/402/102/402_102.html',
  'locator':'Sections 2.4–2.6, smear slides, thin sections, X-ray diffraction',
  'supports':'Назначение микроскопических препаратов и рентгенодифракции; положение образца и относительные оценки состава.',
  'access':'Read full public HTML relevant sections.',
 },
 'odp202_sedimentology': {
  'title':'ODP Leg 202: Sedimentology',
  'url':'https://iodp.tamu.edu/publications/202_IR/chap_02/c2_3.htm',
  'locator':'Preparation of Surfaces of Core Halves; Sediment Disturbance; Color; Bioturbation',
  'supports':'Размазывание поверхности, нарушения бурением, окисление окраски и ограничения распознавания ходов в однородном осадке.',
  'access':'Read complete public HTML chapter.',
 },
 'iodp398_methods': {
  'title':'IODP Expedition 398 methods',
  'url':'https://publications.iodp.org/proceedings/398/102/398_102.html',
  'locator':'Section 2.5.4, Bioturbation; Section 2.8, Smear slides',
  'supports':'Биотурбационный индекс измеряет нарушение первичной ткани; ходы могут перекрываться и плохо различаться; шкалы экспедиционно обусловлены.',
  'access':'Read public HTML sections 2.5.3–2.8.',
 },
 'odp106_classification': {
  'title':'Ocean Drilling Program: Geology 106 Laboratory — Sedimentary rocks and sedimentary rock classification',
  'url':'https://www-odp.tamu.edu/curation/gcr/geol106lab/classifications.htm',
  'locator':'Final paragraphs, Primary and Secondary sedimentary structures',
  'supports':'Нагрузочные текстуры, внедрение песка в мягкий ил, конволюции при движении флюидов или оползании.',
  'access':'Read complete public HTML teaching page. Simplified historical classification not adopted wholesale.',
 },
 'noaa_reef': {
  'title':'NOAA National Ocean Service: What is a coral reef made of?',
  'url':'https://oceanservice.noaa.gov/facts/coralmadeof.html',
  'locator':'Calcium-carbonate skeletons and successive colonies; updated 2026-09-23',
  'supports':'Карбонатный скелет полипов, накопление колоний и органическое построение рифа.',
  'access':'Read complete public HTML fact page.',
 },
 'iodp361_summary': {
  'title':'Hall et al. (2017), IODP Expedition 361 summary',
  'url':'https://publications.iodp.org/proceedings/361/101/361_101.html',
  'locator':'Sites U1475 and U1476, Sedimentology',
  'supports':'Конкретные наблюдения пиритизированных ходов и зелёных диагенетических слоёв, содержащих пирит и глауконит.',
  'access':'Read public HTML site sedimentology sections. Case study, not universal diagnostic rule.',
 },
 'tpu_nedolivko_2018': {
  'title':'Н. М. Недоливко, ТПУ (2018). Индивидуальное домашнее задание №1: построение литологической колонки',
  'url':'https://portal.tpu.ru/SHARED/n/NEDOLIVKO/disc1/Tab2/IDZ_1.pdf',
  'locator':'PDF pages 3 and 6–9; definitions of пр. and в.к.; interval descriptions',
  'supports':'Российский пример раздельной записи проходки, выноса, особенностей слоистости и растительных остатков; легенда колонки.',
  'access':'Read public PDF text, including definitions and examples. No images extracted or paired.',
 },
 'heald_1955_abstract': {
  'title':'Milton T. Heald (1955). Stylolites in Sandstones. The Journal of Geology 63(2):101–114',
  'url':'https://www.journals.uchicago.edu/doi/abs/10.1086/626237',
  'locator':'Public abstract only; doi:10.1086/626237',
  'supports':'Растворение под давлением, швы и сутурные контакты, неодинаковая ориентировка стилолитов и отсутствие простой связи с количеством цемента.',
  'access':'Read public abstract. Paywalled full article was not accessed or downloaded.',
 },
 'kgs_concretion': {
  'title':'Kansas Geological Survey / GeoKansas: Concretion',
  'url':'https://geokansas.ku.edu/concretion',
  'locator':'Concretion; Septarian Concretions',
  'supports':'Диагенетические минеральные стяжения, разнообразие формы и времени образования; заполненные трещины септарий.',
  'access':'Read complete public HTML geology-survey page.',
 },
 'kgs_flint_hills': {
  'title':'Kansas Geological Survey / GeoKansas: Rocks and Minerals of the Flint Hills',
  'url':'https://geokansas.ku.edu/rocks-and-minerals-flint-hills',
  'locator':'Chert; Limestone; Geodes',
  'supports':'Микрокристаллический кварц в кремне, раковистый излом, конкреции и пласты в известняке; различие карбоната и кремнезёма.',
  'access':'Read complete public HTML geology-survey page.',
 },
 'psu_reading_sedimentary': {
  'title':'Pennsylvania State University, GEOSC 10: Reading Sedimentary Rocks',
  'url':'https://courses.ems.psu.edu/geosc10/node/2100',
  'locator':'Mud Cracks — Capitol Reef National Park; text and transcript',
  'supports':'Высыхание и заполнение трещин; их сужение вниз как признак направления стратиграфического верха.',
  'access':'Read public HTML lesson and transcript. Broad simplified environment claims not adopted.',
 },
}

# Each block contains four new training questions and one paraphrase holdout.
# Answers are original Russian paraphrases, deliberately short and narrowly scoped.
BLOCKS = [
 ('rock_naming','bgs_rcs',[
  ('Можно ли заменить литологическое название словом «русловой»?', 'Нет. «Русловой» обозначает предполагаемое происхождение. Литологическое название выбирают по наблюдаемым составу и строению, а обстановку записывают отдельно.'),
  ('Зачем к корневому названию породы добавлять уточнения?', 'Корневое название задаёт основной тип породы. Уточнения сообщают дополнительные наблюдаемые признаки, не подменяя основную классификацию.'),
  ('Нужна ли известная обстановка, чтобы назвать осадочную породу?', 'Для описательного названия обычно достаточно признаков образца или шлифа. Предположение об обстановке не должно служить единственным основанием названия.'),
  ('Почему в базе керна нужен общий словарь названий?', 'Одинаковые названия должны иметь одинаковые определения. Иначе объединение описаний из разных скважин смешивает разные классы пород.'),
 ],('Песчаник назвали дельтовым только из-за положения скважины. Как уточнить запись?', 'Сначала указать описательное название по признакам породы. «Дельтовый» оставить отдельной интерпретацией с собственным обоснованием.')),
 ('laboratory_evidence','iodp402_methods',[
  ('Что добавляет мазковый препарат к описанию осадка?', 'Микроскопия помогает оценить размеры, состав и относительное содержание обломочных, биогенных и аутигенных компонентов, плохо различимых на общем фото.'),
  ('Зачем сохранять точное положение мазкового препарата в керне?', 'Микроскопический результат относится к конкретному отобранному образцу. Его положение нужно связать с описываемым интервалом, а не переносить на весь керн.'),
  ('Какой препарат используют для изучения сцементированной породы?', 'Описание сцементированной породы дополняют шлифом. Он позволяет рассматривать компоненты и структуру под петрографическим микроскопом.'),
  ('Как рентгенодифракция помогает при сомнительном минеральном составе?', 'Она проверяет минеральные фазы по дифракционным признакам. Это дополнительное аналитическое основание, а не название минерала, угаданное по цвету фотографии.'),
 ],('Есть фото керна и анализ одной порошковой пробы. Можно распространить состав на всю колонку?', 'Нет. Анализ характеризует отобранную пробу; сначала сопоставьте её положение с интервалом. Для остальных участков нужны собственные основания.')),
 ('preparation_artifacts','odp202_sedimentology',[
  ('Почему свежая продольная поверхность мягкого керна бывает смазанной?', 'Разделение проволокой может размазывать мягкий осадок и скрывать тонкие детали. Такой след подготовки не следует автоматически считать природной текстурой.'),
  ('Почему снимки одного свежераспиленного керна могут менять окраску со временем?', 'После контакта с воздухом идут поверхностные реакции окисления. Поэтому цвет спустя время может отличаться от цвета свежего осадка.'),
  ('Что означает описание drilling biscuits?', 'Это разделённые бурением куски сравнительно твёрдого керна, иногда окружённые буровой суспензией. Их границы не следует автоматически принимать за осадочные контакты.'),
  ('Почему незаметные ходы в однородном осадке не доказывают отсутствие биотурбации?', 'Распознавание ходов в однородных тёмных глинах и светлых илах ограничено. Надёжнее отметить отсутствие различимых следов, сохранив неопределённость причины однородности.'),
 ],('Вдоль распила тянется гладкая полоса, а внутри образца её нет. Это ламинация?', 'Прежде исключите смазывание при разделении керна. Признак только на подготовленной поверхности недостаточен для подтверждения природной ламинации.')),
 ('ichnofabric','iodp398_methods',[
  ('Что оценивает ихнофабричный индекс: число ходов или нарушение осадка?', 'Прежде всего степень биогенного нарушения первичной ткани, например ламинации. Число видимых ходов само по себе не заменяет эту оценку.'),
  ('При биотурбации ходы обязательно изолированы друг от друга?', 'Нет. Они могут перекрываться и становиться плохо различимыми. Описывайте также сохранность первичной слоистости.'),
  ('Как записывать интенсивность биотурбации без смешения разных шкал?', 'Укажите использованную шкалу и её определение категорий. Одинаковые слова «слабая» или «умеренная» могут иметь разные пороги в разных методиках.'),
  ('Какой признак важен для высокой биотурбации в методике IODP 398?', 'Первичная слоистость полностью нарушена; местами ещё различимы ходы. Отдельно отмечайте наблюдаемые признаки, а не присваивайте категорию по одной пятнистой окраске.'),
 ],('На фото видны перекрывающиеся расплывчатые ходы. Это обязательно слабая биотурбация?', 'Нет. Нечёткость может сопровождать перекрытие ходов. Категорию определяют по нарушению первичной ткани и правилам выбранной шкалы.')),
 ('soft_sediment_deformation','odp106_classification',[
  ('Как образуются нагрузочные выступы песка в глинистом осадке?', 'Более тяжёлый песок погружается в мягкий водонасыщенный нижележащий ил. Возникают направленные вниз выступы на контакте.'),
  ('Что означает структура ball-and-pillow в осадке?', 'Это деформация, связанная с внедрением более тяжёлого осадка в мягкий нижележащий материал. Её рассматривают среди вторичных осадочных структур.'),
  ('Какие процессы способны вызвать конволютную слоистость?', 'Изгибание слойков может сопровождать выход флюида вверх или оползание осадка. По одной форме складочек нельзя уверенно выбрать единственный процесс.'),
  ('Почему смятый мягкий осадок не доказывает землетрясение?', 'Подобная деформация имеет несколько возможных причин, включая движение флюида и оползание. Сначала опишите геометрию, затем проверяйте причину.'),
 ],('Песчаные языки направлены вниз в мягкую глину. Что описать прежде причины?', 'Зафиксируйте выступы на нижнем контакте и деформацию подстилающего ила. Такая геометрия согласуется с нагрузочной деформацией, но требует проверки контекста.')),
 ('reef_framework','noaa_reef',[
  ('Из какого вещества строят скелеты рифообразующие каменистые кораллы?', 'Они выделяют карбонат кальция. Накопление этих скелетов создаёт основу коралловой рифовой постройки.'),
  ('Коралловый риф строится поднимающимся мелким песком?', 'Нет. Каркас рифа нарастает за счёт карбонатных скелетов организмов; новые колонии растут на известковых остатках предыдущих.'),
  ('Что в коралловой колонии является живым, а что минеральным?', 'Полипы — живые организмы. Выделенный ими карбонатный скелет составляет минеральную основу колонии и рифовой постройки.'),
  ('Почему наличие карбонатного скелета не равно простому песчаному слою?', 'Скелет образуется организмом и может входить в растущую постройку. Песчаный слой описывает скопление частиц; это разные признаки.'),
 ],('Верно ли объяснение: «риф — это слои песка, поднимающиеся вверх»?', 'Нет. Коралловая постройка нарастает благодаря образованию карбонатных скелетов и росту последующих колоний на остатках предыдущих.')),
 ('diagenetic_overprint','iodp361_summary',[
  ('Могут ли минеральные изменения подчеркнуть биогенный ход?', 'Да. В керне IODP 361 описаны пиритизированные ходы. Форма хода и последующая минерализация должны учитываться раздельно.'),
  ('Зелёная полоска в керне обязательно является первичным слоем?', 'Нет. В IODP 361 зелёные слои с пиритом и глауконитом интерпретированы как диагенетические изменения. Для другого керна это проверяемая возможность.'),
  ('Можно ли считать пиритизацию и биотурбацию взаимоисключающими признаками?', 'Нет. Биогенные ходы могут позднее пиритизироваться. Минеральное заполнение не исключает биогенную геометрию структуры.'),
  ('Как использовать опубликованный пример глауконитовой зелёной полосы?', 'Как аналог возможного диагенетического изменения. Он не подтверждает глауконит в новом образце без проверки состава.'),
 ],('Внутри похожего на ход пятна обнаружен пирит. След организма исключён?', 'Нет. Известны пиритизированные ходы. Проверьте геометрию структуры и учитывайте пиритизацию как возможное последующее изменение.')),
 ('russian_core_log','tpu_nedolivko_2018',[
  ('Что означают «пр.» и «в.к.» в учебном описании керна ТПУ?', '«Пр.» означает проходку, «в.к.» — вынос керна. Их записывают отдельно: длина извлечённого материала может быть меньше интервала проходки.'),
  ('Зачем в российском описании указывать материал, подчёркивающий слоистость?', 'Это уточняет наблюдение: чередование может выделяться глинистыми, углистыми или сидеритовыми прослойками. Геометрию слоистости описывают отдельно.'),
  ('Как включить растительные остатки в описание керна?', 'Отдельно отметьте растительный детрит, крупные фрагменты или корневые остатки и их расположение. Не сводите разные наблюдения к одному слову «органика».'),
  ('Что нужно приложить к литологической колонке по учебному заданию ТПУ?', 'Заглавие и легенду с расшифровкой использованных условных знаков. Иначе графические обозначения пород и особенностей останутся неоднозначными.'),
 ],('В паспорте проходка 6 м, а вынос 4 м. Нужно ли сделать их одинаковыми?', 'Нет. Это разные величины: интервал проходки и длина извлечённого керна. Сохраняйте обе исходные записи.')),
 ('stylolites','heald_1955_abstract',[
  ('Бывают ли стилолиты в песчаниках?', 'Да. В песчаниках описаны стилолитовые швы и тонкие сутурные контакты между зёрнами, связанные с растворением под давлением.'),
  ('Стилолитовый шов обязательно совпадает с напластованием?', 'Нет. Многие швы параллельны напластованию, но встречаются и швы вдоль наклонных к нему трещин.'),
  ('Как растворение под давлением влияет на обломочные зёрна?', 'Оно может изменять размеры и форму зерновых контактов, угловатость и сферичность. Наблюдаемая форма не всегда сохраняет только условия переноса.'),
  ('Количество стилолитовых швов прямо задаёт количество кварцевого цемента?', 'Нет. Растворённый кремнезём может участвовать в цементации, но простого локального соответствия между объёмом растворения и количеством цемента нет.'),
 ],('Тонкий сутурный шов в песчанике объявили первичной ламинацией. Что проверить?', 'Проверьте признаки растворения под давлением и изменённых контактов зёрен. Стилолитовый шов не равнозначен первичному осадочному слойку.')),
 ('concretions','kgs_concretion',[
  ('Округлое плотное включение обязательно является окаменелостью?', 'Нет. Конкреции могут быть округлыми минеральными стяжениями и напоминать окаменелости. Нужны признаки строения и состава, а не одна форма.'),
  ('Конкреции образуются только до уплотнения осадка?', 'Нет. Они часто растут в осадке, но могут формироваться и в затвердевшей породе при осаждении минералов в порах и полостях.'),
  ('Какая форма возможна у конкреций кроме шаровидной?', 'Встречаются овальные, вытянутые, дисковидные и неправильные конкреции. Поэтому округлость не является обязательным критерием.'),
  ('Что представляют собой минерализованные трещины септарии?', 'Это сеть трещин внутри конкреции, заполненных минералами, например кальцитом. Их нельзя автоматически принимать за трещиноватость всей вмещающей толщи.'),
 ],('Плотный округлый объект похож на яйцо. Можно сразу записать ископаемое яйцо?', 'Нет. Сначала исключите конкрецию: минеральные стяжения часто имитируют такие формы. Биогенное происхождение требует собственных признаков.')),
 ('chert_and_limestone','kgs_flint_hills',[
  ('Чем кремень chert по составу отличается от известняка?', 'Кремень состоит из микроскопического кварца, то есть кремнезёма; известняк преимущественно карбонатный, обычно кальцитовый. Сходный цвет не означает одинакового состава.'),
  ('Какой излом характерен для кремня?', 'Часто наблюдается раковистый излом с острыми краями. Это полезный признак, но состав и другие свойства также следует проверять.'),
  ('Кремень внутри известняка бывает только отдельной галькой?', 'Нет. Он может присутствовать в виде конкреций или протяжённых прослоев. Сам факт нахождения кремня не доказывает перенос гальки.'),
  ('Кристаллы на стенках полости обязательно являются обломочными зёрнами?', 'Нет. В жеодах минералы осаждаются из раствора на стенках полости. Такое заполнение отличается от накопления обломочного материала.'),
 ],('В известняке увидели тёмные кремнистые стяжения. Это обязательно конгломерат?', 'Нет. Кремень может образовывать конкреции и прослои в известняке. Для конгломерата нужно подтвердить обломочное скопление.')),
 ('mudcracks_way_up','psu_reading_sedimentary',[
  ('Как трещины усыхания помогают определить стратиграфический верх?', 'Типичная трещина шире у бывшей поверхности и сужается вниз. При сохранной геометрии это помогает установить направление первоначального верха слоя.'),
  ('Почему древние трещины усыхания могут быть заполнены другим осадком?', 'После высыхания последующее отложение осадка заполняет трещины. Такое заполнение может сохранить их форму при превращении осадка в породу.'),
  ('Нужно ли отличать современный верх фото от первоначального верха слоя?', 'Да. Порода может быть перевёрнута после накопления. Направление верха устанавливают по сохранным геологическим признакам, а не по ориентации снимка.'),
  ('Что прежде описать у предполагаемой трещины усыхания в разрезе?', 'Её расширение к поверхности, сужение вниз и характер заполнения. Эти наблюдения важнее одного внешнего сходства с линией на фото.'),
 ],('Заполненные трещины сужаются к нижней стороне образца. Какую гипотезу проверить?', 'При подтверждении трещин усыхания более широкая сторона указывает к бывшей поверхности слоя. Проверьте сохранность геометрии и ориентацию образца.')),
]

def main():
    train, holdout = [], []
    source_words = Counter()
    for topic, source_id, questions, validation in BLOCKS:
        for split, pairs in [('train', questions), ('holdout', [validation])]:
            for index, (question, answer) in enumerate(pairs, 1):
                source_words[source_id] += len((question+' '+answer).split())
                source = {key:value for key,value in SOURCES[source_id].items() if key!='supports'}
                source.update(source_id=source_id, accessed_date=DATE)
                row = {
                    'conversations':[{'from':'human','value':question},{'from':'gpt','value':answer}],
                    'metadata':{
                        'record_id':f'theory20261025_{topic}_{split}_{index:02d}',
                        'record_type':'text_only_knowledge', 'topic':topic,
                        'source_group':'theory_sources_20261025', 'split':split,
                        'training_approved':split=='train', 'geology_approved':False,
                        'target_language':'ru',
                        'review_status':'user_requested_source_checked_qa_not_geologist_reviewed',
                        'sources':[source],
                        'scope':'Original Russian theory paraphrase only; no image label, depth, facies or expert approval implied.',
                        'source_reuse':'Short factual paraphrase; no verbatim source passages; public access does not imply blanket reuse permission.',
                    },
                }
                (train if split=='train' else holdout).append(row)
    if any(count>200 for count in source_words.values()):
        raise ValueError(f'Source-derived question/answer word budget exceeded: {source_words}')
    # Actual sibling VLM layout is one level above the application repository.
    old = ROOT.parents[2]/'vlm_core_v1/data/knowledge/theory_training_v5.jsonl'
    old_questions = set()
    if old.exists():
        old_questions={json.loads(line)['conversations'][0]['value'].casefold().strip(' ?.!') for line in old.read_text(encoding='utf-8').splitlines() if line.strip()}
    all_rows = train+holdout
    ids=[r['metadata']['record_id'] for r in all_rows]
    qs=[r['conversations'][0]['value'].casefold().strip(' ?.!') for r in all_rows]
    if len(ids)!=len(set(ids)) or len(qs)!=len(set(qs)):
        raise ValueError('Duplicate questions or identifiers')
    overlap=sorted(set(qs)&old_questions)
    if overlap: raise ValueError(f'Exact old training question duplicate: {overlap}')
    for name, rows in [('train',train),('holdout',holdout)]:
        (ROOT/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    (ROOT/'sources.json').write_text(json.dumps({k:{**v,'accessed_date':DATE,'geology_approved':False} for k,v in SOURCES.items()},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    summary={
        'schema':'theory_sources_20261025_v1','train_rows':len(train),'holdout_rows':len(holdout),
        'topics':len(BLOCKS),'read_sources':len(SOURCES),'exact_old_train_question_overlap':len(overlap),
        'old_train_manifest_found':old.exists(),
        'source_question_answer_words':dict(source_words),'geology_approved':False,
        'holdout_scope':'New phrasings of related source concepts, not topic-independent or expert-scored knowledge accuracy.',
        'limitations':['No photo pairs or new facies labels.', 'Short source-checked assistant paraphrases, not geologist approval.',
                       'Several concepts overlap earlier theory; holdout is a regression/paraphrase check.',
                       'Heald 1955 uses public abstract only; no paywall access.',
                       'USGS and GIN RAN URLs that failed direct reading were excluded.',
                       'No blanket download or training permission inferred from a public webpage.'],
        'sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['train.jsonl','holdout.jsonl','sources.json']},
    }
    if (ROOT/'overrides.json').exists():
        corrections=json.loads((ROOT/'overrides.json').read_text(encoding='utf-8'))
        summary['correction_overrides']={'rows':len(corrections),'record_ids':list(corrections),
            'additional_read_sources':2, 'original_training_unchanged':True,
            'note':'TN5 source checked locally; supplementary IODP303306 HTML read. Separate from the 48 new train rows.'}
        for name in ['overrides.json','overrides_sources.json']:
            summary['sha256'][name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
