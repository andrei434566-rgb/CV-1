"""Record incomplete coverage and failed generalisation explicitly, not a 16-class score."""
from pathlib import Path
import json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.domain.bedding_types import PROFILES

def main():
 r=ROOT/'open_datasets/bedding_20261106/expansion';model=ROOT/'models/bedding/extension_20261106_v2'
 out=ROOT/'research_reports/bedding_20261106_expansion';out.mkdir(parents=True,exist_ok=True)
 data=json.loads((r/'dataset.json').read_text(encoding='utf8'))
 card=json.loads((model/'config.json').read_text(encoding='utf8'))
 rows=data['examples'];status=[]
 for name,*_ in PROFILES:
  counts={s:sum(v['positive_type']==name and v['split']==s for v in rows) for s in ('train','val','test')}
  note='Обучен только исследовательский дополнительный признак; автоматическое назначение отключено.' if name in card['fitted_types'] else 'Точного обученного класса в дополнительном модуле нет.'
  if name=='Градационная':note+=' Положительные пары сохранены; нет проверенных отрицательных примеров. Massive не означает отсутствие градации.'
  if name=='Пологонаклонная':note+=' В прежней CNN есть исходная группа Low angle, требующая отдельной проверки ориентации.'
  if name=='Субгоризонтальная':note+=' Прежняя Parallel lamination не доказывает истинную горизонтальность.'
  if name in ('Таблитчатая крупная косая','Троговая крупная косая','Крупная косая'):
   note+=' Найдены широкие подписи cross/plate/trough bedding; размер серии в конкретном фото не подтверждён.'
  if name=='Косо-волнистая':note+=' Найдена точная русская подпись: Мамадалиев, УГГУ, 2013, с.73, скв.299П, обр.2. Фото малоразмерное; не допущено к обучению.'
  if name=='Мелкая косая':note+=' Есть current/climbing ripple lamination; точная привязка к размерной категории проекта требует проверки.'
  status.append({'type':name,'new_positive_roi_support':counts,'automatic_assignment':False,'note':note})
 (out/'coverage_16.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8')
 urls={
  'benxi.pdf':'https://mdpi-res.com/d_attachment/fractalfract/fractalfract-10-00374/article_deploy/fractalfract-10-00374.pdf',
  'pinghu.pdf':'https://mdpi-res.com/d_attachment/energies/energies-15-07214/article_deploy/energies-15-07214.pdf',
  'xujiahe.pdf':'https://mdpi-res.com/d_attachment/sustainability/sustainability-16-04304/article_deploy/sustainability-16-04304.pdf',
  'sydney.pdf':'https://mdpi-res.com/d_attachment/geosciences/geosciences-11-00273/article_deploy/geosciences-11-00273.pdf',
  'longtan.pdf':'https://mdpi-res.com/d_attachment/jmse/jmse-14-01797/article_deploy/jmse-14-01797.pdf',
  'ordos.pdf':'https://mdpi-res.com/d_attachment/energies/energies-17-05978/article_deploy/energies-17-05978.pdf',
  'ural_cosowavy.pdf':'https://www.ursmu.ru/userfiles/media/default/ugpd/ugpd-2013/02-2-geologiya-geofizika-i-geoekologiya-litologiya.pdf',
  'zubair.pdf':'https://ijs.uobaghdad.edu.iq/index.php/eijs/article/download/606/356',
  'chengjiang.html':'https://www.nature.com/articles/s41467-022-29246-z',
  'chengjiang_supp.pdf':'https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-022-29246-z/MediaObjects/41467_2022_29246_MOESM1_ESM.pdf',
  'chengjiang_fig2.png':'https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41467-022-29246-z/MediaObjects/41467_2022_29246_Fig2_HTML.png',
  'gulong.html':'https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.999135/full',
  'gulong_fig4.webp':'https://www.frontiersin.org/files/Articles/999135/xml-images/feart-10-999135-g004.webp',
  'sangonghe.html':'https://link.springer.com/article/10.1007/s13202-018-0599-0',
  'sangonghe_fig3.jpg':'https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs13202-018-0599-0/MediaObjects/13202_2018_599_Fig3_HTML.jpg',
 }
 assets=[{'file':n,'url':u,'bytes':(r/n).stat().st_size,'sha256':hashlib.sha256((r/n).read_bytes()).hexdigest()} for n,u in urls.items()]
 rendering={
  'benxi_fig4.png':{'source':'benxi.pdf','page':8,'scale_to':2200},
  'chengjiang_s1.png':{'source':'chengjiang_supp.pdf','page':1,'scale_to':2200},
  'pinghu_fig10.png':{'source':'pinghu.pdf','page':14,'scale_to':2600},
  'sydney_fig5.png':{'source':'sydney.pdf','page':11,'scale_to':2400},
 }
 (r/'source_provenance.json').write_text(json.dumps({'assets':assets,'rendering':rendering,
    'source_citations':data['sources'],'licence_evidence':'Article rights sections / first PDF page. Ural retained for review only.',
    'retrieved_on':'2026-11-06'},ensure_ascii=False,indent=2),encoding='utf8')
 text=['# Дополнение базы слоистости и контроль дообучения — 06.11.2026','',
 'Найдены и скачаны новые авторские фотографии керна с описаниями. Подготовлено 16 сопоставленных фрагментов для 7 типов: 12 для исследовательского обучения, 4 для проверки. Два фрагмента Opalinus Clay были отобраны в предыдущем проходе, но до этого не участвовали в обучении. Фрагменты одной фотографии/скважины не считаются независимыми образцами.','',
 '**Результат проверки: новый модуль не принят для автоматического описания. Надёжное распознавание всех 16 типов пока не получено. Рабочие модели пород, YOLO и исходная CNN слоистости сохранены без изменений.**','',
 '## Что выполнено','',
 '* Просмотрены статьи Chengjiang, Benxi, Pinghu, Sydney Basin, Gulong, Sangonghe, Longtan, Xujiahe, Ordos, Zubair и российский материал УГГУ. Исходные документы, URL и SHA256 сохранены в source_provenance.json.',
 '* Каждый допущенный фрагмент сопоставлен с авторской подписью и конкретной частью рисунка. Схемы, шлифы, обнажения, трассированные белыми линиями фото и сомнительные подписи не стали учебными метками.',
 '* Есть точная русская подпись косо-волнистой слоистости: УГГУ, 2013, Мамадалиев А.Н., скв.299П Северо-Конитлорского месторождения, образец 2, с.73. Фотография слишком мала и плохо передаёт геометрию; сохранена как материал для проверки, в обучение не включена.',
 '* Показательная ошибка источника: Sangonghe Figure 3d в подписи назван parallel bedding, а в тексте также упомянут под graded bedding. Этот участок исключён.',
 '* Для plate/trough/cross bedding сохранено исходное широкое определение. Размер «крупная» не присвоен только из-за такого названия.',
 '', '## Что обучено','',
 'Шесть независимых линейных сигмоидных классификаторов признаков поверх замороженной ConvNeXt-Tiny. Это не LLM/VLM и не генеративная модель. Исходные веса CNN и её прежний выходной слой не изменяются. Новые веса хранятся отдельно в head.pt (около 23 КБ). Градационная слоистость присутствует в словаре и положительных примерах, но соответствующий выход не обучен и выдаёт null.',
 '', 'Положительный пример подтверждает только названное автором свойство. Все остальные свойства остаются неизвестными. Они не превращаются в отрицательные метки. Отрицательные примеры для видимых типов слойчатости взяты только из исходного авторского класса Massive; это узкая и недостаточная группа контрпримеров. Для градационной слоистости Massive не является отрицательной меткой: массивный слой может иметь изменение зернистости.',
 '', 'Импортированы 12 положительных ROI: 9 участвуют в оптимизации шести признаков, а 3 градационных остаются без обучения из-за отсутствия подтверждённых отрицательных примеров. Также использованы 279 авторских Massive ROI для шести признаков, masked balanced BCE, 120 заранее заданных эпох, замороженные признаки 768, AdamW. Аугментации и искусственное увеличение числа независимых примеров не использованы. Контроль: 61 Massive ROI в валидации; 42 Massive ROI и 4 положительных литературных фрагмента в тестовом архиве. Из них два проверяют обученные признаки, два градационных остаются неоценёнными. Известные скважины и неразрешённые группы источников не пересекаются между частями. При отсутствии конкретного номера скважины фрагменты объединены всей статьёй, а не объявлены новыми скважинами.',
 '', 'Режим research_only: он показывает, что дополнительное обучение технически выполнено, но не означает пригодность для геологического диагноза.',
 '', '## Фактическая проверка переноса','',
 'При заранее заданном пороге срабатывания 0.8 принятых положительных примеров 0 из 2 для двух обученных типов с контролем (флазерная и линзовидная). Два градационных фрагмента не оценивались: этот признак не обучен. Это отказ проверки, а не улучшение качества. На 42 контрольных массивных ROI ложных срабатываний при этом пороге нет, но они не заменяют проверку на похожих типах слоистости. Общая accuracy/F1 «для 16 классов» не рассчитывается.',
 '', '| Контрольный фрагмент | Авторский признак | Оценка дополнительного модуля |', '|---|---|---:|']
 predictions=json.loads((model/'predictions.json').read_text(encoding='utf8'))
 for v in predictions:
  if v['split']=='test' and v['positive_type']:
   score=v['scores'][v['positive_type']];display='не обучен / null' if score is None else f'{score:.4f}'
   text.append(f"| {v['id']} | {v['positive_type']} | {display} |")
 text += ['', 'Оценки не калиброваны как вероятности диагноза. Автоматически назначать свойство по одному контрольному примеру нельзя. Порог после просмотра теста не меняли. Малое число примеров, смещение между публикациями, недостаток независимых скважин и проверенных контрпримеров остаются существенными ограничениями.',
          '', 'Первый промежуточный эксперимент extension_20261106 ошибочно использовал Massive как отрицательную метку градации. Он помечен INVALID_DATA_LABELS.txt и исключён из использования. В текущей версии extension_20261106_v2 эта ошибка исправлена до нового обучения; старые показатели для градации недействительны.',
          '', '## Покрытие всех 16 значений','',
          'Числа ниже относятся только к дополнительной точной разметке этого эксперимента, а не к старым широким группам RF100. Ноль означает отсутствие подходящей точной разметки; это не нулевая вероятность типа.',
          '', '| Тип | Обучение, ROI | Контроль, положительные ROI | Статус |','|---|---:|---:|---|']
 for v in status:
  text.append(f"| {v['type']} | {v['new_positive_roi_support']['train']} | {v['new_positive_roi_support']['test']} | {v['note']} |")
 text += ['', 'Для промышленного подключения требуются проверенные примеры каждого типа из нескольких скважин, контрольные положительные примеры и контрпримеры похожих структур, независимая проверка всех 16 типов, калибровка порога и отчёт об отсутствии ухудшения других моделей. Даже 5/3/3 примера в проверках кода — лишь минимальный технический барьер, а не доказательство достаточности датасета.',
          '', '## Сохранность и проверки','',
          '* SHA256 классификатора пород: 0f980c18cbd42db9f2fb3fdf3b5b50f5ca1cc7a21df6a8573d9c564f830f1ed0 — без изменений.',
          '* SHA256 исходной CNN слоистости: 57b19deec05489e8766ee8084a327203deafb7c0a31d1af31956979dd0cc67d5 — без изменений.',
          '* SHA256 best.pt YOLO: 049ea5dc5d4018b6dafc3be6a90d6e6d2d5f664e225fcca12ba78d24c7e8358f — без изменений.',
          '* Прежние логиты CNN до и после обучения дополнительного модуля совпали побитово на контрольном пакете из 14 ROI.',
          '* 63 теста прошли; native_settings_unchanged=True. Проверены неизвестные метки, запрет вывода отсутствия градации из Massive, градиенты по неизвестным свойствам, утечки групп/дубликатов, закрытие автоматического подключения, прежние шаблоны и классификаторы.',
          '* Копия на E не изменялась. Автоматическое подключение дополнительного модуля выключено; добавлен NOT_FOR_PRODUCTION.txt.',
          '', '## Файлы','',
          '* open_datasets/bedding_20261106/expansion/dataset.json — пары с исходной подписью, координатами ROI, группой источника и контрольными суммами.',
          '* open_datasets/bedding_20261106/expansion/crops — точные вырезки, прошедшие визуальную сверку.',
          '* open_datasets/bedding_20261106/expansion/exported_contact_sheet.jpg — просмотр всех 16 вырезок.',
          '* models/bedding/extension_20261106_v2 — отдельные веса, параметры, предсказания и отчёт сохранности.',
          '* research_reports/bedding_20261106_expansion/coverage_16.json — полный список пробелов по 16 типам.',
          '', '## Источники','']
 for v in data['sources'].values():text.append('* '+v['citation']+' — '+v['url']+'; '+v['license'])
 text+=['* Мамадалиев А.Н., УГГУ, 2013, с.73–74 — '+urls['ural_cosowavy.pdf']+'; материал для проверки, в обучение не допущен.','']
 (out/'RESULTS_RU.md').write_text('\n'.join(text),encoding='utf8')
 print(str(out/'RESULTS_RU.md'))

if __name__=='__main__':main()
