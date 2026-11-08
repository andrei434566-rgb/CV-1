"""Report completed research results without changing active model selection."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.domain.rock_types import ROCK_PROFILES,training_status
from app.infrastructure.supervised_learning import classification_metrics

def common_metrics(logits,targets,rows,source):
    selected=[i for i,r in enumerate(rows) if r['source']==source]
    n=logits.shape[1];names=[str(i) for i in range(n)]
    result=classification_metrics(targets[selected].tolist(),logits[selected].argmax(1).tolist(),names)
    result['common_four_macro_f1']=sum(result['per_class'][i]['f1'] for i in (0,2,5,7))/4
    return result

def main():
    import torch
    folder=ROOT/'models/vision/rock_taxonomy_coal_20261105'
    card=json.loads((folder/'config.json').read_text(encoding='utf8'))
    parts=json.loads((folder/'dataset.json').read_text(encoding='utf8'))
    audit=json.loads((folder/'coal_audit.json').read_text(encoding='utf8'))
    best=json.loads((folder/'best_validation.json').read_text(encoding='utf8'))
    source_results=json.loads((folder/'test_by_source.json').read_text(encoding='utf8'))
    old=json.loads((ROOT/'models/vision/vision_20261104_143516/config.json').read_text(encoding='utf8'))
    m=source_results['mendeley'];common=sum(m['per_class'][i]['f1'] for i in (0,2,5,7))/4
    report={'model':folder.name,'sha256':card['model_sha256'],'generative':False,
            'noncommercial_only':True,'split_counts':{s:len(r) for s,r in parts.items()},
            'validation':card['validation'],'test':card['test'],'test_by_source':source_results,
            'common_mendeley_test':{'old_accuracy':old['test']['accuracy'],'new_accuracy':m['accuracy'],
                                    'old_macro_f1':old['test']['macro_f1'],'new_macro_f1':common},
            'selected_head_epoch':best['epoch'],'test_unseen_wells_demonstrated':False}
    out=ROOT/'research_reports/rock_taxonomy_20261105';out.mkdir(parents=True,exist_ok=True)
    (out/'efficientnet_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    text=['# Распознавание пород CV-1 · 5–6 ноября 2026',
          'Локальная EfficientNet-B0 без LLM/VLM. Пользователь подтвердил только некоммерческие исследования. '
          'Содержит данные DCID (CC BY-NC 4.0), Mendeley Lithology (CC BY 4.0), Wenyu Wang Coal (CC BY 4.0).',
          '## Что реально обучено',
          'Полное дообучение восьми исходных групп, затем замороженный CNN и новая линейная голова '
          'для девяти групп с повтором всех учебных примеров. Это не десятиклассовая модель всех запрошенных пород.',
          '| Тип из справочника | Состояние |','|---|---|']
    text+=['|'+p['name']+'|'+training_status(p,card)+'|' for p in ROCK_PROFILES]
    text+=['','Песчаник и алевролит имеют прямые исходные метки. Mudstone/shale шире русского «аргиллит». '
           'Уголь — экспериментальная переносимая текстурная категория, а не доказанная точность на керне. '
           'Широкая группа crystalline включает также железистые породы BIF, не устанавливает фундамент. '
           'Углистый алевролит и брекчии остаются отдельными отрицательными примерами для ваших целевых названий.',
           '## Разделение данных',
           f"Обучение: {len(parts['train'])}; контроль: {len(parts['val'])}; тест: {len(parts['test'])}. "
           'Для DCID использованы только оригинальные DCID-512-35, без DCID-7 и RWDA-дубликатов. '
           'Один файл с нарушенной CRC исключён. Скважины DCID неизвестны: контроль и тест разделены по изображениям, '
           'а не гарантированно по скважинам. Поэтому высокая оценка может отражать сходство фрагментов одного источника.',
           'Для угля варианты одного родительского фото не пересекают выборки: '
           +', '.join(s+': '+str(len(g))+' фото' for s,g in audit['parent_photo_groups'].items())+'. '
           'Нет идентификаторов физических образцов; независимость по образцам не установлена.',
           'Mendeley: прежние разные учебные, контрольные и тестовые скважины; тест повторно используется в исследовании. '
           'Контрольная выборка служила выбору эпохи, температуры и проверке сохранения прежних групп. '
           'Тест в выборе новой головы не участвовал.',
           '## Результат',
           f"Контроль macro-F1: {card['validation']['macro_f1']:.2%}. Тест macro-F1: {card['test']['macro_f1']:.2%}; accuracy: {card['test']['accuracy']:.2%}.",
           '| Исходная группа | Тестовых примеров | Precision | Recall | F1 |','|---|---:|---:|---:|---:|']
    text+=['|'+card['class_names_ru'][i]+'|'+str(r['support'])+f"|{r['precision']:.2%}|{r['recall']:.2%}|{r['f1']:.2%}|" for i,r in enumerate(card['test']['per_class'])]
    text+=['',f"На общих четырёх типах Mendeley: прежний ConvNeXt-Tiny — accuracy {old['test']['accuracy']:.2%}, "
           f"macro-F1 {old['test']['macro_f1']:.2%}; новый классификатор — accuracy {m['accuracy']:.2%}, macro-F1 {common:.2%}. "
           'Предсказания новых посторонних групп на этих снимках засчитаны ошибками, а не отброшены.',
           'Эти показатели не доказывают 95% на других скважинах и не относятся к распознаванию фаций.',
           '## В приложении',
           'Карточка участка → «Признаки пород и возможности модели…». Доступны все десять описаний, '
           'реальная поддержка модели, ограничения и источники по отдельной кнопке. Проценты рядом с реальными '
           'иллюстрациями исходных классов; это оценки классификатора, а не измеренная вероятность правильности. '
           'Геометрия YOLO, автоматический цвет и правки геолога сохраняются. Фации сами из этих меток не выводятся.',
           '## Воспроизводимость',
           '`scripts/train_rock_taxonomy.py --output <новая папка> --epochs 6 --batch 24 --workers 2 --noncommercial`\n\n'
           '`scripts/extend_rock_taxonomy_coal.py --base <первая модель> --output <новая папка>`\n\n'
           'Манифесты, SHA-256, оригинальные классы и группы родительских фото лежат в папках запусков на C:. '
           'Образцы и сырьё не включаются в архив рабочей модели. Копия на E: не изменялась.',
           'Источники: [DCID](https://github.com/JiayuLi1120/drill-core-image-dataset), '
           '[Mendeley Lithology](https://data.mendeley.com/datasets/b6rj29mrss/1), '
           '[Coal](https://zenodo.org/records/10702879).']
    formatted=[]
    for i,part in enumerate(text):
        if i and not (part.startswith('|') and text[i-1].startswith('|')):formatted.append('')
        formatted.append(part)
    (out/'EFFICIENTNET_RESULTS_RU.md').write_text('\n'.join(formatted)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k in ('model','split_counts','common_mendeley_test','selected_head_epoch')},ensure_ascii=False))

if __name__=='__main__':main()
