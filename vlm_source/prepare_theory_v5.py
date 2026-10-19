"""Build a source-attributed Russian theory supplement and independent holdout.

Answers are original paraphrases of the cited methods. They are text knowledge,
not photo-derived labels. No held-out card is included in the training file.
"""

from __future__ import annotations

import json
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = (SOURCE_ROOT if SOURCE_ROOT.name == "vlm_core_v1"
        else SOURCE_ROOT.parent.parent / "vlm_core_v1")
KNOWLEDGE = ROOT / "data" / "knowledge"
APP_THEORY = ((SOURCE_ROOT.parent if SOURCE_ROOT.name == "vlm_source"
               else ROOT.parent / "kern_analyzer_app") / "app" / "resources" / "theory")
SOURCES = {
    "iodp329": {"title": "IODP Expedition 329, Methods: visual core descriptions", "url": "https://publications.iodp.org/proceedings/329/102/102_2.htm"},
    "iodp381": {"title": "IODP Expedition 381, Methods: lithostratigraphy and drilling disturbance", "url": "https://publications.iodp.org/proceedings/381/102/381_102.html"},
    "iodp395": {"title": "IODP Expedition 395, Methods: core imaging and drilling disturbance", "url": "https://publications.iodp.org/proceedings/395/102/395_102.html"},
    "bre": {"title": "Большая российская энциклопедия: Обломочные горные породы", "url": "https://bigenc.ru/c/oblomochnye-gornye-porody-312eed"},
    "usgs_log": {"title": "USGS: Standardized method for logging drill core", "url": "https://pubs.usgs.gov/publication/tm5E1/full"},
}


def card(identifier: str, topic: str, question: str, alternate: str,
         answer: str, sources: str, *, holdout: bool = False) -> dict:
    return {"id": f"v5_{identifier}", "topic": topic, "question": question,
            "aliases": [alternate], "answer": answer,
            "sources": sources.split(), "holdout": holdout}


CARDS = [
    card("bio_homogeneous", "bioturbation", "Если керн однородный и ходы не видны, биотурбации точно нет?", "Почему нельзя поставить нулевую биотурбацию по однородному керну?", "Нет. Полная переработка осадка организмами тоже может дать однородный вид. Запиши «явных ходов не наблюдается» и оцени интенсивность только там, где первичная слоистость или следы позволяют это сделать.", "iodp329"),
    card("bio_index", "bioturbation", "Что означает ихнофабричный индекс и одинакова ли его шкала во всех публикациях?", "Какая шкала нужна для оценки биотурбации керна?", "Ихнофабричный индекс описывает видимую степень биогенной переработки и сохранность первичной текстуры. Укажи методику вместе с баллом: экспедиция IODP 329 применяла шкалу 1–5, а IODP 381 — 0–6; число без названия шкалы неоднозначно.", "iodp329 iodp381"),
    card("bio_percent", "bioturbation", "Можно ли определить процент биотурбации по одному узкому фрагменту фото?", "Как осторожно оценивать интенсивность биотурбации по керну?", "Оцени видимую долю переработанного интервала в выбранной шкале и запиши, какой участок осмотрен. Узкий или смазанный фрагмент может не представлять весь интервал; при неоднозначности опиши ходы и нарушенную слоистость без численного балла.", "iodp329 iodp381"),
    card("bio_burrows", "bioturbation", "Какие признаки ходов организмов в керне записывать отдельно от степени биотурбации?", "Как документировать биогенные ходы?", "Запиши форму и ориентацию ходов, их размеры, частоту, границы и отличие заполнения от вмещающей породы, а также какие первичные слойки сохранились. Интенсивность переработки и описание самих структур — разные наблюдения.", "iodp329 usgs_log"),
    card("biscuiting", "drilling", "Что такое biscuiting в керне и почему оно может имитировать слоистость?", "Повторяющиеся диски керна — осадочные пласты?", "Biscuiting — распад керна при вращательном бурении на примерно повторяющиеся диски. Между ними могут попасть буровой раствор и перетёртый материал, образуя ложные прослои. Проверь повторяемость дисков, смятие кромок и связь границ с внутренней слоистостью.", "iodp381 iodp395"),
    card("fall_in", "drilling", "Что означает fall-in в верхней части керна?", "Можно ли обломки у верха керна уверенно привязать к глубине отбора?", "Fall-in — обломки, попавшие в верхнюю часть керна из вышележащих стенок скважины при бурении. Если это возможно, пометь материал как потенциально переотложенный при отборе и не используй его без проверки для глубинной фациальной привязки.", "iodp395"),
    card("soupy", "drilling", "Что означает soupy для сохранности текстуры мягкого осадка?", "Если осадок в трубке разжижен, можно ли описывать первичную слоистость?", "Soupy описывает сильно водонасыщенный, разжиженный интервал, где исходная слоистость утрачена при отборе. Отметь нарушение и не восстанавливай по такому виду первичные контакты и мощности прослоев.", "iodp381 iodp395"),
    card("void", "drilling", "Как описать пустой промежуток в колонке керна?", "Можно ли пустоту между кусками керна назвать фацией?", "Пустой промежуток пометь как отсутствие кернового материала с интервалом и причиной, если она известна. Не заполняй его выдуманным литотипом или фацией; соседние куски сами по себе не доказывают состав утраченного участка.", "iodp395 usgs_log"),
    card("smear", "method", "Почему поверхность распила может скрывать тонкую слоистость?", "Смазанная поверхность керна означает массивную текстуру?", "При раскалывании и обработке мягкий материал может размазываться по срезу и закрывать тонкие структуры. Запиши качество поверхности и проверь очищенный участок или другой срез, прежде чем считать породу массивной.", "iodp329 iodp381"),
    card("color_oxidation", "method", "Почему цвет керна после хранения может отличаться от цвета свежего среза?", "Можно ли старое фото считать точным цветом свежего керна?", "Окисление и высыхание после извлечения меняют оттенок осадка; освещение и баланс белого добавляют различия. Фиксируй дату и состояние поверхности, а важный цветовой признак сверяй со свежим срезом и эталоном.", "iodp381 iodp395"),
    card("size_schemes", "classification", "Почему в справочниках граница песка и алеврита бывает 0,05 и 0,063 мм?", "Как совместить русскую и шкалу Вентворта по песку?", "Это разные принятые гранулометрические шкалы. Российская БРЭ приводит для песка нижнюю границу 0,05 мм; в описаниях IODP по Вентворту используется 0,063 мм. В таблице укажи выбранную шкалу и не смешивай пороги внутри одного проекта.", "bre iodp381"),
    card("clay_size_mineral", "classification", "Глинистая размерность частиц доказывает наличие глинистых минералов?", "Чем глинистая фракция отличается от минералов группы глин?", "Нет. Размер частиц и минеральный состав — разные признаки: в IODP 329 слово clay по размерности не означает автоматически иллит или смектит. Для минерального названия нужны микроскопия или анализ состава.", "iodp329"),
    card("lamina_cm", "structures", "Когда тонкий прослой называют ламиной, а когда слоем?", "Как записать толщину ламинации?", "В методиках IODP 329 и 381 ламины обычно тоньше 1 см; более мощные единицы описывают как слои. Лучше запиши фактическую толщину в миллиметрах или сантиметрах, чтобы определение было проверяемым.", "iodp329 iodp381"),
    card("visual_grains", "method", "Можно ли по общему снимку кернового ящика оценить сортировку и окатанность мелкого песка?", "Когда нужен шлиф для оценки формы мелких зерен?", "Для мелких зерен обычный снимок часто не даёт разрешения. IODP 329 определяла форму и сортировку на препаратах и шлифах при большем увеличении. По ящику укажи только действительно видимые крупные компоненты и отметь ограничение.", "iodp329"),
    card("lithification", "classification", "Может ли степень литификации измениться после отбора керна?", "Почему рыхлость после высыхания не всегда исходное свойство?", "Да. В IODP 381 отмечено, что материал может стать твёрже при высыхании или изменении температуры. Указывай состояние во время осмотра и не выводи исходную степень цементации только из позднего фото.", "iodp381"),
    card("photo_observation", "method", "Что должно идти первым в описании спорной фотографии керна?", "Как отделять наблюдение от предположения о фации?", "Сначала запиши видимые цвет, границы, зернистость, слоистость, включения и сохранность поверхности. Затем отдельно предложи литотип и возможную фацию с причиной и уверенностью; то, что нельзя различить на снимке, оставь неустановленным.", "usgs_log iodp329"),
    card("pseudo_beds", "drilling", "Может ли тёмная полоска между дисками керна быть буровым шламом?", "Как отличить буровую псевдопрослойку между дисками?", "Да. При biscuiting между дисками может внедриться буровой раствор или подвижный осадок. Оцени смятые края, одинаковый шаг дисков и отсутствие продолжения полосы внутри целых кусков; одной окраски для диагноза недостаточно.", "iodp381", holdout=True),
    card("bio_no_trace", "bioturbation", "Если на фото нет отдельного хода, можно ли считать керн биотурбированным?", "Отсутствие заметных ходов доказывает сохранность первичной структуры?", "Не всегда. Сильное перемешивание может уничтожить узнаваемые ходы и слоистость. Если доказательств нет, зафиксируй «явные следы не обнаружены», а происхождение однородности оставь открытым.", "iodp329", holdout=True),
    card("redox_color", "method", "Как поступить, если два снимка одного керна имеют разный оттенок?", "Почему цвет фото до и после хранения не совпал?", "Сверь освещение, баланс белого, влажность, свежесть среза и возможное окисление. В итоговом описании укажи состояние и источник наблюдения; не объясняй различие фацией без других признаков.", "iodp381 iodp395", holdout=True),
    card("russian_boundary", "classification", "Алеврит 0,055 мм по какой шкале может попасть в песок?", "Можно ли назвать 0,055 мм одинаково по всем шкалам?", "При границе 0,05 мм в российской схеме это песчаная размерность, а при границе 0,063 мм по Вентворту — алевритовая. Укажи принятую шкалу и метод измерения.", "bre iodp381", holdout=True),
    card("top_debris", "drilling", "У верха рейса лежат чужеродные обломки. Их можно включить в фациальную последовательность?", "Что проверить для обломков, упавших в керноприёмник?", "Сначала исключи fall-in из вышележащих стенок скважины. Пока происхождение и глубина не подтверждены, пометь обломки как сомнительные и не используй их как надёжную фациальную метку.", "iodp395", holdout=True),
]


def row(item: dict, question: str, suffix: str, approved: bool) -> dict:
    return {
        "conversations": [{"from": "human", "value": question},
                          {"from": "gpt", "value": item["answer"]}],
        "metadata": {
            "record_id": item["id"] + suffix,
            "record_type": "text_only_knowledge", "topic": item["topic"],
            "target_language": "ru", "training_approved": approved,
            "geology_approved": False,
            "review_status": "source_checked_original_russian_paraphrase_not_geologist_reviewed",
            "source_group": "core_theory_v5",
            "sources": [{"source_id": sid, **SOURCES[sid]} for sid in item["sources"]],
            "scope": "Text-only geology theory; not a photo label or facies ground truth.",
        },
    }


def main() -> None:
    if len({item["id"] for item in CARDS}) != len(CARDS):
        raise ValueError("Duplicate theory card ID")
    if any(not item["answer"] or any(sid not in SOURCES for sid in item["sources"])
           for item in CARDS):
        raise ValueError("Incomplete theory card")
    KNOWLEDGE.mkdir(parents=True, exist_ok=True)
    APP_THEORY.mkdir(parents=True, exist_ok=True)
    train = []
    holdout = []
    for item in CARDS:
        if item["holdout"]:
            holdout.append(row(item, item["question"], "", False))
        else:
            train.extend((row(item, item["question"], "_a", True),
                          row(item, item["aliases"][0], "_b", True)))
    previous = [json.loads(line) for line in
                (KNOWLEDGE / "theory_training_v4.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    previous_holdout = [json.loads(line) for line in
                        (KNOWLEDGE / "theory_holdout_v4.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()]
    combined_train = previous + train
    combined_holdout = previous_holdout + holdout
    for name, records in (("theory_training_v5.jsonl", combined_train),
                          ("theory_holdout_v5.jsonl", combined_holdout)):
        (KNOWLEDGE / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                            for x in records), encoding="utf-8")
    (KNOWLEDGE / "theory_cards_v5.json").write_text(
        json.dumps({"sources": SOURCES, "cards": CARDS}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    previous_app = json.loads((APP_THEORY / "theory_cards_v4.json").read_text(encoding="utf-8"))
    (APP_THEORY / "theory_cards_v5.json").write_text(
        json.dumps({"sources": {**previous_app["sources"], **SOURCES},
                    "cards": [*previous_app["cards"], *CARDS]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    for topic in sorted({item["topic"] for item in CARDS}):
        lines = [f"# Справочник: {topic}", "", "Теория по источникам; не метки фотографий.", ""]
        for item in CARDS:
            if item["topic"] == topic:
                lines.extend([f"## {item['question']}", "", item["answer"], ""])
        content = "\n".join(lines) + "\n"
        (KNOWLEDGE / f"theory_{topic}_v5_ru.md").write_text(content, encoding="utf-8")
        (APP_THEORY / f"theory_{topic}_v5_ru.md").write_text(content, encoding="utf-8")
    print(json.dumps({"new_cards": len(CARDS), "new_training_rows": len(train),
                      "combined_training_rows": len(combined_train),
                      "combined_holdout_rows": len(combined_holdout)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
