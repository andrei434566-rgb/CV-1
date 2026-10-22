"""Immediate Russian replies to simple conversational turns, without loading a VLM."""

from __future__ import annotations

import re


def _normalize(message: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^а-яёa-z0-9 ]+", " ", message.casefold())).strip()


def quick_reply(message: str, *, has_history: bool = False) -> str | None:
    """Handle only unambiguous short phrases; all substantive questions go to Qwen."""
    phrase = _normalize(message)
    if not phrase or len(phrase) > 100:
        return None
    if re.fullmatch(r"(?:(?:а|ну) )?(?:ты )?(?:работаешь|здесь|тут|на связи|доступен|готов)(?: ли)?", phrase):
        return "Да, я здесь. Могу обсудить керн, проверить выбранное фото, найти материалы в локальных папках и помочь с описанием фаций."
    if re.fullmatch(r"(?:привет|здравствуй|здравствуйте|добрый день|добрый вечер|доброе утро|хай)(?: как дела)?", phrase):
        return "Здравствуйте! Я на связи. Выберите фото или задайте вопрос — помогу с описанием керна и фаций."
    if re.fullmatch(r"(?:спасибо|благодарю|спс|ок спасибо|понял спасибо)", phrase):
        return "Пожалуйста. Если есть сомнительный слой или фото, покажите его — разберём признаки по шагам."
    if re.fullmatch(r"(?:(?:а|ну) )?(?:чем (?:ты )?(?:можешь )?помочь|как (?:ты )?можешь помочь|чем поможешь|что (?:ты )?умеешь|какие (?:у тебя )?возможности|расскажи о (?:своих )?возможностях|что (?:ты )?можешь(?: делать)?)(?: мне)?", phrase):
        return (
            "Могу обсудить признаки на выбранном фото, предложить черновик описания фации, "
            "объяснить литологию и обстановки осадконакопления, найти выдержки в локальных папках "
            "и помочь проверить итоговый Excel. Неуверенные выводы отмечу отдельно."
        )
    if re.fullmatch(r"(?:кто ты|расскажи о себе|представься|что ты за помощник)", phrase):
        return (
            "Я локальный ИИ-помощник Kern Analyzer. Отвечаю по-русски, могу учитывать выбранное фото "
            "и материалы из указанных папок. Геологические выводы по снимку предлагаю для проверки специалистом."
        )
    if phrase == "расскажи" and not has_history:
        return "О чём рассказать? Могу объяснить текстуры керна, признаки фаций, выбранное фото или работу с отчётом."
    return None


def needs_selected_image(message: str) -> bool:
    phrase = _normalize(message)
    if re.search(
        r"\b(?:фото|фотограф\w*|сним\w*|изображен\w*|видно|видишь|"
        r"что здесь|что на нем|что на ней|этот слой|эту фацию|этот керн|данный образец)\b",
        phrase,
    ):
        return True
    return bool(re.fullmatch(r"опиши(?: его| ее| это| керн| слой| фацию| образец)?", phrase))


def needs_local_knowledge(message: str) -> bool:
    phrase = _normalize(message)
    return bool(re.search(
        r"\b(?:керн\w*|фаци\w*|литолог\w*|пород\w*|осадк\w*|разрез\w*|слоист\w*|"
        r"текстур\w*|зернист\w*|песчан\w*|глин\w*|извест\w*|карбонат\w*|русел\w*|"
        r"диаген\w*|биотурб\w*|биоген\w*|ихно\w*|литотип\w*|контакт\w*|"
        r"буров\w*|окислен\w*|ламинац\w*|псевдопросло\w*|"
        r"алеврол\w*|аргилл\w*|долом\w*|"
        r"мергел\w*|конглом\w*|брекч\w*|псамм\w*|алеврит\w*|пелит\w*|"
        r"ангидр\w*|гипс\w*|кремнист\w*|глинист\w*|"
        r"турбидит\w*|микрит\w*|вакст\w*|пакст\w*|грейнст\w*|"
        r"аркоз\w*|граувак\w*|туф\w*|эвапорит\w*|флиш\w*|"
        r"сортиров\w*|окатан\w*|матриц\w*|цемент\w*|"
        r"справоч\w*|скваж\w*|месторожд\w*|"
        r"shale|mudstone|limestone|sandstone|siltstone|"
        r"conglomerate|breccia|bioturbation|ichnofabric|biscuiting|fall|soupy|"
        r"источник\w*|документ\w*|файл\w*|папк\w*)\b",
        phrase,
    ))


def wants_sources(message: str) -> bool:
    """Show source paths and links only when the question explicitly asks for them."""
    phrase = _normalize(message)
    return bool(re.search(
        r"\b(?:источник\w*|ссылк\w*|литератур\w*|библиограф\w*|"
        r"откуда (?:это|данные|информация|взял\w*)|по как\w* публикаци\w*)\b",
        phrase,
    ))


def wants_open_datasets(message: str) -> bool:
    phrase = _normalize(message)
    return bool(re.search(r"\b(?:открой|покажи|добавь|загрузи|скачай)\b", phrase)
                and re.search(r"\b(?:датасет\w*|учебн\w* материал\w*|открыты\w* набор\w*)\b", phrase))


def wants_folder_search(message: str) -> bool:
    """Send explicit file-search requests to the local index, not to the VLM."""
    phrase = _normalize(message)
    return bool(re.search(r"\b(?:найди|найдите|поищи|поищите|ищи|ищите|поиск|найти)\b", phrase))


def wants_photo_file(message: str) -> bool:
    """Distinguish a request for an existing photo file from image analysis."""
    phrase = _normalize(message)
    if not re.search(r"\b(?:фото|фотограф\w*|сним\w*|изображен\w*)\b", phrase):
        return False
    if re.search(r"\b(?:ссылк\w*|путь|адрес|файл\w*|скачать)\b", phrase):
        return True
    if re.search(r"\b(?:описан\w*|опиши|признак\w*|структур\w*)\b", phrase):
        return False
    return bool(re.search(
        r"\b(?:найди|найдите|найти|"
        r"поищи|поищите|ищи|ищите|где|дай|дайте|покажи|покажите|"
        r"пришли|пришлите|скинь|отправь|открой|откройте)\b",
        phrase,
    ))


def hide_unsolicited_sources(answer: str) -> str:
    """Remove citations generated by the VLM when the user did not ask for them."""
    answer = re.split(r"(?im)^\s*(?:источники|ссылки|литература|найденные источники)\s*:", answer, maxsplit=1)[0]
    answer = re.sub(r"(?i)https?://[^\s)\]>]+", "", answer)
    answer = re.sub(r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s)\]>]+", "", answer)
    return answer.strip()
