"""Create a compact local review page for Kootenai River photos and log text."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "data/external_raw/usgs_kootenai_river_2011"


def main() -> None:
    links = json.loads((ROOT / "photo_link_index.json").read_text(encoding="utf-8"))
    downloads = json.loads((ROOT / "photo_download_inventory.json").read_text(encoding="utf-8"))
    depth_review = json.loads((ROOT / "depth_review_candidates.json").read_text(encoding="utf-8"))
    by_url = {photo["url"]: photo for photo in downloads["photos"]}
    depth_by_page_url = {(page["pdf_page"], photo["url"]): photo
                         for page in depth_review["pages"] for photo in page["linked_photos"]}
    cards = []
    photo_pages = 0
    for page in links["pages"]:
        if not page["core_id"]:
            continue
        available = [by_url[url] for url in page["photo_links"] if by_url[url]["status"] == "downloaded"]
        if available:
            photo_pages += 1
        link_items = []
        for ordinal, url in enumerate(page["photo_links"], 1):
            record = by_url[url]
            label = html.escape(Path(url).name)
            estimate = depth_by_page_url.get((page["pdf_page"], url), {}).get("layout_estimate_m")
            depth_note = (f" · примерно {estimate[0]:.2f}–{estimate[1]:.2f} м по расположению ссылки"
                          if estimate else " · глубина по ссылке не определена")
            if record["status"] == "downloaded":
                correction = " · URL исправлен по официальному хосту" if record.get("source_link_corrected") else ""
                link_items.append(f"<li>{ordinal:02d}. <a href='{html.escape(record['local_file'])}'>{label}</a>"
                                  f" · {record['size_px'][0]}×{record['size_px'][1]}{depth_note}{correction}</li>")
            else:
                link_items.append(f"<li>{ordinal:02d}. {label} · недоступен (404)</li>")
        first_image = (f"<a href='{html.escape(available[0]['local_file'])}'>"
                       f"<img src='{html.escape(available[0]['local_file'])}' alt='Фото керна {html.escape(page['core_id'])}'></a>"
                       if available else "<p>Фото по ссылкам на этой странице недоступны.</p>")
        cards.append(f"<section><h2>{html.escape(page['core_id'])} · стр. {page['pdf_page']}</h2>"
                     f"<p>Доступно фото: {len(available)} из {len(page['photo_links'])}. "
                     "Совпадение фото и текста пока только по странице керна, не по глубинному интервалу.</p>"
                     + first_image + "<details><summary>Все фото этого керна</summary><ol>"
                     + "".join(link_items) + "</ol></details>"
                     + "<details><summary>Исходный текст журнала</summary><pre>"
                     + html.escape(page["source_text_en"]) + "</pre></details></section>")
    document = ("<!doctype html><html lang='ru'><meta charset='utf-8'><title>USGS Kootenai River — керн и журналы</title>"
                "<style>body{font:15px system-ui;max-width:1300px;margin:2rem auto;background:#f4f6f9;color:#223}"
                "section{background:white;border:1px solid #ccd;padding:1.3rem;margin:1.3rem 0}"
                "img{max-width:100%;max-height:450px}li{margin:.3rem 0}pre{white-space:pre-wrap;line-height:1.3}"
                "details{margin:1rem 0}summary{cursor:pointer;font-weight:600}</style>"
                "<h1>USGS Kootenai River: фотографии и литологические журналы</h1>"
                "<p>Источник: USGS SIR 2011-5006, DOI 10.3133/sir20115006. В приложении A журналы содержат "
                "описания и ссылки на отдельные фото. Страница не определяет точный интервал каждого снимка и "
                "не создаёт обучающие подписи. Плёнка и состояние образца могут скрывать диагностические признаки.</p>"
                + "".join(cards) + "</html>")
    (ROOT / "review_index.html").write_text(document, encoding="utf-8")
    summary = {"core_log_pages": len(cards), "pages_with_available_photos": photo_pages,
               "unique_photo_links": downloads["summary"]["unique_links"],
               "photos_downloaded": downloads["summary"]["downloaded"],
               "broken_links": downloads["summary"]["failed"], "approved_training_pairs": 0}
    (ROOT / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
