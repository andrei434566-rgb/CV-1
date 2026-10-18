"""Check the downloaded USGS DS542 source triads without creating labels."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data/external_raw/usgs_chandeleur_ds542"


def main() -> None:
    source = json.loads((DEST / "download_inventory.json").read_text(encoding="utf-8"))
    inspected = []
    errors = []
    for record in source["records"]:
        well_id = record["well_id"]
        assets = record.get("local_assets", {})
        if set(assets) != {"photo_pdf", "description_jpg", "digitized_log_txt"}:
            errors.append({"well_id": well_id, "reason": "missing source asset category"})
            continue
        if any(asset["status"] == "error" for asset in assets.values()):
            errors.append({"well_id": well_id, "reason": "download failed"})
            continue
        try:
            paths = {key: ROOT / asset["path"] for key, asset in assets.items()}
            with paths["photo_pdf"].open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    raise ValueError("photo is not PDF")
            page_count = len(PdfReader(str(paths["photo_pdf"]), strict=False).pages)
            with Image.open(paths["description_jpg"]) as image:
                if image.format != "JPEG":
                    raise ValueError("description is not JPEG")
                width, height = image.size
                image.verify()
            raw_log = paths["digitized_log_txt"].read_text(encoding="latin-1")
            rows = list(csv.DictReader(raw_log.splitlines(), delimiter="\t"))
            if not rows or "Depth_start" not in rows[0] or "Depth_end" not in rows[0]:
                raise ValueError("digitized log lacks depth columns")
            depths = []
            for row in rows:
                for key in ("Depth_start", "Depth_end"):
                    try:
                        depths.append(float(row[key]))
                    except (TypeError, ValueError):
                        pass
            if not depths:
                raise ValueError("digitized log has no numeric depths")
            inspected.append({"well_id": well_id, "photo_pages": page_count,
                              "description_pixels": [width, height],
                              "digital_log_rows": len(rows),
                              "log_depth_range_raw": [min(depths), max(depths)],
                              "training_approved": False})
        except Exception as exc:  # Record per-source faults instead of aborting the audit.
            error = {"well_id": well_id, "reason": str(exc)}
            if well_id == "07SCC44CI":
                error["source_recheck"] = (
                    "Official PDF was re-downloaded 2026-10-13; SHA-256 matched the first copy "
                    "(864329b3aaea5b0b44978ad92f0ec21b240c7801cd76bf76ca53829863189c34). "
                    "Both pypdf and Poppler reject its page tree. Exclude this photo."
                )
            errors.append(error)
    result = {
        "source_id": "usgs_chandeleur_ds542",
        "core_count": len(source["records"]),
        "inspected_count": len(inspected),
        "error_count": len(errors),
        "photo_page_total": sum(row["photo_pages"] for row in inspected),
        "by_site": dict(Counter(row["well_id"][-2:] for row in inspected)),
        "training_approved": False,
        "reason": "This audit checks file integrity and source ID; page depths and descriptions still need alignment",
        "cores": inspected,
        "errors": errors,
    }
    (DEST / "content_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    by_id = {row["well_id"]: row for row in inspected}
    table_rows = []
    for record in source["records"]:
        well_id = record["well_id"]
        details = by_id.get(well_id)
        state = "Требуется разметка интервалов" if details else "Ошибка PDF; исключён из кандидатов"
        links = []
        for key, title in (("photo_pdf", "Фото PDF"),
                           ("description_jpg", "Лист описания"),
                           ("digitized_log_txt", "Журнал TXT")):
            asset = record.get("local_assets", {}).get(key)
            if asset and "path" in asset:
                target = (ROOT / asset["path"]).relative_to(DEST).as_posix()
                links.append(f'<a href="{html.escape(target, quote=True)}">{title}</a>')
        table_rows.append(
            f'<tr><td>{html.escape(well_id)}</td><td>{details["photo_pages"] if details else "ошибка"}</td>'
            f'<td>{details["digital_log_rows"] if details else "—"}</td>'
            f'<td>{" · ".join(links)}</td><td>{state}</td></tr>'
        )
    page = """<!doctype html><html lang="ru"><meta charset="utf-8"><title>USGS DS542 — проверка керна</title>
<style>body{font:15px/1.5 system-ui;margin:2rem;max-width:1250px;color:#17212b}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd6de;padding:.45rem;text-align:left}
thead{position:sticky;top:0;background:#eaf2f6}a{color:#075a91}</style>
<h1>USGS DS542 — исходные материалы для разметки</h1>
<p>Фото, лист описания и журнал относятся к одному ID керна. Для каждой фотостраницы нужно отдельно проверить глубину, наблюдаемые признаки, русский текст и условия использования. Автоматического допуска в обучение нет.</p>
<p>Официальные <a href="https://pubs.usgs.gov/ds/542/HTML/Data_Table.html">таблица данных</a> и <a href="https://pubs.usgs.gov/ds/542/HTML/methods.html">методика</a>.</p>
<table><thead><tr><th>Керн</th><th>Страниц фото</th><th>Строк журнала</th><th>Исходные файлы</th><th>Состояние</th></tr></thead><tbody>
""" + "\n".join(table_rows) + "</tbody></table></html>"
    (DEST / "review_index.html").write_text(page, encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("core_count", "inspected_count", "error_count", "photo_page_total")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
