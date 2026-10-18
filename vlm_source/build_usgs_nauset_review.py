"""Create a source-faithful photo/log review page without assigning interval crops."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/external_raw/usgs_nauset_marsh"


def main() -> None:
    inventory = json.loads((DATA / "inventory.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (DATA / "source_log_rows_quarantine.jsonl")
            .read_text(encoding="utf-8").splitlines() if line]
    by_site = defaultdict(list)
    for row in rows:
        by_site[row["site"]].append(row)
    sections = []
    for site, photos in sorted(inventory["photo_composites"].items()):
        src = Path(photos["image_path"]).resolve().as_uri()
        table_rows = []
        for row in by_site[site]:
            depth = row["interval_cmbsf_original"] or "не указан отдельно"
            table_rows.append(
                f'<tr><td>{row["source_row_number"]}</td><td>{html.escape(depth)}</td>'
                f'<td>{html.escape(row["description_original_en"])}</td>'
                f'<td>{html.escape(row["depositional_environment_original_en"])}</td></tr>'
            )
        sections.append(
            f'<section><h2>{html.escape(site)}</h2><p>Полный снимок: '
            f'<a href="{html.escape(src, quote=True)}">открыть</a>. '
            f'Размер {photos["dimensions_px"][0]} × {photos["dimensions_px"][1]} px.</p>'
            f'<img src="{html.escape(src, quote=True)}" alt="Фото {html.escape(site)}">'
            '<table><thead><tr><th>Строка CSV</th><th>Интервал, см</th>'
            '<th>Описание источника</th><th>Обстановка источника</th></tr></thead><tbody>'
            + ''.join(table_rows) + '</tbody></table></section>'
        )
    output = DATA / "review.html"
    output.write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<title>USGS Nauset Marsh: фото и журнал</title>'
        '<style>body{font:16px system-ui;margin:24px auto;max-width:1200px;padding:0 18px}'
        'section{margin:40px 0;border-top:3px solid #345;padding-top:15px}'
        'img{display:block;width:100%;height:auto;border:1px solid #777}'
        'table{border-collapse:collapse;width:100%;margin-top:18px}'
        'td,th{border:1px solid #bbb;padding:6px;vertical-align:top;text-align:left}'
        'tr:nth-child(even){background:#f4f4f4}</style>'
        '<h1>USGS Nauset Marsh: исходные фото и описания</h1>'
        '<p>Соответствие установлено только по скважине. Строки не связаны с точными пикселями; '
        'пустой интервал CSV не восстанавливается по соседним строкам. '
        'Подтверждённых пар для обучения: 0. Описания и обстановки приведены дословно на английском.</p>'
        + ''.join(sections) + '</html>', encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
