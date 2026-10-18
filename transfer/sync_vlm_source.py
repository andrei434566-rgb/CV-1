"""Copy versioned VLM code into the local model folder without touching weights."""

from __future__ import annotations

import shutil
from pathlib import Path


APP = Path(__file__).resolve().parents[1]
SOURCE = APP / "vlm_source"
DESTINATION = APP.parent / "vlm_core_v1"
SUFFIXES = {".py", ".ps1", ".md", ".txt", ".json", ".js", ".mjs", ".html"}


def main() -> None:
    if not SOURCE.is_dir():
        raise SystemExit(f"Не найдены исходники VLM: {SOURCE}")
    if not DESTINATION.is_dir():
        raise SystemExit("Сначала установите локальную VLM: transfer/install_cv1.ps1")

    count = 0
    for source in SOURCE.rglob("*"):
        if not source.is_file() or source.is_symlink() or source.suffix.lower() not in SUFFIXES:
            continue
        relative = source.relative_to(SOURCE)
        destination = DESTINATION / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise SystemExit(f"Отказ перезаписывать ссылку: {destination}")
        if destination.is_file() and destination.read_bytes() == source.read_bytes():
            continue
        shutil.copy2(source, destination)
        count += 1
    print(f"Исходники VLM обновлены: {count} файлов. Веса и данные не затронуты.")


if __name__ == "__main__":
    main()
