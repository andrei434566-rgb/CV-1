"""Run read-only local file search without freezing the Qt window or loading Qwen."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.infrastructure.local_knowledge_search import search_folders, search_photo_files


class LocalSearchWorker(QThread):
    results_ready = Signal(object)
    search_failed = Signal(str)

    def __init__(self, query: str, folders: list[str], parent=None, *,
                 photo_only: bool = False, known_photos: list[tuple[str, str]] | None = None):
        super().__init__(parent)
        self.query = query
        self.folders = list(folders)
        self.photo_only = photo_only
        self.known_photos = list(known_photos or [])

    def run(self) -> None:
        try:
            stats: dict[str, int | bool] = {}
            if self.photo_only:
                hits = search_photo_files(
                    self.query, self.folders, self.known_photos, limit=8, stats=stats,
                )
            else:
                hits = search_folders(self.query, self.folders, limit=8, stats=stats)
            self.results_ready.emit({"hits": hits, "stats": stats})
        except Exception as exc:
            self.search_failed.emit(f"Не удалось завершить поиск: {exc}")
