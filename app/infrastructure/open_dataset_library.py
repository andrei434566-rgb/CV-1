"""Persistent local library for external sources and their prepared datasets."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.infrastructure.open_dataset_sources import discover_local_artifacts, get_source, list_sources


def read_state(root: Path, source_id: str) -> dict:
    get_source(source_id)
    path = Path(root) / source_id / "library.json"
    if not path.is_file():
        return {"source_id": source_id, "local_files": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("source_id") != source_id:
        raise ValueError("Повреждена запись библиотеки учебных материалов")
    return value


def write_state(root: Path, source_id: str, state: dict) -> None:
    get_source(source_id)
    folder = Path(root) / source_id
    folder.mkdir(parents=True, exist_ok=True)
    state = {**state, "source_id": source_id, "updated_at": datetime.now().isoformat(timespec="seconds")}
    temporary = folder / ("library." + uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(folder / "library.json")


def available_files(root: Path, source_id: str, app_root: Path) -> list[Path]:
    state = read_state(root, source_id)
    paths = [Path(value) for value in state.get("local_files", [])]
    paths.extend(discover_local_artifacts(source_id, Path(root), app_root))
    # The download API writes into this source's archives directory.
    archives = Path(root) / source_id / "archives"
    if archives.is_dir():
        paths.extend(path for path in archives.iterdir() if path.is_file()
                     and not path.name.endswith((".part", ".json", ".tmp")))
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file() and not path.name.endswith((".part", ".tmp")):
            unique.setdefault(str(path.resolve()).casefold(), path.resolve())
    return list(unique.values())


def register_files(root: Path, source_id: str, paths: list[Path]) -> dict:
    state = read_state(root, source_id)
    existing = {str(Path(value).resolve()).casefold(): str(Path(value).resolve())
                for value in state.get("local_files", []) if Path(value).is_file()}
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file() or path.name.endswith((".part", ".tmp")):
            raise ValueError(f"Файл не готов к импорту: {path.name}")
        existing[str(path).casefold()] = str(path)
    state["local_files"] = list(existing.values())
    fingerprints = [{"path": value, "bytes": Path(value).stat().st_size,
                     "mtime_ns": Path(value).stat().st_mtime_ns} for value in sorted(existing.values())]
    if state.get("input_fingerprints") != fingerprints:
        state.pop("preparation", None)
    state["input_fingerprints"] = fingerprints
    write_state(root, source_id, state)
    return state


def prepare_source(root: Path, source_id: str, paths: list[Path], *, progress=None, cancel=None) -> dict:
    if not paths:
        raise ValueError("Сначала скачайте материалы или выберите уже имеющиеся архивы")
    destination = (Path(root) / source_id / "prepared"
                   / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}")
    def notify(event):
        if progress:
            progress(event if isinstance(event, dict) else {"message": str(event)})
    if source_id in {"ukcs", "iodp362"}:
        from app.infrastructure.open_dataset_intervals import prepare_interval_dataset
        result = prepare_interval_dataset(source_id, paths, destination, progress=notify, cancel=cancel)
    else:
        from app.infrastructure.open_dataset_import import prepare_dataset
        result = prepare_dataset(source_id, paths, destination, progress=notify, cancel=cancel)
    result = {**result, "output_dir": str(destination), "source_id": source_id}
    state = register_files(root, source_id, paths)
    state["preparation"] = result
    write_state(root, source_id, state)
    return result


def last_preparation(root: Path, source_id: str) -> dict | None:
    state = read_state(root, source_id)
    for fingerprint in state.get("input_fingerprints", []):
        path = Path(fingerprint["path"])
        if path.is_file() and (path.stat().st_size != fingerprint["bytes"]
                               or path.stat().st_mtime_ns != fingerprint["mtime_ns"]):
            return None
    value = state.get("preparation")
    if isinstance(value, dict):
        folder = Path(str(value.get("output_dir") or "__missing__"))
        # Prefer the self-contained copied run, even if its old location still exists.
        replacement = Path(root) / source_id / "prepared" / folder.name
        if replacement.is_dir() and replacement.resolve() != folder.resolve():
            previous = folder
            folder = replacement
            def relocated(path_value):
                if isinstance(path_value, dict):
                    return {key: relocated(item) for key, item in path_value.items()}
                if isinstance(path_value, list):
                    return [relocated(item) for item in path_value]
                if isinstance(path_value, str):
                    try:
                        return str(folder / Path(path_value).relative_to(previous))
                    except ValueError:
                        pass
                return path_value
            value = relocated(value)
            value["output_dir"] = str(folder)
        manifest = Path(str(value.get("manifest") or folder / "manifest.jsonl"))
        if folder.is_dir() and manifest.is_file() and any((folder / name).is_file() for name in ("summary.json", "report.json")):
            return value
    return None


def identify_archive(path: Path) -> str | None:
    name = Path(path).name.casefold()
    if name in {"training_subset.zip", "validation_subset.zip", "test_subset.zip"}:
        return "unibo"
    if name == "dcid.zip":
        return "dcid"
    if name in {"q204_data.zip", "q205_data.zip"}:
        return "ukcs"
    if name == "assets.zip":
        return "corebreakout"
    if name.startswith("362") or "u1480" in name or "u1481" in name:
        return "iodp362"
    if "b6rj29mrss" in name or "lithology" in name:
        return "mendeley"
    return None


def prepared_knowledge_folders(root: Path) -> list[Path]:
    """Only indexed source documents are searchable, never expert-review metadata."""
    folders = []
    for source in list_sources():
        try:
            prepared = last_preparation(root, source.id)
        except (OSError, ValueError, KeyError):
            continue
        if prepared and prepared.get("knowledge_dir"):
            folder = Path(prepared["knowledge_dir"])
            if folder.is_dir():
                folders.append(folder)
    return folders
