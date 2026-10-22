"""Public core datasets and bounded, resumable downloads (standard library only).

Downloading does not approve labels for training, extract archives, or run their
contents.  Byte counts are decimal bytes.  Catalog checksums identify upstream
artifacts, while licence notes retain restrictions on their constituent images.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_USER_AGENT = "KernAnalyzer-OpenDatasets/1.0 (+public-source-metadata)"
_CHUNK_BYTES = 1024 * 1024
DISK_MARGIN_BYTES = 1024 ** 3
_MAX_METADATA_BYTES = 8 * 1024 * 1024


class DatasetDownloadError(RuntimeError):
    """The artifact was not completed or could not be verified."""


class DownloadCancelled(DatasetDownloadError):
    """Cancellation preserves a resumable .part, never a completed artifact."""


def _safe_filename(name: str) -> str:
    if (not name or name in {".", ".."} or name[-1:] in {".", " "}
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.split(".", 1)[0].upper() in {
                "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }):
        raise ValueError("Недопустимое имя файла в метаданных источника")
    return name


def _https_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Источник должен использовать публичный HTTPS URL без учётных данных")
    return url


@dataclass(frozen=True)
class ArtifactDescriptor:
    dataset_id: str
    filename: str
    url: str
    expected_bytes: int | None = None
    checksum: str = ""
    checksum_algorithm: str = "sha256"
    metadata_url: str = ""
    verified: bool = False
    label_ru: str = ""

    def __post_init__(self):
        _safe_filename(self.filename)
        _https_url(self.url)
        if self.expected_bytes is not None and self.expected_bytes <= 0:
            raise ValueError("Размер файла должен быть положительным")
        if self.checksum_algorithm not in {"sha256", "md5"}:
            raise ValueError("Поддерживаются контрольные суммы SHA-256 и MD5")
        if self.checksum and not re.fullmatch(
                r"[0-9a-fA-F]{%d}" % (64 if self.checksum_algorithm == "sha256" else 32),
                self.checksum):
            raise ValueError("Недопустимая контрольная сумма")


@dataclass(frozen=True)
class DatasetSource:
    id: str
    name_ru: str
    roles: tuple[str, ...]
    primary_url: str
    license_id: str
    noncommercial: bool
    notes: str
    artifacts: tuple[ArtifactDescriptor, ...] = ()
    local_folder: str = ""
    metadata_url: str = ""


_COREBREAKOUT_API = "https://api.github.com/repos/rgmyr/corebreakout/releases/tags/v0.2"
_UNIBO = "https://amsacta.unibo.it/id/eprint/7308/"
_UKCS_API = "https://api.figshare.com/v2/articles/14265689"
_DCID_API = "https://huggingface.co/api/datasets/168sir/drill-core-image-dataset/tree/main?expand=true"
_MENDELEY = "https://data.mendeley.com/datasets/b6rj29mrss/1"
_MENDELEY_VERSIONS = "https://data.mendeley.com/public-api/datasets/b6rj29mrss/versions"
_IODP = "https://publications.iodp.org/proceedings/362/"
_IODP_VCD_API = "https://zenodo.org/api/records/3752077"
_IODP_IMAGES_API = "https://zenodo.org/api/records/3752105"


def _artifact(source, name, url, size=None, checksum="", algorithm="sha256", metadata="", label=""):
    return ArtifactDescriptor(source, name, url, size, checksum, algorithm, metadata, True, label)


_SOURCES = (
    DatasetSource(
        "corebreakout", "Столбики керна — CoreBreakout / BGS", ("columns",),
        "https://github.com/rgmyr/corebreakout/releases/tag/v0.2",
        "MIT (код); OGL-UK-3.0 (фото BGS)", False,
        "Фото ящиков и LabelMe-контуры col/tray. Метки столбиков не являются фациями. "
        "Сохранять авторство BGS/UKRI и условия OGL для фото; MIT кода не заменяет лицензию фото. "
        "Архив также содержит старые модели: загрузчик их не запускает.",
        (_artifact("corebreakout", "assets.zip",
                   "https://github.com/rgmyr/corebreakout/releases/download/v0.2/assets.zip",
                   763582851, metadata=_COREBREAKOUT_API, label="Фото и контуры столбиков"),),
        "corebreakout", _COREBREAKOUT_API,
    ),
    DatasetSource(
        "unibo", "Фации осадочного керна — Университет Болоньи", ("facies",),
        _UNIBO, "CC-BY-NC-SA-4.0", True,
        "Фото и маски шести исходных классов. Некоммерческое использование и ShareAlike. "
        "Исходные части сохраняются в паспорте; для обучения создаётся разделение по скважинам. "
        "Классы не заменяют наш справочник фаций.",
        tuple(_artifact("unibo", name, _UNIBO + str(index) + "/" + name, size,
                        metadata=_UNIBO, label=label)
              for index, name, size, label in (
                  (1, "Training_subset.zip", 617913868, "Обучающая часть"),
                  (2, "Validation_subset.zip", 105850345, "Проверочная часть"),
                  (3, "Test_subset.zip", 116620306, "Независимая тестовая часть"),
              )),
        "unibo_facies_7308", _UNIBO,
    ),
    DatasetSource(
        "ukcs", "Керн Северного моря — UKCS Q204/Q205", ("lithology", "facies"),
        "https://figshare.com/articles/dataset/UKCS_Q204_Q205_Subsurface_Data_products_for_Machine_Learning_Study/14265689",
        "MIT (запись Figshare); OGL-UK-3.0 (фото BGS)", False,
        "Глубинно привязанные изображения, LAS и исходные интервальные метки. "
        "Сохранять атрибуцию BGS/UKRI для фото; переводы классов требуют проверки геолога. "
        "NPY необходимо читать без allow_pickle.",
        (_artifact("ukcs", "Q204_data.zip", "https://ndownloader.figshare.com/files/27039890",
                   3534798588, "2957abe45d046c7206b4c8c55d769fceb6aa26eedf20f8f29ceb8f28211ecf41",
                   metadata=_UKCS_API, label="Фото, глубины и исходные метки"),),
        "ukcs_q204_q205_coremdlr", _UKCS_API,
    ),
    DatasetSource(
        "iodp362", "Фото и описания по глубине — IODP 362", ("description", "lithology"),
        _IODP + "362title.html", "CC0-1.0 (данные Zenodo); CC-BY-4.0 (публикация)", False,
        "Excel DESClogik и VCD по U1480/U1481. В VCD есть фото и интервальные наблюдения. "
        "Часть выводов подтверждена шлифами/XRD и не определяется только по фото. "
        "По умолчанию: таблицы VCD и небольшие фотоархивы U1480A/B/C (около 179 МБ). "
        "Остальные большие архивы фото подключаются отдельно.",
        (_artifact("iodp362", "362-DESCVCD.zip",
                   _IODP_VCD_API + "/files/362-DESCVCD.zip/content", 2558460,
                   "255e6add2f1b4b3df7be24391e0e5833", "md5", _IODP_VCD_API,
                   "Исходные Excel-описания VCD"),
         *(_artifact("iodp362", name, _IODP_IMAGES_API + "/files/" + name + "/content",
                     size, checksum, "md5", _IODP_IMAGES_API, "Фото " + name.split("-")[1])
           for name, size, checksum in (
               ("362-U1480A-LSIMG.zip", 48401606, "9c7ef68f0d995272bccef5f1b32960b8"),
               ("362-U1480B-LSIMG.zip", 60568652, "7d7ba97e3e8f0a1355c53567d885f044"),
               ("362-U1480C-LSIMG.zip", 67711796, "2ee798466d0bf1f9bf2e888266c1b539"),
           ))),
        "iodp362", _IODP + "datasets.html",
    ),
    DatasetSource(
        "dcid", "Распознавание пород — DCID", ("lithology",),
        "https://huggingface.co/datasets/168sir/drill-core-image-dataset",
        "CC-BY-NC-4.0", True,
        "Фрагменты 512×512, варианты на 7 и 35 литологических классов. "
        "Только некоммерческое использование. Классы пород не являются масками фаций или полными описаниями.",
        (_artifact("dcid", "DCID.zip",
                   "https://huggingface.co/datasets/168sir/drill-core-image-dataset/resolve/main/DCID.zip",
                   3819926174, "dd9672224c1da3aaa2d6b0c684deb67c1c9a76f1b26b40f86ee40ff13146d61f",
                   metadata=_DCID_API, label="Фрагменты и классы пород"),),
        "dcid", _DCID_API,
    ),
    DatasetSource(
        "mendeley", "Порода и посторонние предметы — Mendeley Lithology", ("lithology",),
        _MENDELEY, "CC-BY-4.0", False,
        "Фрагменты 2×2 см: sandstone, limestone, shale, garbage. "
        "Пустые лотки, дерево и металл полезны как отрицательные примеры. "
        "Исходные train/val/test пересекаются по скважинам — требуется новое разделение по скважинам.",
        (_artifact("mendeley", "b6rj29mrss-1.zip",
                   "https://data.mendeley.com/public-api/zip/b6rj29mrss/download/1",
                   1199024256, "6db325de3c829dd2c0ada6a1528d94479e845f8c82e2703a6f65479a44e2f912",
                   metadata=_MENDELEY, label="Четыре класса фрагментов"),),
        "mendeley_lithology", _MENDELEY_VERSIONS,
    ),
)
_ALIASES = {"ukcs_q204_q205": "ukcs", "mendeley_lithology": "mendeley",
            "unibo_facies": "unibo", "iodp_362": "iodp362"}


def list_sources() -> tuple[DatasetSource, ...]:
    return _SOURCES


def get_source(source_id: str) -> DatasetSource:
    canonical = _ALIASES.get(source_id, source_id)
    for source in _SOURCES:
        if source.id == canonical:
            return source
    raise KeyError(f"Неизвестный открытый набор: {source_id}")


def _cancelled(cancel) -> bool:
    if cancel is None:
        return False
    return bool(cancel.is_set() if hasattr(cancel, "is_set") else cancel())


def _check_cancel(cancel):
    if _cancelled(cancel):
        raise DownloadCancelled("Загрузка остановлена; незавершённый файл сохранён для продолжения")


def _open(url: str, *, headers=None, timeout=30, method="GET"):
    request_headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"}
    request_headers.update(headers or {})
    response = urlopen(Request(_https_url(url), headers=request_headers, method=method), timeout=timeout)
    if urlparse(response.geturl()).scheme != "https":
        response.close()
        raise DatasetDownloadError("Источник перенаправил загрузку на небезопасный протокол")
    return response


def _metadata_json(url: str, *, timeout=30, cancel=None):
    _check_cancel(cancel)
    with _open(url, timeout=timeout) as response:
        data = response.read(_MAX_METADATA_BYTES + 1)
    _check_cancel(cancel)
    if len(data) > _MAX_METADATA_BYTES:
        raise DatasetDownloadError("Слишком большой ответ API метаданных")
    return json.loads(data.decode("utf-8"))


def _head_size(artifact: ArtifactDescriptor, *, timeout=30, cancel=None) -> ArtifactDescriptor:
    _check_cancel(cancel)
    with _open(artifact.url, timeout=timeout, method="HEAD") as response:
        size = response.headers.get("Content-Length", "")
    _check_cancel(cancel)
    if not str(size).isdigit() or int(size) <= 0:
        raise DatasetDownloadError(f"Источник не сообщил точный размер {artifact.filename}")
    return replace(artifact, expected_bytes=int(size))


def discover_api_artifacts(source_id: str, *, timeout=30, cancel=None,
                           include_large_images=False) -> tuple[ArtifactDescriptor, ...]:
    """Refresh file identities from primary APIs; never download their bodies.

    Mendeley's whole-dataset ZIP uses its fixed version URL and previously
    verified SHA-256. Its versions endpoint is small; the full files endpoint
    exceeds 80 MB, so the latter is deliberately not fetched by the UI.
    """
    source = get_source(source_id)
    if source.id == "corebreakout":
        payload = _metadata_json(_COREBREAKOUT_API, timeout=timeout, cancel=cancel)
        return tuple(_artifact(source.id, entry["name"], entry["browser_download_url"], int(entry["size"]),
                               metadata=_COREBREAKOUT_API, label="Фото и контуры столбиков")
                     for entry in payload.get("assets", []) if entry.get("name") == "assets.zip")
    if source.id == "ukcs":
        payload = _metadata_json(_UKCS_API, timeout=timeout, cancel=cancel)
        return tuple(_artifact(source.id, entry["name"], entry["download_url"], int(entry["size"]),
                               entry.get("computed_md5", ""), "md5", _UKCS_API,
                               "Фото, глубины и исходные метки")
                     for entry in payload.get("files", []) if entry.get("name") == "Q204_data.zip")
    if source.id == "dcid":
        payload = _metadata_json(_DCID_API, timeout=timeout, cancel=cancel)
        return tuple(_artifact(source.id, "DCID.zip", source.artifacts[0].url, int(entry["size"]),
                               entry.get("lfs", {}).get("oid", ""), metadata=_DCID_API,
                               label="Фрагменты и классы пород")
                     for entry in payload if entry.get("path") == "DCID.zip")
    if source.id == "mendeley":
        payload = _metadata_json(_MENDELEY_VERSIONS, timeout=timeout, cancel=cancel)
        versions = payload if isinstance(payload, list) else [payload]
        if not any(entry.get("version") == 1 and entry.get("available") for entry in versions):
            raise DatasetDownloadError("Mendeley больше не подтверждает доступность версии 1")
        return source.artifacts
    if source.id == "iodp362":
        artifacts = []
        selected_names = {item.filename for item in source.artifacts}
        for record_id, label in ((3752077, "Исходные таблицы VCD"), (3752105, "Исходные фото секций")):
            api_url = f"https://zenodo.org/api/records/{record_id}"
            payload = _metadata_json(api_url, timeout=timeout, cancel=cancel)
            for entry in payload.get("files", []):
                name = entry.get("key", "")
                if not name.lower().endswith((".zip", ".csv", ".xlsx", ".tsv", ".pdf")):
                    continue
                if not include_large_images and name not in selected_names:
                    continue
                digest = entry.get("checksum", "")
                algorithm, checksum = digest.split(":", 1) if ":" in digest else ("sha256", "")
                artifacts.append(_artifact(source.id, name,
                    entry["links"].get("self") or entry["links"]["content"], int(entry["size"]),
                    checksum, algorithm, api_url, label))
        return tuple(artifacts)
    return tuple(_head_size(item, timeout=timeout, cancel=cancel) for item in source.artifacts)


def resolve_artifacts(source_id: str, *, timeout=30, cancel=None, refresh=False,
                      include_large_images=False) -> tuple[ArtifactDescriptor, ...]:
    source = get_source(source_id)
    if refresh or include_large_images:
        result = discover_api_artifacts(source.id, timeout=timeout, cancel=cancel,
                                        include_large_images=include_large_images)
    else:
        result = tuple(item if item.expected_bytes is not None else
                       _head_size(item, timeout=timeout, cancel=cancel) for item in source.artifacts)
    if not result:
        raise DatasetDownloadError(f"В метаданных {source.name_ru} не найдены ожидаемые файлы")
    return result


def discover_local_artifacts(source_id: str, storage_dir=None, app_root=None) -> list[Path]:
    """Find existing named archives only in explicit storage/project locations.

    No recursive scan of the user's disks and no implication of label approval.
    File integrity is checked by verify_artifact or by the download worker.
    """
    source = get_source(source_id)
    root = Path(app_root) if app_root else Path(__file__).resolve().parents[2]
    directories = [root / "data" / "open_datasets" / source.id / "archives",
                   root.parent / "vlm_core_v1" / "data" / "external_raw" / source.local_folder,
                   root / "vlm_core_v1" / "data" / "external_raw" / source.local_folder,
                   root / "data" / "external_raw" / source.local_folder]
    if storage_dir is not None:
        storage = Path(storage_dir)
        directories[:0] = [storage, storage / "archives", storage / source.id,
                           storage / source.id / "archives", storage / source.local_folder,
                           storage / "external_raw" / source.local_folder]
    names = {item.filename for item in source.artifacts}
    found = {}
    for directory in directories:
        for name in names:
            candidate = directory / name
            if candidate.is_file() and not candidate.is_symlink():
                resolved = candidate.resolve()
                found[str(resolved).casefold()] = resolved
    return sorted(found.values(), key=lambda path: str(path).casefold())


def verify_artifact(path, artifact: ArtifactDescriptor, *, cancel=None) -> bool:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        return False
    if artifact.expected_bytes is not None and path.stat().st_size != artifact.expected_bytes:
        return False
    if artifact.checksum:
        digest = hashlib.new(artifact.checksum_algorithm)
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK_BYTES):
                _check_cancel(cancel)
                digest.update(chunk)
        return digest.hexdigest().lower() == artifact.checksum.lower()
    return artifact.expected_bytes is not None


def _write_metadata(path: Path, payload: dict):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.is_symlink():
        raise DatasetDownloadError("Временный файл метаданных является ссылкой")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _disk_space(directory: Path, remaining: int):
    if shutil.disk_usage(directory).free < max(remaining, 0) + DISK_MARGIN_BYTES:
        raise DatasetDownloadError("Недостаточно места: нужен размер оставшейся загрузки и ещё 1 ГБ")


def download_artifact(artifact: ArtifactDescriptor, archives_dir, *, cancel=None,
                      progress: Callable[[dict], None] | None = None, timeout=30,
                      resume=True) -> Path:
    """Download with HTTP Range, exact byte count, checksum and atomic promotion.

    A cancelled/network-interrupted transfer remains ``filename.part`` with its
    source identity in ``filename.part.json``. Existing unrelated files are not
    overwritten. Unknown-length/chunked responses require a successful HEAD.
    """
    _check_cancel(cancel)
    if timeout <= 0:
        raise ValueError("Таймаут должен быть положительным")
    directory = Path(archives_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / _safe_filename(artifact.filename)
    part = directory / (artifact.filename + ".part")
    metadata_path = directory / (artifact.filename + ".part.json")
    for candidate in (destination, part, metadata_path):
        if candidate.is_symlink() or candidate.resolve().parent != directory:
            raise DatasetDownloadError("Путь загрузки выходит за выбранную папку или является ссылкой")
    if artifact.expected_bytes is None:
        artifact = _head_size(artifact, timeout=timeout, cancel=cancel)
    expected = artifact.expected_bytes
    assert expected is not None

    def report(count, status, resumed=False):
        if progress:
            progress({"dataset_id": artifact.dataset_id, "filename": artifact.filename,
                      "downloaded_bytes": count, "total_bytes": expected,
                      "resumed": resumed, "status": status})

    if destination.exists():
        report(destination.stat().st_size, "verifying")
        if not verify_artifact(destination, artifact, cancel=cancel):
            raise DatasetDownloadError("Файл с таким именем уже существует, но не прошёл проверку")
        report(expected, "complete")
        return destination
    identity = {"url": artifact.url, "expected_bytes": expected,
                "checksum": artifact.checksum.lower(), "algorithm": artifact.checksum_algorithm}
    metadata = {}
    offset = part.stat().st_size if part.exists() else 0
    if offset and resume:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DatasetDownloadError("Для частичного файла нет пригодных метаданных; начните заново") from exc
        if any(metadata.get(key) != value for key, value in identity.items()):
            raise DatasetDownloadError("Частичный файл относится к другой версии источника")
        if offset > expected:
            raise DatasetDownloadError("Частичный файл больше ожидаемого размера")
        if offset == expected:
            report(offset, "verifying", True)
            if not verify_artifact(part, artifact, cancel=cancel):
                raise DatasetDownloadError("Контрольная сумма частичного файла не совпала; начните заново")
            _check_cancel(cancel)
            os.replace(part, destination)
            metadata_path.unlink(missing_ok=True)
            report(expected, "complete", True)
            return destination
    elif not resume:
        offset = 0
    _disk_space(directory, expected - offset)
    headers = {}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        validator = metadata.get("etag") or metadata.get("last_modified")
        if validator:
            headers["If-Range"] = validator
        elif not artifact.checksum:
            # Without a digest or an upstream validator, a changed resource of
            # the same length could silently join bytes from two versions.
            offset = 0
            headers = {}
    try:
        _check_cancel(cancel)
        with _open(artifact.url, headers=headers, timeout=timeout) as response:
            status = response.status
            if response.headers.get("Content-Encoding", "identity").lower() not in {"", "identity"}:
                raise DatasetDownloadError("Сервер применил неожиданное сжатие потока")
            if status == 206:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match:
                    raise DatasetDownloadError("Сервер не подтвердил диапазон продолжения")
                start, end, total = map(int, match.groups())
                if start != offset or total != expected or end < start or end >= expected:
                    raise DatasetDownloadError("Диапазон или размер продолжения не совпадает с исходным файлом")
                if (offset and metadata.get("etag") and response.headers.get("ETag")
                        and metadata["etag"] != response.headers["ETag"]):
                    raise DatasetDownloadError("Содержимое источника изменилось во время продолжения")
                response_bytes = end - start + 1
            elif status == 200:
                offset = 0  # A server that ignores Range must be restarted, never appended.
                response_bytes = expected
            else:
                raise DatasetDownloadError(f"Неожиданный HTTP-статус: {status}")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdigit() or int(length) != response_bytes):
                raise DatasetDownloadError("HTTP-размер файла не совпадает с опубликованными метаданными")
            if not length and not artifact.checksum:
                raise DatasetDownloadError("Поток не имеет длины или контрольной суммы для проверки")
            _disk_space(directory, expected - offset)
            _write_metadata(metadata_path, {**identity, "etag": response.headers.get("ETag", ""),
                                            "last_modified": response.headers.get("Last-Modified", "")})
            digest = hashlib.new(artifact.checksum_algorithm)
            if offset and artifact.checksum:
                with part.open("rb") as prior:
                    while chunk := prior.read(_CHUNK_BYTES):
                        _check_cancel(cancel)
                        digest.update(chunk)
            downloaded = offset
            report(downloaded, "downloading", bool(offset))
            with part.open("ab" if offset else "wb") as output:
                next_space_check = downloaded + 32 * _CHUNK_BYTES
                while True:
                    _check_cancel(cancel)
                    chunk = response.read(min(_CHUNK_BYTES, expected - downloaded + 1))
                    if not chunk:
                        break
                    if downloaded + len(chunk) > expected:
                        raise DatasetDownloadError("Источник прислал больше байтов, чем опубликовано")
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_space_check:
                        _disk_space(directory, expected - downloaded)
                        next_space_check = downloaded + 32 * _CHUNK_BYTES
                    report(downloaded, "downloading", bool(offset))
                output.flush()
                os.fsync(output.fileno())
            if downloaded != expected:
                raise DatasetDownloadError(f"Загрузка неполная: {downloaded} из {expected} байт; можно продолжить")
            report(downloaded, "verifying", bool(offset))
            if artifact.checksum and digest.hexdigest().lower() != artifact.checksum.lower():
                raise DatasetDownloadError("Контрольная сумма не совпала; завершённый файл не создан")
            _check_cancel(cancel)
        os.replace(part, destination)
        metadata_path.unlink(missing_ok=True)
        report(expected, "complete", bool(offset))
        return destination
    except DownloadCancelled:
        report(part.stat().st_size if part.exists() else 0, "cancelled", bool(offset))
        raise
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise DatasetDownloadError(f"Загрузка не завершена, её можно продолжить: {exc}") from exc
