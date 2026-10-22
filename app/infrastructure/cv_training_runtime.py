"""Select an existing offline CUDA Python when a user starts CV training.

Importing this module neither imports torch nor starts a process. Call the
selector from the training worker, not while opening a window. No packages or
weights are installed, and the isolated probe blocks outbound connections.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.runtime_paths import application_root


_PREFIX = "__KERN_CV_RUNTIME__"
_PROBE = r'''
import importlib
import json
import socket

def offline(*args, **kwargs):
    raise OSError("Network disabled during local CV runtime probe")

socket.create_connection = offline
socket.getaddrinfo = offline
socket.socket.connect = offline
socket.socket.connect_ex = offline
modules, errors = {}, {}
for name in ("torch", "ultralytics", "yaml"):
    try:
        modules[name] = importlib.import_module(name)
    except Exception as exc:
        errors[name] = type(exc).__name__ + ": " + str(exc)
result = {"ok": not errors, "errors": errors, "cuda_available": False,
          "device_name": "", "cuda_error": "", "versions": {}}
for name, module in modules.items():
    result["versions"][name] = str(getattr(module, "__version__", "unknown"))
torch = modules.get("torch")
if torch is not None:
    result["cuda_version"] = str(torch.version.cuda or "")
    try:
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            # Test the driver and compiled kernels with one float, not a model.
            torch.ones(1, device="cuda").sum().item()
            torch.cuda.synchronize()
            result["device_name"] = str(torch.cuda.get_device_name(0))
    except Exception as exc:
        result["cuda_available"] = False
        result["cuda_error"] = type(exc).__name__ + ": " + str(exc)
print("__KERN_CV_RUNTIME__" + json.dumps(result, ensure_ascii=True), flush=True)
'''


class CvTrainingRuntimeError(RuntimeError):
    """No existing interpreter can run the requested local CV training."""


class CvTrainingRuntimeCancelled(RuntimeError):
    """The user cancelled runtime detection before training began."""


@dataclass(frozen=True)
class CvTrainingRuntime:
    python: Path
    versions: dict[str, str]
    cuda_available: bool
    cuda_version: str
    device_name: str


def _check_cancel(cancel) -> None:
    if cancel is not None and (cancel() if callable(cancel) else cancel.is_set()):
        raise CvTrainingRuntimeCancelled("Проверка окружения обучения отменена")


def _candidates(root: Path, vlm_config) -> list[Path]:
    paths = []
    # In a frozen build sys.executable is Kern_Analyzer.exe, not a Python CLI.
    if not getattr(sys, "frozen", False) and sys.executable:
        paths.append(Path(sys.executable))
    paths.extend((root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"))
    configured = vlm_config.get("python") if isinstance(vlm_config, Mapping) else vlm_config
    if configured:
        paths.append(Path(configured))
    sibling = root.parent / "vlm_core_v1" / ".venv"
    paths.extend((sibling / "Scripts" / "python.exe", sibling / "bin" / "python"))
    result, seen = [], set()
    for path in paths:
        path = path.expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _stop(process) -> None:
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.communicate(timeout=1)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _probe(python: Path, root: Path, timeout: float, cancel) -> tuple[dict | None, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
    try:
        process = subprocess.Popen(
            [str(python), "-I", "-c", _PROBE], cwd=str(root), env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        return None, f"не удалось запустить Python: {exc}"
    deadline = time.monotonic() + timeout
    try:
        while True:
            _check_cancel(cancel)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _stop(process)
                return None, f"проверка превысила {timeout:g} с"
            try:
                stdout, stderr = process.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        _stop(process)
        raise
    if process.returncode:
        detail = str(stderr or stdout or "нет диагностического сообщения").strip()[-1000:]
        return None, f"Python завершил проверку с кодом {process.returncode}: {detail}"
    for line in reversed(str(stdout or "").splitlines()):
        if not line.startswith(_PREFIX):
            continue
        try:
            result = json.loads(line[len(_PREFIX):])
        except ValueError:
            break
        if isinstance(result, dict) and isinstance(result.get("ok"), bool):
            return result, ""
        break
    return None, "Python не вернул корректный результат проверки библиотек"


def select_cv_training_runtime(
    vlm_config: Mapping | str | Path | None = None, *, root: Path | None = None,
    timeout_seconds: float = 180.0, total_timeout_seconds: float = 300.0,
    require_cuda: bool = True, cancel=None,
) -> CvTrainingRuntime:
    """Return an existing usable runtime or raise an actionable Russian error.

    ``vlm_config`` accepts the existing ``_local_vlm_config()`` mapping or its
    Python path. The total budget covers all candidates; each running probe can
    be cancelled through a threading.Event or a callable. Cold CUDA imports can
    take several minutes on a busy disk; run this in the cancellable training
    worker. Probes are not cached because dependencies and the CUDA driver can
    change between training runs.
    """
    if not (0 < timeout_seconds <= 300 and 0 < total_timeout_seconds <= 300):
        raise ValueError("Время проверки окружения должно быть больше 0 и не больше 300 секунд")
    root = Path(root if root is not None else application_root()).resolve()
    if not root.is_dir():
        raise CvTrainingRuntimeError(f"Не найдена папка приложения: {root}")
    deadline = time.monotonic() + total_timeout_seconds
    failures, install_python = [], None
    for python in _candidates(root, vlm_config):
        _check_cancel(cancel)
        if not python.is_file():
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failures.append("Общее время проверки окружений исчерпано.")
            break
        result, error = _probe(python, root, min(timeout_seconds, remaining), cancel)
        _check_cancel(cancel)
        if result is None:
            failures.append(f"{python}: {error}")
            continue
        if result.get("cuda_available") is True:
            install_python = install_python or python
        errors = result.get("errors", {})
        if not result["ok"]:
            detail = "; ".join(f"{name}: {message}" for name, message in errors.items()) if isinstance(errors, dict) else str(errors)
            failures.append(f"{python}: не загружаются библиотеки: {detail or 'torch / ultralytics / PyYAML'}")
            continue
        if require_cuda and result.get("cuda_available") is not True:
            detail = str(result.get("cuda_error") or "PyTorch не видит доступную NVIDIA CUDA")
            failures.append(f"{python}: {detail}")
            continue
        versions = result.get("versions", {})
        if not isinstance(versions, dict) or not all(name in versions for name in ("torch", "ultralytics", "yaml")):
            failures.append(f"{python}: неполный результат проверки библиотек")
            continue
        return CvTrainingRuntime(python=python, versions={str(k): str(v) for k, v in versions.items()},
                                 cuda_available=result.get("cuda_available") is True,
                                 cuda_version=str(result.get("cuda_version") or ""),
                                 device_name=str(result.get("device_name") or ""))
    _check_cancel(cancel)
    requirements = root / "requirements.txt"
    if not requirements.is_file() and (root / "_internal" / "requirements.txt").is_file():
        requirements = root / "_internal" / "requirements.txt"
    message = "Не найдено готовое локальное окружение для обучения CV"
    message += " с NVIDIA CUDA." if require_cuda else "."
    if failures:
        message += "\n\n" + "\n".join(failures)
    command = (("& " if os.name == "nt" else "") + f'"{install_python}"' if install_python else "python")
    message += ("\n\nВыберите Python с установленной CUDA-сборкой PyTorch в настройках помощника. "
                "Затем активируйте это CUDA-окружение и установите зависимости проекта:\n"
                f'{command} -m pip install -r "{requirements}"\n'
                "requirements.txt сам по себе не гарантирует поддержку CUDA. "
                "Приложение не скачивало и не устанавливало пакеты.")
    raise CvTrainingRuntimeError(message)


def resolve_cv_training_python(vlm_config: Mapping | str | Path | None = None, **kwargs) -> Path:
    """Convenience API for subprocess/QProcess callers that only need a path."""
    return select_cv_training_runtime(vlm_config, **kwargs).python
