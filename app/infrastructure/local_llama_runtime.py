"""Managed offline GPU inference; the only endpoint is an owned loopback child."""
from __future__ import annotations

import atexit
import base64
import ctypes
import hashlib
from io import BytesIO
import json
import math
import os
import secrets
from pathlib import Path
import socket
import subprocess
import struct
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

if __package__:
    from .local_vlm_modes import normalize_model_request
else:
    from local_vlm_modes import normalize_model_request


DESCRIPTION_FIELDS = ("description", "lithology", "texture", "facies_name", "facies_association", "depositional_environment")
DESCRIPTION_SCHEMA = {
    "type": "object", "properties": {name: {"type": "string", "minLength": 1, "maxLength": 1000 if name == "description" else 160} for name in DESCRIPTION_FIELDS},
    "required": list(DESCRIPTION_FIELDS), "additionalProperties": False,
}


def available_runtime(request: dict) -> tuple[Path, dict] | None:
    if str(request.get("inference_backend") or "auto") == "transformers":
        return None
    effective = normalize_model_request(request)
    # An official GGUF must never silently replace a specialist adapter or a
    # locally merged checkpoint. Those continue through the original runtime.
    base = effective["base"].replace("\\", "/")
    if effective["adapter"] or effective["theory_adapter"]:
        return None
    configured = request.get("turbo_root") or os.environ.get("CV1_TURBO_ROOT")
    sibling = next((parent / "runtime" / "turbo" for parent in Path(effective["base"]).parents
                    if parent.name.casefold() == "vlm_core_v1"), None)
    root = Path(configured or sibling or (Path(__file__).resolve().parents[3] / "vlm_core_v1" / "runtime" / "turbo"))
    try:
        manifest = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
        if manifest.get("base_model") != "Qwen/Qwen3-VL-2B-Instruct":
            return None
        official_cache = "models--qwen--qwen3-vl-2b-instruct/snapshots/" in base.casefold()
        converted_source = manifest.get("converted_from")
        same_source = bool(converted_source and Path(effective["base"]).resolve() == Path(converted_source).resolve())
        if not official_cache and not same_source:
            return None
        if manifest.get("base_revision") and Path(effective["base"]).name != manifest["base_revision"]:
            return None
        for key in ("server", "model", "mmproj"):
            target = (root / manifest[key]).resolve()
            if not target.is_relative_to(root.resolve()) or not target.is_file():
                return None
        return root.resolve(), manifest
    except (OSError, ValueError, KeyError, TypeError):
        return None


def openai_messages(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            converted = []
            for part in content:
                if part.get("type") == "image":
                    buffer = BytesIO()
                    part["image"].save(buffer, format="PNG")
                    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                    converted.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}})
                else:
                    converted.append(dict(part))
            content = converted
        result.append({"role": message["role"], "content": content})
    return result


class LocalLlamaRuntime:
    def __init__(self, lora_path: str | Path | None = None):
        # Explicit research opt-in only. Application model selection never sets
        # this automatically; conversion/base compatibility is checked by the
        # separate preparation tool before the caller supplies a candidate.
        self.lora_path = Path(lora_path).expanduser().resolve() if lora_path is not None else None
        self._loaded_lora_identity = None
        self.process = None
        self.root = None
        self.manifest = None
        self.endpoint = ""
        self.api_key = ""
        self.log = None
        self.job_handle = None
        self.opener = build_opener(ProxyHandler({}))
        atexit.register(self.close)

    def _candidate_identity(self):
        if self.lora_path is None:
            return None
        path = Path(self.lora_path).expanduser().resolve()
        if "," in str(path):
            raise ValueError("Путь LoRA содержит запятую, которую runtime трактует как разделитель адаптеров.")
        if not path.is_file():
            raise ValueError("Файл исследовательского LoRA GGUF не найден.")
        if path.stat().st_size <= 24:
            raise ValueError("Исследовательский LoRA GGUF не содержит данных после заголовка.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            header = stream.read(24)
            if len(header) != 24:
                raise ValueError("Некорректный заголовок исследовательского LoRA GGUF.")
            magic, version, tensors, metadata = struct.unpack("<4sIQQ", header)
            if magic != b"GGUF" or version not in (2, 3) or tensors < 1 or metadata < 1:
                raise ValueError("Ожидается непустой исследовательский адаптер GGUF версии 2 или 3.")
            digest.update(header)
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        # Include bytes, not merely a mutable path, in owned-server reuse.
        return str(path), digest.hexdigest()

    def start(self, root: Path, manifest: dict) -> float:
        candidate_identity = self._candidate_identity()
        if candidate_identity is not None and manifest.get("release") != "b11913":
            raise ValueError("Исследовательский LoRA поддерживается только проверенным runtime b11913.")
        if (self.root == root and self.manifest == manifest and self._loaded_lora_identity == candidate_identity
                and self.process is not None and self.process.poll() is None):
            return 0.0
        self.close()
        started = monotonic()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.endpoint = f"http://127.0.0.1:{port}"
        self.api_key = secrets.token_hex(32)
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        self.log = (logs / f"worker-{os.getpid()}.log").open("ab", buffering=0)
        command = [str(root / manifest["server"]), "-m", str(root / manifest["model"]),
                   "--mmproj", str(root / manifest["mmproj"]), "--host", "127.0.0.1", "--port", str(port),
                   "-ngl", "99", "-c", "16384", "-np", "4", "--flash-attn", "on", "--jinja",
                   "--api-key", self.api_key]
        if candidate_identity is not None:
            command += ["--lora", candidate_identity[0], "--lora-init-without-apply"]
        self.process = subprocess.Popen(command, cwd=str((root / manifest["server"]).parent),
                                        stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if os.name == "nt":
            try:
                self.job_handle = _own_child_process(self.process)
            except Exception:
                self.close()
                raise
        self.root, self.manifest = root, manifest
        self._loaded_lora_identity = candidate_identity
        while monotonic() - started < 120:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.close()
                raise RuntimeError(f"Локальный GPU runtime завершился с кодом {code}. Проверьте журнал runtime/logs.")
            try:
                with self.opener.open(self.endpoint + "/health", timeout=1) as response:
                    if response.status == 200:
                        return monotonic() - started
            except (OSError, URLError):
                sleep(0.2)
        self.close()
        raise RuntimeError("Локальная Qwen не загрузилась в ускоренном runtime за 2 минуты.")

    def complete(self, messages: list[dict], *, max_tokens: int, schema: dict | None = None,
                 lora: list[dict] | None = None) -> dict:
        payload = {"messages": openai_messages(messages), "temperature": 0, "max_tokens": max_tokens,
                   "stream": False, "cache_prompt": True, "seed": 0, "repeat_penalty": 1.08}
        if lora is not None or self.lora_path is not None:
            if lora is not None and not isinstance(lora, list):
                raise ValueError("LoRA запроса должна быть списком id/scale.")
            selected, ids = [], set()
            for entry in lora or []:
                if (not isinstance(entry, dict) or set(entry) != {"id", "scale"}
                        or type(entry["id"]) is not int or entry["id"] < 0 or entry["id"] in ids
                        or type(entry["scale"]) not in (int, float)):
                    raise ValueError("Некорректный или повторный id/scale LoRA запроса.")
                try:
                    scale = float(entry["scale"])
                except (OverflowError, ValueError) as exc:
                    raise ValueError("Некорректный scale LoRA запроса.") from exc
                if not math.isfinite(scale):
                    raise ValueError("Scale LoRA запроса должен быть конечным числом.")
                if self.lora_path is None:
                    raise ValueError("Для выбора LoRA сначала явно загрузите исследовательский адаптер.")
                ids.add(entry["id"])
                selected.append({"id": entry["id"], "scale": scale})
            # b11913 server-context.cpp:1778 falls back to startup scales for
            # an EMPTY request map, despite the README's empty-list semantics.
            # The constructor loads one adapter (id 0); explicit zero keeps a
            # control request on base weights even if GET reports scale 1.
            # https://github.com/ggml-org/llama.cpp/blob/b11913/tools/server/server-context.cpp#L1776
            payload["lora"] = selected or ([{"id": 0, "scale": 0.0}] if self.lora_path is not None else [])
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "core_description", "strict": True, "schema": schema}}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(self.endpoint + "/v1/chat/completions", data=body,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        try:
            with self.opener.open(request, timeout=180) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-1200:]
            raise RuntimeError(f"Ошибка локального GPU runtime: {detail}") from exc

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.job_handle is not None:
            ctypes.windll.kernel32.CloseHandle(self.job_handle)
            self.job_handle = None
        self.root = self.manifest = None
        self._loaded_lora_identity = None


def _own_child_process(process):
    """Windows closes this job on worker exit, including a forced QProcess kill."""
    from ctypes import wintypes
    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", IoCounters), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
    kernel = ctypes.windll.kernel32
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle, int(process._handle)):
        if handle:
            kernel.CloseHandle(handle)
        raise RuntimeError("Не удалось привязать локальный runtime к жизненному циклу приложения.")
    return handle
