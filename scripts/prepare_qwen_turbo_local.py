"""Prepare a GPU GGUF runtime from existing Qwen weights, without re-downloading them."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4
import zipfile

import requests

TAG = "b11913"
ASSET = "llama-b11913-bin-win-vulkan-x64.zip"
ASSET_SIZE = 37140477
ASSET_SHA256 = "d9b27dfe8addb0782c59c194604dbab69ccef8493c8964a435d762c4be45e176"
MIN_GGUF_BYTES = 100_000_000


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def source_fingerprint(base):
    """Hash converter inputs, including changed weights at an unchanged path."""
    base = Path(base)
    suffixes = {".safetensors", ".bin", ".json", ".model", ".tiktoken", ".jinja", ".py"}
    prefixes = ("tokenizer", "processor", "preprocessor", "vocab", "merges", "special_tokens", "added_tokens", "chat_template")
    files = {}
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base)
        if any(part.startswith(".") for part in relative.parts) or not path.is_file():
            continue
        if path.suffix.casefold() in suffixes or path.name.casefold().startswith(prefixes):
            files[relative.as_posix()] = sha256(path)
    if "config.json" not in files or not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise RuntimeError("В исходной модели не найдены config.json и фактические файлы весов.")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"algorithm": "sha256", "digest": hashlib.sha256(encoded).hexdigest(), "files": files}


def valid_gguf(path):
    try:
        if not path.is_file() or path.stat().st_size <= MIN_GGUF_BYTES:
            return False
        with path.open("rb") as stream:
            return stream.read(4) == b"GGUF"
    except OSError:
        return False


def cached_model_pair(root, base, fingerprint):
    """Reuse a pair only when both its source and output bytes are unchanged."""
    try:
        manifest = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
        if (Path(manifest["converted_from"]).resolve() != base.resolve()
                or manifest.get("source_fingerprint") != fingerprint
                or manifest.get("release") != TAG or manifest.get("quantization") != "Q8_0"):
            return None
        pair = tuple((root / manifest[key]).resolve() for key in ("model", "mmproj"))
        hashes = manifest["sha256"]
        if not isinstance(hashes, dict) or pair[0] == pair[1]:
            return None
        if all(path.is_relative_to(root.resolve()) and valid_gguf(path)
               and hashes.get(path.name) == sha256(path) for path in pair):
            return pair
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def prepare_model_pair(root, base, source):
    """Publish immutable generations so a failed conversion keeps the old pair."""
    fingerprint = source_fingerprint(base)
    pair = cached_model_pair(root, base, fingerprint)
    if pair is not None:
        return pair[0], pair[1], fingerprint
    model_root = root / "model"
    model_root.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=model_root)).resolve()
    if not stage.is_relative_to(model_root.resolve()):
        raise RuntimeError("Временная папка конвертации находится вне runtime/model.")
    targets = (stage / "Qwen3VL-2B-Instruct-Q8_0.gguf", stage / "mmproj-Qwen3VL-2B-Instruct-F16.gguf")
    try:
        for target, options in zip(targets, (["--outtype", "q8_0"], ["--mmproj", "--outtype", "f16"])):
            subprocess.run([sys.executable, str(source / "convert_hf_to_gguf.py"), str(base), "--outfile", str(target), *options], check=True)
            if not valid_gguf(target):
                raise RuntimeError("Конвертер не создал полный GGUF: " + target.name)
        if source_fingerprint(base) != fingerprint:
            raise RuntimeError("Исходные веса или настройки изменились во время конвертации. Повторите подготовку.")
        generation = model_root / (fingerprint["digest"][:16] + "-" + uuid4().hex)
        stage.replace(generation)
        return generation / targets[0].name, generation / targets[1].name, fingerprint
    finally:
        # Only these two explicitly created files can be removed on failure.
        for target in targets:
            target.unlink(missing_ok=True)
        if stage.is_dir():
            stage.rmdir()


def fetch_runtime(root):
    target = root / "downloads" / ASSET
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == ASSET_SHA256:
        return target
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{ASSET}"
    head = requests.head(url, allow_redirects=True, timeout=30)
    head.raise_for_status()
    direct = head.url
    step = (ASSET_SIZE + 15) // 16
    def part(index):
        begin, end = index * step, min(ASSET_SIZE, (index + 1) * step) - 1
        path = target.with_suffix(f".part{index:02d}")
        if path.exists() and path.stat().st_size == end - begin + 1:
            return path
        with requests.get(direct, headers={"Range": f"bytes={begin}-{end}"}, stream=True, timeout=(30, 90)) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError("Сервер не поддерживает проверяемую загрузку по частям.")
            with path.open("wb") as stream:
                for chunk in response.iter_content(128 * 1024):
                    stream.write(chunk)
        if path.stat().st_size != end - begin + 1:
            raise RuntimeError("Неполный фрагмент runtime.")
        print(f"Runtime: {index + 1}/16 фрагментов получен", flush=True)
        return path
    with ThreadPoolExecutor(max_workers=16) as pool:
        paths = list(pool.map(part, range(16)))
    with target.open("wb") as stream:
        for path in paths:
            with path.open("rb") as source:
                shutil.copyfileobj(source, stream)
    if sha256(target) != ASSET_SHA256:
        raise RuntimeError("Не совпал SHA256 GPU runtime.")
    return target


def fetch_converter(root):
    source = root / "source"
    metadata = []
    for name in ("conversion", "gguf-py/gguf", "convert_hf_to_gguf.py"):
        response = requests.get(f"https://api.github.com/repos/ggml-org/llama.cpp/contents/{name}?ref={TAG}", timeout=30)
        response.raise_for_status()
        payload = response.json()
        metadata.extend(payload if isinstance(payload, list) else [payload])
    metadata = [item for item in metadata if item["type"] == "file"]
    def fetch(item):
        destination = source / item["path"]
        def valid(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == item["sha"]
        if destination.is_file() and valid(destination.read_bytes()):
            return
        response = requests.get(item["download_url"], timeout=60)
        response.raise_for_status()
        if not valid(response.content):
            raise RuntimeError("Не совпал Git SHA проверяемого конвертера: " + item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(fetch, metadata))
    print("Официальный конвертер получен и проверен", flush=True)
    return source


def main():
    parser = argparse.ArgumentParser()
    parent = Path(__file__).resolve().parents[2] / "vlm_core_v1"
    parser.add_argument("--base", type=Path)
    parser.add_argument("--destination", type=Path, default=parent / "runtime" / "turbo")
    args = parser.parse_args()
    if args.base is None:
        snapshots = list((parent / "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots").glob("*"))
        args.base = next((path for path in snapshots if (path / "config.json").is_file()), None)
    if args.base is None or not (args.base / "config.json").is_file():
        raise RuntimeError("Укажите --base с уже скачанными весами Qwen3-VL-2B.")
    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_vl" or config.get("text_config", {}).get("hidden_size") != 2048:
        raise RuntimeError("Этот установщик рассчитан на Qwen3-VL-2B; указана другая архитектура.")
    root = args.destination.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        runtime = pool.submit(fetch_runtime, root)
        converter = pool.submit(fetch_converter, root)
        archive_path, source = runtime.result(), converter.result()
    binary_root = root / "bin"
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if not (binary_root / entry.filename).resolve().is_relative_to(binary_root.resolve()):
                raise RuntimeError("Архив содержит путь вне папки runtime.")
        archive.extractall(binary_root)
    if importlib.util.find_spec("sentencepiece") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece"], check=True)
    model, projector, fingerprint = prepare_model_pair(root, args.base, source)
    servers = list(binary_root.rglob("llama-server.exe"))
    if len(servers) != 1 or not model.is_file() or not projector.is_file():
        raise RuntimeError("Подготовлены не все компоненты runtime.")
    manifest = {"schema_version": 1, "runtime": "llama.cpp", "release": TAG, "gpu_backend": "vulkan",
                "server": str(servers[0].relative_to(root)), "model": str(model.relative_to(root)),
                "mmproj": str(projector.relative_to(root)), "base_model": "Qwen/Qwen3-VL-2B-Instruct",
                "base_revision": args.base.name, "converted_from": str(args.base.resolve()), "quantization": "Q8_0",
                "source_fingerprint": fingerprint,
                "sha256": {model.name: sha256(model), projector.name: sha256(projector)}}
    temporary_manifest = root / "runtime.json.partial"
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_manifest.replace(root / "runtime.json")
    print(json.dumps({"installed": str(root), "base_revision": args.base.name}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
