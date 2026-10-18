"""Build the portable model assets for a private CV-1 GitHub release.

Only runtime weights, local knowledge and VLM source are included. Training
photos, external source archives, Python environments and build caches are not.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


APP = Path(__file__).resolve().parents[1]
WORKSPACE = APP.parent
VLM = WORKSPACE / "vlm_core_v1"
SNAPSHOT = (
    VLM
    / "model_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots"
    / "89644892e4d85e24eaac8bacfd4f463576704203"
)
WEIGHT = SNAPSHOT / "model.safetensors"
OUT = Path(__file__).resolve().parent / "out"
PART_BYTES = 1_500_000_000


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_member(archive: zipfile.ZipFile, path: Path) -> None:
    archive.write(path, path.relative_to(WORKSPACE).as_posix())


def support_files() -> list[Path]:
    files: list[Path] = []
    files.append(APP / "transfer/QWEN_APACHE_2.0.txt")
    root_suffixes = {".py", ".md", ".json", ".txt", ".ps1"}
    files.extend(p for p in VLM.iterdir() if p.is_file() and p.suffix.lower() in root_suffixes)
    for folder in (VLM / "scripts", VLM / "data/knowledge"):
        files.extend(p for p in folder.rglob("*") if p.is_file())
    for folder in (VLM / "portable_trainer", VLM / "review_tool"):
        files.extend(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in {".py", ".ps1", ".md", ".txt", ".json", ".js", ".mjs", ".html"}
        )
    uv = VLM / "portable_trainer/vendor/uv.exe"
    if uv.is_file():
        files.append(uv)
    files.extend(p for p in SNAPSHOT.iterdir() if p.is_file() and p != WEIGHT)
    for folder in (VLM / "outputs").iterdir():
        if folder.is_dir() and folder.name.startswith("core_vlm_"):
            files.extend(p for p in folder.iterdir() if p.is_file())
    files.extend(p for p in (APP / "models").rglob("*") if p.is_file())
    return sorted(set(files), key=lambda p: str(p).casefold())


def main() -> None:
    if not WEIGHT.is_file():
        raise FileNotFoundError(f"Base Qwen weights missing: {WEIGHT}")
    if not (VLM / "outputs/core_vlm_v8_single_column_lora/adapter_model.safetensors").is_file():
        raise FileNotFoundError("Active VLM adapter missing")
    OUT.mkdir(parents=True, exist_ok=True)
    support = OUT / "cv1-runtime.zip"
    files = support_files()
    with zipfile.ZipFile(support, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4, allowZip64=True) as archive:
        for path in files:
            archive_member(archive, path)
    print(f"Runtime archive: {support.name}; files={len(files)}; bytes={support.stat().st_size}", flush=True)

    parts: list[dict[str, object]] = []
    with WEIGHT.open("rb") as source:
        number = 1
        while True:
            first = source.read(min(8 * 1024 * 1024, PART_BYTES))
            if not first:
                break
            target = OUT / f"cv1-qwen3vl-2b-model.part{number:03d}"
            digest = hashlib.sha256()
            written = 0
            with target.open("wb") as sink:
                sink.write(first)
                digest.update(first)
                written += len(first)
                while written < PART_BYTES:
                    block = source.read(min(8 * 1024 * 1024, PART_BYTES - written))
                    if not block:
                        break
                    sink.write(block)
                    digest.update(block)
                    written += len(block)
            parts.append({"name": target.name, "bytes": written, "sha256": digest.hexdigest()})
            print(f"Model part {number}: {written} bytes", flush=True)
            number += 1

    manifest = {
        "schema": "cv1_runtime_v1",
        "base_model": "Qwen/Qwen3-VL-2B-Instruct",
        "base_model_license": "Apache-2.0",
        "base_model_source": "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct",
        "model_relative_path": WEIGHT.relative_to(WORKSPACE).as_posix(),
        "model_bytes": WEIGHT.stat().st_size,
        "model_sha256": checksum(WEIGHT),
        "active_vlm_adapter": "vlm_core_v1/outputs/core_vlm_v8_single_column_lora",
        "runtime_archive": {"name": support.name, "bytes": support.stat().st_size, "sha256": checksum(support)},
        "parts": parts,
        "included_vlm_adapters": len({p.parent for p in files if p.name == "adapter_model.safetensors"}),
        "included_yolo_weights": [p.relative_to(WORKSPACE).as_posix() for p in files if p.suffix == ".pt"],
        "excluded": ["source photos", "training datasets", "downloaded external archives", "virtual environments", "build caches"],
    }
    app_zip = OUT / "Kern_Analyzer-Windows.zip"
    if app_zip.is_file():
        manifest["app_archive"] = {"name": app_zip.name, "bytes": app_zip.stat().st_size, "sha256": checksum(app_zip)}
    path = OUT / "cv1-runtime-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Manifest: {path}", flush=True)


if __name__ == "__main__":
    main()
