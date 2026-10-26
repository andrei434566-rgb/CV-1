"""Offline model selection and portable LoRA compatibility checks.

This module deliberately imports neither torch nor transformers. Invalid local
paths and incompatible model identities must fail before CUDA memory is used.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


MODEL_MODES = ("base", "lora")


def normalize_model_request(config: dict) -> dict[str, str]:
    """Return effective adapter paths without changing persisted settings.

    A request without ``model_mode`` keeps the historical separate photo and
    text adapter choices. Explicit modes use one specialist adapter by default;
    the optional text adapter remains an advanced override.
    """
    base = str(config.get("base") or "").strip()
    adapter = str(config.get("adapter") or "").strip()
    theory_adapter = str(config.get("theory_adapter") or "").strip()
    mode = str(config.get("model_mode") or "").strip().casefold()
    if not mode:
        return {
            "model_mode": "lora" if adapter or theory_adapter else "base",
            "base": base, "adapter": adapter, "theory_adapter": theory_adapter,
        }
    if mode not in MODEL_MODES:
        raise ValueError("Неизвестный режим ИИ. Выберите базовую модель или модель с дообучением.")
    if mode == "base":
        adapter = theory_adapter = ""
    elif not adapter:
        raise ValueError("Выбран режим с дообучением: укажите папку специализированного LoRA-адаптера.")
    else:
        theory_adapter = theory_adapter or adapter
    return {"model_mode": mode, "base": base, "adapter": adapter, "theory_adapter": theory_adapter}


def adapter_for_operation(config: dict, operation: str) -> tuple[str, str]:
    effective = normalize_model_request(config)
    name = "description" if operation in {"describe", "photo_chat"} else "theory"
    return name, effective["adapter" if name == "description" else "theory_adapter"]


def _read_json(path: Path, title: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать {title}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Неверный формат {title}: {path}")
    return value


def _identity(value: str) -> str:
    """Recognize HF IDs and Windows/Linux cache paths after moving a model."""
    value = value.replace("\\", "/").rstrip("/")
    cached = re.search(r"(?:^|/)models--([^/]+)--([^/]+)(?:/|$)", value, re.I)
    if cached:
        return f"{cached.group(1)}/{cached.group(2)}".casefold()
    qwen = re.search(r"(?:^|/)(Qwen3-VL-\d+[BM](?:-[A-Za-z0-9.]+)*)(?:/|$)", value, re.I)
    if qwen:
        return ("qwen/" + qwen.group(1)).casefold()
    if re.fullmatch(r"[^/ :]+/[^/ :]+", value):
        return value.casefold()
    return ""


def _revision(value: str, config: dict | None = None) -> str:
    matched = re.search(r"(?:^|[/\\])snapshots[/\\]([0-9a-f]{40})(?:[/\\]|$)", value, re.I)
    candidate = matched.group(1) if matched else str((config or {}).get("_commit_hash") or "")
    return candidate.casefold() if re.fullmatch(r"[0-9a-f]{40}", candidate, re.I) else ""


def _referenced_config(reference: str, adapter: Path) -> Path | None:
    if not reference:
        return None
    path = Path(reference)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path] + [parent / path for parent in list(adapter.parents)[:4]]
    return next((candidate / "config.json" for candidate in candidates if (candidate / "config.json").is_file()), None)


def _architecture_signature(config: dict) -> tuple:
    text = config.get("text_config") or {}
    vision = config.get("vision_config") or {}
    return (
        config.get("model_type"), tuple(config.get("architectures") or []),
        *(text.get(key) for key in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size")),
        *(vision.get(key) for key in ("hidden_size", "intermediate_size", "depth", "out_hidden_size", "patch_size")),
    )


def validate_local_models(config: dict) -> dict:
    """Validate architecture and available base ID/revision evidence offline.

    A portable adapter may include ``cv1_adapter_manifest.json`` with
    ``base_model_id``, ``base_revision`` (commit hash) and/or
    ``base_config_sha256``. Without it, PEFT's original HF/cache reference is
    used. The adapter's metadata is never rewritten to hide a mismatch.
    """
    effective = normalize_model_request(config)
    base = Path(effective["base"])
    if not effective["base"] or not base.is_dir() or not (base / "config.json").is_file():
        raise ValueError("Локальная папка Qwen3-VL не найдена. Укажите её в настройках ИИ.")
    base_config = _read_json(base / "config.json", "конфигурацию базовой модели")
    if base_config.get("model_type") != "qwen3_vl" or "Qwen3VLForConditionalGeneration" not in base_config.get("architectures", []):
        raise ValueError("Этот локальный модуль поддерживает Qwen3-VL. Qwen2.5-VL и другие архитектуры требуют отдельного модуля.")
    base_identity = _identity(str(base)) or _identity(str(base_config.get("_name_or_path") or ""))
    base_revision = _revision(str(base), base_config)
    base_sha = hashlib.sha256((base / "config.json").read_bytes()).hexdigest()
    reports = []
    for folder in dict.fromkeys((effective["adapter"], effective["theory_adapter"])):
        if not folder:
            continue
        adapter = Path(folder)
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError(f"Локальный LoRA-адаптер не найден: {adapter}")
        metadata = _read_json(adapter / "adapter_config.json", "конфигурацию LoRA")
        if str(metadata.get("peft_type") or "").upper() != "LORA":
            raise ValueError(f"Выбранный адаптер не является LoRA: {adapter}")
        if metadata.get("task_type") not in {None, "CAUSAL_LM"}:
            raise ValueError(f"Адаптер предназначен для другой задачи: {adapter}")
        review_path = adapter / "acceptance_review.json"
        if not review_path.is_file() and adapter.name.casefold() == "adapter":
            review_path = adapter.parent / "acceptance_review.json"
        if review_path.is_file():
            review = _read_json(review_path, "результат проверки адаптера")
            if str(review.get("status") or "").casefold().startswith("rejected"):
                raise ValueError("Этот экспериментальный адаптер отклонён по результатам проверки. Выберите базовую модель или другой проверенный адаптер.")
        manifest_path = adapter / "cv1_adapter_manifest.json"
        manifest = _read_json(manifest_path, "паспорт адаптера") if manifest_path.is_file() else {}
        reference = str(metadata.get("base_model_name_or_path") or "")
        expected_identity = _identity(str(manifest.get("base_model_id") or reference))
        # A mutable alias such as "main" must not hide the original snapshot
        # hash present in PEFT's cache path.
        expected_revision = next((str(value).casefold() for value in (
            manifest.get("base_revision"), metadata.get("revision"), _revision(reference)
        ) if re.fullmatch(r"[0-9a-f]{40}", str(value or ""), re.I)), "")
        expected_sha = str(manifest.get("base_config_sha256") or "").casefold()
        reference_config = _referenced_config(reference, adapter)
        same_config = bool(reference_config and hashlib.sha256(reference_config.read_bytes()).hexdigest() == base_sha)
        if reference_config and _architecture_signature(_read_json(reference_config, "исходную конфигурацию LoRA")) != _architecture_signature(base_config):
            raise ValueError("Размеры и архитектура базовой модели не совпадают с исходной конфигурацией LoRA.")
        if expected_identity and base_identity and expected_identity != base_identity:
            raise ValueError(f"LoRA обучена для {expected_identity}, выбрана {base_identity}. Базовые модели должны совпадать.")
        if expected_sha and expected_sha != base_sha:
            raise ValueError("Конфигурация базовой модели не совпадает с паспортом LoRA-адаптера.")
        if re.fullmatch(r"[0-9a-f]{40}", expected_revision, re.I) and base_revision and expected_revision.casefold() != base_revision:
            raise ValueError("Версия базовой модели не совпадает с версией, на которой обучался LoRA-адаптер.")
        if not (same_config or expected_sha == base_sha or (expected_identity and base_identity and expected_identity == base_identity)):
            raise ValueError("Нельзя подтвердить базовую модель LoRA. Сохраните исходное имя папки Qwen или добавьте cv1_adapter_manifest.json с base_model_id и base_config_sha256.")
        reports.append({
            "adapter": str(adapter), "base_model_id": expected_identity or base_identity,
            "revision_verified": bool(base_revision and re.fullmatch(r"[0-9a-f]{40}", expected_revision, re.I)),
            "base_config_sha256": base_sha,
        })
    return {**effective, "base_model_id": base_identity, "base_revision": base_revision, "base_config_sha256": base_sha, "adapters": reports}


def write_adapter_manifest(base_path: str | Path, adapter_path: str | Path) -> Path:
    """Save portable base-config provenance after a new adapter is trained.

    This records metadata compatibility only. It neither hashes the large
    weight files nor certifies the adapter's geology, quality or acceptance.
    Existing PEFT files and independent acceptance reviews remain untouched.
    """
    base, adapter = Path(base_path), Path(adapter_path)
    report = validate_local_models({
        "model_mode": "lora", "base": str(base), "adapter": str(adapter),
    })
    adapter_report = report["adapters"][0]
    manifest = {
        "schema": "cv1_adapter_manifest_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": report["base_model_id"] or adapter_report["base_model_id"] or None,
        "base_revision": report["base_revision"] or None,
        "base_config_sha256": report["base_config_sha256"],
        "base_config_hash_scope": "SHA-256 of local config.json bytes only; weight files are not hashed or verified",
        "metadata_only": True,
        "weights_verified": False,
        "review_status": "pending_review",
        "geologist_approved": False,
        "quality_metrics": None,
        "acceptance_note": "Training completion does not establish geological accuracy or approve deployment. Review on independent examples is still required.",
    }
    destination = adapter / "cv1_adapter_manifest.json"
    temporary = adapter / f".cv1_adapter_manifest_{uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
