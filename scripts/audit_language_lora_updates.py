"""Measure language LoRA updates on CPU without constructing a model or BA.

Only selected safetensors tensors are read. The ratios describe weight updates;
they do not establish the cause of a quality failure or measure model accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


class TensorReader:
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            size = int.from_bytes(stream.read(8), "little")
            if size <= 0 or size > 100 * 1024 * 1024:
                raise ValueError("Invalid safetensors header size")
            self.header = json.loads(stream.read(size))
        self.offset = size + 8

    def tensor(self, key: str) -> np.ndarray:
        item = self.header[key]
        start, end = item["data_offsets"]
        if start < 0 or end < start:
            raise ValueError("Invalid tensor offsets")
        with self.path.open("rb") as stream:
            stream.seek(self.offset + start)
            raw = stream.read(end - start)
        if len(raw) != end - start:
            raise ValueError("Incomplete tensor data")
        if item["dtype"] == "BF16":
            result = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
        else:
            dtype = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}.get(item["dtype"])
            if dtype is None:
                raise ValueError("Unsupported floating-point dtype: " + item["dtype"])
            result = np.frombuffer(raw, dtype=dtype).astype(np.float64 if item["dtype"] == "F64" else np.float32)
        result = result.reshape(item["shape"])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite tensor: " + key)
        return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(base_folder: Path, adapter_folder: Path, *, targets: str = "qv") -> dict:
    base_file = base_folder / "model.safetensors"
    adapter_file = adapter_folder / "adapter_model.safetensors"
    base, adapter = TensorReader(base_file), TensorReader(adapter_file)
    config = json.loads((adapter_folder / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("use_rslora") or config.get("use_dora") or config.get("rank_pattern") or config.get("alpha_pattern"):
        raise ValueError("Audit currently requires uniform ordinary LoRA scaling")
    if config["r"] <= 0 or config["lora_alpha"] <= 0:
        raise ValueError("Audit requires positive rank and alpha")
    scale = config["lora_alpha"] / config["r"]
    samples = []
    for key in sorted(adapter.header):
        if not key.endswith(".lora_A.weight") or ".language_model." not in key:
            continue
        if targets == "qv" and not key.endswith((".q_proj.lora_A.weight", ".v_proj.lora_A.weight")):
            continue
        a = adapter.tensor(key).astype(np.float64)
        b = adapter.tensor(key.replace(".lora_A.weight", ".lora_B.weight")).astype(np.float64)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or a.shape[0] != config["r"]:
            raise ValueError("Incompatible LoRA matrices: " + key)
        # ||BA||F^2 = trace((B^T B)(AA^T)); both products are rank x rank.
        delta_norm = float(scale * np.sqrt(max(float(np.trace((b.T @ b) @ (a @ a.T))), 0.0)))
        base_key = key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
        weight = base.tensor(base_key)
        if tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError("Base/adapter shape mismatch: " + key)
        base_norm = float(np.linalg.norm(weight.astype(np.float64)))
        if base_norm == 0:
            raise ValueError("Undefined relative update for a zero base matrix")
        samples.append({"module": base_key.removesuffix(".weight"), "base_norm": base_norm,
                        "delta_norm": delta_norm, "relative_delta": delta_norm / base_norm,
                        "lora_B_rms": float(np.sqrt(np.mean(b * b)))})
    if not samples:
        raise ValueError("No matching language LoRA matrices")
    ratios = [row["relative_delta"] for row in samples]
    return {"adapter_name": adapter_folder.parent.name, "adapter_weights_sha256": digest(adapter_file),
            "adapter_config_sha256": digest(adapter_folder / "adapter_config.json"),
            "targets": targets, "modules": len(samples), "lora_scale": scale,
            "relative_delta_min_median_max": [min(ratios), float(np.median(ratios)), max(ratios)],
            "lora_B_rms_median": float(np.median([row["lora_B_rms"] for row in samples])),
            "module_measurements": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapters", type=Path, nargs="+", required=True)
    parser.add_argument("--targets", choices=("qv", "all"), default="qv")
    parser.add_argument("--include-modules", action="store_true", help="Include every per-module measurement")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema": "qwen_language_adapter_update_audit_v1", "gpu_used": False,
              "base_snapshot": args.base.name, "base_config_sha256": digest(args.base / "config.json"),
              "method": "Frobenius norm of (alpha/r)BA divided by base matrix norm; rank products avoid materializing BA",
              "base_loading": "Selected local safetensors tensors only; no model construction, torch or GPU",
              "adapters": [audit(args.base, folder, targets=args.targets) for folder in args.adapters],
              "limitations": ["Weight ratios are descriptive, not proof of why responses did or did not change",
                              "The complete base model weight file is not rehashed by this audit",
                              "No model generation or geological accuracy is measured"]}
    if not args.include_modules:
        for row in report["adapters"]:
            row.pop("module_measurements")
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(json.dumps({"adapters": [{key: row[key] for key in (
        "adapter_name", "modules", "relative_delta_min_median_max")} for row in report["adapters"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
