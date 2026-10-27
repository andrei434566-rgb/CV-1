"""Plan or convert a checked local language LoRA to a separate candidate GGUF.

No base-weight merge, server launch, settings update or model activation occurs.
The default is a read-only plan; --convert creates a new immutable output folder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.infrastructure.local_vlm_modes import validate_local_models
from scripts.audit_language_lora_updates import TensorReader
from scripts.prepare_qwen_turbo_local import source_fingerprint

TAG = "b11913"
COMMIT = "7aca0d2ea8d92b56911233b822db01f9fbb1ae50"
LOCK_SHA256 = "1db62d1d4d2ced0934c52e2e47a7fe059d2ca2a16598ca591cee026a229b2e2a"
LOCK = Path(__file__).with_name("llama_lora_converter_b11913.lock.json")
BASE_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
LEGACY_PAIR = {
    "base_weights": "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0",
    "model": "e5fa40ecfcf36151e3aecf8ecd6d3ce2f20c4a1f74ee6891270a2dab018b4dd6",
    "mmproj": "28fab81bf7c927aee76ac8d678047926b72ab87348a135662ab9d86fdedd32a9",
}
GGUF_NAMES = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
              "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
LORA_KEY = re.compile(r"base_model\.model\.model\.language_model\.layers\.(\d+)\."
                      r"(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_([AB])\.weight")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_source(source: Path) -> dict:
    if sha256(LOCK) != LOCK_SHA256:
        raise ValueError("Converter source lock bytes differ from the reviewed lock")
    lock = read_json(LOCK)
    if lock.get("commit") != COMMIT or lock.get("tag") != TAG:
        raise ValueError("Converter version mismatch")
    for name, hashes in lock["files"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha256(path) != hashes["sha256"]:
            raise ValueError("Pinned converter file differs or is absent: " + name)
    expected = {name for name in lock["files"] if name != "convert_lora_to_gguf.py"}
    actual = {path.relative_to(source).as_posix()
              for folder in (source / "conversion", source / "gguf-py/gguf") for path in folder.glob("*.py")}
    if actual != expected:
        raise ValueError("Converter import package contains unexpected or missing Python modules")
    return {"tag": TAG, "commit": COMMIT, "source_lock_sha256": LOCK_SHA256,
            "verified_source_files": len(lock["files"]),
            "converter_sha256": lock["files"]["convert_lora_to_gguf.py"]["sha256"]}


def language_tensor_names(adapter: Path, base_config: dict) -> set[str]:
    config = read_json(adapter / "adapter_config.json")
    if (config.get("peft_type") != "LORA" or config.get("bias", "none") != "none"
            or any(config.get(key) for key in ("use_rslora", "use_dora", "rank_pattern", "alpha_pattern",
                                               "modules_to_save", "fan_in_fan_out", "use_qalora",
                                               "alora_invocation_tokens", "invocation_string"))
            or config.get("r", 0) <= 0 or config.get("lora_alpha", 0) <= 0):
        raise ValueError("Only ordinary language-only LoRA with uniform positive scaling is supported")
    tensors = TensorReader(adapter / "adapter_model.safetensors")
    pairs, names = {}, set()
    for key in tensors.header:
        if key == "__metadata__":
            continue
        match = LORA_KEY.fullmatch(key)
        if not match:
            raise ValueError("Unsupported, visual or non-LoRA tensor: " + key)
        layer, branch, projection, side = match.groups()
        if int(layer) >= base_config["text_config"]["num_hidden_layers"] or (
                (branch == "self_attn") != (projection in {"q_proj", "k_proj", "v_proj", "o_proj"})):
            raise ValueError("Language module/architecture mismatch")
        tensor = tensors.tensor(key)
        if tensor.ndim != 2 or tensor.shape[0 if side == "A" else 1] != config["r"]:
            raise ValueError("LoRA tensor rank/shape mismatch")
        pairs.setdefault((layer, projection), set()).add(side)
        names.add(f"blk.{layer}.{GGUF_NAMES[projection]}.weight.lora_{side.lower()}")
    if not pairs or any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("Incomplete LoRA A/B pairs")
    return names


def build_plan(base: Path, adapter: Path, runtime: Path, source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing conversion folder")
    converter = verify_source(source)
    compatibility = validate_local_models({"base": str(base), "adapter": str(adapter), "model_mode": "lora"})
    passport = read_json(adapter / "cv1_adapter_manifest.json")
    training = read_json(adapter.parent / "run_config.json")
    runtime_config = read_json(runtime / "runtime.json")
    if training.get("status") != "candidate_requires_geological_validation":
        raise ValueError("A completed training run is required before conversion")
    if (compatibility["base_revision"] != BASE_REVISION or passport.get("base_revision") != BASE_REVISION
            or runtime_config.get("base_revision") != BASE_REVISION
            or runtime_config.get("base_model") != "Qwen/Qwen3-VL-2B-Instruct"
            or runtime_config.get("release") != TAG or runtime_config.get("quantization") != "Q8_0"):
        raise ValueError("The candidate and runtime must use the pinned Qwen3-VL-2B base revision")
    fingerprint = source_fingerprint(base)
    if fingerprint != training.get("model_source_fingerprint"):
        raise ValueError("Actual HF base bytes differ from the frozen training base")
    if passport.get("base_config_sha256") != fingerprint["files"]["config.json"]:
        raise ValueError("Adapter passport base configuration hash differs")
    # The original project runtime predates source_fingerprint. Only its exact
    # pinned production model pair is accepted as a documented legacy case.
    if runtime_config.get("source_fingerprint") is not None:
        if runtime_config["source_fingerprint"] != fingerprint:
            raise ValueError("GGUF runtime and HF training source fingerprints differ")
        relationship = "matching_full_source_fingerprint"
    else:
        if fingerprint["files"].get("model.safetensors") != LEGACY_PAIR["base_weights"]:
            raise ValueError("Legacy runtime cannot establish compatibility with this HF weight file")
        relationship = "pinned_original_project_HF_and_GGUF_pair"
    runtime_hashes = {}
    for kind in ("model", "mmproj"):
        path = (runtime / runtime_config[kind]).resolve()
        if not path.is_relative_to(runtime.resolve()):
            raise ValueError("Runtime model path escapes the selected runtime")
        actual = sha256(path)
        if actual != runtime_config.get("sha256", {}).get(path.name):
            raise ValueError("Runtime model bytes differ from its manifest")
        if relationship.startswith("pinned_") and actual != LEGACY_PAIR[kind]:
            raise ValueError("Legacy GGUF pair differs from the pinned project baseline")
        runtime_hashes[kind] = actual
    names = language_tensor_names(adapter, read_json(base / "config.json"))
    return {"schema": "qwen_lora_gguf_conversion_plan_v1", "status": "conversion_only_not_model_acceptance",
            "converter": converter, "base_revision": BASE_REVISION, "base_fingerprint": fingerprint,
            "runtime_compatibility": relationship, "runtime_sha256": runtime_hashes,
            "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
            "training_run_config_sha256": sha256(adapter.parent / "run_config.json"),
            "outtype": "f16", "expected_tensor_names": sorted(names),
            "active_in_application": False, "geologist_approved": False,
            "scope": "Separate research candidate; no merge, settings edit or server launch"}


def inspect_gguf(path: Path, source: Path, expected_names: set[str], alpha: float) -> dict:
    package = (source / "gguf-py").resolve()
    sys.path.insert(0, str(package))
    gguf = importlib.import_module("gguf")
    if not Path(gguf.__file__).resolve().is_relative_to(package):
        raise ValueError("Another gguf package shadows the pinned reader")
    reader = gguf.GGUFReader(str(path))
    value = lambda name: reader.get_field(name).contents() if reader.get_field(name) else None
    if (value("general.architecture") != "qwen3vl" or value("general.type") != "adapter"
            or value("adapter.type") != "lora" or value("adapter.lora.alpha") != float(alpha)
            or {tensor.name for tensor in reader.tensors} != expected_names):
        raise ValueError("Converted GGUF metadata or complete language tensor set differs")
    return {"architecture": "qwen3vl", "type": "adapter", "adapter_type": "lora",
            "lora_alpha": float(alpha), "tensor_count": len(reader.tensors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--convert", action="store_true")
    args = parser.parse_args()
    args.source = args.source or args.runtime / "source"
    plan = build_plan(args.base, args.adapter, args.runtime, args.source, args.output_dir)
    if not args.convert:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "adapter-f16.gguf"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1")
    try:
        subprocess.run([sys.executable, str((args.source / "convert_lora_to_gguf.py").resolve()),
                        "--base", str(args.base.resolve()), "--outtype", "f16", "--outfile", str(output.resolve()),
                        str(args.adapter.resolve())], cwd=args.source, env=env, check=True)
        if (sha256(args.adapter / "adapter_model.safetensors") != plan["adapter_sha256"]
                or sha256(args.adapter / "adapter_config.json") != plan["adapter_config_sha256"]
                or sha256(args.base / "config.json") != plan["base_fingerprint"]["files"]["config.json"]
                or verify_source(args.source) != plan["converter"]):
            raise ValueError("Adapter or converter inputs changed during conversion")
        metadata = inspect_gguf(output, args.source, set(plan["expected_tensor_names"]),
                                read_json(args.adapter / "adapter_config.json")["lora_alpha"])
        plan.update(status="converted_candidate_not_accepted", file=output.name,
                    bytes=output.stat().st_size, sha256=sha256(output), metadata=metadata)
    except BaseException as error:
        (args.output_dir / "failure.json").write_text(json.dumps({"status": "failed", "error": str(error)}), encoding="utf-8")
        raise
    with (args.output_dir / "conversion_manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: plan[key] for key in ("status", "file", "bytes", "sha256", "active_in_application")}))


if __name__ == "__main__":
    main()
