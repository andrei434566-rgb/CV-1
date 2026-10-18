"""Extract descriptions from official external core logs for human review.

PDF descriptions use embedded text extraction. Raster log sheets use the base
Qwen3-VL-2B model as a transcription assistant, never the fine-tuned adapter.
All resulting rows remain unapproved and are not wired into training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROJECT = Path(__file__).resolve().parent
DEFAULT_BASE = (
    PROJECT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
)
PAIR_FILE = PROJECT / "data" / "external_candidate_pairs.jsonl"
DEFAULT_OUTPUT = PROJECT / "data" / "external_log_transcriptions.jsonl"


def read_pairs() -> list[dict]:
    return [json.loads(line) for line in PAIR_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def transcribe_image(model, processor, image_path: Path, dataset: str, max_side: int, max_tokens: int) -> str:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        # USGS descriptive log sheets place the prose in a right-hand column.
        # Crop away the plotted core to preserve text resolution and reduce
        # visual tokens; Midnight Pass uses a wider description column.
        if dataset != "usgs_2009_apalachicola_bay":
            width, height = image.size
            left_fraction = 0.24 if dataset == "usgs_2024_330_midnight_pass" else 0.34
            image = image.crop((int(width * left_fraction), int(height * 0.04), width, int(height * 0.96)))
        image.thumbnail((max_side, max_side))
    prompt = (
        "This is a geologist-authored sediment core log sheet. Transcribe only the printed "
        "lithologic description text, preserving the source language, depth intervals, "
        "grain-size terms, Munsell color codes, structures, contacts, and abbreviations. "
        "Do not interpret the photograph, do not summarize, translate, correct, or add "
        "information. Omit scale headings and symbol legends. Mark genuinely unreadable "
        "words as [illegible]. Return plain text only."
    )
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
    }]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    completion = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(completion, skip_special_tokens=True)[0].strip()


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        # Reuse the desktop's bundled PDF reader when this project venv lacks it.
        bundled_site = (Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime"
                        / "dependencies" / "python" / "Lib" / "site-packages")
        if bundled_site.exists():
            sys.path.insert(0, str(bundled_site))
            from pypdf import PdfReader
        else:
            raise RuntimeError("Install pypdf to extract text from PDF core logs.")
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"[page {index}]\n{text}")
    return "\n\n".join(pages)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", help="Process only this external dataset ID")
    parser.add_argument("--limit", type=int, help="Limit the number of distinct log files")
    parser.add_argument("--max-image-side", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    parser.add_argument("--append", action="store_true", help="Append rows instead of replacing output")
    parser.add_argument("--resume", action="store_true", help="Skip log files already present in output")
    args = parser.parse_args()

    pairs = read_pairs()
    unique_logs: dict[str, dict] = {}
    for row in pairs:
        log_path = row.get("log_path")
        if not log_path or (args.dataset and row["dataset"] != args.dataset):
            continue
        unique_logs.setdefault(log_path, row)
    selected = list(unique_logs.items())
    if args.limit is not None:
        selected = selected[:args.limit]
    if args.resume and args.output.exists():
        old_rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
        completed = {row.get("log_path") for row in old_rows if row.get("transcript", "").strip()}
        # Keep successful transcriptions and discard empty extraction attempts
        # so resume can retry those sources without accumulating duplicates.
        retained = [row for row in old_rows if row.get("transcript", "").strip()]
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained),
            encoding="utf-8",
        )
        selected = [(path, row) for path, row in selected if path not in completed]

    image_logs = [(path, row) for path, row in selected if Path(path).suffix.lower() not in {".pdf"}]
    model = processor = None
    if image_logs:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA-enabled PyTorch is required for log-image transcription.")
        processor = AutoProcessor.from_pretrained(args.base_model, local_files_only=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.base_model, torch_dtype=torch.float16, attn_implementation="sdpa",
            device_map="cuda", local_files_only=True,
        )
        model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append or args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        for index, (relative_log, pair) in enumerate(selected, start=1):
            log_path = PROJECT / relative_log
            suffix = log_path.suffix.lower()
            try:
                if suffix == ".pdf":
                    transcript = extract_pdf(log_path)
                    method = "pypdf_embedded_text" if transcript else "pypdf_found_no_embedded_text_scanned_pdf"
                else:
                    transcript = transcribe_image(model, processor, log_path, pair["dataset"], args.max_image_side, args.max_new_tokens)
                    method = "Qwen3-VL-2B-Instruct zero-shot transcription"
                record = {
                    "dataset": pair["dataset"],
                    "core_key": pair.get("core_key"),
                    "log_path": relative_log,
                    "photo_paths": [r["photo_path"] for r in pairs if r.get("log_path") == relative_log],
                    "doi": pair["doi"],
                    "metadata_url": pair["metadata_url"],
                    "transcription_method": method,
                    "transcript": transcript,
                    "transcription_review_status": "needs_geologist_review" if transcript else "requires_OCR_or_manual_transcription",
                    "training_approved": False,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"[{index}/{len(selected)}] {pair['dataset']} {pair.get('core_key')} ({method}; {len(transcript)} chars)")
            except Exception as error:
                print(f"ERROR {relative_log}: {type(error).__name__}: {error}", file=sys.stderr)
                output.write(json.dumps({
                    "dataset": pair["dataset"], "core_key": pair.get("core_key"),
                    "log_path": relative_log, "doi": pair["doi"],
                    "transcription_method": "failed", "transcript": "",
                    "transcription_review_status": "failed", "training_approved": False,
                    "error": f"{type(error).__name__}: {error}",
                }, ensure_ascii=False) + "\n")
                output.flush()


if __name__ == "__main__":
    main()
