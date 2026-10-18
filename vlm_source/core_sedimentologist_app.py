"""Local, review-first core photo description and report application.

Run with the project's Python environment and open http://127.0.0.1:8765.
No network requests are made by this application.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import nullcontext
import hashlib
import html
import io
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
SESSIONS = ROOT / "data" / "app_sessions"
REVIEWS = ROOT / "data" / "geologist_review_ledger.jsonl"
PAGES = ROOT / "data" / "knowledge" / "gazprom_facies_models_pages.jsonl"
R390_CANDIDATES = ROOT / "data" / "r390_review_candidates.jsonl"
R390_REVIEWS = ROOT / "data" / "r390_geologist_reviews.jsonl"
BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
ADAPTERS = {
    "v5": ROOT / "outputs" / "core_vlm_v5_interval_crops_lora",
    "v8": ROOT / "outputs" / "core_vlm_v8_single_column_lora",
}
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
_model_lock = threading.Lock()
_review_lock = threading.Lock()
_model = None
_processor = None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def session_path(session_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise ValueError("Неверный идентификатор записи")
    return SESSIONS / session_id / "record.json"


def load_model(mode: str = "base"):
    global _model, _processor
    from peft import PeftModel

    if _model is None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("Для локального анализа нужна CUDA и доступная видеокарта")
        if not BASE.is_dir():
            raise RuntimeError("Базовая модель не найдена")
        _processor = AutoProcessor.from_pretrained(BASE)
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
        )
        _model.eval()
    if mode != "base":
        adapter = ADAPTERS[mode]
        if not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"Адаптер {mode} не найден")
        if not isinstance(_model, PeftModel):
            _model = PeftModel.from_pretrained(_model, adapter, adapter_name=mode)
        elif mode not in _model.peft_config:
            _model.load_adapter(adapter, adapter_name=mode, is_trainable=False)
        _model.set_adapter(mode)
        _model.eval()
    return _model, _processor


def extract_description(raw: str) -> tuple[str, bool]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    def segment_text(data: object) -> str:
        if not isinstance(data, dict):
            return ""
        segments = data.get("description_segments")
        if not isinstance(segments, list) or not segments:
            return ""
        if not all(isinstance(segment, dict)
                   and isinstance(segment.get("top_m"), (int, float))
                   and isinstance(segment.get("base_m"), (int, float))
                   and isinstance(segment.get("description"), str)
                   and segment["description"].strip()
                   for segment in segments):
            return ""
        return "\n".join(segment["description"].strip() for segment in segments)

    try:
        data = json.loads(cleaned)
        if description := segment_text(data):
            return description, True
    except (ValueError, AttributeError, TypeError):
        pass
    normalized = cleaned.replace("“", '"').replace("”", '"')
    if normalized != cleaned:
        try:
            if description := segment_text(json.loads(normalized)):
                return description, False
        except (ValueError, AttributeError, TypeError):
            pass
    match = re.search(r'"description"\s*:\s*"((?:\\.|[^"\\])*)', cleaned)
    if match:
        try:
            return json.loads('"' + match.group(1) + '"'), False
        except ValueError:
            pass
    return cleaned[:3000], False


def describe(image: Image.Image, top_m: float, base_m: float, mode: str) -> tuple[str, str, bool]:
    import torch

    model, processor = load_model(mode)
    preview = image.copy()
    preview.thumbnail((448, 448))
    prompt = (
        "Это увеличенный фрагмент керна внутри описываемого слоя "
        f"{top_m:.2f}–{base_m:.2f} м; на фото показан участок примерно "
        f"{top_m:.2f}–{base_m:.2f} м. Составь краткое нейтральное литологическое "
        "описание на русском языке по непосредственно видимым признакам. Верни только JSON "
        "с ключом description_segments и одним объектом с top_m, base_m и description. "
        "Не назначай среду осадконакопления и не придумывай свойства, которых нет в "
        "видимой части. Игнорируй номера образцов и надписи на фото."
    )
    messages = [{"role": "user", "content": [{"type": "image", "image": preview}, {"type": "text", "text": prompt}]}]
    prepared = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prepared], images=[preview], return_tensors="pt").to("cuda")
    adapter_context = model.disable_adapter() if mode == "base" and hasattr(model, "disable_adapter") else nullcontext()
    with torch.inference_mode(), adapter_context:
        generated = model.generate(
            **inputs, max_new_tokens=220, do_sample=False,
            repetition_penalty=1.12, no_repeat_ngram_size=4,
        )
    raw = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    description, valid_json = extract_description(raw)
    return description, raw, valid_json


def chat_reply(question: str, history: list[dict], image: Image.Image | None, mode: str, answer_style: str) -> str:
    """Answer a local conversation turn; images are supplied only for the current turn."""
    import torch

    model, processor = load_model(mode)
    fast = answer_style == "fast"
    messages = [{"role": "system", "content": (
        "Ты ИИ-помощник геолога-седиментолога. Отвечай всегда по-русски. "
        "Разделяй наблюдения по фотографии и геологическую интерпретацию. "
        "Если признак не виден или данных недостаточно, прямо скажи об этом. "
        "Не выдумывай глубину, масштаб, минералогию или обстановку осадконакопления. "
        "Отвечай на вопрос пользователя, а не формируй JSON, если его не попросили. "
        + ("Отвечай кратко, обычно в 3–5 предложениях." if fast else "При необходимости дай подробный разбор.")
    )}]
    for turn in history[-6 if fast else -8:]:
        messages.append({"role": turn["role"], "content": turn["content"][:1200 if fast else 2000]})
    current = [{"type": "text", "text": question}]
    images = None
    if image is not None:
        preview = image.copy()
        preview.thumbnail((640, 640) if fast else (768, 768))
        current.insert(0, {"type": "image", "image": preview})
        images = [preview]
    messages.append({"role": "user", "content": current})
    prepared = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [prepared], "return_tensors": "pt"}
    if images is not None:
        kwargs["images"] = images
    inputs = processor(**kwargs).to("cuda")
    adapter_context = model.disable_adapter() if mode == "base" and hasattr(model, "disable_adapter") else nullcontext()
    with torch.inference_mode(), adapter_context:
        generated = model.generate(**inputs, max_new_tokens=220 if fast else 384, do_sample=False)
    return processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()


def relevant_pages(query: str, limit: int = 5) -> list[dict]:
    terms = [word for word in re.findall(r"[а-яёa-z]{4,}", query.lower()) if word not in {"порода", "керна", "слоев"}]
    if not terms or not PAGES.exists():
        return []
    scored = []
    with PAGES.open(encoding="utf-8") as source:
        for line in source:
            page = json.loads(line)
            text = page.get("text", "")
            if len(text) < 200:
                continue
            lower = text.lower()
            score = sum(min(lower.count(term), 8) for term in terms)
            if score:
                first = min((lower.find(term) for term in terms if term in lower), default=0)
                snippet = " ".join(text[max(0, first - 100):first + 260].split())
                scored.append((score, {"pdf_page": page["pdf_page"], "book_page": page.get("book_page"), "chapter": page.get("chapter_title"), "snippet": snippet}))
    return [item for _, item in sorted(scored, key=lambda pair: pair[0], reverse=True)[:limit]]


def r390_candidates() -> dict[str, dict]:
    if not R390_CANDIDATES.exists():
        return {}
    return {
        row["record_id"]: row
        for line in R390_CANDIDATES.read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line))
    }


def report_html(record: dict) -> str:
    review = record.get("review") or {}
    observation = review.get("observation") or record.get("draft_description") or "Описание не сформировано"
    interpretation = review.get("interpretation") or "Не установлена"
    status = "Подтверждено геологом" if review.get("approved") else "Черновик, требуется проверка геолога"
    image_path = SESSIONS / record["id"] / "photo.jpg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    esc = lambda value: html.escape(str(value))
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Описание керна {esc(record['id'])}</title>
<style>body{{font:15px/1.5 Arial,sans-serif;color:#142231;max-width:900px;margin:45px auto;padding:0 25px}}
h1{{font-size:27px}}h2{{margin-top:30px;border-bottom:1px solid #cbd5df;padding-bottom:6px}}img{{max-width:100%;max-height:850px;object-fit:contain;border:1px solid #cbd5df}}
.meta{{background:#f3f6f8;padding:16px}}.warning{{background:#fff0d5;padding:12px}}pre{{white-space:pre-wrap;font:inherit}}@media print{{button{{display:none}}body{{margin:15mm auto}}}}</style></head><body>
<button onclick="print()">Печать / сохранить PDF</button><h1>Описание керна</h1><div class="meta"><b>Скважина:</b> {esc(record.get('well') or 'не указана')}<br>
<b>Интервал:</b> {esc(record['top_m'])}–{esc(record['base_m'])} м<br><b>Дата обработки:</b> {esc(record['created_at'])}<br><b>Дата съёмки:</b> {esc(record.get('photo_taken_at') or 'не указана')}<br><b>Условия съёмки:</b> {esc(record.get('lighting') or 'не указаны')}<br><b>Статус:</b> {esc(status)}<br>
<b>Модель:</b> {esc(record.get('model') or 'не указана')}</div>
<h2>Фотография</h2><img alt="Фотография керна" src="data:image/jpeg;base64,{encoded}">
<h2>Наблюдение по фотографии</h2><pre>{esc(observation)}</pre><h2>Интерпретация среды осадконакопления</h2><pre>{esc(interpretation)}</pre>
<h2>Основания интерпретации</h2><pre>{esc(review.get('evidence') or 'Не указаны')}</pre>
<h2>Примечания и ограничения</h2><pre>{esc(review.get('notes') or 'Глубинная привязка введена пользователем. Интерпретация требует сопоставления с полным разрезом и исходными данными.')}</pre>
<p><b>Проверил:</b> {esc(review.get('reviewer') or 'не указан')}<br><b>Фото SHA-256:</b> {esc(record['photo_sha256'])}<br>
<b>Идентификатор записи:</b> {esc(record['id'])}</p>
<div class="warning">Автоматическое описание является черновиком. Статус подтверждения относится только к правкам, явно сохранённым геологом.</div>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, payload: bytes, content_type: str, status: int = 200, filename: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value: dict, status: int = 200) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        try:
            if url.path == "/":
                self.send_bytes((ROOT / "review_tool" / "sedimentologist.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/chat":
                self.send_bytes((ROOT / "review_tool" / "vlm_chat.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/r390":
                self.send_bytes((ROOT / "review_tool" / "r390_review.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/r390_candidates":
                rows = list(r390_candidates().values())
                self.send_json({"records": rows, "count": len(rows)})
            elif re.fullmatch(r"/api/r390_image/r390_\d{3}", url.path):
                record_id = url.path.rsplit("/", 1)[-1]
                candidate = r390_candidates().get(record_id)
                if not candidate:
                    raise FileNotFoundError(record_id)
                with Image.open(candidate["image"]) as opened:
                    preview = ImageOps.exif_transpose(opened).convert("RGB")
                    preview.thumbnail((950, 1500))
                    stream = io.BytesIO()
                    preview.save(stream, format="JPEG", quality=82)
                self.send_bytes(stream.getvalue(), "image/jpeg")
            elif url.path == "/api/knowledge":
                query = parse_qs(url.query).get("q", [""])[0][:150]
                self.send_json({"results": relevant_pages(query), "source": "Фациальные модели, пользовательский PDF"})
            elif re.fullmatch(r"/api/report/[0-9a-f]{32}\.html", url.path):
                session_id = url.path.split("/")[-1][:-5]
                body = report_html(read_json(session_path(session_id))).encode("utf-8")
                self.send_bytes(body, "text/html; charset=utf-8", filename=f"core_report_{session_id[:8]}.html")
            elif re.fullmatch(r"/api/record/[0-9a-f]{32}", url.path):
                self.send_json(read_json(session_path(url.path.rsplit("/", 1)[-1])))
            else:
                self.send_json({"error": "Не найдено"}, 404)
        except (ValueError, FileNotFoundError) as error:
            self.send_json({"error": str(error)}, 404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_UPLOAD_BYTES:
                self.send_json({"error": "Файл слишком большой или запрос пустой"}, 413)
                return
            payload = json.loads(self.rfile.read(length))
            if self.path == "/api/analyze":
                self.analyze(payload)
            elif self.path == "/api/chat":
                self.chat(payload)
            elif self.path == "/api/review":
                self.review(payload)
            elif self.path == "/api/r390_review":
                self.r390_review(payload)
            else:
                self.send_json({"error": "Не найдено"}, 404)
        except (ValueError, KeyError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception as error:
            self.send_json({"error": f"Ошибка локального анализа: {error}"}, 500)

    def chat(self, payload: dict) -> None:
        question = str(payload.get("message", "")).strip()
        if not 1 <= len(question) <= 4000:
            raise ValueError("Введите вопрос длиной до 4000 символов")
        mode = str(payload.get("model_mode", "base"))
        if mode not in {"base", "v8", "v5"}:
            raise ValueError("Неизвестный режим модели")
        answer_style = str(payload.get("answer_style", "fast"))
        if answer_style not in {"fast", "detailed"}:
            raise ValueError("Неизвестный режим ответа")
        history = payload.get("history", [])
        if not isinstance(history, list) or len(history) > 16 or any(
            not isinstance(turn, dict)
            or turn.get("role") not in {"user", "assistant"}
            or not isinstance(turn.get("content"), str)
            for turn in history
        ):
            raise ValueError("Неверный формат истории диалога")
        image = None
        image_data = payload.get("image")
        if image_data:
            raw = base64.b64decode(str(image_data).split(",", 1)[-1], validate=True)
            if len(raw) > 20 * 1024 * 1024:
                raise ValueError("Изображение превышает 20 МБ")
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Поддерживаются только JPEG, PNG и WebP")
                if opened.width * opened.height > 120_000_000:
                    raise ValueError("Слишком большое изображение")
                image = ImageOps.exif_transpose(opened).convert("RGB")
        with _model_lock:
            answer = chat_reply(question, history, image, mode, answer_style)
        self.send_json({"answer": answer, "model": mode, "answer_style": answer_style})

    def analyze(self, payload: dict) -> None:
        top_m, base_m = float(payload["top_m"]), float(payload["base_m"])
        if not (0 <= top_m < base_m <= 15000):
            raise ValueError("Проверьте глубины интервала")
        mode = str(payload.get("model_mode", "base"))
        if mode not in {"v5", "v8", "base"}:
            raise ValueError("Неизвестный режим модели")
        photo_taken_at = str(payload.get("photo_taken_at", ""))[:10]
        if photo_taken_at and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", photo_taken_at):
            raise ValueError("Дата съёмки должна иметь формат ГГГГ-ММ-ДД")
        lighting = str(payload.get("lighting", "не указаны"))
        if lighting not in {"дневной свет", "ультрафиолет", "другое", "не указаны"}:
            raise ValueError("Неверные условия съёмки")
        image_data = payload["image"].split(",", 1)[-1]
        raw = base64.b64decode(image_data, validate=True)
        if len(raw) > 20 * 1024 * 1024:
            raise ValueError("Изображение превышает 20 МБ")
        with Image.open(io.BytesIO(raw)) as opened:
            source_format = opened.format
            image = ImageOps.exif_transpose(opened).convert("RGB")
        extension = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}.get(source_format)
        if extension is None:
            raise ValueError("Поддерживаются только JPEG, PNG и WebP")
        if image.width < 100 or image.height < 100 or image.width * image.height > 120_000_000:
            raise ValueError("Неподходящий размер изображения")
        with _model_lock:
            description, model_raw, valid_json = describe(image, top_m, base_m, mode)
        session_id = uuid.uuid4().hex
        directory = SESSIONS / session_id
        directory.mkdir(parents=True, exist_ok=False)
        original_path = directory / f"photo_source.{extension}"
        original_path.write_bytes(raw)
        report_image = image.copy()
        report_image.thumbnail((1600, 2400))
        report_image.save(directory / "photo.jpg", "JPEG", quality=85)
        record = {
            "id": session_id, "created_at": now(), "well": str(payload.get("well", ""))[:100],
            "top_m": top_m, "base_m": base_m, "photo_sha256": hashlib.sha256(raw).hexdigest(),
            "photo_name": str(payload.get("name", ""))[:200],
            "photo_original_path": str(original_path),
            "photo_taken_at": photo_taken_at, "lighting": lighting,
            "draft_description": description, "model_raw": model_raw,
            "model_output_valid_json": valid_json,
            "model": {
                "v5": "Qwen3-VL-2B-Instruct + LoRA v5 (экспериментальная)",
                "v8": "Qwen3-VL-2B-Instruct + LoRA v8 (экспериментальная)",
                "base": "Qwen3-VL-2B-Instruct (базовая)",
            }[mode],
            "review": None, "training_approved": False,
        }
        write_json(directory / "record.json", record)
        self.send_json(record)

    def review(self, payload: dict) -> None:
        path = session_path(str(payload["id"]))
        record = read_json(path)
        observation = str(payload.get("observation", "")).strip()[:8000]
        interpretation = str(payload.get("interpretation", "")).strip()[:4000]
        reviewer = str(payload.get("reviewer", "")).strip()[:160]
        approved = payload.get("approved") is True
        if not observation:
            raise ValueError("Нужно заполнить наблюдение")
        if approved and not reviewer:
            raise ValueError("Для подтверждения укажите имя геолога")
        review = {
            "observation": observation, "interpretation": interpretation,
            "evidence": str(payload.get("evidence", "")).strip()[:4000],
            "notes": str(payload.get("notes", "")).strip()[:4000],
            "reviewer": reviewer, "approved": approved, "reviewed_at": now(),
        }
        record["review"] = review
        record["training_approved"] = approved
        write_json(path, record)
        if approved:
            REVIEWS.parent.mkdir(parents=True, exist_ok=True)
            with REVIEWS.open("a", encoding="utf-8") as ledger:
                ledger.write(json.dumps({
                    "review_id": uuid.uuid4().hex, "record_id": record["id"],
                    "image": record["photo_original_path"],
                    "top_m": record["top_m"], "base_m": record["base_m"],
                    "well": record["well"], "photo_sha256": record["photo_sha256"],
                    "photo_taken_at": record.get("photo_taken_at"), "lighting": record.get("lighting"),
                    "draft": record["draft_description"], "correction": observation,
                    "model": record["model"],
                    "interpretation": interpretation, "reviewer": reviewer,
                    "evidence": review["evidence"],
                    "approved_at": review["reviewed_at"], "training_approved": True,
                }, ensure_ascii=False) + "\n")
        self.send_json({"saved": True, "approved": approved, "report_url": f"/api/report/{record['id']}.html"})

    def r390_review(self, payload: dict) -> None:
        record_id = str(payload["record_id"])
        candidate = r390_candidates().get(record_id)
        if not candidate:
            raise ValueError("Кадр скважины 390 не найден")
        observation = str(payload.get("observation", "")).strip()[:8000]
        reviewer = str(payload.get("reviewer", "")).strip()[:160]
        top_m, base_m = float(payload["top_m"]), float(payload["base_m"])
        if not observation or not reviewer or not (0 <= top_m < base_m <= 15000):
            raise ValueError("Для подтверждения нужны описание, геолог и корректный интервал")
        review = {
            "review_id": uuid.uuid4().hex,
            "record_id": record_id,
            "well_id": "390",
            "image": candidate["image"],
            "photo_report_run_interval_m": candidate["image_depth_interval_m"],
            "reviewed_image_interval_m": [top_m, base_m],
            "source_description": candidate["source_description"],
            "source_report": candidate["description_source"],
            "observation": observation,
            "interpretation": str(payload.get("interpretation", "")).strip()[:4000],
            "reviewer": reviewer,
            "reviewed_at": now(),
            "geologist_reviewed": True,
            "training_approved": False,
            "training_blocker": "Figure contains several core columns or views; crop the reviewed area and align its depth before supervised image training.",
            "approval_basis": "Geologist checked the figure and supplied an interval-specific description; image-region alignment remains separate.",
        }
        R390_REVIEWS.parent.mkdir(parents=True, exist_ok=True)
        with _review_lock, R390_REVIEWS.open("a", encoding="utf-8") as ledger:
            ledger.write(json.dumps(review, ensure_ascii=False) + "\n")
        self.send_json({"saved": True, "review_id": review["review_id"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Локальный ИИ-седиментолог: http://127.0.0.1:{args.port}", flush=True)

    def warmup_base() -> None:
        try:
            with _model_lock:
                load_model("base")
            print("Базовая Qwen3-VL-2B загружена и готова к диалогу", flush=True)
        except Exception as error:
            print(f"Предварительная загрузка VLM не удалась: {error}", flush=True)

    threading.Thread(target=warmup_base, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
