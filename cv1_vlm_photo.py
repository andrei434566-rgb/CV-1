"""Independent local VLM geological photo description; does not use YOLO or invent depths."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from PIL import Image, ImageOps

PROMPT = """Ты эксперт по описанию осадочных пород по фотографиям керна.
Сначала определи, действительно ли на изображении виден керн. Если нет — скажи об этом и не выдумывай описание.
Если керн виден, опиши его на русском языке:
1. Общее строение изображения и ориентировочное число визуально различимых участков (без точных границ).
2. Для каждого визуально различимого участка — расположение на изображении, цвет, предполагаемая литология (если можно различить), зернистость, слоистость, текстуры, включения и видимые контакты.
3. Возможные фациальные признаки и гипотезы об условиях осадконакопления, явно отделяя интерпретации от наблюдений.
4. Что нельзя достоверно установить по фотографии.
Не придумывай глубины, масштабы, номера фаций, минералогию и лабораторные свойства.
Не утверждай, что геологические границы определены точно. Не используй сведения, которых нет на изображении.
Ответ структурируй по разделам. """

def analyze(photo: str | Path, output: str | Path, model: str = "qwen2.5vl:7b",
            endpoint: str = "http://127.0.0.1:11434", max_dimension: int = 1800) -> dict:
    photo, output = Path(photo).expanduser().resolve(), Path(output).expanduser().resolve()
    if not photo.is_file():
        raise FileNotFoundError(photo)
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(photo) as src:
        image = ImageOps.exif_transpose(src).convert("RGB")
        image.thumbnail((max_dimension, max_dimension))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    payload = json.dumps({
        "model": model, "prompt": PROMPT,
        "images": [base64.b64encode(buffer.getvalue()).decode("ascii")],
        "stream": False, "options": {"temperature": 0.2}
    }, ensure_ascii=False).encode("utf-8")
    request = Request(endpoint.rstrip("/") + "/api/generate", data=payload,
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=600) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Нет ответа Ollama. Проверьте, что Ollama запущена и модель загружена: " + str(exc)) from exc
    description = str(result.get("response") or "").strip()
    if not description:
        raise RuntimeError("Ollama вернула пустое описание. Проверьте, что выбрана VLM с поддержкой изображений.")
    record = {
        "photo": str(photo), "model": model, "image_size_sent": list(image.size),
        "description": description, "segmentation_used": False,
        "facies_verified": False, "depths_verified": False,
        "note": "Exploratory VLM description, not expert-confirmed facies or pixel segmentation."
    }
    report = output / "vlm_photo_report.json"
    report.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "vlm_description.txt").write_text(description, encoding="utf-8")
    return {"report": str(report), "description_file": str(output / "vlm_description.txt"),
            "description": description, "model": model}
