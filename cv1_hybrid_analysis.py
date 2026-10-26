"""CV-1 hybrid inference: YOLO class polygons, optional local Ollama VLM descriptions.

No depth inference, no automatic relabeling. Run:
 python cv1_hybrid_analysis.py photo.jpg --model best.pt --vlm-model qwen2.5vl:7b
"""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

def describe_with_ollama(image_bytes: bytes, class_name: str, model: str, endpoint: str, timeout: int = 180) -> str:
    prompt = (
        "Ты геолог-седиментолог. Опиши ТОЛЬКО видимые признаки участка керна на фотографии. "
        "Класс фации, определённый отдельной моделью сегментации: " + class_name + ". "
        "Не меняй класс и не выдумывай глубину, минералы, пористость или обстановку осадконакопления. "
        "Раздели ответ на: литология (если различима), цвет, зернистость (если различима), "
        "слоистость, текстуры, включения, контакты (если видны), ограничения наблюдения. "
        "Если признак нельзя определить визуально, так и напиши. Ответ по-русски."
    )
    payload = json.dumps({"model": model, "prompt": prompt,
                          "images": [base64.b64encode(image_bytes).decode("ascii")],
                          "stream": False}, ensure_ascii=False).encode("utf-8")
    req = Request(endpoint.rstrip("/") + "/api/generate", data=payload,
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as response:
        return str(json.load(response).get("response") or "").strip()

def analyze(photo: str | Path, model_path: str | Path, output: str | Path,
            vlm_model: str = "", endpoint: str = "http://127.0.0.1:11434",
            confidence: float = 0.25) -> dict:
    from ultralytics import YOLO
    from PIL import Image, ImageDraw
    photo = Path(photo).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not photo.is_file() or not model_path.is_file():
        raise FileNotFoundError("Нужны существующие фотография и best.pt")
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(photo) as im:
        original = im.convert("RGB")
    # Ultralytics expects three image channels; TIFF/PNG may be RGBA or grayscale.
    import numpy as np
    rgb_array = np.ascontiguousarray(np.asarray(original, dtype=np.uint8))
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"RGB conversion failed: shape={rgb_array.shape}")
    # A CHW float tensor bypasses image loaders that may add an alpha channel.
    import torch
    tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).contiguous()
    if tensor.shape[1] != 3:
        raise ValueError(f"YOLO input must have 3 channels, got {tuple(tensor.shape)}")
    yolo = YOLO(str(model_path))
    try:
        result = yolo.predict(source=tensor, conf=confidence, verbose=False)[0]
    except RuntimeError as exc:
        if "channels" in str(exc) or "groups=" in str(exc):
            raise RuntimeError(
                f"YOLO channel mismatch despite verified RGB tensor {tuple(tensor.shape)}. "
                "The selected best.pt may contain an incompatible model architecture "
                "or an internal 4-channel layer. Check that it is a segmentation checkpoint "
                "trained with 3-channel RGB images. Original error: " + str(exc)
            ) from exc
        raise
    if result.masks is None or result.boxes is None:
        polygons = []
    else:
        polygons = result.masks.xy
    records = []
    for i, points in enumerate(polygons):
        cls = int(result.boxes.cls[i].item())
        score = float(result.boxes.conf[i].item())
        name = str(result.names.get(cls, cls))
        xy = [(float(x), float(y)) for x, y in points]
        if len(xy) < 3:
            continue
        mask = Image.new("L", original.size, 0)
        ImageDraw.Draw(mask).polygon(xy, fill=255)
        bbox = mask.getbbox()
        if bbox is None:
            continue
        crop = Image.new("RGB", original.size, (255, 255, 255))
        crop.paste(original, (0, 0), mask)
        crop = crop.crop(bbox)
        image_path = output / f"segment_{i:03d}.png"
        crop.save(image_path)
        item = {"segment_id": i, "class_id": cls, "class_name": name,
                "confidence": round(score, 4), "polygon_pixels": xy,
                "crop": str(image_path), "description": None,
                "vlm_status": "not_requested" if not vlm_model else "pending",
                "class_source": "yolo"}
        if vlm_model:
            try:
                stream = io.BytesIO()
                crop.save(stream, format="PNG")
                item["description"] = describe_with_ollama(stream.getvalue(), name, vlm_model, endpoint)
                item["vlm_status"] = "completed" if item["description"] else "empty_response"
            except (URLError, TimeoutError, OSError, ValueError) as exc:
                item["vlm_status"] = "error"
                item["vlm_error"] = str(exc)
        records.append(item)
    report = {"photo": str(photo), "yolo_model": str(model_path),
              "vlm_model": vlm_model or None, "segments": records,
              "depths_verified": False,
              "note": "YOLO classes are preserved. VLM describes visible evidence only; no depth matching."}
    report_file = output / "hybrid_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"segments": len(records), "vlm_completed": sum(x["vlm_status"] == "completed" for x in records),
            "vlm_errors": sum(x["vlm_status"] == "error" for x in records),
            "report": str(report_file), "output": str(output)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path)
    parser.add_argument("--model", type=Path, default=Path("best.pt"))
    parser.add_argument("--output", type=Path, default=Path("_cv1_hybrid_output"))
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    print(json.dumps(analyze(args.photo, args.model, args.output, args.vlm_model,
                             args.endpoint, args.conf), ensure_ascii=False, indent=2))
