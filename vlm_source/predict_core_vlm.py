"""Run the experimental core-description LoRA adapter on one core photo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from long_core_mosaic import prepare as prepare_long_core


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE = (
    PROJECT_ROOT
    / "model_cache"
    / "hub"
    / "models--Qwen--Qwen3-VL-2B-Instruct"
    / "snapshots"
    / "89644892e4d85e24eaac8bacfd4f463576704203"
)
DEFAULT_ADAPTER = PROJECT_ROOT / "outputs" / "core_vlm_v3_knowledge_lora"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--top-m", type=float, required=True)
    parser.add_argument("--base-m", type=float, required=True)
    parser.add_argument(
        "--containing-layer-top-m",
        type=float,
        help="Top depth of the full source layer when the input image is an interval crop.",
    )
    parser.add_argument(
        "--containing-layer-base-m",
        type=float,
        help="Base depth of the full source layer when the input image is an interval crop.",
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--base-only", action="store_true", help="Use the base Qwen without a project LoRA adapter.")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--tile-wide-photo", action="store_true",
                        help="Stack a very wide, single-row core into ordered panels before inference.")
    parser.add_argument("--plain-description", action="store_true",
                        help="Experimental one-sentence visual description without JSON formatting.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required for this model.")

    processor = AutoProcessor.from_pretrained(args.base_model)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model = base if args.base_only else PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    with Image.open(args.image) as opened:
        if args.tile_wide_photo:
            image, tile_count = prepare_long_core(opened, args.max_image_side)
        else:
            image = opened.convert("RGB")
            image.thumbnail((args.max_image_side, args.max_image_side))
            tile_count = 1

    if (args.containing_layer_top_m is None) != (args.containing_layer_base_m is None):
        parser.error("Pass both containing-layer depth arguments together.")
    if args.containing_layer_top_m is not None:
        prompt = (
            "Это увеличенный фрагмент керна внутри описываемого слоя "
            f"{args.containing_layer_top_m:.2f}–{args.containing_layer_base_m:.2f} м; "
            f"на фото показан участок примерно {args.top_m:.2f}–{args.base_m:.2f} м. "
            "Составь краткое нейтральное литологическое описание на русском языке по непосредственно "
            "видимым признакам. Верни только JSON с ключом description_segments и одним объектом "
            "с top_m, base_m и description. Не назначай среду осадконакопления и не придумывай "
            "свойства, которых нет в видимой части. Игнорируй номера образцов и надписи на фото."
        )
    else:
        prompt = (
            f"Опиши только непосредственно различимые на фотографии признаки керна "
            f"в пределах интервала {args.top_m}–{args.base_m} м. "
            "Верни только корректный JSON с ключом description_segments; для каждого "
            "сегмента укажи top_m, base_m и краткое описание на русском языке. "
            "Если внутренние границы слоёв нельзя уверенно привязать к шкале глубин, "
            "верни один сегмент на весь заданный интервал. Не угадывай состав, "
            "зернистость, минералы, крепость, пористость, трещины или биотурбацию, "
            "если это не различимо на фото. Не выдумывай ссылки на номера описаний, "
            "образцов или источники. Не назначай среду осадконакопления."
        )
    if tile_count > 1:
        prompt = (
            f"Это одно очень длинное фото керна {args.top_m}–{args.base_m} м, "
            f"разрезанное на {tile_count} горизонтальных фрагмента. Они расположены "
            "сверху вниз в том же порядке, что слева направо на исходном фото. "
            "Не считай границы фрагментов геологическими контактами. "
            "Опиши только признаки, уверенно видимые на самих фрагментах. "
            "Верни только JSON с ключом description_segments и ровно одним объектом: "
            f"top_m={args.top_m}, base_m={args.base_m}, description — краткий русский текст. "
            "Не назначай породе название, размер зерна, состав или фацию, если они не "
            "различимы; не придумывай внутренние глубинные границы."
        )
    if args.plain_description:
        layout = (f"Это один керн, разделённый на {tile_count} последовательных полос сверху вниз. "
                  if tile_count > 1 else "")
        prompt = (layout + "Одним коротким предложением по-русски назови только хорошо видимые "
                  "на поверхности признаки материала. Не указывай глубины, фацию, точный размер "
                  "зёрен или состав, если их нельзя уверенно определить на снимке. Без JSON.")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    completion = generated[:, inputs["input_ids"].shape[1] :]
    result = processor.batch_decode(completion, skip_special_tokens=True)[0].strip()
    if result.startswith("```"):
        result = result.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    if args.plain_description:
        print(result)
        return
    try:
        result = json.dumps(json.loads(result), ensure_ascii=False, indent=2)
    except json.JSONDecodeError as error:
        print(f"WARNING: model output is not valid JSON ({error}).", file=sys.stderr)
    print(result)


if __name__ == "__main__":
    main()
