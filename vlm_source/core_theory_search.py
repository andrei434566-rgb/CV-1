"""Offline, source-cited search over the user's facies book and curated theory cards.

Examples:
  python core_theory_search.py build
  python core_theory_search.py search "градационная слоистость"
  python core_theory_search.py ask "Как отличать слоистость от повреждения керна?"

The index contains text from a user-provided copyrighted book: keep it local.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
KNOWLEDGE = ROOT / "data" / "knowledge"
INDEX = KNOWLEDGE / "core_theory_local.sqlite3"
BOOK = KNOWLEDGE / "gazprom_facies_models_pages.jsonl"
V1 = KNOWLEDGE / "text_knowledge_training_v1.jsonl"
V2 = KNOWLEDGE / "theory_cards_v2.json"
DS542 = KNOWLEDGE / "theory_cards_usgs_ds542.json"
LILY = KNOWLEDGE / "theory_cards_iodp_lily.json"
TM5E1 = KNOWLEDGE / "theory_cards_usgs_tm5e1.json"
DS1061 = KNOWLEDGE / "theory_cards_usgs_ds1061.json"
BASE = ROOT / "model_cache" / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / "89644892e4d85e24eaac8bacfd4f463576704203"
STOPWORDS = {
    "какие", "какой", "какая", "какое", "когда", "почему", "зачем", "нужно",
    "можно", "керна", "керне", "керн", "чтобы", "который", "между", "после",
    "описать", "описание", "описании", "фото", "снимке", "снимок",
}


def chunks(text: str, limit: int = 1050, overlap: int = 120):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        if end < len(text):
            boundary = text.rfind(". ", start + limit // 2, end)
            if boundary > start:
                end = boundary + 1
        yield text[start:end].strip()
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def build_index(path: Path = INDEX) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.execute("CREATE VIRTUAL TABLE docs USING fts5(title, body, source_id UNINDEXED, locator UNINDEXED, url UNINDEXED, kind UNINDEXED, tokenize='unicode61')")
    counts = {"book_chunks": 0, "old_cards": 0, "new_cards": 0}

    with BOOK.open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            if page["extraction_status"] != "text":
                continue
            for number, fragment in enumerate(chunks(page["text"]), start=1):
                locator = f"PDF с. {page['pdf_page']}"
                if page["book_page"] is not None:
                    locator += f", книга с. {page['book_page']}"
                locator += f", фрагмент {number}"
                db.execute(
                    "INSERT INTO docs (title,body,source_id,locator,url,kind) VALUES (?,?,?,?,?,?)",
                    (page["chapter_title"] or "Фациальные модели", fragment,
                     page["source_id"], locator, "", "user_book"),
                )
                counts["book_chunks"] += 1

    with V1.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            question, answer = (entry["value"] for entry in row["conversations"])
            sources = row["metadata"].get("sources", [])
            source = sources[0] if sources else {}
            db.execute(
                "INSERT INTO docs (title,body,source_id,locator,url,kind) VALUES (?,?,?,?,?,?)",
                (question, answer, source.get("source_id", "curated_v1"),
                 source.get("locator", ""), "", "curated_card"),
            )
            counts["old_cards"] += 1

    for card_file in (V2, DS542, LILY, TM5E1, DS1061):
        corpus = json.loads(card_file.read_text(encoding="utf-8"))
        sources = {source["id"]: source for source in corpus["sources"]}
        for card in corpus["cards"]:
            source = sources[card["source"]]
            db.execute(
                "INSERT INTO docs (title,body,source_id,locator,url,kind) VALUES (?,?,?,?,?,?)",
                (card["question"], card["answer"], source["id"],
                 card["locator"], source["url"], "curated_card"),
            )
            counts["new_cards"] += 1
    db.commit()
    db.close()
    return counts


def search(query: str, path: Path = INDEX, limit: int = 6) -> list[dict]:
    words = re.findall(r"[а-яёa-z0-9]{4,}", query.lower())
    terms = [word[:5] + "*" if len(word) > 5 else word + "*"
             for word in words if word not in STOPWORDS]
    terms = list(dict.fromkeys(terms))[:8]
    if not terms:
        return []
    fts_query = " OR ".join(terms)
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT title, snippet(docs,1,'','',' … ',55), source_id, locator, url, kind, bm25(docs,2.0,1.0) "
            "FROM docs WHERE docs MATCH ? ORDER BY bm25(docs,2.0,1.0) LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    return [dict(zip(("title", "excerpt", "source_id", "locator", "url", "kind", "score"), row))
            for row in rows]


def ask(query: str, hits: list[dict], adapter: Path | None, max_new_tokens: int) -> str:
    if not hits:
        return "В локальном справочнике не найдены подходящие материалы."
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch required for local Qwen inference; 'search' works without CUDA")
    prompt_sources = "\n".join(
        f"[{i}] {h['title']} ({h['source_id']}; {h['locator']}; {h['url'] or 'локальная книга'}): {h['excerpt']}"
        for i, h in enumerate(hits, start=1)
    )
    prompt = (
        "Ответь по-русски в 2–4 предложениях на вопрос о керне, используя только приведённые выдержки. "
        "Отделяй непосредственно наблюдаемые признаки от интерпретации и не делай вывод "
        "о конкретной фотографии, если её нет. Если материалов недостаточно, скажи об этом. "
        "В конце укажи номера использованных источников в квадратных скобках.\n\n"
        f"Вопрос: {query}\n\nИсточники:\n{prompt_sources}"
    )
    processor = AutoProcessor.from_pretrained(BASE)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="cuda"
    )
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "search", "ask"))
    parser.add_argument("query", nargs="?")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()
    if args.mode == "build":
        print(json.dumps(build_index(), ensure_ascii=False))
        return
    if not args.query:
        parser.error("query required for search/ask")
    if not INDEX.is_file():
        build_index()
    hits = search(args.query, limit=args.limit)
    if args.mode == "search":
        print(json.dumps(hits, ensure_ascii=False, indent=2))
    else:
        print(ask(args.query, hits, args.adapter, args.max_new_tokens))
        print("\nНайденные материалы для проверки:")
        for index, hit in enumerate(hits, 1):
            print(f"[{index}] {hit['source_id']} — {hit['locator']} {hit['url']}")


if __name__ == "__main__":
    main()
