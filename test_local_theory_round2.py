"""Offline CPU checks for reference delivery, not learned VLM accuracy."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from app.infrastructure import local_theory_lookup as lookup
from app.infrastructure.local_chat_intents import needs_selected_image
from app.infrastructure.local_knowledge_search import search_folders

ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "app/resources/theory"
BASE = RESOURCES / "theory_cards_v5.json"
SUPPLEMENT = RESOURCES / "theory_cards_round2.json"
SOURCE = ROOT / "vlm_source/theory_round2"
TEMP_ROOT = ROOT / "open_datasets/test_tmp"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


class LocalTheoryRound2Tests(unittest.TestCase):
    def setUp(self):
        lookup._load_cards.cache_clear()
        self.addCleanup(lookup._load_cards.cache_clear)
        self.source_rows = jsonl(SOURCE / "train.jsonl")
        self.supplement = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))
        self.base = json.loads(BASE.read_text(encoding="utf-8"))

    def temp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        context = tempfile.TemporaryDirectory(prefix="theory_reference_", dir=TEMP_ROOT)
        self.addCleanup(context.cleanup)
        return Path(context.name)

    def test_all_48_source_train_answers_are_exactly_accessible_without_model(self):
        self.assertEqual(len(self.source_rows), 48)
        self.assertEqual(len(self.supplement["cards"]), 48)
        for row in self.source_rows:
            question, answer = [turn["value"] for turn in row["conversations"]]
            with self.subTest(record_id=row["metadata"]["record_id"]):
                self.assertEqual(lookup.lookup_theory_answer(question, BASE), answer)
        loaded = lookup._load_cards(str(BASE))
        established_ids = {card["id"] for card in [*self.base["cards"], *self.supplement["cards"]]}
        self.assertEqual(len(established_ids), 175)
        self.assertTrue(established_ids.issubset({card["id"] for card in loaded["cards"]}))

    def test_supplement_has_no_holdout_case_study_or_invented_aliases(self):
        expected = {row["metadata"]["record_id"] for row in self.source_rows}
        self.assertEqual({card["id"] for card in self.supplement["cards"]}, expected)
        prohibited = [*jsonl(SOURCE / "holdout.jsonl"), *jsonl(SOURCE / "case_study_russia_train.jsonl"),
                      *jsonl(SOURCE / "case_study_russia_holdout.jsonl")]
        self.assertEqual(len(prohibited), 16)  # 13 heldout + 3 case-study train records.
        resource_texts = [SUPPLEMENT.read_text(encoding="utf-8"),
                          (RESOURCES / "theory_round2_ru.md").read_text(encoding="utf-8")]
        for row in prohibited:
            for text in resource_texts:
                self.assertNotIn(row["metadata"]["record_id"], text)
        for card in self.supplement["cards"]:
            self.assertIs(card["holdout"], False)
            self.assertIs(card["exact_only"], True)
            self.assertEqual(card["aliases"], [])
            self.assertEqual(card["metadata"]["origin_split"], "train")
            self.assertIs(card["metadata"]["geology_approved"], False)

    def test_source_urls_are_shown_only_when_requested(self):
        for row in self.source_rows:
            question, answer = [turn["value"] for turn in row["conversations"]]
            default = lookup.lookup_theory_answer(question, BASE)
            with_sources = lookup.lookup_theory_answer(question, BASE, include_sources=True)
            with self.subTest(record_id=row["metadata"]["record_id"]):
                self.assertEqual(default, answer)
                self.assertNotRegex(default, r"https?://|[A-Za-z]:[\\/]")
                self.assertNotIn("Источники:", default)
                self.assertTrue(with_sources.startswith(answer + "\n\nИсточники:\n"))
                for source in row["metadata"]["sources"]:
                    self.assertIn(source["title"], with_sources)
                    self.assertIn(source["url"], with_sources)

    def test_exact_question_with_explicit_source_suffix_retains_exact_only_match(self):
        for row in self.source_rows:
            question, answer = [turn["value"] for turn in row["conversations"]]
            with self.subTest(record_id=row["metadata"]["record_id"]):
                result = lookup.lookup_theory_answer(question + " Дай источники.", BASE, include_sources=True)
                self.assertTrue(result.startswith(answer + "\n\nИсточники:\n"))
                self.assertIn(row["metadata"]["sources"][0]["url"], result)
        self.assertEqual(lookup._question_without_source_suffix("Как описать осадок?\nУкажите ссылки, пожалуйста."),
                         "Как описать осадок")
        self.assertEqual(lookup._question_without_source_suffix("Что означает источник осадка?"),
                         "Что означает источник осадка?")

    def test_exact_only_cards_do_not_answer_reworded_or_broader_queries(self):
        only_new = lookup._merge_reference_supplement({"sources": {}, "cards": []}, self.supplement)
        with patch.object(lookup, "_load_cards", return_value=only_new):
            for row in self.source_rows:
                question = row["conversations"][0]["value"]
                self.assertIsNone(lookup.lookup_theory_answer("Объясни подробнее: " + question, BASE))
            self.assertIsNone(lookup.lookup_theory_answer("Что такое порода и как она образуется?", BASE))

    def test_legacy_fuzzy_matching_is_not_changed_by_48_exact_only_cards(self):
        card = self.base["cards"][0]
        question = "Поясни: " + card["question"]
        merged = lookup._merge_reference_supplement(self.base, self.supplement)
        with patch.object(lookup, "_load_cards", return_value=self.base):
            old_answer = lookup.lookup_theory_answer(question, BASE)
        self.assertEqual(old_answer, card["answer"])
        with patch.object(lookup, "_load_cards", return_value=merged):
            self.assertEqual(lookup.lookup_theory_answer(question, BASE), old_answer)

    def test_canonical_mottling_question_and_open_alias_use_corrected_sources(self):
        questions = ["Пятнистая окраска автоматически означает ходы организмов?",
                     "Когда пятнистость не следует записывать как биотурбацию?",
                     "Почему пятнистость не обязательно биотурбация?"]
        for question in questions:
            answer = lookup.lookup_theory_answer(question, BASE)
            self.assertIn("отдельные ходы не обязательны", answer)
            self.assertNotIn("Для биотурбации нужны форма ходов", answer)
            self.assertFalse(answer.startswith("Нет."))
            self.assertNotIn("https://", answer)
            sourced = lookup.lookup_theory_answer(question + " Дай источники.", BASE, include_sources=True)
            self.assertIn("https://iodp.tamu.edu/publications/TN/TNote_5.pdf", sourced)

    def test_both_false_mottling_md_passages_are_corrected_and_attributed(self):
        for path in [RESOURCES / "theory_structures_v4_ru.md",
                     ROOT.parent / "vlm_core_v1/data/knowledge/theory_structures_v4_ru.md"]:
            # The installed local knowledge mirror is outside the repo in CI.
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("Для биотурбации нужны форма ходов и их отношение к ламинациям", text)
            self.assertIn("отдельные ходы не обязательны при полной переработке осадка", text)
            self.assertIn("https://iodp.tamu.edu/publications/TN/TNote_5.pdf", text)

    def test_new_md_preserves_each_question_answer_and_context(self):
        bundled = (RESOURCES / "theory_round2_ru.md").read_text(encoding="utf-8")
        self.assertEqual(bundled.count("\n## "), 48)
        for row in self.source_rows:
            question, answer = [turn["value"] for turn in row["conversations"]]
            self.assertIn("## " + question, bundled)
            self.assertIn(answer, bundled)
            self.assertIn("ID: " + row["metadata"]["record_id"], bundled)
            for source in row["metadata"]["sources"]:
                self.assertIn(source["title"] + " — " + source["url"], bundled)
        mirror = ROOT.parent / "vlm_core_v1/data/knowledge/theory_round2_ru.md"
        if mirror.exists():
            self.assertEqual(mirror.read_text(encoding="utf-8"), bundled)

    def test_explicit_photo_requests_are_not_replaced_by_reference_answers(self):
        for question in ["Какие признаки биотурбации на этом снимке?",
                         "Опиши пятнистость по прикрепленному изображению.",
                         "Определи породу на загруженном фото."]:
            self.assertTrue(needs_selected_image(question))
            self.assertIsNone(lookup.lookup_theory_answer(question, BASE))
        # Inspect the actual UI routing without importing Qt or starting a model.
        tree = ast.parse((ROOT / "app/ui/windows/main_window.py").read_text(encoding="utf-8-sig"))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_send_vlm_chat")
        parents = {child: parent for parent in ast.walk(function) for child in ast.iter_child_nodes(parent)}
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "lookup_theory_answer"]
        self.assertTrue(calls)
        for call in calls:
            node, guarded = call, False
            while node in parents:
                node = parents[node]
                if isinstance(node, ast.If) and "not photo_request" in ast.unparse(node.test):
                    guarded = True
            self.assertTrue(guarded)

    def test_supplement_failure_and_missing_file_preserve_legacy_answers(self):
        folder = self.temp()
        path = folder / "theory_cards_v5.json"
        fixture = {"sources": {}, "cards": [{"id": "fixture", "question": "Проверочный старый вопрос?",
                                               "answer": "Старый справочный ответ.", "sources": []}]}
        path.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
        self.assertEqual(lookup.lookup_theory_answer("Проверочный старый вопрос?", path), "Старый справочный ответ.")
        for content in ("not json", "[]"):
            path.with_name("theory_cards_round2.json").write_text(content, encoding="utf-8")
            lookup._load_cards.cache_clear()
            self.assertEqual(lookup.lookup_theory_answer("Проверочный старый вопрос?", path), "Старый справочный ответ.")

    def test_unchecked_holdout_case_scope_or_broad_alias_supplement_cards_are_refused(self):
        mutations = [lambda c: c.update(holdout=True),
                     lambda c: c.update(aliases=["Что такое порода?"]),
                     lambda c: c["metadata"].update(origin_split="holdout"),
                     lambda c: c["metadata"].update(origin_source_group="russia_yamal_case_study_2025"),
                     lambda c: c["metadata"].update(geology_approved=True),
                     lambda c: c["metadata"].update(review_status="not_checked")]
        for mutate in mutations:
            supplement = copy.deepcopy(self.supplement)
            first = supplement["cards"][0]
            mutate(first)
            merged = lookup._merge_reference_supplement(self.base, supplement)
            self.assertNotIn(first["id"], {c["id"] for c in merged["cards"]})

    def test_generic_search_reads_new_md_but_not_training_or_card_json(self):
        folder = self.temp()
        new_md = folder / "theory_round2_ru.md"
        new_md.write_text((RESOURCES / "theory_round2_ru.md").read_text(encoding="utf-8"), encoding="utf-8")
        for name in ["theory_training_v5.jsonl", "theory_cards_v5.json", "theory_holdout_v5.jsonl"]:
            (folder / name).write_text("ball-and-pillow test marker that must be skipped", encoding="utf-8")
        hits = search_folders("ball pillow", [str(folder)])
        self.assertTrue(hits)
        self.assertTrue(all(Path(hit["path"]).name == "theory_round2_ru.md" for hit in hits))
        self.assertTrue(any("ball-and-pillow" in hit["excerpt"] for hit in hits))

    def test_pyinstaller_datas_includes_the_app_resource_tree(self):
        tree = ast.parse((ROOT / "Kern_Analyzer.spec").read_text(encoding="utf-8-sig"))
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "datas" for target in node.targets))
        self.assertIn(("app", "app"), ast.literal_eval(assignment.value))
        self.assertTrue(SUPPLEMENT.is_file())
        self.assertTrue((RESOURCES / "theory_round2_ru.md").is_file())


if __name__ == "__main__":
    unittest.main()
