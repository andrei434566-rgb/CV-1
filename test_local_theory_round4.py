"""Train-only offline reference delivery; these tests do not score VLM weights."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.infrastructure import local_theory_lookup as lookup


ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "app/resources/theory"
BASE = RESOURCES / "theory_cards_v5.json"
SUPPLEMENT = RESOURCES / "theory_cards_round4.json"
SOURCE = ROOT / "vlm_source/theory_round4/new_source_cards.json"


class LocalTheoryRound4Tests(unittest.TestCase):
    def setUp(self):
        lookup._load_cards.cache_clear()
        self.addCleanup(lookup._load_cards.cache_clear)
        self.source = json.loads(SOURCE.read_text(encoding="utf-8"))
        self.supplement = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))
        self.train = [card for card in self.source["cards"] if card["split"] == "train"]
        self.validation = [card for card in self.source["cards"] if card["split"] == "validation"]

    def test_all_33_train_answers_are_available_without_loading_vlm(self):
        self.assertEqual(len(self.train), 33)
        self.assertEqual(len(self.supplement["cards"]), 33)
        self.assertEqual(self.supplement["source_manifest_sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        for card in self.train:
            with self.subTest(identity=card["id"]):
                self.assertEqual(lookup.lookup_theory_answer(card["question"], BASE), card["answer"])

    def test_validation_ids_questions_and_targets_are_not_shipped_as_reference(self):
        resource = SUPPLEMENT.read_text(encoding="utf-8")
        loaded_ids = {card["id"] for card in lookup._load_cards(str(BASE))["cards"]}
        self.assertEqual(len(self.validation), 8)
        self.assertEqual({card["id"] for card in self.supplement["cards"]}, {card["id"] for card in self.train})
        for card in self.validation:
            with self.subTest(identity=card["id"]):
                self.assertNotIn(card["id"], resource)
                self.assertNotIn(card["question"], resource)
                self.assertNotIn(card["answer"], resource)
                self.assertNotIn(card["id"], loaded_ids)

    def test_delivery_is_exact_source_checked_reference_without_expert_certification(self):
        self.assertEqual(self.supplement["delivery"], "offline_reference_answers_not_model_weight_training")
        self.assertIs(self.supplement["geologist_reviewed"], False)
        for card in self.supplement["cards"]:
            self.assertIs(card["exact_only"], True)
            self.assertIs(card["holdout"], False)
            self.assertEqual(card["aliases"], [])
            meta = card["metadata"]
            self.assertEqual(meta["origin_split"], "train")
            self.assertEqual(meta["origin_source_group"], "theory_sources_20261027")
            self.assertEqual(meta["reference_scope"], "general_theory")
            self.assertTrue(meta["review_status"].startswith("source_checked"))
            self.assertIs(meta["geology_approved"], False)

    def test_sources_are_added_only_on_request(self):
        card = self.train[0]
        answer = lookup.lookup_theory_answer(card["question"], BASE)
        self.assertEqual(answer, card["answer"])
        self.assertNotIn("Источники:", answer)
        self.assertNotRegex(answer, r"https?://")
        sourced = lookup.lookup_theory_answer(card["question"] + " Дай источники.", BASE, include_sources=True)
        self.assertTrue(sourced.startswith(card["answer"] + "\n\nИсточники:\n"))
        self.assertIn(self.source["sources"][card["sources"][0]]["url"], sourced)

    def test_established_answers_and_aliases_keep_their_original_matching(self):
        original = json.loads(BASE.read_text(encoding="utf-8"))
        round2 = json.loads((RESOURCES / "theory_cards_round2.json").read_text(encoding="utf-8"))
        original = lookup._merge_reference_supplement(original, round2)
        for card in original["cards"]:
            for question in [card["question"], *card.get("aliases", [])]:
                with self.subTest(identity=card["id"], question=question):
                    with patch.object(lookup, "_load_cards", return_value=original):
                        before = lookup.lookup_theory_answer(question, BASE)
                    self.assertEqual(lookup.lookup_theory_answer(question, BASE), before)

    def test_new_cards_do_not_change_fuzzy_ranking(self):
        only_new = lookup._merge_reference_supplement(
            {"cards": [], "sources": {}}, self.supplement, expected_source_group="theory_sources_20261027",
        )
        self.assertEqual(len(only_new["cards"]), 33)
        with patch.object(lookup, "_load_cards", return_value=only_new):
            for card in self.train:
                self.assertIsNone(lookup.lookup_theory_answer("Объясни подробнее: " + card["question"], BASE))

    def test_wrong_group_holdout_or_false_expert_flag_is_rejected(self):
        mutations = [lambda c: c["metadata"].update(origin_source_group="theory_sources_20261025"),
                     lambda c: c["metadata"].update(origin_split="validation"),
                     lambda c: c.update(holdout=True), lambda c: c["metadata"].update(geology_approved=True)]
        for mutate in mutations:
            supplement = copy.deepcopy(self.supplement)
            card = supplement["cards"][0]
            mutate(card)
            merged = lookup._merge_reference_supplement(
                {"cards": [], "sources": {}}, supplement, expected_source_group="theory_sources_20261027",
            )
            self.assertNotIn(card["id"], {value["id"] for value in merged["cards"]})

    def test_bad_round4_file_does_not_break_established_answers(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "theory_cards_v5.json"
            base.write_text(json.dumps({"sources": {}, "cards": [{"id": "old", "question": "Старый проверочный вопрос?",
                                                                    "answer": "Прежний ответ.", "sources": []}]}), encoding="utf-8")
            base.with_name("theory_cards_round4.json").write_text("not json", encoding="utf-8")
            self.assertEqual(lookup.lookup_theory_answer("Старый проверочный вопрос?", base), "Прежний ответ.")


if __name__ == "__main__":
    unittest.main()
