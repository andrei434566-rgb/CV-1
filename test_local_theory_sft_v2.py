"""Checked TRAIN reference delivery only; these checks never score neural weights."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from app.infrastructure import local_theory_lookup as lookup
from scripts.export_geology_sft_v2_reference import build

ROOT = Path(__file__).resolve().parent
RESOURCES = ROOT / "app/resources/theory"
BASE = RESOURCES / "theory_cards_v5.json"
SUPPLEMENT = RESOURCES / "theory_cards_sft_v2.json"
SOURCE = ROOT / "vlm_source/geology_sft_v2_20261027"
GROUP = "geology_sft_v2_20261027"


class LocalTheorySftV2Tests(unittest.TestCase):
    def setUp(self):
        lookup._load_cards.cache_clear()
        self.addCleanup(lookup._load_cards.cache_clear)
        self.supplement = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))
        self.train = json.loads((SOURCE / "training_cards.json").read_text(encoding="utf-8"))["cards"]

    def old_corpus(self):
        corpus = json.loads(BASE.read_text(encoding="utf-8"))
        for name, group in (("theory_cards_round2.json", "theory_sources_20261025"),
                            ("theory_cards_round4.json", "theory_sources_20261027")):
            corpus = lookup._merge_reference_supplement(
                corpus, json.loads((RESOURCES / name).read_text(encoding="utf-8")), expected_source_group=group)
        return corpus

    def test_resource_is_reproducible_train_only_source_attributed_and_not_weight_learning(self):
        self.assertEqual(self.supplement, build())
        self.assertEqual(self.supplement["source_manifest_sha256"], hashlib.sha256((SOURCE / "training_cards.json").read_bytes()).hexdigest())
        self.assertEqual(self.supplement["source_registry_sha256"], hashlib.sha256((SOURCE / "source_registry.json").read_bytes()).hexdigest())
        self.assertEqual(self.supplement["delivery"], "offline_reference_answers_not_model_weight_training")
        self.assertEqual(self.supplement["counts"], {"cards": 110, "sources": 9, "validation_cards_included": 0})
        originals = {card["id"]: card for card in self.train}
        for card in self.supplement["cards"]:
            with self.subTest(identity=card["id"]):
                self.assertEqual(card["question"], originals[card["id"]]["question"])
                self.assertEqual(card["answer"], originals[card["id"]]["answer"])
                self.assertIs(card["exact_only"], True)
                self.assertIs(card["holdout"], False)
                self.assertEqual(card["aliases"], [])
                self.assertEqual(card["metadata"]["origin_split"], "train")
                self.assertEqual(card["metadata"]["origin_source_group"], GROUP)
                self.assertEqual(card["metadata"]["reference_scope"], "general_theory")
                self.assertIs(card["metadata"]["geology_approved"], False)
                self.assertTrue(all(identity.startswith("geo_sft_v2_") for identity in card["sources"]))
        self.assertIs(self.supplement["geologist_reviewed"], False)
        for source in self.supplement["sources"].values():
            self.assertTrue(source["license"] and source["attribution"])

    def test_all_110_answers_are_available_without_loading_neural_weights(self):
        for card in self.train:
            with self.subTest(identity=card["id"]):
                self.assertEqual(lookup.lookup_theory_answer(card["question"], BASE), card["answer"])

    def test_source_urls_are_only_added_when_requested(self):
        for card in (self.train[0], self.train[50], self.train[-1]):
            with self.subTest(identity=card["id"]):
                answer = lookup.lookup_theory_answer(card["question"], BASE)
                self.assertEqual(answer, card["answer"])
                self.assertNotRegex(answer, r"https?://")
                sourced = lookup.lookup_theory_answer(card["question"] + " Дай источники.", BASE, include_sources=True)
                self.assertTrue(sourced.startswith(card["answer"] + "\n\nИсточники:\n"))
                for identity in next(item for item in self.supplement["cards"] if item["id"] == card["id"])["sources"]:
                    self.assertIn(self.supplement["sources"][identity]["url"], sourced)

    def test_new_reference_cannot_fuzzily_answer_changed_properties_or_numbers(self):
        isolated = lookup._merge_reference_supplement(
            {"cards": [], "sources": {}}, self.supplement, expected_source_group=GROUP)
        self.assertEqual(len(isolated["cards"]), 110)
        with patch.object(lookup, "_load_cards", return_value=isolated):
            for card in self.train:
                self.assertIsNone(lookup.lookup_theory_answer("Объясни подробнее: " + card["question"], BASE))
            numeric = [card for card in self.train if any(char.isdigit() for char in card["question"])]
            self.assertTrue(numeric)
            for card in numeric:
                index = next(index for index, char in enumerate(card["question"]) if char.isdigit())
                changed = card["question"][:index] + "999" + card["question"][index + 1:]
                self.assertIsNone(lookup.lookup_theory_answer(changed, BASE))

    def test_all_frozen_challenge_lookups_remain_identical_and_no_answers_are_delivered(self):
        challenge = json.loads((SOURCE / "evaluation_challenge.json").read_text(encoding="utf-8"))["items"]
        self.assertEqual(len(challenge), 16)
        previous = self.old_corpus()
        resource = SUPPLEMENT.read_text(encoding="utf-8")
        for card in challenge:
            with self.subTest(identity=card["id"]):
                with patch.object(lookup, "_load_cards", return_value=previous):
                    before = lookup.lookup_theory_answer(card["question"], BASE)
                self.assertEqual(lookup.lookup_theory_answer(card["question"], BASE), before)
                self.assertNotIn(card["id"], resource)
                self.assertNotIn(card["question"], resource)
                self.assertNotIn(card["reference_answer"], resource)


if __name__ == "__main__":
    unittest.main()
