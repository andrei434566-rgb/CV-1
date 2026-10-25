"""CPU-only checks using synthetic provenance fixtures, never model weights."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_qwen_theory_round2 as preparer

TEMP_ROOT = Path(__file__).resolve().parent / "open_datasets/test_tmp"


def theory(identifier, question, *, training, split=True, answer="Проверочный текст."):
    return {
        "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}],
        "metadata": {"record_id": identifier, "record_type": "text_only_knowledge",
                     "training_approved": training, "geology_approved": False, "target_language": "ru",
                     "review_status": "source_checked_synthetic_fixture_not_real_geology",
                     "sources": [{"source_id": "synthetic_fixture", "url": "https://example.test/source",
                                  "title": "Synthetic test fixture, not a real source"}],
                     **({"split": "train" if training else "holdout"} if split else {})},
    }


def conversation(identifier, question, *, training):
    row = theory(identifier, question, training=training)
    row["metadata"]["record_type"] = "authored_conversation_not_geological_label"
    row["metadata"].pop("sources")
    return row


class TheoryRound2Tests(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="qwen_theory_round2_", dir=TEMP_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = {
            "legacy_theory_train": [theory("old_train", "Старый учебный вопрос?", training=True, split=False)],
            "legacy_theory_holdout": [theory("old_holdout", "Старый проверочный вопрос?", training=False, split=False)],
            "new_theory_train": [theory("new_train", "Новый учебный вопрос?", training=True)],
            "new_theory_holdout": [theory("new_holdout", "Новый проверочный вопрос?", training=False)],
            "conversation_train": [conversation("new_dialogue", "Добрый день, помощник!", training=True)],
            "conversation_holdout": [conversation("new_dialogue_holdout", "Отзовись, пожалуйста.", training=False)],
        }
        self.paths = {name: self.root / f"{name}.jsonl" for name in self.inputs}
        self.system_source = self.root / "system.py"
        self.system_source.write_text('raise RuntimeError("must not execute")\nsystem_text = ("Русский " "помощник.")\n', encoding="utf-8")

    def build(self, *, system_file=None, answer_overrides=None,
              additional_theory_train=None, additional_theory_holdout=None):
        for name, rows in self.inputs.items():
            self.paths[name].write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return preparer.build_plan(**self.paths, system_source=self.system_source, system_file=system_file,
                                   answer_overrides=answer_overrides, additional_theory_train=additional_theory_train,
                                   additional_theory_holdout=additional_theory_holdout)

    def write_overrides(self, overrides):
        path = self.root / "overrides.json"
        path.write_text(json.dumps(overrides, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def override():
        return {"answer": "Исправленный текст фикстуры.", "reason": "Synthetic provenance correction test.",
                "sources": [{"source_id": "new_fixture", "url": "https://example.test/correction",
                             "title": "Synthetic source, not real geological material"}],
                "review_status": "source_checked_synthetic_fixture_not_real_geology", "geology_approved": False}

    def test_small_plan_preserves_honest_provenance_and_partitions(self):
        originals = copy.deepcopy(self.inputs)
        exports, summary = self.build()
        self.assertEqual(self.inputs, originals)
        self.assertEqual(summary["counts"]["mixed"], 3)
        self.assertEqual(summary["counts"]["holdout_theory"], 2)
        self.assertEqual(summary["counts"]["holdout_conversation"], 5)
        self.assertEqual(summary["system_prompt"]["text"], "Русский помощник.")
        self.assertFalse(summary["test_split_used"])
        self.assertEqual(summary["geologist_reviewed_rows"], 0)
        for row in exports["mixed"]:
            self.assertTrue(row["metadata"]["training_approved"])
            self.assertFalse(row["metadata"]["geology_approved"])
            self.assertEqual(row["metadata"]["split"], "train")
            self.assertFalse(row.get("image"))
        self.assertTrue(all(row["metadata"]["evaluation_kind"] == "casual_guard" for row in exports["holdout_conversation"]))
        self.assertEqual(summary["inputs"]["new_theory_train"]["sha256"], preparer.file_hash(self.paths["new_theory_train"]))

    def test_new_rows_need_explicit_correct_split(self):
        self.inputs["new_theory_train"][0]["metadata"].pop("split")
        with self.assertRaisesRegex(ValueError, "Unexpected split"):
            self.build()

    def test_test_rows_are_refused_even_with_misleading_train_split(self):
        self.inputs["new_theory_train"][0]["metadata"]["source_split"] = "test"
        with self.assertRaisesRegex(ValueError, "Sealed test"):
            self.build()

    def test_holdout_cannot_be_marked_for_training(self):
        self.inputs["new_theory_holdout"][0]["metadata"]["training_approved"] = True
        with self.assertRaisesRegex(ValueError, "training_approved"):
            self.build()

    def test_approval_is_boolean_not_truthiness(self):
        self.inputs["new_theory_train"][0]["metadata"]["training_approved"] = 1
        with self.assertRaisesRegex(ValueError, "training_approved"):
            self.build()

    def test_media_and_image_placeholders_are_refused(self):
        for change in ({"image": "missing.png"}, {"images": ["missing.png"]}, {"video": "missing.mp4"}):
            with self.subTest(change=change):
                row = theory("row", "Вопрос?", training=True)
                row.update(change)
                with self.assertRaisesRegex(ValueError, "text-only"):
                    preparer.validated_rows([row], training=True, theory=True)
        row = theory("row", "<image> Опиши?", training=True)
        with self.assertRaisesRegex(ValueError, "Image placeholder"):
            preparer.validated_rows([row], training=True, theory=True)

    def test_expert_claims_and_missing_status_are_refused(self):
        for value in (True, None, "false", 0):
            with self.subTest(value=value):
                row = theory("row", "Вопрос?", training=True)
                row["metadata"]["geology_approved"] = value
                with self.assertRaisesRegex(ValueError, "geology_approved"):
                    preparer.validated_rows([row], training=True, theory=True)

    def test_sources_must_be_structurally_valid_and_source_checked(self):
        mutations = [
            {"sources": []}, {"sources": [{"source_id": "fixture"}]},
            {"sources": [{"source_id": "fixture", "url": "file:///tmp/source", "title": "fixture"}]},
            {"sources": [{"source_id": "fixture", "url": "https://example.test"}]},
            {"review_status": "not_reviewed"},
        ]
        for change in mutations:
            with self.subTest(change=change):
                row = theory("row", "Вопрос?", training=True)
                row["metadata"].update(change)
                with self.assertRaises(ValueError):
                    preparer.validated_rows([row], training=True, theory=True)

    def test_incomplete_pairs_and_wrong_roles_are_refused(self):
        for turns in ([{"from": "human", "value": "Один ход"}],
                      [{"from": "gpt", "value": "Вопрос"}, {"from": "human", "value": "Ответ"}],
                      [{"from": "human", "value": "Вопрос"}, {"from": "gpt", "value": " "}]):
            with self.subTest(turns=turns):
                row = theory("row", "Вопрос?", training=True)
                row["conversations"] = turns
                with self.assertRaises(ValueError):
                    preparer.validated_rows([row], training=True, theory=True)

    def test_normalized_question_overlap_across_partitions_is_refused(self):
        self.inputs["new_theory_train"][0]["conversations"][0]["value"] = " Ёлки: какая разница? "
        self.inputs["new_theory_holdout"][0]["conversations"][0]["value"] = "ЕЛКИ какая разница!!!"
        with self.assertRaisesRegex(ValueError, "Normalized question occurs"):
            self.build()

    def test_ids_are_unique_even_when_questions_differ(self):
        self.inputs["new_theory_holdout"][0]["metadata"]["record_id"] = "old_train"
        with self.assertRaisesRegex(ValueError, "Duplicate record IDs"):
            self.build()

    def test_duplicate_training_formulations_keep_first_with_exclusion_log(self):
        duplicate = copy.deepcopy(self.inputs["legacy_theory_train"][0])
        duplicate["metadata"]["record_id"] = "old_train_alternate"
        duplicate["conversations"][0]["value"] = "Старый учебный вопрос!!!"
        self.inputs["legacy_theory_train"].append(duplicate)
        exports, summary = self.build()
        self.assertEqual(len(exports["train_theory"]), 2)
        self.assertEqual(summary["input_counts"]["legacy_theory_train"], 2)
        self.assertEqual(summary["used_counts"]["legacy_theory_train"], 1)
        self.assertEqual(summary["input_partition_checks"]["train_duplicate_question_excess_rows"], 1)
        self.assertEqual(summary["partition_checks"]["train_duplicate_question_excess_rows"], 0)
        self.assertEqual(summary["training_question_exclusions"][0]["excluded_id"], "old_train_alternate")
        self.assertEqual(summary["training_question_exclusions"][0]["kept_id"], "old_train")
        self.assertEqual(summary["training_question_exclusions"][0]["reason"], "duplicate_question_keep_first")
        self.assertEqual(len(summary["training_question_exclusions"][0]["excluded_answer_sha256"]), 64)
        self.assertEqual(preparer.record_id(exports["train_theory"][0]), "old_train")

    def test_deduplication_keeps_first_answer_without_merging_new_facts(self):
        duplicate = copy.deepcopy(self.inputs["legacy_theory_train"][0])
        duplicate["metadata"]["record_id"] = "old_train_later_version"
        duplicate["conversations"][1]["value"] = "Иной ответ с дополнительными подробностями."
        self.inputs["legacy_theory_train"].append(duplicate)
        exports, summary = self.build()
        kept = exports["train_theory"][0]
        self.assertEqual(kept["conversations"][1]["value"], "Проверочный текст.")
        self.assertNotEqual(summary["training_question_exclusions"][0]["excluded_answer_sha256"],
                            summary["training_question_exclusions"][0]["kept_answer_sha256"])
        self.assertEqual(len(self.inputs["legacy_theory_train"]), 2)

    def test_deduplication_cannot_hide_an_input_id_collision(self):
        self.inputs["legacy_theory_train"].append(copy.deepcopy(self.inputs["legacy_theory_train"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate record IDs"):
            self.build()

    def test_duplicate_holdout_formulations_are_refused(self):
        self.inputs["new_theory_holdout"][0]["conversations"][0]["value"] = "Старый проверочный вопрос!!!"
        with self.assertRaisesRegex(ValueError, "Duplicate normalized questions within holdout"):
            self.build()

    def test_open_question_yes_no_preface_edit_is_minimal_and_auditable(self):
        before = "Нет. Сохранившаяся часть ответа без изменения фактов."
        self.inputs["legacy_theory_train"][0]["conversations"] = [
            {"from": "human", "value": "Как различить признаки?"}, {"from": "gpt", "value": before}]
        exports, summary = self.build()
        edited = exports["train_theory"][0]
        self.assertEqual(edited["conversations"][1]["value"], before[len("Нет. "):])
        self.assertEqual(self.inputs["legacy_theory_train"][0]["conversations"][1]["value"], before)
        edit = edited["metadata"]["preparation_edits"][0]
        self.assertEqual(edit["old_answer_sha256"], hashlib.sha256(before.encode()).hexdigest())
        self.assertEqual(edit["new_answer_sha256"], hashlib.sha256(before[len("Нет. "):].encode()).hexdigest())
        self.assertEqual(summary["preparation_edits"][0]["record_id"], "old_train")
        self.assertEqual(summary["preparation_edits"][0]["stream"], "legacy_theory_train")

    def test_closed_questions_and_nonliteral_prefixes_remain_unchanged(self):
        for question, answer in [("Можно ли это утверждать?", "Нет. Данных мало."),
                                 ("Есть ли доказательства?", "Да. Их достаточно."),
                                 ("Как объяснить?", "Нет, это другая ситуация."),
                                 ("Что написать?", "Нет.")]:
            with self.subTest(question=question, answer=answer):
                row = theory("row", question, training=True, answer=answer)
                self.assertEqual(preparer.sanitize_open_question_answers([row], stream="test_fixture"), [])
                self.assertEqual(row["conversations"][1]["value"], answer)
        row = theory("row", "В чём разница?", training=True, answer="Да. Отличается форма.")
        self.assertEqual(len(preparer.sanitize_open_question_answers([row], stream="test_fixture")), 1)
        self.assertEqual(row["conversations"][1]["value"], "Отличается форма.")

    def test_explicit_system_prompt_is_preserved(self):
        system_file = self.root / "explicit_system.txt"
        system_file.write_text("Другой сохранённый системный текст.", encoding="utf-8")
        _, summary = self.build(system_file=system_file)
        self.assertEqual(summary["system_prompt"]["text"], "Другой сохранённый системный текст.")

    def test_source_checked_answer_override_preserves_old_evidence(self):
        original = copy.deepcopy(self.inputs["legacy_theory_train"][0])
        replacement = self.override()
        exports, summary = self.build(answer_overrides=self.write_overrides({"old_train": replacement}))
        row = exports["train_theory"][0]
        self.assertEqual(row["conversations"][1]["value"], replacement["answer"])
        self.assertEqual(row["metadata"]["sources"], replacement["sources"])
        edit = row["metadata"]["preparation_edits"][0]
        self.assertEqual(edit["old_answer"], original["conversations"][1]["value"])
        self.assertEqual(edit["old_sources"], original["metadata"]["sources"])
        self.assertEqual(edit["reason"], replacement["reason"])
        self.assertEqual(summary["answer_overrides_applied"][0]["record_id"], "old_train")
        self.assertEqual(self.inputs["legacy_theory_train"][0], original)
        self.assertIn("answer_overrides", summary["inputs"])

    def test_unchecked_or_falsely_expert_approved_overrides_are_refused(self):
        for change in ({"review_status": "not_checked"}, {"sources": []},
                       {"geology_approved": True}, {"reason": ""}, {"answer": ""}):
            with self.subTest(change=change):
                value = self.override()
                value.update(change)
                with self.assertRaises(ValueError):
                    self.build(answer_overrides=self.write_overrides({"old_train": value}))

    def test_unknown_override_id_is_refused_instead_of_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "Unknown answer override"):
            self.build(answer_overrides=self.write_overrides({"unknown_record": self.override()}))

    def test_additional_theory_files_are_separate_validated_provenance_inputs(self):
        train_paths = [self.root / "additional_train_1.jsonl", self.root / "additional_train_2.jsonl"]
        for number, path in enumerate(train_paths):
            path.write_text(json.dumps(theory(f"extra_train_{number}", f"Дополнительный учебный вопрос {number}?",
                                             training=True), ensure_ascii=False) + "\n", encoding="utf-8")
        holdout = self.root / "additional_holdout.jsonl"
        holdout.write_text(json.dumps(theory("extra_holdout", "Дополнительный проверочный вопрос?", training=False),
                                      ensure_ascii=False) + "\n", encoding="utf-8")
        exports, summary = self.build(additional_theory_train=train_paths, additional_theory_holdout=[holdout])
        self.assertEqual(len(exports["mixed"]), 5)
        self.assertEqual(len(exports["holdout_theory"]), 3)
        self.assertEqual(summary["input_counts"]["additional_theory_train"], 2)
        self.assertEqual(summary["used_counts"]["additional_theory_holdout"], 1)
        self.assertEqual(summary["inputs"]["additional_theory_train_01"]["sha256"], preparer.file_hash(train_paths[0]))
        self.assertEqual(summary["inputs"]["additional_theory_train_02"]["sha256"], preparer.file_hash(train_paths[1]))
        self.assertEqual(summary["inputs"]["additional_theory_holdout_01"]["sha256"], preparer.file_hash(holdout))

    def test_additional_holdout_obeys_identity_question_and_test_guards(self):
        path = self.root / "extra_holdout.jsonl"
        for row in [theory("old_train", "Отдельный вопрос?", training=False),
                    theory("extra_holdout", "СТАРЫЙ УЧЕБНЫЙ ВОПРОС!!!", training=False),
                    theory("extra_holdout", "Ещё один вопрос?", training=False)]:
            with self.subTest(row=row):
                if row["conversations"][0]["value"] == "Ещё один вопрос?":
                    row["metadata"]["source_split"] = "test"
                path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.build(additional_theory_holdout=[path])

    def test_immutable_atomic_export_preserves_sources(self):
        exports, summary = self.build()
        hashes = {name: preparer.file_hash(path) for name, path in self.paths.items()}
        output = self.root / "готовые_манифесты"
        preparer.write_plan(output, exports, summary)
        self.assertTrue((output / "mixed.jsonl").is_file())
        saved = [json.loads(line) for line in (output / "mixed.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(saved, exports["mixed"])
        self.assertEqual((output / "system_prompt.txt").read_text(encoding="utf-8"), summary["system_prompt"]["text"])
        with self.assertRaises(FileExistsError):
            preparer.write_plan(output, exports, summary)
        self.assertEqual(hashes, {name: preparer.file_hash(path) for name, path in self.paths.items()})

    def test_failed_export_cleans_only_its_own_stage(self):
        exports, summary = self.build()
        output = self.root / "failed_export"
        sentinel = self.root / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with patch.object(Path, "rename", side_effect=OSError("synthetic failure")):
            with self.assertRaisesRegex(OSError, "synthetic failure"):
                preparer.write_plan(output, exports, summary)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(".failed_export_*")))
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_repository_conversation_replay_is_bounded_and_disjoint(self):
        train = preparer.read_rows(preparer.CONVERSATION_ROOT / "conversation_train.jsonl", training=True, theory=False)
        heldout = preparer.read_rows(preparer.CONVERSATION_ROOT / "conversation_holdout.jsonl", training=False, theory=False)
        old = preparer.validated_rows(preparer.friendly_rows()[1], training=False, theory=False, allow_unsplit=True)
        self.assertEqual(len(train), 60)
        self.assertEqual(len(heldout), 8)
        checks = preparer.check_partitions(train, [*old, *heldout])
        self.assertEqual(checks["train_duplicate_question_excess_rows"], 0)
        self.assertEqual(checks["train_holdout_normalized_question_overlap"], 0)

    def test_repository_source_pack_builds_without_gpu_or_downloads(self):
        source = preparer.CONVERSATION_ROOT
        exports, summary = preparer.build_plan(
            preparer.THEORY_ROOT / "theory_training_v5.jsonl",
            preparer.THEORY_ROOT / "theory_holdout_v5.jsonl",
            source / "train.jsonl", source / "holdout.jsonl",
            source / "conversation_train.jsonl", source / "conversation_holdout.jsonl",
            answer_overrides=source / "overrides.json",
        )
        self.assertEqual(len(exports["mixed"]), 289)
        self.assertEqual(len(exports["holdout_theory"]), 36)
        self.assertEqual(len(exports["holdout_conversation"]), 12)
        self.assertEqual(summary["geologist_reviewed_rows"], 0)
        self.assertEqual(summary["partition_checks"]["train_holdout_normalized_question_overlap"], 0)
        self.assertEqual({edit["excluded_id"] for edit in summary["training_question_exclusions"]},
                         {"v4_sand_vs_silt_a", "v4_carbonate_wacke_pack_a"})


if __name__ == "__main__":
    unittest.main()
