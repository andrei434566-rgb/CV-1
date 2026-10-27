"""CPU checks for local base/LoRA selection; no model weights or downloads."""

import hashlib
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.infrastructure.local_vlm_modes import adapter_for_operation, normalize_model_request, validate_local_models, write_adapter_manifest


class ModelModeTests(unittest.TestCase):
    def test_base_mode_ignores_previously_saved_adapter_paths(self):
        result = normalize_model_request({"model_mode": "base", "base": "base", "adapter": "missing", "theory_adapter": "old"})
        self.assertEqual((result["adapter"], result["theory_adapter"]), ("", ""))
        self.assertEqual(adapter_for_operation(result, "describe"), ("description", ""))

    def test_one_specialist_adapter_is_used_for_photo_and_text(self):
        config = {"model_mode": "lora", "adapter": "specialist"}
        self.assertEqual(adapter_for_operation(config, "describe"), ("description", "specialist"))
        self.assertEqual(adapter_for_operation(config, "chat"), ("theory", "specialist"))

    def test_legacy_config_keeps_separate_paths_and_unadapted_text(self):
        self.assertEqual(adapter_for_operation({"adapter": "photo"}, "chat"), ("theory", ""))
        self.assertEqual(adapter_for_operation({"adapter": "photo", "theory_adapter": "text"}, "chat"), ("theory", "text"))

    def test_explicit_invalid_modes_and_missing_adapter_fail(self):
        for config in ({"model_mode": "server"}, {"model_mode": "lora"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_model_request(config)


class LocalCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.revision = "a" * 40
        self.base = self.root / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / self.revision
        self.base.mkdir(parents=True)
        self.base_config = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl", "text_config": {"hidden_size": 2048}}
        self.write(self.base / "config.json", self.base_config)
        self.adapter = self.root / "specialist"
        self.adapter.mkdir()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"fake weights; metadata validation only")
        self.metadata = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": "Qwen/Qwen3-VL-2B-Instruct"}
        self.write(self.adapter / "adapter_config.json", self.metadata)
        self.config = {"model_mode": "lora", "base": str(self.base), "adapter": str(self.adapter)}

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_hf_identity_is_portable_and_no_download_is_needed(self):
        report = validate_local_models(self.config)
        self.assertEqual(report["base_model_id"], "qwen/qwen3-vl-2b-instruct")
        self.assertEqual(len(report["adapters"]), 1)
        self.assertFalse(report["adapters"][0]["revision_verified"])

    def test_windows_original_cache_path_can_move_to_another_pc(self):
        self.metadata["base_model_name_or_path"] = rf"C:\old\models--Qwen--Qwen3-VL-2B-Instruct\snapshots\{self.revision}"
        self.write(self.adapter / "adapter_config.json", self.metadata)
        self.assertTrue(validate_local_models(self.config)["adapters"][0]["revision_verified"])

    def test_wrong_model_size_is_rejected(self):
        self.metadata["base_model_name_or_path"] = "Qwen/Qwen3-VL-8B-Instruct"
        self.write(self.adapter / "adapter_config.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "должны совпадать"):
            validate_local_models(self.config)

    def test_wrong_qwen_architecture_is_rejected_even_in_base_mode(self):
        self.base_config["model_type"] = "qwen2_5_vl"
        self.write(self.base / "config.json", self.base_config)
        with self.assertRaisesRegex(ValueError, "Qwen3-VL"):
            validate_local_models({**self.config, "model_mode": "base"})

    def test_known_revision_mismatch_is_rejected(self):
        self.metadata["revision"] = "b" * 40
        self.write(self.adapter / "adapter_config.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "Версия базовой"):
            validate_local_models(self.config)

    def test_revision_alias_does_not_hide_original_snapshot_hash(self):
        self.metadata["revision"] = "main"
        self.metadata["base_model_name_or_path"] = r"C:\old\models--Qwen--Qwen3-VL-2B-Instruct\snapshots" + "\\" + "b" * 40
        self.write(self.adapter / "adapter_config.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "Версия базовой"):
            validate_local_models(self.config)

    def test_available_original_config_checks_actual_model_dimensions(self):
        original = self.root / "original"
        original.mkdir()
        original_config = {**self.base_config, "text_config": {"hidden_size": 4096}}
        self.write(original / "config.json", original_config)
        self.metadata["base_model_name_or_path"] = str(original)
        self.write(self.adapter / "adapter_config.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "Размеры и архитектура"):
            validate_local_models(self.config)

    def test_missing_adapter_is_ignored_in_base_mode(self):
        (self.adapter / "adapter_model.safetensors").unlink()
        self.assertEqual(validate_local_models({**self.config, "model_mode": "base"})["adapters"], [])
        with self.assertRaisesRegex(ValueError, "не найден"):
            validate_local_models(self.config)

    def test_rejected_experiment_is_not_activated(self):
        self.write(self.adapter / "acceptance_review.json", {"status": "rejected_for_general_assistant"})
        with self.assertRaisesRegex(ValueError, "отклонён"):
            validate_local_models(self.config)

    def test_parent_review_is_applied_to_adapter_subfolder(self):
        nested = self.adapter / "adapter"
        nested.mkdir()
        (nested / "adapter_model.safetensors").write_bytes(b"fake")
        self.write(nested / "adapter_config.json", self.metadata)
        self.write(self.adapter / "acceptance_review.json", {"status": "rejected_for_general_assistant"})
        with self.assertRaisesRegex(ValueError, "отклонён"):
            validate_local_models({**self.config, "adapter": str(nested)})

    def test_manifest_confirms_renamed_base_folder_by_config_hash(self):
        renamed = self.root / "local_base"
        self.base.rename(renamed)
        sha = hashlib.sha256((renamed / "config.json").read_bytes()).hexdigest()
        self.write(self.adapter / "cv1_adapter_manifest.json", {"base_model_id": "Qwen/Qwen3-VL-2B-Instruct", "base_config_sha256": sha})
        self.assertEqual(validate_local_models({**self.config, "base": str(renamed)})["adapters"][0]["base_config_sha256"], sha)

    def test_mismatched_manifest_hash_is_rejected(self):
        self.write(self.adapter / "cv1_adapter_manifest.json", {"base_config_sha256": "f" * 64})
        with self.assertRaisesRegex(ValueError, "паспортом"):
            validate_local_models(self.config)

    def test_unverifiable_arbitrary_paths_fail_with_actionable_error(self):
        renamed = self.root / "local_base"
        self.base.rename(renamed)
        with self.assertRaisesRegex(ValueError, "cv1_adapter_manifest.json"):
            validate_local_models({**self.config, "base": str(renamed)})

    def test_written_manifest_is_portable_metadata_with_pending_review(self):
        original_metadata = (self.adapter / "adapter_config.json").read_bytes()
        original_weights = (self.adapter / "adapter_model.safetensors").read_bytes()
        path = write_adapter_manifest(self.base, self.adapter)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["base_config_sha256"], hashlib.sha256((self.base / "config.json").read_bytes()).hexdigest())
        self.assertEqual(manifest["base_revision"], self.revision)
        self.assertEqual(manifest["review_status"], "pending_review")
        self.assertTrue(manifest["metadata_only"])
        self.assertFalse(manifest["weights_verified"])
        self.assertFalse(manifest["geologist_approved"])
        self.assertIsNone(manifest["quality_metrics"])
        self.assertEqual((self.adapter / "adapter_config.json").read_bytes(), original_metadata)
        self.assertEqual((self.adapter / "adapter_model.safetensors").read_bytes(), original_weights)
        renamed_base, renamed_adapter = self.root / "copied_base", self.root / "copied_adapter"
        self.base.rename(renamed_base)
        self.adapter.rename(renamed_adapter)
        report = validate_local_models({"model_mode": "lora", "base": str(renamed_base), "adapter": str(renamed_adapter)})
        self.assertEqual(report["adapters"][0]["base_config_sha256"], manifest["base_config_sha256"])

    def test_writer_does_not_stamp_incompatible_adapter_as_matching(self):
        self.metadata["base_model_name_or_path"] = "Qwen/Qwen3-VL-8B-Instruct"
        self.write(self.adapter / "adapter_config.json", self.metadata)
        with self.assertRaisesRegex(ValueError, "должны совпадать"):
            write_adapter_manifest(self.base, self.adapter)
        self.assertFalse((self.adapter / "cv1_adapter_manifest.json").exists())

    def test_writer_does_not_overwrite_rejected_experiment_or_acceptance_review(self):
        review = self.adapter / "acceptance_review.json"
        self.write(review, {"status": "rejected_for_general_assistant", "raw_metrics_preserved": True})
        previous = review.read_bytes()
        with self.assertRaisesRegex(ValueError, "отклонён"):
            write_adapter_manifest(self.base, self.adapter)
        self.assertEqual(review.read_bytes(), previous)
        self.assertFalse((self.adapter / "cv1_adapter_manifest.json").exists())


class RunnerSelectionTests(unittest.TestCase):
    def make_runner(self, answer="Готов."):
        from app.infrastructure.local_vlm_runner import LocalModel

        class Inputs(dict):
            def to(self, device):
                return self

        runner = LocalModel()
        runner.load = Mock()
        runner.processor = Mock()
        runner.processor.return_value = Inputs(input_ids=np.array([[1, 2]]))
        runner.processor.batch_decode.return_value = [answer]
        runner.model = Mock()
        runner.model.generation_config = SimpleNamespace(eos_token_id=99)
        runner.model.generate.return_value = np.array([[1, 2, 5, 99]])
        runner.model.disable_adapter.return_value = nullcontext()
        return runner

    def answer(self, runner, request):
        torch = SimpleNamespace(inference_mode=lambda: nullcontext(), cuda=SimpleNamespace(synchronize=lambda: None))
        with patch.dict("sys.modules", {"torch": torch}):
            return runner.answer(request, [])

    def test_explicit_base_request_does_not_select_stale_adapter(self):
        runner = self.make_runner()
        result = self.answer(runner, {"model_mode": "base", "base": "base", "adapter": "old", "operation": "chat", "question": "Работаешь?"})
        runner.model.set_adapter.assert_not_called()
        self.assertEqual(result["model_info"], {"model_mode": "base", "base": "base", "adapter": ""})

    def test_lora_mode_uses_same_adapter_for_ordinary_text(self):
        runner = self.make_runner()
        result = self.answer(runner, {"model_mode": "lora", "base": "base", "adapter": "specialist", "operation": "chat", "question": "Что такое песчаник?"})
        runner.model.set_adapter.assert_called_once_with("theory")
        self.assertEqual(result["model_info"]["adapter"], "specialist")

    def test_legacy_photo_only_adapter_is_disabled_for_ordinary_text(self):
        runner = self.make_runner()
        result = self.answer(runner, {"base": "base", "adapter": "photo", "operation": "chat", "question": "Привет"})
        runner.model.set_adapter.assert_not_called()
        runner.model.disable_adapter.assert_called_once()
        self.assertEqual(result["model_info"]["model_mode"], "base")

    def test_description_schema_preserves_geometry_boundary(self):
        fields = {key: "" for key in ("facies_name", "lithology", "texture", "facies_association", "depositional_environment", "description")}
        fields["description"] = "Серый слоистый фрагмент."
        runner = self.make_runner(json.dumps(fields, ensure_ascii=False))
        result = self.answer(runner, {"model_mode": "base", "base": "base", "operation": "describe", "context": "Интервал 1"})
        prompt = runner.processor.apply_chat_template.call_args.args[0][-1]["content"][0]["text"]
        self.assertIn("не меняй границы", prompt)
        self.assertEqual(result["fields"], fields)

    def test_bad_local_metadata_fails_before_gpu_runtime_import(self):
        from app.infrastructure.local_vlm_runner import LocalModel
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            with self.assertRaisesRegex(ValueError, "папка Qwen3-VL"):
                LocalModel().load("missing", "", "", "base")

    def test_malformed_description_fields_are_not_coerced_to_text(self):
        runner = self.make_runner('{"facies_name": {"x": 1}, "description": "Текст"}')
        result = self.answer(runner, {"model_mode": "base", "base": "base", "operation": "describe"})
        self.assertNotIn("fields", result)


class SettingsModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def test_settings_keep_paths_but_disable_them_in_base_mode(self):
        from app.ui.dialogs.local_vlm_settings_dialog import LocalVlmSettingsDialog
        dialog = LocalVlmSettingsDialog({"model_mode": "base", "adapter": "saved", "theory_adapter": "text"})
        self.addCleanup(dialog.close)
        self.assertFalse(dialog.adapter.isEnabled())
        self.assertEqual(dialog.values()["adapter"], "saved")
        dialog.model_mode.setCurrentIndex(dialog.model_mode.findData("lora"))
        self.assertTrue(dialog.adapter.isEnabled())
        self.assertTrue(dialog.theory_adapter.isEnabled())
        self.assertEqual(dialog.values()["model_mode"], "lora")

    def test_invalid_saved_config_can_be_opened_and_repaired(self):
        from app.ui.dialogs.local_vlm_settings_dialog import LocalVlmSettingsDialog
        dialog = LocalVlmSettingsDialog({"model_mode": "lora", "adapter": ""})
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.values()["model_mode"], "lora")


if __name__ == "__main__":
    unittest.main()
