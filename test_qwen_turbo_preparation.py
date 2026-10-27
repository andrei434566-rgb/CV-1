"""Installer cache identity and atomic publication without downloads or a GPU."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import prepare_qwen_turbo_local as installer


class QwenTurboPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.base = self.folder / "base"
        self.base.mkdir()
        self.config = {"model_type": "qwen3_vl", "text_config": {"hidden_size": 2048}}
        (self.base / "config.json").write_text(json.dumps(self.config), encoding="utf-8")
        for name in ("model.safetensors", "tokenizer.json", "preprocessor_config.json"):
            (self.base / name).write_bytes(b"original " + name.encode())
        self.root = self.folder / "runtime"
        self.source = self.folder / "converter"
        self.archive = self.folder / "runtime.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr("llama-server.exe", b"fake server")
        self.output_number = 0

    def convert(self, command, **kwargs):
        self.output_number += 1
        target = Path(command[command.index("--outfile") + 1])
        target.write_bytes(b"GGUF" + str(self.output_number).encode() + b" converted output")
        return subprocess.CompletedProcess(command, 0)

    def run_installer(self, *, base=None, convert=None):
        args = ["prepare", "--base", str(base or self.base), "--destination", str(self.root)]
        with (patch.object(installer, "MIN_GGUF_BYTES", 4),
              patch.object(installer, "fetch_runtime", return_value=self.archive),
              patch.object(installer, "fetch_converter", return_value=self.source),
              patch.object(installer.importlib.util, "find_spec", return_value=object()),
              patch.object(installer.subprocess, "run", side_effect=convert or self.convert) as calls,
              patch.object(installer.sys, "argv", args), contextlib.redirect_stdout(io.StringIO())):
            installer.main()
            return calls.call_count

    def manifest(self):
        return json.loads((self.root / "runtime.json").read_text(encoding="utf-8"))

    def pair(self, manifest=None):
        manifest = manifest or self.manifest()
        return tuple(self.root / manifest[key] for key in ("model", "mmproj"))

    def test_identical_source_and_verified_pair_are_reused(self):
        self.assertEqual(self.run_installer(), 2)
        previous = self.manifest()
        self.assertEqual(set(previous["source_fingerprint"]["files"]),
                         {"config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json"})
        self.assertEqual(self.run_installer(), 0)
        self.assertEqual(previous, self.manifest())

    def test_changes_at_same_path_rebuild_both_outputs(self):
        self.run_installer()
        for name in ("model.safetensors", "config.json", "tokenizer.json", "preprocessor_config.json"):
            with self.subTest(name=name):
                previous = self.manifest()
                old_pair = self.pair(previous)
                original_bytes = tuple(path.read_bytes() for path in old_pair)
                if name == "config.json":
                    (self.base / name).write_text(json.dumps(dict(self.config, changed=True)), encoding="utf-8")
                else:
                    (self.base / name).write_bytes(b"changed " + name.encode())
                self.assertEqual(self.run_installer(), 2)
                current = self.manifest()
                self.assertNotEqual(previous["source_fingerprint"], current["source_fingerprint"])
                self.assertNotEqual(old_pair, self.pair(current))
                self.assertEqual(original_bytes, tuple(path.read_bytes() for path in old_pair))

    def test_another_source_path_requires_conversion_even_for_identical_bytes(self):
        self.run_installer()
        other = self.folder / "another-base"
        shutil.copytree(self.base, other)
        self.assertEqual(self.run_installer(base=other), 2)
        self.assertEqual(self.manifest()["converted_from"], str(other.resolve()))

    def test_changed_output_hash_rebuilds_the_entire_pair(self):
        self.run_installer()
        previous = self.manifest()
        model, _ = self.pair(previous)
        data = model.read_bytes()
        model.write_bytes(data[:-1] + b"!")
        self.assertEqual(self.run_installer(), 2)
        current = self.manifest()
        self.assertNotEqual(previous["model"], current["model"])
        self.assertNotEqual(previous["mmproj"], current["mmproj"])
        for path in self.pair(current):
            self.assertEqual(current["sha256"][path.name], installer.sha256(path))

    def test_legacy_manifest_without_source_fingerprint_is_not_reused(self):
        self.run_installer()
        old = self.manifest()
        old.pop("source_fingerprint")
        (self.root / "runtime.json").write_text(json.dumps(old), encoding="utf-8")
        self.assertEqual(self.run_installer(), 2)

    def test_second_conversion_failure_preserves_old_manifest_and_pair(self):
        self.run_installer()
        manifest_bytes = (self.root / "runtime.json").read_bytes()
        old_pair = self.pair()
        old_bytes = tuple(path.read_bytes() for path in old_pair)
        (self.base / "model.safetensors").write_bytes(b"new weights")
        count = 0

        def fail_projector(command, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise subprocess.CalledProcessError(1, command)
            return self.convert(command, **kwargs)

        with self.assertRaises(subprocess.CalledProcessError):
            self.run_installer(convert=fail_projector)
        self.assertEqual((self.root / "runtime.json").read_bytes(), manifest_bytes)
        self.assertEqual(tuple(path.read_bytes() for path in old_pair), old_bytes)
        self.assertFalse(list((self.root / "model").glob(".prepare-*")))

    def test_source_change_during_conversion_is_not_published(self):
        self.run_installer()
        manifest_bytes = (self.root / "runtime.json").read_bytes()
        (self.base / "model.safetensors").write_bytes(b"new weights")

        def mutate_source(command, **kwargs):
            result = self.convert(command, **kwargs)
            (self.base / "tokenizer.json").write_bytes(str(self.output_number).encode())
            return result

        with self.assertRaisesRegex(RuntimeError, "изменились во время"):
            self.run_installer(convert=mutate_source)
        self.assertEqual((self.root / "runtime.json").read_bytes(), manifest_bytes)


if __name__ == "__main__":
    unittest.main()
