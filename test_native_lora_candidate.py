"""CPU-only research LoRA startup and request isolation; no native child runs."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.infrastructure import local_llama_runtime as native


class Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b'{"choices":[{"message":{"content":"ok"}}]}'


class Opener:
    def __init__(self):
        self.payloads = []

    def open(self, request, timeout):
        if hasattr(request, "data"):
            self.payloads.append(json.loads(request.data))
        return Response()


class NativeLoraCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = {"release": "b11913", "server": "server.exe", "model": "model.gguf", "mmproj": "mmproj.gguf"}
        self.spawn = self.enterContext(patch.object(native.subprocess, "Popen"))
        self.spawn.return_value.poll.return_value = None
        self.enterContext(patch.object(native, "_own_child_process", return_value=None))
        reservation = self.enterContext(patch.object(native.socket, "socket"))
        reservation.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 51234)

    def candidate(self, name="adapter.gguf", payload=b"candidate-body"):
        path = self.root / name
        path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 2, 2) + payload)
        return path

    def runtime(self, path=None):
        runtime = native.LocalLlamaRuntime(lora_path=path)
        runtime.opener = Opener()
        runtime.endpoint = "http://127.0.0.1:51234"
        runtime.api_key = "test-key"
        self.addCleanup(runtime.close)
        return runtime

    def test_default_start_and_completion_keep_existing_base_behavior(self):
        runtime = self.runtime()
        runtime.start(self.root, self.manifest)
        command = self.spawn.call_args.args[0]
        self.assertNotIn("--lora", command)
        self.assertNotIn("--lora-init-without-apply", command)
        runtime.complete([{"role": "user", "content": "base"}], max_tokens=650)
        payload = runtime.opener.payloads[-1]
        self.assertNotIn("lora", payload)
        self.assertEqual(payload["max_tokens"], 650)
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["repeat_penalty"], 1.08)

    def test_explicit_candidate_is_loaded_disabled_and_default_request_is_base(self):
        path = self.candidate()
        runtime = self.runtime(path)
        runtime.start(self.root, self.manifest)
        command = self.spawn.call_args.args[0]
        self.assertEqual(command[command.index("--lora") + 1], str(path.resolve()))
        self.assertIn("--lora-init-without-apply", command)
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("-np") + 1], "4")
        runtime.complete([{"role": "user", "content": "control"}], max_tokens=10)
        self.assertEqual(runtime.opener.payloads[-1]["lora"], [{"id": 0, "scale": 0.0}])

    def test_reuse_checks_adapter_bytes_even_at_same_size_and_timestamp(self):
        path = self.candidate(payload=b"body-A")
        runtime = self.runtime(path)
        runtime.start(self.root, self.manifest)
        first_identity = runtime._loaded_lora_identity
        self.assertEqual(runtime.start(self.root, self.manifest), 0)
        self.assertEqual(self.spawn.call_count, 1)
        stamp = path.stat()
        path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 2, 2) + b"body-B")
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        runtime.start(self.root, self.manifest)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertNotEqual(runtime._loaded_lora_identity, first_identity)
        self.assertEqual(runtime._loaded_lora_identity[0], first_identity[0])

    def test_changing_candidate_path_restarts_even_when_bytes_match(self):
        runtime = self.runtime(self.candidate("first.gguf"))
        runtime.start(self.root, self.manifest)
        runtime.lora_path = self.candidate("second.gguf")
        runtime.start(self.root, self.manifest)
        self.assertEqual(self.spawn.call_count, 2)
        self.assertEqual(self.spawn.call_args.args[0][-2], str(runtime.lora_path.resolve()))

    def test_invalid_candidate_is_refused_before_discarding_a_running_base(self):
        runtime = self.runtime()
        runtime.start(self.root, self.manifest)
        process = runtime.process
        runtime.lora_path = self.root / "missing.gguf"
        with self.assertRaisesRegex(ValueError, "не найден"):
            runtime.start(self.root, self.manifest)
        process.terminate.assert_not_called()
        self.assertIs(runtime.process, process)
        self.assertEqual(self.spawn.call_count, 1)

    def test_damaged_header_and_unverified_runtime_version_never_start(self):
        path = self.root / "bad.gguf"
        invalid = [b"GGUF", struct.pack("<4sIQQ", b"GGUF", 3, 2, 2),
                   struct.pack("<4sIQQ", b"nope", 3, 2, 2) + b"body",
                   struct.pack("<4sIQQ", b"GGUF", 1, 2, 2) + b"body",
                   struct.pack("<4sIQQ", b"GGUF", 3, 0, 2) + b"body",
                   struct.pack("<4sIQQ", b"GGUF", 3, 2, 0) + b"body"]
        for content in invalid:
            path.write_bytes(content)
            runtime = self.runtime(path)
            with self.subTest(content=content), self.assertRaises(ValueError):
                runtime.start(self.root, self.manifest)
        runtime = self.runtime(self.candidate())
        with self.assertRaisesRegex(ValueError, "b11913"):
            runtime.start(self.root, dict(self.manifest, release="other"))
        self.spawn.assert_not_called()

    def test_per_request_lora_is_copied_and_never_becomes_global_state(self):
        runtime = self.runtime(self.candidate())
        selected = [{"id": 0, "scale": 1}]
        original = copy.deepcopy(selected)
        runtime.complete([{"role": "user", "content": "candidate"}], max_tokens=12, lora=selected,
                         schema={"type": "object"})
        runtime.complete([{"role": "user", "content": "base"}], max_tokens=12, lora=[])
        runtime.complete([{"role": "user", "content": "omitted"}], max_tokens=12)
        self.assertEqual(selected, original)
        self.assertEqual([p["lora"] for p in runtime.opener.payloads],
                         [[{"id": 0, "scale": 1.0}], [{"id": 0, "scale": 0.0}], [{"id": 0, "scale": 0.0}]])
        self.assertIn("response_format", runtime.opener.payloads[0])
        self.assertNotIn("lora_adapters", runtime.opener.payloads[0])

    def test_base_control_can_explicitly_disable_adapters_without_loading_one(self):
        runtime = self.runtime()
        runtime.complete([{"role": "user", "content": "base"}], max_tokens=10, lora=[])
        self.assertEqual(runtime.opener.payloads[-1]["lora"], [])
        with self.assertRaisesRegex(ValueError, "явно загрузите"):
            runtime.complete([], max_tokens=10, lora=[{"id": 0, "scale": 1}])

    def test_malformed_or_nonfinite_lora_selection_is_rejected_before_request(self):
        runtime = self.runtime(self.candidate())
        invalid = [(), {}, [{"id": True, "scale": 1}], [{"id": 0.0, "scale": 1}],
                   [{"id": -1, "scale": 1}], [{"id": 0, "scale": True}],
                   [{"id": 0, "scale": "1"}], [{"id": 0, "scale": float("nan")}],
                   [{"id": 0, "scale": float("inf")}], [{"id": 0, "scale": 10 ** 500}],
                   [{"id": 0, "scale": 1, "extra": 0}],
                   [{"id": 0, "scale": 1}, {"id": 0, "scale": 0}]]
        for selected in invalid:
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                runtime.complete([], max_tokens=10, lora=selected)
        self.assertEqual(runtime.opener.payloads, [])

    def test_concurrent_base_and_candidate_requests_keep_independent_selection(self):
        runtime = self.runtime(self.candidate())

        def answer(index):
            return runtime.complete([{"role": "user", "content": str(index)}], max_tokens=12,
                                    lora=[{"id": 0, "scale": 1}] if index % 2 else [])

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(answer, range(20)))
        self.assertEqual(len(results), 20)
        self.assertEqual(len(runtime.opener.payloads), 20)
        for payload in runtime.opener.payloads:
            index = int(payload["messages"][0]["content"])
            self.assertEqual(payload["lora"], [{"id": 0, "scale": 1.0 if index % 2 else 0.0}])

    def test_explicit_zero_overrides_b11913_global_scale_one_fallback(self):
        # Pinned external semantics: parse_lora_request stores zero as an entry;
        # launch_slot_with_task uses startup configuration only for an empty map.
        # server-common.cpp:281-291 and server-context.cpp:1776-1792 (b11913).
        def pinned_effective_scale(selection):
            config = {entry["id"]: entry["scale"] for entry in selection}
            return config.get(0, 0.0) if config else 1.0

        self.assertEqual(pinned_effective_scale([]), 1.0)
        runtime = self.runtime(self.candidate())
        for selection in (None, []):
            runtime.complete([{"role": "user", "content": "base control"}], max_tokens=12, lora=selection)
            payload = runtime.opener.payloads[-1]
            self.assertTrue(payload["lora"])
            self.assertEqual(pinned_effective_scale(payload["lora"]), 0.0)
        runtime.complete([{"role": "user", "content": "candidate"}], max_tokens=12,
                         lora=[{"id": 0, "scale": 1.0}])
        self.assertEqual(pinned_effective_scale(runtime.opener.payloads[-1]["lora"]), 1.0)


if __name__ == "__main__":
    unittest.main()
