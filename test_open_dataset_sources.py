"""Download integrity, cancellation and source restrictions without network IO."""

from __future__ import annotations

import hashlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.infrastructure import open_dataset_sources as datasets


class FakeResponse(io.BytesIO):
    def __init__(self, body=b"", *, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body)), "ETag": '"version-1"', **(headers or {})}

    def geturl(self):
        return "https://example.org/core.zip"


def artifact(body=b"abcdef"):
    return datasets.ArtifactDescriptor("example", "core.zip", "https://example.org/core.zip",
        len(body), hashlib.sha256(body).hexdigest())


class OpenDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".open-dataset-test-", dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.disk = patch.object(datasets.shutil, "disk_usage", return_value=SimpleNamespace(free=20 * 1024 ** 3))
        self.disk.start()
        self.addCleanup(self.disk.stop)

    def test_six_catalog_sources_keep_noncommercial_restrictions(self):
        self.assertEqual({"corebreakout", "unibo", "ukcs", "iodp362", "dcid", "mendeley"},
                         {source.id for source in datasets.list_sources()})
        self.assertEqual({"unibo", "dcid"},
                         {source.id for source in datasets.list_sources() if source.noncommercial})
        self.assertIn("SA", datasets.get_source("unibo").license_id)
        self.assertTrue(all(item.expected_bytes for item in datasets.resolve_artifacts("iodp362")))

    def test_success_checks_digest_and_atomically_creates_final_file(self):
        reports = []
        with patch.object(datasets, "_open", return_value=FakeResponse(b"abcdef")):
            path = datasets.download_artifact(artifact(), self.root, progress=reports.append)
        self.assertEqual(b"abcdef", path.read_bytes())
        self.assertFalse((self.root / "core.zip.part").exists())
        self.assertFalse((self.root / "core.zip.part.json").exists())
        self.assertEqual("complete", reports[-1]["status"])

    def test_cancel_then_resume_uses_range_and_preserves_original_bytes(self):
        body = b"abcdefghij"
        cancelled = threading.Event()
        def progress(event):
            if event["status"] == "downloading" and event["downloaded_bytes"] >= 4:
                cancelled.set()
        with patch.object(datasets, "_CHUNK_BYTES", 4), \
             patch.object(datasets, "_open", return_value=FakeResponse(body)):
            with self.assertRaises(datasets.DownloadCancelled):
                datasets.download_artifact(artifact(body), self.root, cancel=cancelled, progress=progress)
        self.assertEqual(b"abcd", (self.root / "core.zip.part").read_bytes())
        self.assertFalse((self.root / "core.zip").exists())
        response = FakeResponse(body[4:], status=206, headers={"Content-Range": "bytes 4-9/10"})
        with patch.object(datasets, "_open", return_value=response) as network:
            path = datasets.download_artifact(artifact(body), self.root)
        self.assertEqual("bytes=4-", network.call_args.kwargs["headers"]["Range"])
        self.assertEqual(body, path.read_bytes())

    def _partial(self, body=b"abcdef", partial=b"abc"):
        item = artifact(body)
        (self.root / "core.zip.part").write_bytes(partial)
        datasets._write_metadata(self.root / "core.zip.part.json", {
            "url": item.url, "expected_bytes": len(body), "checksum": item.checksum,
            "algorithm": item.checksum_algorithm, "etag": '"version-1"',
        })
        return item

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        item = self._partial()
        with patch.object(datasets, "_open", return_value=FakeResponse(b"abcdef")):
            path = datasets.download_artifact(item, self.root)
        self.assertEqual(b"abcdef", path.read_bytes())

    def test_wrong_range_is_rejected_without_modifying_partial(self):
        item = self._partial()
        with patch.object(datasets, "_open", return_value=FakeResponse(b"def", status=206,
                headers={"Content-Range": "bytes 2-4/6"})):
            with self.assertRaises(datasets.DatasetDownloadError):
                datasets.download_artifact(item, self.root)
        self.assertEqual(b"abc", (self.root / "core.zip.part").read_bytes())

    def test_short_stream_is_not_promoted_to_completed_file(self):
        with patch.object(datasets, "_open", return_value=FakeResponse(b"abc", headers={"Content-Length": "6"})):
            with self.assertRaises(datasets.DatasetDownloadError):
                datasets.download_artifact(artifact(), self.root)
        self.assertEqual(b"abc", (self.root / "core.zip.part").read_bytes())
        self.assertFalse((self.root / "core.zip").exists())

    def test_checksum_mismatch_is_never_promoted(self):
        with patch.object(datasets, "_open", return_value=FakeResponse(b"xxxxxx")):
            with self.assertRaises(datasets.DatasetDownloadError):
                datasets.download_artifact(artifact(), self.root)
        self.assertFalse((self.root / "core.zip").exists())

    def test_oversized_stream_and_wrong_content_length_are_rejected(self):
        for response in (FakeResponse(b"abcdefg", headers={"Content-Length": "6"}),
                         FakeResponse(b"abcdefg")):
            with self.subTest(headers=response.headers), patch.object(datasets, "_open", return_value=response):
                with self.assertRaises(datasets.DatasetDownloadError):
                    datasets.download_artifact(artifact(), self.root, resume=False)
                self.assertFalse((self.root / "core.zip").exists())

    def test_existing_wrong_file_is_preserved(self):
        target = self.root / "core.zip"
        target.write_bytes(b"manual")
        with patch.object(datasets, "_open") as network:
            with self.assertRaises(datasets.DatasetDownloadError):
                datasets.download_artifact(artifact(), self.root)
        network.assert_not_called()
        self.assertEqual(b"manual", target.read_bytes())

    def test_low_disk_space_is_checked_before_network_and_keeps_margin(self):
        with patch.object(datasets.shutil, "disk_usage", return_value=SimpleNamespace(free=datasets.DISK_MARGIN_BYTES)), \
             patch.object(datasets, "_open") as network:
            with self.assertRaises(datasets.DatasetDownloadError):
                datasets.download_artifact(artifact(), self.root)
        network.assert_not_called()

    def test_source_identity_mismatch_cannot_resume_a_different_download(self):
        self._partial(body=b"xxxxxx")
        with patch.object(datasets, "_open") as network:
            with self.assertRaises(datasets.DatasetDownloadError):
                datasets.download_artifact(artifact(), self.root)
        network.assert_not_called()

    def test_traversal_and_reserved_filenames_rejected(self):
        for filename in ("../escape.zip", "..\\escape.zip", "C:escape.zip", "CON.zip", "x."):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                datasets.ArtifactDescriptor("bad", filename, "https://example.org/file")

    def test_local_archives_discovered_in_sibling_and_user_storage(self):
        app = self.root / "app_checkout"
        sibling = self.root / "vlm_core_v1" / "data" / "external_raw" / "unibo_facies_7308"
        sibling.mkdir(parents=True)
        first = sibling / "Training_subset.zip"
        first.write_bytes(b"archive")
        storage = self.root / "storage"
        storage.mkdir()
        second = storage / "Test_subset.zip"
        second.write_bytes(b"archive")
        (storage / "Validation_subset.zip.part").write_bytes(b"incomplete")
        self.assertEqual({first.resolve(), second.resolve()},
                         set(datasets.discover_local_artifacts("unibo", storage, app)))

    def test_refresh_figshare_uses_file_size_and_published_md5(self):
        payload = {"files": [{"name": "Q204_data.zip", "size": 123,
                    "download_url": "https://ndownloader.figshare.com/files/1",
                    "computed_md5": "0" * 32}]}
        with patch.object(datasets, "_metadata_json", return_value=payload):
            item, = datasets.resolve_artifacts("ukcs", refresh=True)
        self.assertEqual((123, "md5", "0" * 32),
                         (item.expected_bytes, item.checksum_algorithm, item.checksum))


if __name__ == "__main__":
    unittest.main()
