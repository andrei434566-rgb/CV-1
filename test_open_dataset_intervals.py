"""Tiny source fixtures exercise registration, original text and review gating."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from openpyxl import Workbook
from PIL import Image

from app.infrastructure.open_dataset_intervals import (
    DatasetPreparationCancelled, _normalized_ooxml, prepare_interval_dataset,
)


def _npy(values) -> bytes:
    stream = io.BytesIO()
    np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def _rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class OpenIntervalDatasetTests(unittest.TestCase):
    def test_iodp362_real_lsimg_columns_match_exact_section_archive_half(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (12, 32), "gray").save(root / "section_trim.jpg")
            (root / "362-U1480B-LSIMG.csv").write_text(
                "Exp,Site,Hole,Core,Type,Sect,A/W,Top depth CSF-A (m),Bottom depth CSF-A (m),"
                "Top depth CSF-B (m),Bottom depth CSF-B (m),Uncropped image filename,Cropped image filename\n"
                "362,U1480,B,1,H,2,A,1.53,3.03,1.461,2.893,missing_uncropped.jpg,section_trim.jpg\n",
                encoding="utf-8")
            book = Workbook()
            sheet = book.active
            sheet.append(["Sample", "Top Depth [m]", "Bottom Depth [m]", "Lithology principal name", "General interval comment"])
            sheet.append(["362-U1480B-1H-2-A", 1.7, 1.9, "silty clay [S54]", "Source text remains English."])
            sheet.append(["362-U1480B-1H-2-W", 1.7, 1.9, "sand", "Working half must not match archive half."])
            sheet.append(["362-U1480B-1H-3-A", 1.7, 1.9, "ash", "Another section must not match."])
            book.save(root / "362_macro_U1480B.xlsx")
            report = prepare_interval_dataset("iodp362", [root], root / "out")
            row, = _rows(report["manifest"])
            self.assertEqual("silty clay [S54]", row["source_label"])
            self.assertEqual("Source text remains English.", row["original_description"])
            meta = row["metadata"]
            self.assertEqual("CSF-A", meta["depth_reference"])
            self.assertEqual((1.53, 3.03), (meta["photo_depth_top_m"], meta["photo_depth_bottom_m"]))
            self.assertEqual("362-U1480B-LSIMG.csv", Path(meta["photo_source_path"]).name)
            self.assertEqual(3, len(meta["photo_source_field_selections"]))
            self.assertIn("standard_sample_measurement", meta["source_depth_reference_convention"]["rule"])
            self.assertTrue(meta["needs_interval_crop"])
            self.assertFalse(meta["pixel_alignment_verified"])
            self.assertFalse(meta["training_approved"])
            documents = _rows(report["source_documents"])
            photo_doc = next(d for d in documents if "CSF-B" in d["text"])
            self.assertEqual("1.461", photo_doc["original_cells"]["Top depth CSF-B (m) [10]"])

    def test_generic_iodp_table_never_assumes_lsimg_depth_scale_or_matches_partial_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (8, 16), "gray").save(root / "section.jpg")
            (root / "362-U1480A-LSIMG.csv").write_text(
                "Exp,Site,Hole,Core,Type,Sect,A/W,Top depth CSF-A (m),Bottom depth CSF-A (m),Cropped image filename\n"
                "362,U1480,A,1,H,1,A,0,1.5,section.jpg\n", encoding="utf-8")
            book = Workbook()
            sheet = book.active
            sheet.append(["Sample", "Top Depth [m]", "Bottom Depth [m]", "Lithology"])
            sheet.append(["362-U1480A-1H-1-A", 0.1, 0.2, "sand"])
            book.save(root / "unrecognized_export.xlsx")
            report = prepare_interval_dataset("iodp362", [root], root / "out")
            self.assertEqual(0, report["counts"]["paired_candidates"])
            row, = _rows(report["intervals_manifest"])
            self.assertNotIn("source_depth_reference_convention", row)

    def test_iodp_lowercase_ooxml_parts_preserve_source_cells_and_record_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            book = Workbook()
            sheet = book.active
            sheet.title = "sediment"
            sheet.append(["Sample", "Top [cm]", "Bottom [cm]", "Top Depth [m]", "Bottom Depth [m]",
                          "Lithology principal name", "General interval comment"])
            sheet.append(["362-U1480A-1H-1-A", 0, 6, 0, 0.06, "ash/tuff [Leg202]", "Original graded ash."])
            valid = io.BytesIO()
            book.save(valid)
            broken = io.BytesIO()
            with zipfile.ZipFile(io.BytesIO(valid.getvalue())) as source, zipfile.ZipFile(broken, "w") as target:
                for entry in source.infolist():
                    name = "[content_types].xml" if entry.filename == "[Content_Types].xml" else entry.filename
                    target.writestr(name, source.read(entry))
                # A shared-string part is optional for openpyxl-authored inline
                # strings, but its known lowercase alias must also normalize.
                target.writestr("xl/sharedstrings.xml", '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>')
            archive = root / "362-DESCVCD.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("data_by_hole/362_macro_U1480A.xlsx", broken.getvalue())
            source_before = archive.read_bytes()
            report = prepare_interval_dataset("iodp362", [archive], root / "out")
            row, = _rows(report["intervals_manifest"])
            self.assertEqual("U1480A", row["well_id"])
            self.assertEqual((0, 0.06), (row["depth_top_m"], row["depth_bottom_m"]))
            self.assertEqual("ash/tuff [Leg202]", row["source_label"])
            self.assertEqual("Original graded ash.", row["original_description"])
            self.assertEqual("ash/tuff [Leg202]", row["original_cells"]["Lithology principal name [6]"])
            self.assertEqual({"[Content_Types].xml", "xl/sharedStrings.xml"},
                             {change["to"] for change in row["source_workbook_normalizations"]})
            self.assertEqual(0, report["counts"]["paired_candidates"])
            self.assertFalse(row["training_approved"])
            self.assertEqual(source_before, archive.read_bytes())

    def test_ooxml_case_conflicts_and_uncompressed_limits_are_rejected(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("[Content_Types].xml", "original")
            target.writestr("[content_types].xml", "conflicting")
        with self.assertRaisesRegex(ValueError, "conflicting_case"):
            _normalized_ooxml(archive.getvalue())
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as target:
            target.writestr("../outside.xml", "unsafe")
        with self.assertRaisesRegex(ValueError, "unsafe_workbook_part"):
            _normalized_ooxml(archive.getvalue())
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as target:
            target.writestr("xl/workbook.xml", "abc")
        with patch("app.infrastructure.open_dataset_intervals.MAX_DOCUMENT_BYTES", 2):
            with self.assertRaisesRegex(ValueError, "workbook_uncompressed_size_or_entry_limit"):
                _normalized_ooxml(archive.getvalue())

    def test_ukcs_preserves_original_code_and_exact_registered_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "Q204_data.zip"
            image = np.zeros((24, 4, 3), dtype=np.uint8)
            image[:12, :, 0], image[12:, :, 1] = 120, 200
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("Q204/W-1_labels.npy", _npy(np.array([b"s"] * 12 + [b"sh"] * 12)))
                target.writestr("Q204/W-1_depth.npy", _npy(100 + np.arange(24) * 0.001))
                target.writestr("Q204/W-1_image.npy", _npy(image))
                target.writestr("../outside.txt", "malicious")
            report = prepare_interval_dataset("ukcs_q204_q205", [archive], root / "review")
            records = _rows(report["manifest"])
            intervals = _rows(report["intervals_manifest"])
            self.assertEqual({"s", "sh"}, {r["source_label"] for r in records})
            self.assertEqual(2, report["counts"]["intervals"])
            self.assertEqual(2, report["counts"]["paired_candidates"])
            self.assertEqual(0, report["counts"]["training_approved"])
            self.assertFalse(report["readiness"]["can_train"])
            self.assertIn("unsafe_archive_member", report["unmatched_reasons"])
            self.assertFalse((root / "outside.txt").exists())
            first = next(r for r in records if r["source_label"] == "s")
            with Image.open(root / "review" / first["image"]) as photo:
                np.testing.assert_array_equal(np.asarray(photo), image[:12])
            self.assertEqual("s", first["conversations"][1]["value"])
            self.assertEqual("", first["original_description"])
            self.assertFalse(first["metadata"]["training_approved"])
            self.assertEqual(100.011, intervals[0]["depth_bottom_m"])
            self.assertEqual("first_and_last_observed_row_not_interval_edges", intervals[0]["depth_bounds_kind"])

    def test_ukcs_mismatched_rows_never_create_training_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with zipfile.ZipFile(root / "bad.zip", "w") as target:
                target.writestr("W_labels.npy", _npy(np.array([b"s"] * 10)))
                target.writestr("W_depth.npy", _npy(np.arange(10) / 1000))
                target.writestr("W_image.npy", _npy(np.zeros((9, 3, 3), dtype=np.uint8)))
            report = prepare_interval_dataset("ukcs_q204", [root / "bad.zip"], root / "out")
            self.assertEqual(0, report["counts"]["paired_candidates"])
            self.assertEqual(0, report["counts"]["intervals"])
            self.assertIn("ukcs_alignment_or_preview_failed", report["unmatched_reasons"])

    def test_ukcs_preview_depths_describe_crop_not_long_source_run(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.arange(620 * 3 * 3, dtype=np.uint8).reshape(620, 3, 3)
            with zipfile.ZipFile(root / "Q204.zip", "w") as target:
                target.writestr("W_labels.npy", _npy(np.array([b"s"] * 620)))
                target.writestr("W_depth.npy", _npy(np.arange(620) / 1000))
                target.writestr("W_image.npy", _npy(image))
            report = prepare_interval_dataset("ukcs_q204", [root / "Q204.zip"], root / "out")
            row = _rows(report["manifest"])[0]
            self.assertEqual(0.01, row["metadata"]["depth_top_m"])
            self.assertEqual(0.609, row["metadata"]["depth_bottom_m"])
            self.assertEqual(0.0, row["metadata"]["source_run_depth_top_m"])
            self.assertEqual(0.619, row["metadata"]["source_run_depth_bottom_m"])
            with Image.open(root / "out" / row["image"]) as photo:
                np.testing.assert_array_equal(np.asarray(photo), image[10:610])

    def test_iodp_cross_table_requires_identity_depth_and_preserves_english(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.new("RGB", (8, 16), "gray").save(root / "362-U1480E-1H.jpg")
            (root / "photos.csv").write_text(
                "Site,Hole,Core,Type,Top depth CSF-A (m),Bottom depth CSF-A (m),Image filename\n"
                "U1480,E,1,H,1,2,362-U1480E-1H.jpg\n", encoding="utf-8")
            book = Workbook()
            sheet = book.active
            sheet.append(["Site", "Hole", "Core", "Type", "Top depth CSF-A (m)", "Bottom depth CSF-A (m)", "Full Lithology", "Description"])
            sheet.append(["U1480", "E", 1, "H", 1.2, 1.6, "silty clay", "Original source: burrows observed."])
            sheet.append(["U1480", "F", 1, "H", 1.2, 1.6, "sand", "Different hole must not match."])
            book.save(root / "supplement.xlsx")
            report = prepare_interval_dataset("iodp362", [root], root / "out")
            records = _rows(report["manifest"])
            self.assertEqual(1, len(records))
            self.assertEqual("silty clay", records[0]["source_label"])
            self.assertEqual("Original source: burrows observed.", records[0]["original_description"])
            self.assertTrue(records[0]["metadata"]["needs_interval_crop"])
            self.assertFalse(records[0]["metadata"]["pixel_alignment_verified"])
            self.assertFalse(records[0]["metadata"]["training_approved"])
            self.assertNotIn("conversations", records[0])
            self.assertEqual(2, report["counts"]["intervals"])
            self.assertEqual(3, report["counts"]["documents"])
            self.assertIn("interval_has_no_unique_identifier_depth_photo_match", report["unmatched_reasons"])

    def test_iodp_documents_without_photo_are_searchable_not_pseudo_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "notes.txt").write_text("Original sedimentological notes, not an image label.", encoding="utf-8")
            (root / "ambiguous.csv").write_text(
                "Top depth CSF-A (m),Top depth CSF-B (m),Bottom depth CSF-A (m),Lithology\n1,2,3,sand\n", encoding="utf-8")
            report = prepare_interval_dataset("iodp_362", [root], root / "out")
            self.assertEqual("source_index_only", report["status"])
            self.assertEqual(0, report["counts"]["paired_candidates"])
            self.assertEqual(0, report["counts"]["intervals"])
            docs = _rows(report["source_documents"])
            self.assertTrue(any("Original sedimentological notes" in d["text"] for d in docs))
            self.assertIn("table_row_has_no_unambiguous_metre_depth_interval", report["unmatched_reasons"])

    def test_existing_review_and_cancelled_run_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "review").mkdir()
            (root / "review" / "review.csv").write_text("expert corrections", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_interval_dataset("iodp362", [], root / "review")
            event = threading.Event()
            event.set()
            with self.assertRaises(DatasetPreparationCancelled):
                prepare_interval_dataset("iodp362", [], root / "cancelled", cancel=event)
            self.assertFalse((root / "cancelled").exists())


if __name__ == "__main__":
    unittest.main()
