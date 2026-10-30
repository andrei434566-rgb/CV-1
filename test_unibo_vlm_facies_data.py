"""Reject ambiguous masks and cross-group/pixel leakage before visual training."""
import unittest
import numpy as np
from scripts.prepare_unibo_vlm_facies import (
    exact_mask, eligible_quality, mask_quality, validate_source_groups,
    assert_crop_separation, box_iou, candidate_windows,
)


class UniBoFaciesCropTests(unittest.TestCase):
    def test_palette_does_not_round_unknown_colours_into_a_facies(self):
        rgb = np.array([[(242, 229, 136), (243, 229, 136), (0, 0, 0)]], dtype=np.uint8)
        self.assertEqual(exact_mask(rgb).tolist(), [[1, 255, 0]])

    def test_purity_rejects_mixed_background_and_ignored_regions(self):
        pure = np.ones((100, 100), dtype=np.uint8)
        self.assertTrue(eligible_quality(mask_quality(pure, 1)))
        for replacement in (0, 2, 255):
            mask = pure.copy()
            mask[:10] = replacement
            self.assertFalse(eligible_quality(mask_quality(mask, 1)))
        ignored = pure.copy()
        ignored[0] = 255
        self.assertFalse(eligible_quality(mask_quality(ignored, 1)))

    def test_integral_candidates_do_not_label_a_mixed_boundary_as_pure(self):
        mask = np.ones((256, 256), dtype=np.uint8)
        mask[:, 128:] = 2
        candidates = list(candidate_windows(mask))
        self.assertTrue(candidates)
        self.assertFalse(any(box == (0, 0, 256, 256) for _, box, _ in candidates))
        for label, box, quality in candidates:
            crop = mask[box[1]:box[3], box[0]:box[2]]
            self.assertTrue(eligible_quality(mask_quality(crop, label)))
            self.assertEqual(mask_quality(crop, label), quality)

    def test_source_group_and_original_photo_hash_cannot_cross_splits(self):
        rows = [{"well_group": "S1", "source_sha256": "a", "split": "train"},
                {"well_group": "S2", "source_sha256": "b", "split": "val"}]
        self.assertEqual(validate_source_groups(rows), {"S1": "train", "S2": "val"})
        with self.assertRaisesRegex(ValueError, "S-group"):
            validate_source_groups(rows + [{"well_group": "S1", "source_sha256": "c", "split": "val"}])
        with self.assertRaisesRegex(ValueError, "photo hash"):
            validate_source_groups(rows + [{"well_group": "S3", "source_sha256": "a", "split": "val"}])

    def test_crop_hash_and_test_pixels_are_refused(self):
        def record(group, split, pixel, origin):
            return {"metadata": {"source_group": group, "split": split, "pixel_sha256": pixel, "source_sha256": origin}}
        train, val = record("S1", "train", "p1", "o1"), record("S2", "val", "p2", "o2")
        assert_crop_separation([train, val])
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_crop_separation([train, record("S2", "val", "p1", "o2")])
        with self.assertRaisesRegex(ValueError, "Test"):
            assert_crop_separation([record("S3", "test", "p3", "o3")])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assert_crop_separation([train, train])

    def test_overlap_is_measured_in_the_original_resized_photo(self):
        self.assertEqual(box_iou((0, 0, 128, 128), (0, 0, 128, 128)), 1)
        self.assertEqual(box_iou((0, 0, 128, 128), (128, 0, 256, 128)), 0)
        self.assertGreater(box_iou((0, 0, 128, 128), (64, 0, 192, 128)), .1)


if __name__ == "__main__":
    unittest.main()
