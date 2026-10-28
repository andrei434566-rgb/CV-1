"""Safeguards for evidence decoding, independent from GPU and model weights."""
import unittest

from scripts.finalize_facies_reasoning_v1 import decode_review


class FaciesReportTest(unittest.TestCase):
    def setUp(self):
        cases = [(f"FRV1-{i:02d}__new_facies_reasoning", "new_facies_reasoning") for i in range(1, 13)]
        cases += [(f"old{i}", "closed_book_regression") for i in range(4)]
        cases += [(f"chat{i}", "conversation") for i in range(4)]
        self.blind = [{"record_id": identity, "evaluation_mode": group} for identity, group in cases]
        self.key = {identity: {"A": "baseline", "B": "candidate"} for identity, _ in cases}
        def result(category):
            return {"category": category, "completed": True, "reason": "test", "flags": []}
        self.review = {"reviewer": "test", "rows": [
            {"record_id": identity, "A": result("correct"), "B": result("partial")}
            for identity, _ in cases]}

    def test_exploratory_case_is_not_counted_as_formal(self):
        counts, _, _ = decode_review(self.blind, self.key, self.review)
        self.assertEqual(counts["new_facies_formal"]["baseline"]["correct"], 11)
        self.assertEqual(counts["new_facies_formal"]["candidate"]["partial"], 11)
        self.assertEqual(counts["new_facies_exploratory"]["baseline"]["correct"], 1)

    def test_refuses_missing_duplicate_and_invalid_ratings(self):
        cases = self.review["rows"]
        for bad in (cases[:-1], cases[:-1] + [cases[0]], cases[:-1] + [{**cases[-1], "A": {}}]):
            with self.subTest(last=bad[-1]), self.assertRaises(ValueError):
                decode_review(self.blind, self.key, {"reviewer": "test", "rows": bad})

    def test_refuses_invalid_identity_key(self):
        identity = self.blind[0]["record_id"]
        self.key[identity] = {"A": "baseline", "B": "baseline"}
        with self.assertRaises(ValueError):
            decode_review(self.blind, self.key, self.review)


if __name__ == "__main__":
    unittest.main()
