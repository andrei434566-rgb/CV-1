"""Colour suggestions are conservative and only use actual crop pixels."""
import unittest

import numpy as np
from PIL import Image

from app.infrastructure.core_color_hint import estimate_core_color


def bgr_image(array):
    return Image.fromarray(array[:, :, ::-1].copy(), "RGB")


class CoreColorHintTests(unittest.TestCase):
    def test_grey_core_with_brown_tray_and_white_label(self):
        image = np.full((240, 100, 3), (155, 162, 166), dtype=np.uint8)
        image[:, :12] = image[:, -12:] = (55, 90, 140)
        image[90:125, 36:64] = (255, 255, 255)
        result = estimate_core_color(bgr_image(image))
        self.assertIsNotNone(result)
        self.assertIn("серый", result.label)
        self.assertIn("освещения", result.limitation)

    def test_brown_core_is_not_forced_grey(self):
        image = np.full((200, 100, 3), (55, 90, 140), dtype=np.uint8)
        image[:, :8] = image[:, -8:] = (170, 170, 170)
        self.assertEqual(estimate_core_color(bgr_image(image)).label, "коричневый")

    def test_mixed_colours_are_not_reported_as_one_dominant_colour(self):
        image = np.full((200, 100, 3), (160, 160, 160), dtype=np.uint8)
        image[:, 50:] = (55, 90, 140)
        self.assertIsNone(estimate_core_color(bgr_image(image)))

    def test_other_colours_are_not_misrepresented_as_grey_or_brown(self):
        image = Image.new("RGB", (100, 200), (25, 140, 45))
        self.assertIsNone(estimate_core_color(image))

    def test_white_background_and_too_little_rock_are_not_grey(self):
        image = np.full((200, 100, 3), 255, dtype=np.uint8)
        self.assertIsNone(estimate_core_color(bgr_image(image)))
        image[80:90, 45:55] = (140, 140, 140)
        self.assertIsNone(estimate_core_color(bgr_image(image)))

    def test_mask_excludes_brown_outside_polygon(self):
        image = np.full((200, 100, 3), (55, 90, 140), dtype=np.uint8)
        image[:, 38:62] = (155, 160, 165)
        mask = np.zeros((200, 100), dtype=np.uint8)
        mask[:, 38:62] = 255
        result = estimate_core_color(bgr_image(image), mask)
        self.assertIsNotNone(result)
        self.assertIn("серый", result.label)

    def test_invalid_and_tiny_input_returns_no_hint(self):
        self.assertIsNone(estimate_core_color(None))
        self.assertIsNone(estimate_core_color(Image.new("RGB", (5, 5))))
        self.assertIsNone(estimate_core_color(Image.new("RGB", (20, 20)), np.zeros((2, 2))))


if __name__ == "__main__":
    unittest.main()
