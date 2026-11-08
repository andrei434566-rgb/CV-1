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
        result=estimate_core_color(bgr_image(image))
        self.assertTrue(result.mixed)
        self.assertIn('серый',result.label);self.assertIn('коричневый',result.label)
        self.assertAlmostEqual(result.supporting_fraction,.5)

    def test_other_colours_are_not_misrepresented_as_grey_or_brown(self):
        image = Image.new("RGB", (100, 200), (25, 140, 45))
        self.assertEqual(estimate_core_color(image).label,'зелёный')

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

    def test_neutral_warm_grey_and_wide_palette_are_measured_from_pixels(self):
        examples=[((180,178,173),'светло-серый'),((128,128,128),'серый'),((70,70,70),'тёмно-серый'),
                  ((30,30,30),'чёрный'),((235,235,235),'белый'),((200,185,155),'бежевый')]
        for rgb,label in examples:
            with self.subTest(rgb=rgb):
                result=estimate_core_color(Image.new('RGB',(80,180),rgb))
                self.assertEqual(result.label,label);self.assertEqual(result.rgb,rgb)
                self.assertAlmostEqual(sum(row['fraction'] for row in result.components),1.)

    def test_labels_shadows_and_coloured_outside_alpha_do_not_change_core(self):
        image=Image.new('RGBA',(120,240),(140,90,55,0))
        image.paste((140,140,140,255),(20,0,100,240))
        image.paste((230,190,120,255),(30,70,90,130))
        result=estimate_core_color(image,exclude_boxes=[[.25,70/240,.75,130/240]])
        self.assertEqual(result.label,'серый')
        self.assertEqual(result.rgb,(140,140,140))

    def test_mask_and_colour_raster_are_bounded_for_large_images(self):
        result=estimate_core_color(Image.new('RGB',(1600,4200),(128,128,128)))
        self.assertLessEqual(result.usable_pixels,768*2048)

    def test_dark_and_pale_bands_are_not_averaged_into_invented_mid_grey(self):
        image=Image.new('RGB',(100,200),(55,55,55));image.paste((210,210,210),(0,100,100,200))
        result=estimate_core_color(image)
        self.assertTrue(result.mixed)
        self.assertIn('тёмно-серый',result.label);self.assertIn('светло-серый',result.label)

    def test_accepted_expert_colour_is_not_replaced_even_if_it_matches_old_auto(self):
        from app.infrastructure.core_color_hint import apply_photo_color
        attrs={'Цвет':'Серый','__vision_color':'Серый','__color_expert_confirmed':'1'}
        self.assertFalse(apply_photo_color(attrs,{'label':'коричневый'}))
        self.assertEqual(attrs['Цвет'],'Серый')
        self.assertTrue(apply_photo_color(attrs,{'label':'коричневый'},force=True))
        self.assertEqual(attrs['__color_expert_confirmed'],'0')


if __name__ == "__main__":
    unittest.main()
