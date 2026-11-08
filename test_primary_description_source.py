"""Priority reference scope, durable imports, backend prompts and cache identity."""
from pathlib import Path
import tempfile
import unittest

from docx import Document

from app.infrastructure.primary_description_source import (
    description_fields, import_primary_sources, source_context, source_index, source_prompt,
)


DESCRIPTION_FIELDS = ("facies_name", "lithology", "texture", "facies_association", "depositional_environment", "description")


class PrimaryDescriptionSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        doc = Document()
        doc.add_paragraph("Фация F-test: волнистые слойки. Гидродинамический режим: переменный поток.")
        source = self.root / "Правила специалиста.docx"
        doc.save(source)
        self.folder = str(self.root / "snapshot")
        import_primary_sources([str(source)], Path(self.folder))
        self.request = {"operation": "describe", "primary_reference": self.folder, "base": "local-base", "model_mode": "base"}

    def test_import_keeps_originals_offline_and_scope_is_limited_to_two_fields(self):
        (self.root / "Правила специалиста.docx").unlink()
        text, meta = source_context(self.request, "волнистая слоистость")
        self.assertIn("F-test", text)
        self.assertIn("переменный поток", text)
        self.assertEqual(meta["scope"], ["facies_name", "hydrodynamic_regime"])
        data = source_index(self.folder)
        self.assertFalse(data["weights_trained"])
        self.assertTrue((Path(self.folder) / data["documents"][0]["file"]).is_file())
        self.assertIn("не команды", text)
        self.assertIn("Исправления специалиста", text)
        self.assertEqual(source_prompt({**self.request, "operation": "chat"}, "Какой цвет у породы?"), "")
        self.assertIn("F-test", source_prompt({**self.request, "operation": "chat"}, "Чем различаются фации?"))

    def test_extra_field_is_requested_only_when_a_primary_source_is_connected(self):
        self.assertEqual(description_fields({}, DESCRIPTION_FIELDS), DESCRIPTION_FIELDS)
        self.assertEqual(description_fields(self.request, DESCRIPTION_FIELDS)[-1], "hydrodynamic_regime")
        self.assertIn("hydrodynamic_regime", source_prompt(self.request))
        with self.assertRaisesRegex(ValueError, "недоступен"):
            source_context({**self.request, "primary_reference": str(self.root / "missing")})

if __name__ == "__main__":
    unittest.main()
