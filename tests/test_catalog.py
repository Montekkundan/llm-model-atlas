"""The reading catalog must not overstate executable architecture support."""

import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

from atlas import load_preset


ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog" / "model_families.json"


class CatalogTests(unittest.TestCase):
    def test_all_23_unique_numbered_case_studies_have_sources(self):
        catalog = json.loads(CATALOG.read_text())
        self.assertEqual(catalog["schemaVersion"], 1)
        cards = catalog["cards"]
        self.assertEqual(len(cards), 23)
        self.assertEqual([card["lessonNumber"] for card in cards], list(range(51, 74)))
        self.assertEqual(len({card["lessonSlug"] for card in cards}), 23)
        for card in cards:
            with self.subTest(lesson=card["lessonNumber"]):
                self.assertTrue(card["lessonTitle"].strip())
                self.assertTrue(card["definingMechanisms"])
                self.assertTrue(card["sources"])
                for source in card["sources"]:
                    url = urlparse(source["url"])
                    self.assertEqual(url.scheme, "https")
                    self.assertTrue(url.netloc)

    def test_only_actual_tiny_paths_claim_runnable_status(self):
        cards = json.loads(CATALOG.read_text())["cards"]
        runnable = [card for card in cards if card["implementationStatus"] == "runnable_tiny_text_path"]
        self.assertEqual({card["lessonNumber"] for card in runnable}, {52, 53})
        self.assertEqual({card["runnablePreset"] for card in runnable}, {"olmo2", "gemma3"})
        self.assertEqual({path.stem for path in (ROOT / "presets").glob("*.json")}, {"olmo2", "gemma3"})
        for card in runnable:
            with self.subTest(preset=card["runnablePreset"]):
                self.assertIsNotNone(load_preset(card["runnablePreset"]))
                self.assertIn("not published weights", card["implementationNote"])
        for card in cards:
            if card["lessonNumber"] in {52, 53}:
                continue
            with self.subTest(pending=card["lessonNumber"]):
                self.assertEqual(card["implementationStatus"], "documented_pending")
                self.assertIsNone(card["runnablePreset"])
                self.assertIn("no runnable", card["implementationNote"])
        self.assertEqual(next(card for card in cards if card["lessonNumber"] == 72)["kind"], "survey")


if __name__ == "__main__":
    unittest.main()
