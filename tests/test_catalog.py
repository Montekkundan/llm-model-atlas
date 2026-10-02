"""The reading catalog must not overstate executable architecture support."""

import json
import unittest
from pathlib import Path
from urllib.parse import urlparse

from atlas import load_preset
from atlas.case_study import CASE_OPERATORS, run_case


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
        self.assertEqual({card["lessonNumber"] for card in runnable}, {51, 52, 53, 54, 56})
        self.assertEqual({card["runnablePreset"] for card in runnable}, {"deepseek_v3_style", "olmo2", "gemma3", "mistral_small31", "qwen3_dense"})
        self.assertEqual({path.stem for path in (ROOT / "presets").glob("*.json")}, {"deepseek_v3_style", "olmo2", "gemma3", "mistral_small31", "qwen3_dense"})
        for card in runnable:
            with self.subTest(preset=card["runnablePreset"]):
                self.assertIsNotNone(load_preset(card["runnablePreset"]))
                self.assertIn("not published weights", card["implementationNote"].lower())
        for card in cards:
            if card["lessonNumber"] in {51, 52, 53, 54, 56}:
                continue
            with self.subTest(pending=card["lessonNumber"]):
                self.assertEqual(card["implementationStatus"], "runnable_reference_mechanisms")
                self.assertIsNone(card["runnablePreset"])
                self.assertIn("No runnable full-family preset", card["implementationNote"])
                self.assertTrue(card["referenceOperators"])
                self.assertEqual(card["referenceCommand"], f"python -m atlas.case_study {card['lessonNumber']}")
        self.assertEqual(next(card for card in cards if card["lessonNumber"] == 72)["kind"], "survey")

    def test_catalog_lists_the_operators_the_case_runner_executes(self):
        for card in json.loads(CATALOG.read_text())["cards"]:
            with self.subTest(lesson=card["lessonNumber"]):
                operators = list(CASE_OPERATORS[card["lessonNumber"]])
                self.assertEqual(card["referenceOperators"], operators)
                if card["implementationStatus"] == "runnable_reference_mechanisms":
                    self.assertIn("checks: " + ", ".join(operators) + ".", card["implementationNote"])

    def test_documented_duplicate_operator_results(self):
        # README: these lessons print identical checks because they run the same operators.
        groups = {}
        for lesson in range(51, 74):
            groups.setdefault(json.dumps(run_case(lesson)["checks"], sort_keys=True), []).append(lesson)
        self.assertEqual(sorted(group for group in groups.values() if len(group) > 1),
                         [[51, 58, 67], [53, 63], [66, 71]])


if __name__ == "__main__":
    unittest.main()
