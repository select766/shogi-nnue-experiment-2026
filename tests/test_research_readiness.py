import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.check_research_readiness import check_catalog
from train_nnue.research_loop import write_json


class CatalogTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "manifest.json").write_text('{}\n')
        (self.root / "sample.jsonl").write_text('{}\n')
        write_json(self.root / "configs/research_data.json", {"datasets": [{
            "id": "fixture", "manifest": "manifest.json",
            "manifest_sha256": hashlib.sha256(b'{}\n').hexdigest(),
            "paths": {"test": "sample.jsonl"}}]})

    def test_catalog_inputs_present(self):
        self.assertEqual(check_catalog(self.root)["datasets"][0]["id"], "fixture")

    def test_changed_manifest_rejected(self):
        (self.root / "manifest.json").write_text('{"changed": true}\n')
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            check_catalog(self.root)

    def test_empty_input_rejected(self):
        (self.root / "sample.jsonl").write_text('')
        with self.assertRaisesRegex(ValueError, "missing data"):
            check_catalog(self.root)
