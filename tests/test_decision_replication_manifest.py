import tempfile
import unittest
from pathlib import Path

from scripts.prepare_decision_replication_manifest import build_inventory, inspect_file


class ProvenanceTests(unittest.TestCase):
    def test_position_index_is_not_game_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows.jsonl").write_text('{"source_index": 1, "source_root": 2}\n')
            row = inspect_file(root, {"path": "rows.jsonl", "mode": "jsonl_schema"})
            self.assertEqual(row["rows"], 1)
            self.assertEqual(row["rows_with_namespaced_game_id"], 0)

    def test_missing_or_empty_inputs_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_inventory(Path(directory), {
                "inventory": [{"path": "missing"}], "unresolved_requirements": []})
            self.assertEqual(result["status"], "held")
            self.assertFalse(result["exclusion_verified"])
            self.assertFalse(result["formal_matches_authorized"])
            self.assertTrue(any("Missing evidence" in b for b in result["blockers"]))

    def test_large_jsonl_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "large").open("wb") as stream:
                stream.truncate(33 * 1024 * 1024)
            result = inspect_file(root, {"path": "large", "mode": "jsonl_schema"})
            self.assertEqual(result["inspection"], "not_scanned_size_limit")
            self.assertNotIn("sha256", result)
