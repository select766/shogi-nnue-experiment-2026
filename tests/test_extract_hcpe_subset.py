import json
import tempfile
import unittest
from pathlib import Path

from train_nnue.extract_hcpe_subset import load_excluded_sfens, parse_splits


class ExtractHcpeSubsetTest(unittest.TestCase):
    def test_default_splits_remain_backward_compatible(self):
        self.assertEqual(
            parse_splits(None, 1000),
            [("train", 1000), ("val", 1000), ("test", 1000)],
        )

    def test_parses_named_splits(self):
        self.assertEqual(
            parse_splits(["validation=10000", "test=10000"], 1000),
            [("validation", 10000), ("test", 10000)],
        )

    def test_rejects_duplicate_split_names(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_splits(["test=10", "test=20"], 1000)

    def test_loads_excluded_positions_by_sfen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.jsonl"
            path.write_text(
                json.dumps({"sfen": "position-a", "bestmove": "7g7f"}) + "\n"
            )

            self.assertEqual(load_excluded_sfens([path]), {"position-a"})


if __name__ == "__main__":
    unittest.main()
