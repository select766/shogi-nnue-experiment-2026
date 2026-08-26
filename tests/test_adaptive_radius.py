import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_adaptive_radius import (  # noqa: E402
    bootstrap_differences,
    fit_tree,
    load_aligned_details,
    predict_tree,
    select_fixed,
)


class AdaptiveRadiusTest(unittest.TestCase):
    def test_tree_learns_separable_radius(self):
        features = np.asarray([[0.0], [0.1], [0.9], [1.0]])
        labels = np.asarray([0, 0, 1, 1])
        tree = fit_tree(features, labels, max_depth=1, n_classes=2)
        np.testing.assert_array_equal(labels, predict_tree(tree, features))

    def test_fixed_selection_uses_registered_metric_order(self):
        changed = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=float)
        diversity = np.asarray([[2, 3], [2, 3], [2, 3]], dtype=float)
        gains = np.zeros_like(changed)
        self.assertEqual(1, select_fixed(changed, gains, diversity))

    def test_aligned_details_rejects_different_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.jsonl"
            right = Path(directory) / "right.jsonl"
            left.write_text(json.dumps({"source_root": 1}) + "\n")
            right.write_text(json.dumps({"source_root": 2}) + "\n")
            with self.assertRaises(ValueError):
                load_aligned_details([("1", str(left)), ("2", str(right))])

    def test_bootstrap_difference_reports_direction(self):
        result = bootstrap_differences([1, 1, 1], [0, 0, 0], iterations=100)
        self.assertEqual(1.0, result["mean"])
        self.assertGreater(result["bootstrap_95"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
