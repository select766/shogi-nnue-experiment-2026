import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_decision_teacher_cache import build_cache  # noqa: E402


class DecisionTeacherCacheTest(unittest.TestCase):
    def test_hard_target_uses_winning_expert_directions(self):
        row = {
            "utilities_cp": [0, 20, 20, -5, 0, 0, 0, 0, 0],
            "gates": [([0.125] * 8)] * 9,
        }
        cache, changed = build_cache([row])
        self.assertEqual(1, changed)
        np.testing.assert_array_equal(cache[0, 8:], [0.5, 0.5, 0, 0, 0, 0, 0, 0])

    def test_small_gain_retains_base_gate(self):
        base = np.asarray([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
        row = {"utilities_cp": [0, 9, 0, 0, 0, 0, 0, 0, 0], "gates": [base] * 9}
        cache, changed = build_cache([row])
        self.assertEqual(0, changed)
        np.testing.assert_allclose(cache[0, 8:], base)


if __name__ == "__main__":
    unittest.main()
