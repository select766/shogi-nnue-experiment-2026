import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_joint_candidate_directions import (  # noqa: E402
    metrics_for_directions,
    select_direction_set,
)


class JointCandidateDirectionsTest(unittest.TestCase):
    def test_metrics_count_unique_moves_and_best_gain(self):
        moves = np.asarray([[0, 1, 1, 2]], dtype=np.int16)
        utilities = np.asarray([[0, 5, 5, 20]], dtype=float)
        metrics = metrics_for_directions(moves, utilities, (0, 2))
        self.assertEqual(3, metrics["diversity"][0])
        self.assertEqual(20, metrics["gains"][0])
        self.assertTrue(metrics["teacher_changed"][0])

    def test_selection_can_replace_axis_with_joint_direction(self):
        # Columns are base, axis0, axis1, joint0, joint1.  The two joint
        # directions retain the fixed utility while covering more moves.
        moves = np.asarray([
            [0, 1, 1, 2, 3],
            [0, 1, 1, 2, 3],
            [0, 1, 1, 2, 3],
        ], dtype=np.int16)
        utilities = np.asarray([
            [0, 10, 0, 10, 0],
            [0, 10, 0, 10, 0],
            [0, 10, 0, 10, 0],
        ], dtype=float)
        selected, fixed, optimized = select_direction_set(
            moves, utilities, fixed=(0, 1), selection_roots=3, select_count=2
        )
        self.assertEqual((0, 2), selected)
        self.assertGreater(
            optimized["candidate_move_diversity_mean"],
            fixed["candidate_move_diversity_mean"],
        )


if __name__ == "__main__":
    unittest.main()
