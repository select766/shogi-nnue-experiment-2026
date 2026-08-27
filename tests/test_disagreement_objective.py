import unittest

import numpy as np

from scripts.analyze_disagreement_objective import candidate_arrays, select_radius_vector


class DisagreementObjectiveTest(unittest.TestCase):
    def test_pooled_utilities_replace_run_specific_scores(self):
        rows = [
            [{"source_root": 7, "moves": ["a"] * 9, "utilities_cp": [1] * 9}],
            [{"source_root": 7, "moves": ["a"] * 9, "utilities_cp": [2] * 9}],
        ]
        pooled = [{"source_root": 7, "unique_move_scores_cp": {"a": 3}}]
        _, utilities = candidate_arrays(rows, pooled)
        np.testing.assert_array_equal(utilities, 3)

    def test_selects_diverse_vector_under_utility_constraints(self):
        moves = np.empty((2, 2, 9), dtype=object)
        utilities = np.zeros((2, 2, 9), dtype=float)
        moves[:] = "base"
        # Fixed radius index 1 changes both roots but produces only one move.
        moves[:, 1, 1:] = "fixed"
        utilities[:, 1, 1:] = 10
        # Radius index 0 for expert 0 retains utility and adds root-dependent moves.
        moves[0, 0, 1] = "a"
        moves[1, 0, 1] = "b"
        utilities[:, 0, 1] = 10
        vector, fixed, selected = select_radius_vector(
            moves, utilities, selection_roots=1, fixed_index=1
        )
        self.assertEqual(vector[0], 0)
        self.assertGreater(
            selected["candidate_move_diversity_mean"],
            fixed["candidate_move_diversity_mean"],
        )


if __name__ == "__main__":
    unittest.main()
