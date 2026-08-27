import unittest

import numpy as np

from train_nnue.root_search_statistics import (
    parse_usi_multipv_score,
    statistics_from_scores,
)


class RootSearchStatisticsTest(unittest.TestCase):
    def test_parse_cp_and_mate(self):
        self.assertEqual(
            parse_usi_multipv_score("info depth 3 multipv 2 score cp -45 pv 7g7f"),
            (2, -45.0),
        )
        self.assertEqual(
            parse_usi_multipv_score("info multipv 1 score mate -3 pv 7g7f"),
            (1, -31997.0),
        )

    def test_statistics_are_normalized(self):
        features = statistics_from_scores([300, 100, 0, -100])
        np.testing.assert_allclose(features[:4], [0.15, 0.2, 0.4, np.std([300, 100, 0, -100]) / 1000])
        self.assertGreaterEqual(features[4], 0)
        self.assertLessEqual(features[4], 1)
        self.assertEqual(features[5], 1)


if __name__ == "__main__":
    unittest.main()
