import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_search_utility_horizons import (  # noqa: E402
    average_ranks,
    apply_logit_bias,
    classify,
    remove_logit_bias,
    spearman_with_ties,
    wilson_interval,
)
from collect_search_utility_teachers import (  # noqa: E402
    build_candidate_biases,
    load_candidate_biases,
    onnxruntime_library_dir,
    search_candidate,
)


class FakeEngine:
    def __init__(self, lines):
        self.lines = iter(lines)
        self.commands = []

    def send(self, command):
        self.commands.append(command)

    def read_until(self, predicate):
        while True:
            line = next(self.lines)
            if predicate(line):
                return line


class SearchUtilityHorizonStatisticsTest(unittest.TestCase):
    def test_onnxruntime_library_resolution(self):
        expected = REPO_ROOT / "YaneuraOu/extra/onnxruntime/linux/current/lib"
        self.assertEqual(
            expected.resolve(),
            onnxruntime_library_dir(REPO_ROOT / "bin/YaneuraOu-expert-blending"),
        )

    def test_average_ranks_preserves_ties(self):
        np.testing.assert_allclose(average_ranks([30, 10, 10, 20]), [3, 0.5, 0.5, 2])

    def test_spearman_with_ties(self):
        self.assertAlmostEqual(1.0, spearman_with_ties([0, 1, 1, 3], [5, 8, 8, 9]))
        self.assertIsNone(spearman_with_ties([1, 1], [1, 2]))

    def test_logit_bias_round_trip(self):
        base = np.asarray([0.1, 0.2, 0.3, 0.4])
        bias = np.asarray([0.0, 1.0, -0.5, 0.2])
        observed = apply_logit_bias(base, bias)
        np.testing.assert_allclose(base, remove_logit_bias(observed, bias), atol=1e-12)

    def test_candidate_search_drains_after_bestmove_without_overwriting_gate(self):
        engine = FakeEngine([
            "info string blending_weight=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2]",
            "bestmove 7g7f",
            "info string blending_weight=[0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]",
            "readyok",
        ])
        move, gate = search_candidate(engine, "fake", 100, [0] * 8, True)
        self.assertEqual("7g7f", move)
        np.testing.assert_allclose([0.1] * 6 + [0.2, 0.2], gate)
        self.assertEqual("isready", engine.commands[-1])

    def test_candidate_bias_geometries_have_equal_count(self):
        axis = build_candidate_biases(4, "positive-axis", 2.0)
        contrast = build_candidate_biases(4, "cyclic-contrast", 2.0)
        self.assertEqual(5, len(axis))
        self.assertEqual(5, len(contrast))
        np.testing.assert_array_equal([2, -2, 0, 0], contrast[1])
        np.testing.assert_array_equal([0, 0, 0, 0], contrast[0])

    def test_loads_named_candidate_biases_and_prepends_base(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "biases.json"
            path.write_text(json.dumps({"directions": [
                {"name": "joint", "bias": [1, 1, -1, -1]},
            ]}))
            names, biases = load_candidate_biases(path, 4)
        self.assertEqual(["base", "joint"], names)
        np.testing.assert_array_equal([0, 0, 0, 0], biases[0])
        np.testing.assert_array_equal([1, 1, -1, -1], biases[1])

    def test_rejects_duplicate_candidate_bias_names(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "biases.json"
            path.write_text(json.dumps({"directions": [
                {"name": "same", "bias": [1, 0]},
                {"name": "same", "bias": [0, 1]},
            ]}))
            with self.assertRaisesRegex(ValueError, "invalid candidate"):
                load_candidate_biases(path, 2)

    def test_wilson_interval_contains_observed_fraction(self):
        low, high = wilson_interval(60, 100)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.6)

    def test_classify_supported(self):
        primary = {
            "informative_rank_fraction": 0.8,
            "spearman_mean_bootstrap_95": [0.31, 0.7],
            "oracle_winner_set_overlap": {"wilson_95": [0.41, 0.8]},
        }
        long_horizon = {"teacher_changed_wilson_95": [0.06, 0.2]}
        self.assertEqual("supported", classify(primary, long_horizon))

    def test_classify_inconclusive(self):
        primary = {
            "informative_rank_fraction": 0.8,
            "spearman_mean_bootstrap_95": [0.2, 0.7],
            "oracle_winner_set_overlap": {"wilson_95": [0.3, 0.8]},
        }
        long_horizon = {"teacher_changed_wilson_95": [0.04, 0.2]}
        self.assertEqual("inconclusive", classify(primary, long_horizon))


if __name__ == "__main__":
    unittest.main()
