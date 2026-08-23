import os
import unittest

import cshogi

from train_nnue.accuracy_statistics import (
    exact_mcnemar_p_value,
    paired_summary,
    stratified_summary,
    wilson_interval,
)
from train_nnue.compare_accuracy import compare_results
from train_nnue.eval_accuracy import resolve_engine_option_paths
from train_nnue.run_match import elo_diff, parse_options


class ResolveEngineOptionPathsTest(unittest.TestCase):
    def test_resolves_all_repository_path_options(self):
        root = "/workspace/train-nnue"
        options = {
            "Threads": 1,
            "EvalDir": "bin/eval",
            "ExpertBlendingDir": "tmp/expert_blending_release",
        }

        resolved = resolve_engine_option_paths(options, root)

        self.assertEqual(resolved["Threads"], 1)
        self.assertEqual(resolved["EvalDir"], os.path.join(root, "bin/eval"))
        self.assertEqual(
            resolved["ExpertBlendingDir"],
            os.path.join(root, "tmp/expert_blending_release"),
        )
        self.assertEqual(options["EvalDir"], "bin/eval")

    def test_preserves_absolute_paths(self):
        options = {"EvalDir": "/models/eval"}

        resolved = resolve_engine_option_paths(options, "/workspace/train-nnue")

        self.assertEqual(resolved["EvalDir"], "/models/eval")


class AccuracyStatisticsTest(unittest.TestCase):
    def test_wilson_interval(self):
        interval = wilson_interval(644, 1000)

        self.assertAlmostEqual(interval["lower"], 0.613824, places=6)
        self.assertAlmostEqual(interval["upper"], 0.673074, places=6)

    def test_exact_mcnemar_matches_research_report(self):
        self.assertAlmostEqual(exact_mcnemar_p_value(89, 78), 0.439129, places=6)

    def test_paired_counts_and_accuracy_difference(self):
        summary = paired_summary(
            [True, True, False, False],
            [True, False, True, False],
        )

        self.assertEqual(summary["both_correct"], 1)
        self.assertEqual(summary["baseline_only"], 1)
        self.assertEqual(summary["candidate_only"], 1)
        self.assertEqual(summary["both_wrong"], 1)
        self.assertEqual(summary["accuracy_difference"], 0.0)
        self.assertEqual(summary["exact_mcnemar_p_value"], 1.0)

    def test_strata_include_eval_entering_king_and_optional_game_ply(self):
        records = [
            {
                "sfen": cshogi.STARTING_SFEN,
                "bestmove": "7g7f",
                "eval": 10,
                "game_ply": 20,
            },
            {
                "sfen": "4K4/9/9/9/9/9/9/9/4k4 b - 1",
                "bestmove": "5a5b",
                "eval": -3000,
                "game_ply": 120,
            },
        ]

        summary = stratified_summary(records, [True, False])

        self.assertTrue(summary["game_ply"]["available"])
        self.assertEqual(
            summary["game_ply"]["groups"]["early_1_40"]["candidate"]["total"],
            1,
        )
        self.assertEqual(
            summary["absolute_teacher_eval"]["groups"]["winning_2001_plus"]
            ["candidate"]["matches"],
            0,
        )
        self.assertEqual(
            summary["entering_king"]["groups"]["entering_king"]["candidate"]
            ["total"],
            1,
        )

    def test_missing_game_ply_is_explicit(self):
        records = [{"sfen": cshogi.STARTING_SFEN, "bestmove": "7g7f", "eval": 0}]

        summary = stratified_summary(records, [True])

        self.assertFalse(summary["game_ply"]["available"])


class CompareAccuracyTest(unittest.TestCase):
    def test_compares_aligned_results(self):
        records = [
            {"sfen": cshogi.STARTING_SFEN, "bestmove": "7g7f", "eval": 0}
        ]
        baseline = {
            "config": {"engine_options": {"Threads": 1}, "go_params": {"nodes": 1}},
            "details": [
                {
                    "index": 0,
                    "sfen": cshogi.STARTING_SFEN,
                    "expected": "7g7f",
                    "match": False,
                }
            ]
        }
        candidate = {
            "config": {"engine_options": {"Threads": 1}, "go_params": {"nodes": 1}},
            "details": [
                {
                    "index": 0,
                    "sfen": cshogi.STARTING_SFEN,
                    "expected": "7g7f",
                    "match": True,
                }
            ]
        }

        comparison = compare_results(baseline, candidate, records)

        self.assertEqual(comparison["paired"]["candidate_only"], 1)
        self.assertEqual(comparison["paired"]["accuracy_difference"], 1.0)

    def test_rejects_misaligned_results(self):
        records = [
            {"sfen": cshogi.STARTING_SFEN, "bestmove": "7g7f", "eval": 0}
        ]
        detail = {
            "index": 0,
            "sfen": cshogi.STARTING_SFEN,
            "expected": "2g2f",
            "match": True,
        }

        result = {
            "config": {"engine_options": {"Threads": 1}, "go_params": {"nodes": 1}},
            "details": [detail],
        }
        with self.assertRaisesRegex(ValueError, "not aligned"):
            compare_results(result, result, records)

    def test_rejects_different_search_protocols(self):
        result = {
            "config": {"engine_options": {"Threads": 1}, "go_params": {"nodes": 1}},
            "details": [],
        }
        candidate = {
            "config": {"engine_options": {"Threads": 1}, "go_params": {"nodes": 2}},
            "details": [],
        }

        with self.assertRaisesRegex(ValueError, "different go_params"):
            compare_results(result, candidate, [])


class MatchStatisticsTest(unittest.TestCase):
    def test_elo_is_zero_for_equal_score(self):
        elo, standard_error = elo_diff(40, 40, 20)

        self.assertAlmostEqual(elo, 0.0)
        self.assertGreater(standard_error, 0.0)

    def test_engine_options_allow_values_containing_equals(self):
        self.assertEqual(
            parse_options("Threads=1,Command=tool --arg=a=b"),
            {"Threads": "1", "Command": "tool --arg=a=b"},
        )


if __name__ == "__main__":
    unittest.main()
