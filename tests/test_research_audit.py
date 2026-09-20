import math
import unittest

from scripts.audit_research_20260920 import history_probe, paired_summary


def pair(index, black, white, ply=1):
    return [dict(sfen=f"position{index} b - {ply}", engine1_color=color,
                 engine1_result=result)
            for color, result in (("black", black), ("white", white))]


class ResearchAuditTest(unittest.TestCase):
    def test_history_probe_accepts_current_runner_protocol(self):
        probe = history_probe()
        self.assertEqual(probe["plies"], 6)
        self.assertEqual(probe["positions_with_move_history"], 5)

    def test_color_cancellation_has_zero_pair_variance(self):
        result = paired_summary(pair(0, "win", "loss") + pair(1, "loss", "win"))
        self.assertEqual(result["pair_elo_95"], [0, 0])

    def test_correlated_games_use_pair_sample_variance(self):
        result = paired_summary(pair(0, "win", "win") + pair(1, "loss", "loss"))
        self.assertAlmostEqual(result["pair_elo_se"], 800 / math.log(10))

    def test_move_number_does_not_hide_duplicate_position(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_summary(pair(0, "win", "loss") + pair(0, "win", "loss", ply=9))

    def test_incomplete_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete"):
            paired_summary(pair(0, "win", "loss") + pair(1, "win", "loss")[:1])


if __name__ == "__main__":
    unittest.main()
