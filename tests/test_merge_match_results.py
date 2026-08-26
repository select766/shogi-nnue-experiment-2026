import unittest

from scripts.merge_match_results import validate_runs


def make_run(details, wins=1, losses=1, draws=0):
    return {
        "engine1": {"path": "candidate"},
        "engine2": {"path": "control"},
        "search": {"nodes": 100000},
        "games": wins + losses + draws,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "details": details,
    }


class ValidateMatchRunsTest(unittest.TestCase):
    def test_accepts_disjoint_color_reversed_openings(self):
        details = [
            {"sfen": "position-a", "engine1_color": "black"},
            {"sfen": "position-a", "engine1_color": "white"},
        ]

        merged = validate_runs(
            [make_run(details)], expected_games=2, require_color_pairs=True
        )

        self.assertEqual(len(merged), 2)

    def test_rejects_duplicate_sfen_color(self):
        first = make_run([
            {"sfen": "position-a", "engine1_color": "black"},
            {"sfen": "position-a", "engine1_color": "white"},
        ])
        second = make_run([
            {"sfen": "position-a", "engine1_color": "black"},
            {"sfen": "position-a", "engine1_color": "white"},
        ])

        with self.assertRaisesRegex(ValueError, "duplicate SFEN/color"):
            validate_runs([first, second], require_color_pairs=True)

    def test_rejects_incomplete_result(self):
        run = make_run([
            {"sfen": "position-a", "engine1_color": "black"},
        ])

        with self.assertRaisesRegex(ValueError, "details length"):
            validate_runs([run])


if __name__ == "__main__":
    unittest.main()
